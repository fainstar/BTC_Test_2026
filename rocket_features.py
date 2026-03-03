"""
rocket_features.py — ROCKET 特徵工程模組（CuPy GPU 加速版）

使用 ROCKET (RandOm Convolutional KErnel Transform) 對 K 線資料進行特徵萃取。
全部卷積運算在 GPU 上透過 CuPy 執行。

設定：
- 輸入窗口長度 : 每個 kernel 隨機抽取（12～576 根，約 1 小時～2 天）
- Kernel 數量   : 1,000
- 輸入通道      : open, high, low, close, volume（5 維）
- 每 kernel 輸出 2 個特徵：max、PPV（正值比例）
- 總特徵數       : 1,000 × 2 = 2,000
- 標籤           : 下一根 close 是否 > 當前 close（1=漲, 0=跌/平）

參考：Dempster et al. (2020) "ROCKET: Exceptionally fast and accurate time
series classification using random convolutional kernels."
"""
import numpy as np
import cupy as cp
import pandas as pd
from pathlib import Path
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich import box

console = Console()

# ─── 參數 ───────────────────────────────────────────────────────────────────
INPUT_FILE     = "Data/btc_5m_no_outliers.csv"
OUTPUT_FILE    = "Data/btc_5m_rocket_features.parquet"
WIN_LEN_MIN    = 6           # 最短窗口
WIN_LEN_MAX    = 864          # 最長窗口
N_KERNELS      = 4096        # kernel 數量
CHANNELS       = ["open", "high", "low", "close", "volume"]
SEED           = 42

# ─── ROCKET 核心實作 ─────────────────────────────────────────────────────────

def _generate_kernels(n_kernels: int, win_len_min: int, win_len_max: int,
                      n_channels: int, rng: np.random.Generator):
    """
    產生 n_kernels 個隨機卷積核，每個核包含：
      win_len   : 從 [win_len_min, win_len_max] 隨機抽取（log-uniform）
      length    : 從 {7, 9, 11} 隨機選取
      weights   : N(0,1) 零均值化
      bias      : U(-1,1)
      dilation  : 2^k，使卷積感受野不超過該 kernel 的 win_len
      padding   : 0 或 (length//2 * dilation)（各 50%）
      channel   : 隨機選擇一個輸入通道
    """
    candidate_lengths = np.array([7, 9, 11])
    kernels = []
    for _ in range(n_kernels):
        # log-uniform 抽取窗口長度，讓短/中/長窗口都有合理覆蓋
        win_len = int(np.exp(rng.uniform(
            np.log(win_len_min), np.log(win_len_max))))
        win_len = max(win_len, 12)  # 至少 12

        length = rng.choice(candidate_lengths)
        # 確保 kernel 感受野 <= win_len
        max_dilation_exp = np.log2(max((win_len - 1) / (length - 1), 1))
        dilation = int(2 ** rng.uniform(0, max_dilation_exp))
        padding = 0 if rng.random() < 0.5 else (length // 2) * dilation
        weights = rng.standard_normal(length)
        weights -= weights.mean()          # 零均值化
        bias = rng.uniform(-1, 1)
        channel = rng.integers(0, n_channels)
        kernels.append({
            "win_len": win_len,
            "length": length,
            "weights": weights,
            "bias": bias,
            "dilation": dilation,
            "padding": padding,
            "channel": channel,
        })
    return kernels


def _apply_kernel_batch_gpu(X_ch: cp.ndarray, kernel: dict) -> tuple[cp.ndarray, cp.ndarray]:
    """
    GPU 向量化批次卷積：對全部 N 個樣本同時套用 dilated kernel。

    核心技巧：逐 weight 切片累加，避免 3D 中間 tensor。
    全部運算在 GPU 上完成（CuPy）。

    X_ch : CuPy ndarray, shape (N, T) float32
    回傳 (max_vals, ppv_vals)，各 shape (N,) CuPy ndarray
    """
    w     = cp.asarray(kernel["weights"], dtype=cp.float32)   # (k_len,)
    b     = cp.float32(kernel["bias"])
    d     = int(kernel["dilation"])
    pad   = int(kernel["padding"])
    k_len = len(w)

    N, T = X_ch.shape

    if pad > 0:
        X_pad = cp.pad(X_ch, ((0, 0), (pad, pad)),
                       mode="constant", constant_values=cp.float32(0))
    else:
        X_pad = X_ch

    T_pad = X_pad.shape[1]
    n_pos = T_pad - (k_len - 1) * d
    if n_pos <= 0:
        return cp.zeros(N, dtype=cp.float32), cp.zeros(N, dtype=cp.float32)

    # 逐 weight 累加 dilated 卷積輸出
    output = cp.full((N, n_pos), b, dtype=cp.float32)
    for i in range(k_len):
        output += w[i] * X_pad[:, i * d: i * d + n_pos]

    return output.max(axis=1), (output > 0).mean(axis=1)


def apply_rocket(X_gpu: cp.ndarray, kernels: list) -> np.ndarray:
    """
    X_gpu : CuPy ndarray, shape (N, C, T)
    回傳 NumPy ndarray, shape (N, 2 * n_kernels) — 結果搬回 CPU

    所有 kernel 在 GPU 上依序運算（GPU 本身就是大規模並行，
    不需要 ThreadPoolExecutor）。
    """
    N, C, T = X_gpu.shape
    n_k = len(kernels)
    # 在 GPU 上分配 features
    features_gpu = cp.zeros((N, 2 * n_k), dtype=cp.float32)

    t0 = time.perf_counter()
    console.print(f"  [bold green][GPU][/] 套用 {n_k} 個 kernel 到 {N} 個樣本（{C} 通道 × {T} 時步）...")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Kernels", total=n_k)
        for ki, kernel in enumerate(kernels):
            k_win = kernel["win_len"]
            ch    = kernel["channel"]
            x_ch = X_gpu[:, ch, T - k_win:]
            mx, ppv = _apply_kernel_batch_gpu(x_ch, kernel)
            features_gpu[:, 2 * ki]     = mx
            features_gpu[:, 2 * ki + 1] = ppv
            progress.advance(task)

            if (ki + 1) % 200 == 0:
                cp.cuda.Stream.null.synchronize()

    cp.cuda.Stream.null.synchronize()
    console.print(f"  [bold green]✓[/] [GPU] 完成，共耗時 [bold]{time.perf_counter()-t0:.1f}[/] 秒")

    # 搬回 CPU
    return cp.asnumpy(features_gpu)


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def build_rocket_features(
    input_file  : str  = INPUT_FILE,
    output_file : str  = OUTPUT_FILE,
    win_len_min : int  = WIN_LEN_MIN,
    win_len_max : int  = WIN_LEN_MAX,
    n_kernels   : int  = N_KERNELS,
    channels    : list = CHANNELS,
    seed        : int  = SEED,
) -> pd.DataFrame:

    console.rule("[bold magenta]ROCKET 特徵工程[/]")

    rng = np.random.default_rng(seed)
    with console.status("[bold green]載入資料中..."):
        df  = pd.read_csv(input_file, parse_dates=["open_time"])
        df[channels] = df[channels].astype(float)
    n_ch = len(channels)
    console.print(f"  ✓ 載入資料：[bold]{len(df):,}[/] 筆，通道：{channels}")

    # ── Step 1：先產生 kernels（需要最大窗口長度）───────────────────────────
    console.print(f"  產生 [bold]{n_kernels}[/] 個隨機 kernel（窗口 {win_len_min}～{win_len_max}）...")
    kernels = _generate_kernels(n_kernels, win_len_min, win_len_max, n_ch, rng)
    max_win = max(k["win_len"] for k in kernels)
    min_win = min(k["win_len"] for k in kernels)
    avg_win = int(np.mean([k["win_len"] for k in kernels]))

    kern_table = Table(title="Kernel 窗口分佈", box=box.SIMPLE_HEAD)
    kern_table.add_column("統計", style="bold")
    kern_table.add_column("數值", justify="right")
    kern_table.add_row("最短窗口", str(min_win))
    kern_table.add_row("平均窗口", str(avg_win))
    kern_table.add_row("最長窗口", str(max_win))
    console.print(kern_table)

    # ── Step 2：以最大窗口建立滑動窗口 ────────────────────────────────────────
    total     = len(df)
    n_samples = total - max_win - 1      # 預留 2 根給 label_2
    console.print(f"  樣本數：[bold]{n_samples:,}[/]（最大窗口={max_win}，每步滑動 1 根）")

    # 在 GPU 上建立所有窗口並做 z-score 標準化
    raw = cp.asarray(df[channels].values, dtype=cp.float32)  # (total, n_ch) → GPU

    t0 = time.perf_counter()
    X_gpu = cp.zeros((n_samples, n_ch, max_win), dtype=cp.float32)
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("建立滑動窗口", total=n_samples)
        for i in range(n_samples):
            window = raw[i: i + max_win].T  # (C, max_win)
            mu  = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True) + 1e-8
            X_gpu[i] = (window - mu) / std
            progress.advance(task)

    console.print(f"  ✓ 窗口建立完成，耗時 [bold]{time.perf_counter()-t0:.1f}s[/]  "
                  f"GPU 記憶體: [bold]{X_gpu.nbytes / 1e9:.2f}[/] GB")

    # 標籤 label  ：下 1 根 close > 當前 close → 1
    # 標籤 label_2：下 2 根 close > 當前 close → 1
    labels   = np.zeros(n_samples, dtype=np.int8)
    labels_2 = np.zeros(n_samples, dtype=np.int8)
    close_arr = df["close"].values.astype(np.float64)
    for i in range(n_samples):
        cur_close   = close_arr[i + max_win - 1]
        next_close  = close_arr[i + max_win]
        next2_close = close_arr[i + max_win + 1]
        labels[i]   = 1 if next_close  > cur_close else 0
        labels_2[i] = 1 if next2_close > cur_close else 0

    # ── Step 3：套用 ROCKET（GPU）──────────────────────────────────────────────
    feat_matrix = apply_rocket(X_gpu, kernels)

    # 釋放 GPU 記憶體
    console.print("  釋放 GPU 記憶體...")
    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()
    console.print("  ✓ GPU 記憶體已釋放")

    # ── Step 4：組合輸出 DataFrame ────────────────────────────────────────────
    with console.status(f"[bold green]建立 DataFrame（{n_samples:,} × {2*n_kernels:,}）..."):
        t0 = time.perf_counter()
        col_names = []
        for ki in range(n_kernels):
            col_names += [f"rocket_{ki}_max", f"rocket_{ki}_ppv"]

        feat_df = pd.DataFrame(feat_matrix, columns=col_names)

        # 附上時間戳（對應窗口最後一根的 open_time）與標籤
        feat_df.insert(0, "open_time",
                       df["open_time"].iloc[max_win - 1 : max_win - 1 + n_samples].values)
        feat_df["label"]   = labels
        feat_df["label_2"] = labels_2
    console.print(f"  ✓ DataFrame 建立完成，耗時 [bold]{time.perf_counter()-t0:.1f}s[/]")

    # 儲存為 Parquet（比 CSV 快 10 倍以上、檔案小 5 倍以上）
    with console.status(f"[bold green]儲存至 {output_file}..."):
        t0 = time.perf_counter()
        if output_file.endswith(".parquet"):
            feat_df.to_parquet(output_file, index=False, engine="pyarrow")
        else:
            feat_df.to_csv(output_file, index=False)
        save_time = time.perf_counter() - t0

    pos1 = labels.mean() * 100
    pos2 = labels_2.mean() * 100

    # 最終摘要表
    summary_table = Table(title="ROCKET 特徵工程完成", box=box.DOUBLE_EDGE, show_lines=True)
    summary_table.add_column("項目", style="bold")
    summary_table.add_column("數值", justify="right")
    summary_table.add_row("特徵維度", f"{feat_df.shape[1] - 3:,} 維（{n_kernels:,} kernels × 2）")
    summary_table.add_row("樣本數", f"{len(feat_df):,}")
    summary_table.add_row("label 漲比例", f"{pos1:.2f}%")
    summary_table.add_row("label_2 漲比例", f"{pos2:.2f}%")
    summary_table.add_row("儲存耗時", f"{save_time:.1f}s")
    summary_table.add_row("輸出檔案", output_file)
    console.print(summary_table)

    return feat_df


if __name__ == "__main__":
    build_rocket_features()
