"""
svm_classify.py — RandomForest 特徵篩選 + RBF SVM 分類模組

流程：
1. 載入 btc_5m_rocket_features.csv
2. 時間序列切割 train / test（80% / 20%，不打亂順序）
3. StandardScaler 標準化
4. RandomForest 篩選 Top-N 重要特徵
5. Nystroem RBF 近似核映射
6. LinearSVC 訓練
7. 輸出 Accuracy / Precision / Recall / F1 / Confusion Matrix
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

INPUT_FILE       = "Data/btc_5m_rocket_features.parquet"
REPORT_DIR       = "Report"
TRAIN_RATIO      = 0.8
N_SELECT         = 1024     # RF 篩選後保留的特徵數
NYSTROEM_N_COMP  = 4096    # Nystroem 近似維度
RF_N_ESTIMATORS  = 1024     # RF 估計器數量
CONF_THRESHOLDS  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # 信心過濾門檻


def _train_and_eval(X_train, X_test, y_train, y_test, label_name: str):
    """對單一標籤進行 Nystroem + LinearSVC 訓練與評估。"""
    n_up_train = y_train.sum()
    n_up_test  = y_test.sum()

    console.rule(f"[bold cyan]目標：{label_name}[/]")

    # 資料分佈表
    dist_table = Table(title="資料分佈", box=box.SIMPLE_HEAD, show_lines=False)
    dist_table.add_column("集合", style="bold")
    dist_table.add_column("漲", justify="right", style="green")
    dist_table.add_column("跌", justify="right", style="red")
    dist_table.add_column("漲比例", justify="right", style="yellow")
    dist_table.add_row("Train", str(n_up_train), str(len(y_train)-n_up_train),
                        f"{n_up_train/len(y_train)*100:.2f}%")
    dist_table.add_row("Test",  str(n_up_test),  str(len(y_test)-n_up_test),
                        f"{n_up_test/len(y_test)*100:.2f}%")
    console.print(dist_table)

    # Nystroem RBF 近似
    with console.status(f"[bold green]Nystroem RBF（n_components={NYSTROEM_N_COMP}）映射中..."):
        t0 = time.perf_counter()
        nys = Nystroem(kernel="rbf", gamma=1.0 / X_train.shape[1],
                       n_components=NYSTROEM_N_COMP, random_state=42)
        X_tr_rbf = nys.fit_transform(X_train)
        X_te_rbf = nys.transform(X_test)
    console.print(f"  ✓ 映射完成，耗時 [bold]{time.perf_counter()-t0:.1f}s[/]  "
                  f"維度 {X_train.shape[1]} → {X_tr_rbf.shape[1]}")

    # LinearSVC
    with console.status("[bold green]訓練 LinearSVC（max_iter=10240）..."):
        t0 = time.perf_counter()
        clf = LinearSVC(
            dual="auto",
            max_iter=10240,
            random_state=42,
            C=0.01,
            class_weight='balanced'
        )
        clf.fit(X_tr_rbf, y_train)
    console.print(f"  ✓ 訓練完成，耗時 [bold]{time.perf_counter()-t0:.1f}s[/]")

    y_pred_train = clf.predict(X_tr_rbf)
    y_pred_test  = clf.predict(X_te_rbf)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test)
    test_rec  = recall_score(y_test, y_pred_test)
    test_f1   = f1_score(y_test, y_pred_test)
    cm        = confusion_matrix(y_test, y_pred_test)

    # 指標表
    metric_table = Table(title="模型指標", box=box.ROUNDED, show_lines=True)
    metric_table.add_column("指標", style="bold")
    metric_table.add_column("數值", justify="right")
    metric_table.add_row("Train Accuracy",  f"{train_acc:.4f}")
    metric_table.add_row("Test  Accuracy",  f"[bold]{test_acc:.4f}[/]")
    metric_table.add_row("Test  Precision", f"{test_prec:.4f}  [dim](漲的命中率)[/]")
    metric_table.add_row("Test  Recall",    f"{test_rec:.4f}  [dim](漲的召回率)[/]")
    metric_table.add_row("Test  F1 Score",  f"[bold cyan]{test_f1:.4f}[/]")
    console.print(metric_table)

    # 混淆矩陣表
    cm_table = Table(title="Confusion Matrix（實際 × 預測）", box=box.ROUNDED, show_lines=True)
    cm_table.add_column("", style="bold")
    cm_table.add_column("預測跌", justify="right", style="red")
    cm_table.add_column("預測漲", justify="right", style="green")
    cm_table.add_row("實際跌", str(cm[0, 0]), str(cm[0, 1]))
    cm_table.add_row("實際漲", str(cm[1, 0]), str(cm[1, 1]))
    console.print(cm_table)

    # Classification Report
    report = classification_report(y_test, y_pred_test,
                                   target_names=["跌/平", "漲"])
    console.print(Panel(report, title="Classification Report", border_style="dim"))

    # ── 信心過濾 (Confidence Thresholding) ────────────────────────────────
    scores_test = clf.decision_function(X_te_rbf)

    console.rule("[bold yellow]信心過濾分析[/]")
    console.print(f"  decision_function 統計："
                  f"min=[cyan]{scores_test.min():.3f}[/]  "
                  f"max=[cyan]{scores_test.max():.3f}[/]  "
                  f"mean=[cyan]{scores_test.mean():.3f}[/]  "
                  f"std=[cyan]{scores_test.std():.3f}[/]")

    conf_table = Table(title="信心門檻篩選", box=box.SIMPLE_HEAD)
    conf_table.add_column("門檻", justify="right", style="bold")
    conf_table.add_column("覆蓋率", justify="right")
    conf_table.add_column("樣本數", justify="right")
    conf_table.add_column("ACC", justify="right")
    conf_table.add_column("Prec(漲)", justify="right", style="green")
    conf_table.add_column("Rec(漲)", justify="right", style="yellow")
    conf_table.add_column("F1(漲)", justify="right", style="cyan")

    conf_results = []
    for thr in CONF_THRESHOLDS:
        mask = np.abs(scores_test) >= thr
        n_pass = mask.sum()
        if n_pass < 10:
            continue
        coverage = n_pass / len(y_test)
        y_t = y_test[mask]
        y_p = y_pred_test[mask]
        acc  = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec  = recall_score(y_t, y_p, zero_division=0)
        f1   = f1_score(y_t, y_p, zero_division=0)
        conf_table.add_row(f"{thr:.2f}", f"{coverage:.1%}", str(n_pass),
                           f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}")
        conf_results.append({"threshold": thr, "coverage": coverage,
                              "n_samples": n_pass, "acc": acc,
                              "precision": prec, "recall": rec, "f1": f1})
    console.print(conf_table)

    return {
        "clf": clf, "nystroem": nys,
        "train_acc": train_acc, "test_acc": test_acc,
        "test_precision": test_prec, "test_recall": test_rec,
        "test_f1": test_f1, "confusion_matrix": cm,
        "confidence": conf_results,
    }


def run_svm(input_file: str = INPUT_FILE,
            train_ratio: float = TRAIN_RATIO) -> dict:

    console.rule("[bold magenta]SVM 分類流程啟動[/]")

    # ── 1. 載入資料 ──────────────────────────────────────────────────────────
    with console.status("[bold green]載入資料中..."):
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file, parse_dates=["open_time"])
    df["open_time"] = pd.to_datetime(df["open_time"])
    feat_cols = [c for c in df.columns if c.startswith("rocket_")]
    X = df[feat_cols].values.astype(np.float32)
    y1 = df["label"].values.astype(np.int8)
    y2 = df["label_2"].values.astype(np.int8)

    info_table = Table(title="資料概覽", box=box.ROUNDED, show_lines=True)
    info_table.add_column("項目", style="bold")
    info_table.add_column("數值", justify="right")
    info_table.add_row("樣本數", f"{len(df):,}")
    info_table.add_row("特徵維度", f"{X.shape[1]:,}")
    info_table.add_row("label 漲比例", f"{y1.sum():,}  ({y1.mean()*100:.2f}%)")
    info_table.add_row("label_2 漲比例", f"{y2.sum():,}  ({y2.mean()*100:.2f}%)")
    console.print(info_table)

    # ── 2. 時間序列切割（不打亂順序）─────────────────────────────────────────
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y1_train, y1_test = y1[:split_idx], y1[split_idx:]
    y2_train, y2_test = y2[:split_idx], y2[split_idx:]

    split_table = Table(title="時間序列切割", box=box.ROUNDED, show_lines=True)
    split_table.add_column("集合", style="bold")
    split_table.add_column("筆數", justify="right")
    split_table.add_column("起始時間")
    split_table.add_column("結束時間")
    split_table.add_row("Train", f"{len(X_train):,}",
                         str(df['open_time'].iloc[0]),
                         str(df['open_time'].iloc[split_idx-1]))
    split_table.add_row("Test",  f"{len(X_test):,}",
                         str(df['open_time'].iloc[split_idx]),
                         str(df['open_time'].iloc[-1]))
    console.print(split_table)

    # ── 3. StandardScaler（用 train fit，transform 兩者）─────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    console.print("  ✓ 特徵已 [bold]StandardScaler[/] 標準化")

    # ── 4. RandomForest 特徵篩選（用 label 訓練，選 Top-N）───────────────────
    with console.status(f"[bold green]訓練 RandomForest（n_estimators={RF_N_ESTIMATORS}）做特徵篩選..."):
        t0 = time.perf_counter()
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=8,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train_sc, y1_train)
        rf_time = time.perf_counter() - t0

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:N_SELECT]
    top_idx.sort()
    selected_names = [feat_cols[i] for i in top_idx]

    rf_table = Table(title="RandomForest 特徵篩選", box=box.ROUNDED, show_lines=True)
    rf_table.add_column("項目", style="bold")
    rf_table.add_column("數值", justify="right")
    rf_table.add_row("耗時", f"{rf_time:.1f}s")
    rf_table.add_row("篩選特徵數", f"Top-{N_SELECT}")
    rf_table.add_row("最高重要度", f"{importances.max():.6f}")
    rf_table.add_row("RF Train ACC", f"{accuracy_score(y1_train, rf.predict(X_train_sc)):.4f}")
    rf_table.add_row("RF Test  ACC", f"{accuracy_score(y1_test, rf.predict(X_test_sc)):.4f}")
    rf_table.add_row("維度變化", f"{X_train_sc.shape[1]} → {N_SELECT}")
    console.print(rf_table)

    X_train_sel = X_train_sc[:, top_idx]
    X_test_sel  = X_test_sc[:, top_idx]

    # ── 5. 分別對 label 和 label_2 訓練 RBF-SVM ─────────────────────────────
    results = {}
    results["label"]   = _train_and_eval(X_train_sel, X_test_sel,
                                         y1_train, y1_test,
                                         "label (下 1 根漲跌)")
    results["label_2"] = _train_and_eval(X_train_sel, X_test_sel,
                                         y2_train, y2_test,
                                         "label_2 (下 2 根漲跌)")

    # ── 6. 對比摘要 ──────────────────────────────────────────────────────────
    summary_table = Table(title="摘要比較", box=box.DOUBLE_EDGE, show_lines=True)
    summary_table.add_column("目標", style="bold")
    summary_table.add_column("Train ACC", justify="right")
    summary_table.add_column("Test ACC", justify="right")
    summary_table.add_column("Test F1", justify="right", style="bold cyan")
    for name, r in results.items():
        summary_table.add_row(name,
                               f"{r['train_acc']:.4f}",
                               f"{r['test_acc']:.4f}",
                               f"{r['test_f1']:.4f}")
    console.print(summary_table)

    results["scaler"] = scaler

    # ── 7. 產生 Markdown 報告 ───────────────────────────────────────────────
    _generate_md_report(results, df, split_idx, rf_time, importances, rf, 
                        X_train_sc, y1_train, X_test_sc, y1_test)

    return results


def _generate_md_report(results: dict, df: pd.DataFrame, split_idx: int,
                        rf_time: float, importances: np.ndarray,
                        rf, X_train_sc, y1_train, X_test_sc, y1_test):
    """\u7522\u751f Markdown \u5831\u544a\u4e26\u5132\u5b58\u81f3 Report/ \u8cc7\u6599\u593e\u3002"""
    Path(REPORT_DIR).mkdir(exist_ok=True)
    now = datetime.now()
    filename = f"report_{now.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = Path(REPORT_DIR) / filename

    lines = []
    lines.append(f"# SVM \u5206\u985e\u5831\u544a")
    lines.append(f"")
    lines.append(f"> \u7522\u751f\u6642\u9593\uff1a{now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # \u8cc7\u6599\u6982\u89bd
    lines.append(f"## \u8cc7\u6599\u6982\u89bd")
    lines.append(f"")
    lines.append(f"| \u9805\u76ee | \u6578\u503c |")
    lines.append(f"|------|------|")
    lines.append(f"| \u6a23\u672c\u6578 | {len(df):,} |")
    lines.append(f"| Train | {split_idx:,} \u7b46 |")
    lines.append(f"| Test | {len(df) - split_idx:,} \u7b46 |")
    lines.append(f"| Train \u6642\u9593 | {df['open_time'].iloc[0]} \uff5e {df['open_time'].iloc[split_idx-1]} |")
    lines.append(f"| Test \u6642\u9593 | {df['open_time'].iloc[split_idx]} \uff5e {df['open_time'].iloc[-1]} |")
    lines.append(f"")

    # \u8d85\u53c3\u6578
    lines.append(f"## \u8d85\u53c3\u6578")
    lines.append(f"")
    lines.append(f"| \u53c3\u6578 | \u503c |")
    lines.append(f"|------|------|")
    lines.append(f"| RF n_estimators | {RF_N_ESTIMATORS} |")
    lines.append(f"| RF \u7be9\u9078\u7279\u5fb5\u6578 | {N_SELECT} |")
    lines.append(f"| Nystroem n_components | {NYSTROEM_N_COMP} |")
    lines.append(f"| SVM C | 0.01 |")
    lines.append(f"| SVM class_weight | balanced |")
    lines.append(f"| RF \u8a13\u7df4\u8017\u6642 | {rf_time:.1f}s |")
    lines.append(f"| RF \u6700\u9ad8\u91cd\u8981\u5ea6 | {importances.max():.6f} |")
    lines.append(f"| RF Train ACC | {accuracy_score(y1_train, rf.predict(X_train_sc)):.4f} |")
    lines.append(f"| RF Test ACC | {accuracy_score(y1_test, rf.predict(X_test_sc)):.4f} |")
    lines.append(f"")

    # \u5404\u6a19\u7c64\u7d50\u679c
    for label_key in ["label", "label_2"]:
        if label_key not in results or label_key == "scaler":
            continue
        r = results[label_key]
        cm = r["confusion_matrix"]
        title = "\u4e0b 1 \u6839\u6f32\u8dcc" if label_key == "label" else "\u4e0b 2 \u6839\u6f32\u8dcc"

        lines.append(f"## {label_key}\uff08{title}\uff09")
        lines.append(f"")

        # \u6307\u6a19
        lines.append(f"### \u6a21\u578b\u6307\u6a19")
        lines.append(f"")
        lines.append(f"| \u6307\u6a19 | \u6578\u503c |")
        lines.append(f"|------|------|")
        lines.append(f"| Train Accuracy | {r['train_acc']:.4f} |")
        lines.append(f"| **Test Accuracy** | **{r['test_acc']:.4f}** |")
        lines.append(f"| Test Precision | {r['test_precision']:.4f} |")
        lines.append(f"| Test Recall | {r['test_recall']:.4f} |")
        lines.append(f"| **Test F1** | **{r['test_f1']:.4f}** |")
        lines.append(f"")

        # \u6df7\u6dc6\u77e9\u9663
        lines.append(f"### Confusion Matrix")
        lines.append(f"")
        lines.append(f"| | \u9810\u6e2c\u8dcc | \u9810\u6e2c\u6f32 |")
        lines.append(f"|------|------|------|")
        lines.append(f"| **\u5be6\u969b\u8dcc** | {cm[0,0]} | {cm[0,1]} |")
        lines.append(f"| **\u5be6\u969b\u6f32** | {cm[1,0]} | {cm[1,1]} |")
        lines.append(f"")

        # \u4fe1\u5fc3\u904e\u6ffe
        if r.get("confidence"):
            lines.append(f"### \u4fe1\u5fc3\u904e\u6ffe")
            lines.append(f"")
            lines.append(f"| \u9580\u6abb | \u8986\u84cb\u7387 | \u6a23\u672c\u6578 | ACC | Prec(\u6f32) | Rec(\u6f32) | F1(\u6f32) |")
            lines.append(f"|------|------|------|------|------|------|------|")
            for c in r["confidence"]:
                lines.append(f"| {c['threshold']:.2f} | {c['coverage']:.1%} | {c['n_samples']} | "
                             f"{c['acc']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |")
            lines.append(f"")

    # \u6458\u8981\u6bd4\u8f03
    lines.append(f"## \u6458\u8981\u6bd4\u8f03")
    lines.append(f"")
    lines.append(f"| \u76ee\u6a19 | Train ACC | Test ACC | Test F1 |")
    lines.append(f"|------|------|------|------|")
    for name in ["label", "label_2"]:
        if name in results:
            r = results[name]
            lines.append(f"| {name} | {r['train_acc']:.4f} | {r['test_acc']:.4f} | **{r['test_f1']:.4f}** |")
    lines.append(f"")

    md_text = "\n".join(lines)
    filepath.write_text(md_text, encoding="utf-8")
    console.print(f"  \u2713 \u5831\u544a\u5df2\u5132\u5b58\u81f3 [bold green]{filepath}[/]")


if __name__ == "__main__":
    run_svm()
