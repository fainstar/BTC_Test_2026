# BTC 5 分鐘 K 線預測 Pipeline

> 從 Binance API 抓取 BTCUSDT 5m K 線 → 清理 → 異常值處理 → ROCKET 特徵工程（GPU）→ RandomForest 篩選 → RBF SVM 分類 → 信心過濾
>
> 全流程使用 **Rich** 美化終端輸出（進度條、表格、旋轉動畫）

---

## 專案結構

```
CODE/
├── Fetch/
│   └── fetch_data.py              # ① 資料抓取（Binance API → CSV + SQLite）
├── Data/
│   ├── btc_5m.csv                 # 原始 K 線
│   ├── btc_data.db                # SQLite 持久化
│   ├── btc_5m_clean.csv           # 清理後
│   ├── btc_5m_no_outliers.csv     # Winsorize 後
│   └── btc_5m_rocket_features.parquet  # ROCKET 特徵
├── btc_data_pipeline.py           # ② 資料清理 + 異常值處理
├── rocket_features.py             # ③ ROCKET 特徵工程（GPU）
├── svm_classify.py                # ④ RF 篩選 + SVM 分類
├── requirements.txt
└── README.md
```

---

## 資料流總覽

```mermaid
flowchart LR
    A[/"Binance API"/] -->|"fetch_data.py"| B[("btc_5m.csv\n401K 筆")]
    B -->|"btc_data_pipeline.py\nclean + winsorize"| C[("no_outliers.csv\n100K 筆")]
    C -->|"rocket_features.py\nGPU 卷積"| D[("rocket.parquet\n99K × 8192")]
    D -->|"svm_classify.py\nRF→Nystroem→SVM"| E["📊 分類報表"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#533483,stroke:#fff,color:#fff
    style C fill:#533483,stroke:#fff,color:#fff
    style D fill:#533483,stroke:#fff,color:#fff
    style E fill:#e94560,stroke:#fff,color:#fff
```

### 維度變化

```
原始 401K×12 → 清理 100K×6 → ROCKET 99K×8192 → RF篩選 ×1024 → Nystroem ×4096 → 漲/跌 + 信心分數
```

### 執行順序

```bash
python Fetch/fetch_data.py         # ① 抓取原始資料（只需執行一次）
python btc_data_pipeline.py        # ② 清理 + Winsorize
python rocket_features.py          # ③ ROCKET 特徵萃取（需要 GPU + CuPy）
python svm_classify.py             # ④ 訓練 + 評估 + 信心過濾
```

---

## 各檔案功能說明

### 1. `Fetch/fetch_data.py` — 資料抓取

| 項目 | 說明 |
|------|------|
| **用途** | 從 Binance REST API 抓取 BTCUSDT 5 分鐘 K 線歷史資料 |
| **輸入** | Binance `/api/v3/klines` 端點 |
| **輸出** | `Data/btc_5m.csv`（CSV 匯出）、`Data/btc_data.db`（SQLite 持久化） |
| **抓取量** | 預設 100,000 根 K 棒（約 347 天） |

**核心函式：**

| 函式 | 功能 |
|------|------|
| `fetch_5m_btcusdt_latest_n()` | 從最新時間往回追溯，每次 1,000 筆分頁抓取 |
| `save_to_sqlite()` | 以 `open_time` 為主鍵存入 SQLite，自動去重 |
| `load_from_sqlite()` | 從 SQLite 讀取，支援 `limit` 參數取最新 N 筆 |

---

### 2. `btc_data_pipeline.py` — 資料清理 + 異常值處理

| 項目 | 說明 |
|------|------|
| **用途** | 整合清理與 Winsorize 的功能 |
| **輸入** | `Data/btc_5m.csv` |
| **輸出** | `Data/btc_5m_clean.csv` → `Data/btc_5m_no_outliers.csv` |
| **類別** | `BTCDataCleaner` |
| **Rich 輸出** | `console.status()` 載入動畫、Winsorize 調整結果 `Table` |

**流程：**

1. **清理**（`clean()` 方法）
   - 只保留 `open_time, open, high, low, close, volume` 六個欄位
   - 以 `open_time` 去重，按時間排序
   - 數值欄位轉 `float`
   - 401,000 筆 → 100,074 筆

2. **Winsorize**（`winsorize()` 方法）
   - 對 OHLC 四欄做**滾動窗口 IQR 補值**（不移除列）
   - 窗口 = 2,016 根（7 天），IQR 倍數 k = 2.0
   - 超出 $[Q_1 - 2 \cdot IQR, \; Q_3 + 2 \cdot IQR]$ 的值被 clip 到邊界
   - 每欄約調整 ~6,600 筆
   - OHLCV 格式化為小數點四位

---

### 3. `rocket_features.py` — ROCKET 特徵工程（CuPy GPU）

| 項目 | 說明 |
|------|------|
| **用途** | 以隨機卷積核（ROCKET）萃取時序特徵 |
| **輸入** | `Data/btc_5m_no_outliers.csv` |
| **輸出** | `Data/btc_5m_rocket_features.parquet` |
| **硬體需求** | NVIDIA GPU + CuPy |
| **Rich 輸出** | `Progress` 進度條（滑動窗口建立 + Kernel 套用）、Kernel 分佈 `Table`、完成摘要 `Table` |

**關鍵參數：**

| 參數 | 值 | 說明 |
|------|------|------|
| `N_KERNELS` | 4,096 | 隨機卷積核數量 |
| `WIN_LEN_MIN` | 6 | 最短窗口（30 分鐘） |
| `WIN_LEN_MAX` | 864 | 最長窗口（3 天） |
| `CHANNELS` | OHLCV (5 維) | 輸入通道 |
| 輸出維度 | 8,192 | 每核 2 個特徵（max + PPV） |

**核心演算法：**

1. **Kernel 生成**：每個核隨機抽取窗口長度（log-uniform）、核長度（7/9/11）、dilation、padding、通道
2. **GPU 卷積**：使用 CuPy 逐 weight 切片累加進行 dilated convolution
3. **特徵提取**：每核產生 `max`（最大值）和 `PPV`（正值比例）兩個統計量
4. **標準化**：每個滑動窗口內做 Z-score 正規化
5. **標籤**：
   - `label`：下 1 根 close > 當前 close → 1（漲）
   - `label_2`：下 2 根 close > 當前 close → 1（漲）

---

### 4. `svm_classify.py` — 分類模型 + 信心過濾

| 項目 | 說明 |
|------|------|
| **用途** | RandomForest 特徵篩選 → Nystroem RBF 近似 → LinearSVC 分類 |
| **輸入** | `Data/btc_5m_rocket_features.parquet` |
| **輸出** | Rich 終端報表（Accuracy / Precision / Recall / F1 / Confusion Matrix） |
| **Rich 輸出** | `console.status()` 訓練動畫、資料概覽 / 切割 / RF 篩選 / 模型指標 / 混淆矩陣 / 信心門檻 / 摘要比較 `Table`、Classification Report `Panel` |

**關鍵參數：**

| 參數 | 值 | 說明 |
|------|------|------|
| `RF_N_ESTIMATORS` | 1,024 | RandomForest 估計器數量 |
| `N_SELECT` | 1,024 | RF 篩選後保留的特徵數 |
| `NYSTROEM_N_COMP` | 4,096 | Nystroem RBF 近似維度 |
| `C` | 0.01 | SVM 正則化（低 C = 強正則化） |
| `class_weight` | `'balanced'` | 自動平衡漲跌比例 |

**流程：**

1. 時間序列 80/20 切割（不打亂順序）
2. `StandardScaler` 標準化
3. RandomForest 訓練 → 取 Top-1024 重要特徵（8,192 → 1,024 維）
4. `Nystroem(kernel="rbf")` 近似核映射（1,024 → 4,096 維）
5. `LinearSVC` 訓練與預測
6. **信心過濾**：依 `decision_function` 距離超平面的距離，在不同門檻下顯示精確率 vs 覆蓋率

---

## 中間檔案一覽

| 檔案 | 大小（約） | 列數 | 產生者 |
|------|-----------|------|--------|
| `Data/btc_5m.csv` | ~50 MB | 401,000 | `Fetch/fetch_data.py` |
| `Data/btc_data.db` | ~55 MB | 100,000 | `Fetch/fetch_data.py` |
| `Data/btc_5m_clean.csv` | ~8 MB | 100,074 | `btc_data_pipeline.py` |
| `Data/btc_5m_no_outliers.csv` | ~8 MB | 100,074 | `btc_data_pipeline.py` |
| `Data/btc_5m_rocket_features.parquet` | ~500 MB | 99,213 | `rocket_features.py` |

---

## 優化建議

### 特徵工程層面

| # | 建議 | 預期效果 | 難度 |
|---|------|---------|------|
| 1 | **加入金融技術指標**（RSI、MACD、Bollinger Bands、ATR）作為額外通道或獨立特徵 | ROCKET 只看原始 OHLCV，金融因子能提供更高層次的結構資訊 | ★★☆ |
| 2 | **多時間尺度融合**：同時用 5m、15m、1h K 線產生 ROCKET 特徵，再拼接 | 捕捉跨時間尺度的動態 | ★★★ |
| 3 | **滑動窗口建構向量化**：目前 `rocket_features.py` 用 Python for-loop 逐樣本建窗口，改用 `cp.lib.stride_tricks.as_strided` 可大幅加速 | 窗口建構時間預計從數十秒降至 <1 秒 | ★★☆ |
| 4 | **MiniROCKET 替代**：使用固定長度（9）、確定性 dilation、PPV-only，kernel 數可大幅減少且效果相當 | 速度提升 10～50 倍，特徵維度減半 | ★★☆ |

### 標籤設計層面

| # | 建議 | 預期效果 |
|---|------|---------|
| 5 | **三分類**：將漲幅 < 0.05% 歸為「持平」，只訓練漲/跌兩個明確方向 | 消除雜訊標籤，提高模型信號品質 |
| 6 | **回歸目標**：改預測未來 N 根的收益率，再據此制定交易策略 | 保留幅度資訊，避免二元分類丟失收益大小 |
| 7 | **延長預測窗口**：預測 6 根（30 分鐘）或 12 根（1 小時）後的方向 | 更長的窗口通常有更高的訊噪比 |

### 模型層面

| # | 建議 | 預期效果 |
|---|------|---------|
| 8 | **LightGBM / XGBoost 替代 SVM**：原生支援特徵重要度、高維稀疏資料處理更好 | 通常在表格型資料上優於 SVM |
| 9 | **Stacking 集成**：RF + SVM + LightGBM 的預測結果作為 meta-features，再用 Logistic Regression 整合 | 利用模型多樣性提升泛化 |
| 10 | **CalibratedClassifierCV**：對 SVM 輸出做概率校準，使信心分數更可靠 | 讓信心過濾的門檻更有實際概率意義 |

### 驗證與回測層面

| # | 建議 | 預期效果 |
|---|------|---------|
| 11 | **Walk-forward 驗證**：用滾動窗口替代固定 80/20 切割，模擬真實交易條件 | 更貼近實際部署的效能估計 |
| 12 | **加入交易成本模擬**：考慮手續費（0.1%）、滑價後計算淨收益 | 判斷策略是否真正可獲利 |
| 13 | **計算 Sharpe Ratio / Max Drawdown**：從分類結果推算策略收益曲線 | 更全面的風險收益評估 |

### 工程層面

| # | 建議 | 預期效果 |
|---|------|---------|
| 14 | **統一 Pipeline 類別**：將全部流程封裝為單一 class，一行 `Pipeline().run()` 跑完 | 簡化使用、方便參數調優 |
| 15 | **參數配置檔**：將所有超參數抽至 `config.yaml`，腳本讀取配置 | 不同實驗間快速切換，避免改程式碼 |
| 16 | **模型序列化**：用 `joblib.dump()` 儲存訓練好的 scaler / RF / Nystroem / SVC | 推論時不需重新訓練 |
| 17 | **增量資料更新**：`fetch_data.py` 支援只抓取 DB 中最新時間之後的新資料 | 每日更新只需抓數百筆而非全量 |

---

## 環境需求

```
Python 3.11
numpy, pandas, scikit-learn
cupy (需 NVIDIA GPU + CUDA)
requests (Binance API)
pyarrow (Parquet 讀寫)
rich (終端美化輸出)
```

## 硬體

- GPU：NVIDIA GeForce RTX 3080（ROCKET 特徵工程加速）
- ROCKET 4,096 kernels + 30k 樣本約需 2～3 GB GPU 記憶體
