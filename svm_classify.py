"""
svm_classify.py — RandomForest 特徵篩選 + RBF SVM 分類模組

流程：
1. 載入 btc_5m_rocket_features.csv
2. 時間序列切割 train / test（80% / 20%，不打亂順序）
3. StandardScaler 標準化
4. RandomForest 篩選 Top-256 重要特徵
5. Nystroem RBF 近似核映射
6. LinearSVC 訓練
7. 輸出 Accuracy / Precision / Recall / F1 / Confusion Matrix
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import time

INPUT_FILE       = "btc_5m_rocket_features.parquet"
TRAIN_RATIO      = 0.8
N_SELECT         = 1024     # RF 篩選後保留的特徵數
NYSTROEM_N_COMP  = 4096    # Nystroem 近似維度（配合 256 輸入）
RF_N_ESTIMATORS  = 1024     # RF 估計器數量
CONF_THRESHOLDS  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # 信心過濾門檻


def _train_and_eval(X_train, X_test, y_train, y_test, label_name: str):
    """對單一標籤進行 Nystroem + LinearSVC 訓練與評估。"""
    n_up_train = y_train.sum()
    n_up_test  = y_test.sum()
    print(f"\n{'='*55}")
    print(f"  目標：{label_name}")
    print(f"{'='*55}")
    print(f"  Train 漲/跌：{n_up_train}/{len(y_train)-n_up_train}  "
          f"({n_up_train/len(y_train)*100:.2f}% 漲)")
    print(f"  Test  漲/跌：{n_up_test}/{len(y_test)-n_up_test}  "
          f"({n_up_test/len(y_test)*100:.2f}% 漲)")

    # Nystroem RBF 近似
    print(f"  Nystroem RBF（n_components={NYSTROEM_N_COMP}）映射中...")
    t0 = time.perf_counter()
    nys = Nystroem(kernel="rbf", gamma=1.0 / X_train.shape[1],
                   n_components=NYSTROEM_N_COMP, random_state=42)
    X_tr_rbf = nys.fit_transform(X_train)
    X_te_rbf = nys.transform(X_test)
    print(f"  映射完成，耗時 {time.perf_counter()-t0:.1f}s  "
          f"維度 {X_train.shape[1]} → {X_tr_rbf.shape[1]}")

    # LinearSVC
    print(f"  訓練 LinearSVC（max_iter=5000）...")
    t0 = time.perf_counter()
    # 建議修改後的版本
    clf = LinearSVC(
        dual="auto", 
        max_iter=10240,        # 增加疊代次數確保收斂
        random_state=42, 
        C=0.01,                # 核心改動：調低 C 增加正則化，緩解過擬合
        class_weight='balanced' # 核心改動：自動處理標籤不平衡，修正「愛猜漲」的偏差
    )
    clf.fit(X_tr_rbf, y_train)
    print(f"  訓練完成，耗時 {time.perf_counter()-t0:.1f}s")

    y_pred_train = clf.predict(X_tr_rbf)
    y_pred_test  = clf.predict(X_te_rbf)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test)
    test_rec  = recall_score(y_test, y_pred_test)
    test_f1   = f1_score(y_test, y_pred_test)
    cm        = confusion_matrix(y_test, y_pred_test)

    print(f"\n  Train Accuracy : {train_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  Test  Precision: {test_prec:.4f}  (漲的命中率)")
    print(f"  Test  Recall   : {test_rec:.4f}  (漲的召回率)")
    print(f"  Test  F1 Score : {test_f1:.4f}")
    print()
    print(f"  Confusion Matrix (rows=實際, cols=預測):")
    print(f"              預測跌  預測漲")
    print(f"    實際跌    {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"    實際漲    {cm[1,0]:>6}  {cm[1,1]:>6}")
    print()
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred_test,
                                target_names=["跌/平", "漲"]))

    # ── 信心過濾 (Confidence Thresholding) ────────────────────────────────
    scores_test = clf.decision_function(X_te_rbf)   # 距離超平面的帶符號距離
    print(f"  ── 信心過濾分析 ──")
    print(f"  decision_function 統計："
          f"min={scores_test.min():.3f}  max={scores_test.max():.3f}  "
          f"mean={scores_test.mean():.3f}  std={scores_test.std():.3f}")
    print()
    print(f"  {'門檻':>6} {'覆蓋率':>8} {'樣本數':>8} {'ACC':>8} "
          f"{'Prec(漲)':>10} {'Rec(漲)':>10} {'F1(漲)':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

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
        print(f"  {thr:>6.2f} {coverage:>7.1%} {n_pass:>8} {acc:>8.4f} "
              f"{prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
        conf_results.append({"threshold": thr, "coverage": coverage,
                              "n_samples": n_pass, "acc": acc,
                              "precision": prec, "recall": rec, "f1": f1})
    print()

    return {
        "clf": clf, "nystroem": nys,
        "train_acc": train_acc, "test_acc": test_acc,
        "test_precision": test_prec, "test_recall": test_rec,
        "test_f1": test_f1, "confusion_matrix": cm,
        "confidence": conf_results,
    }


def run_svm(input_file: str = INPUT_FILE,
            train_ratio: float = TRAIN_RATIO) -> dict:

    # ── 1. 載入資料 ──────────────────────────────────────────────────────────
    if input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file, parse_dates=["open_time"])
    df["open_time"] = pd.to_datetime(df["open_time"])
    feat_cols = [c for c in df.columns if c.startswith("rocket_")]
    X = df[feat_cols].values.astype(np.float32)
    y1 = df["label"].values.astype(np.int8)
    y2 = df["label_2"].values.astype(np.int8)
    print(f"載入 {len(df)} 筆樣本，特徵維度 {X.shape[1]}")
    print(f"label   分佈：漲={y1.sum()} ({y1.mean()*100:.2f}%)")
    print(f"label_2 分佈：漲={y2.sum()} ({y2.mean()*100:.2f}%)")

    # ── 2. 時間序列切割（不打亂順序）─────────────────────────────────────────
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y1_train, y1_test = y1[:split_idx], y1[split_idx:]
    y2_train, y2_test = y2[:split_idx], y2[split_idx:]
    print(f"\nTrain: {len(X_train)} 筆  |  Test: {len(X_test)} 筆")
    print(f"Train 時間：{df['open_time'].iloc[0]} ～ {df['open_time'].iloc[split_idx-1]}")
    print(f"Test  時間：{df['open_time'].iloc[split_idx]} ～ {df['open_time'].iloc[-1]}")

    # ── 3. StandardScaler（用 train fit，transform 兩者）─────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("\n特徵已 StandardScaler 標準化")

    # ── 4. RandomForest 特徵篩選（用 label 訓練，選 Top-256）─────────────────
    print(f"\n訓練 RandomForest（n_estimators={RF_N_ESTIMATORS}）做特徵篩選...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=8,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train_sc, y1_train)
    rf_time = time.perf_counter() - t0
    print(f"  RF 訓練完成，耗時 {rf_time:.1f}s")

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:N_SELECT]
    top_idx.sort()   # 保持原始順序
    selected_names = [feat_cols[i] for i in top_idx]

    print(f"  篩選 Top-{N_SELECT} 特徵（重要度最高: {importances[np.argsort(importances)[-1]]:.6f}）")
    print(f"  RF Train ACC: {accuracy_score(y1_train, rf.predict(X_train_sc)):.4f}")
    print(f"  RF Test  ACC: {accuracy_score(y1_test, rf.predict(X_test_sc)):.4f}")

    X_train_sel = X_train_sc[:, top_idx]
    X_test_sel  = X_test_sc[:, top_idx]
    print(f"  特徵維度 {X_train_sc.shape[1]} → {X_train_sel.shape[1]}")

    # ── 5. 分別對 label 和 label_2 訓練 RBF-SVM ─────────────────────────────
    results = {}
    results["label"]   = _train_and_eval(X_train_sel, X_test_sel,
                                         y1_train, y1_test,
                                         "label (下 1 根漲跌)")
    results["label_2"] = _train_and_eval(X_train_sel, X_test_sel,
                                         y2_train, y2_test,
                                         "label_2 (下 2 根漲跌)")

    # ── 5. 對比摘要 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  摘要比較")
    print("=" * 55)
    print(f"  {'目標':<22} {'Train ACC':>10} {'Test ACC':>10} {'Test F1':>10}")
    for name, r in results.items():
        print(f"  {name:<22} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}")

    results["scaler"] = scaler
    return results


if __name__ == "__main__":
    run_svm()
