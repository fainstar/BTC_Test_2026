"""
btc_data_pipeline.py — 將 clean_data 與 remove_outliers 合併的資料清理類別

本檔整合了兩個流程：
1. 從 btc_5m.csv 讀取並保留 OHLCV 欄位、去重、排序
2. 以滾動窗口 IQR 進行 Winsorize (k=2.0) 並格式化數據

使用範例：
```bash
python btc_data_pipeline.py
```
會先輸出 `btc_5m_clean.csv`，接著產生 `btc_5m_no_outliers.csv`。
"""

from __future__ import annotations

import pandas as pd
from typing import Iterable, List

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

INPUT_FILE = "Data/btc_5m.csv"
CLEAN_OUTPUT = "Data/btc_5m_clean.csv"
OUTLIER_OUTPUT = "Data/btc_5m_no_outliers.csv"
KEEP_COLS = ["open_time", "open", "high", "low", "close", "volume"]
WINDOW = 2016
K = 2.0
CHECK_COLS = ["open", "high", "low", "close"]


class BTCDataCleaner:
    def __init__(self,
                 input_file: str = INPUT_FILE,
                 clean_output: str = CLEAN_OUTPUT,
                 outlier_output: str = OUTLIER_OUTPUT,
                 keep_cols: Iterable[str] = KEEP_COLS,
                 window: int = WINDOW,
                 k: float = K,
                 check_cols: Iterable[str] = CHECK_COLS):
        self.input_file = input_file
        self.clean_output = clean_output
        self.outlier_output = outlier_output
        self.keep_cols = list(keep_cols)
        self.window = window
        self.k = k
        self.check_cols = list(check_cols)

    def clean(self) -> pd.DataFrame:
        with console.status(f"[bold green]載入 {self.input_file}..."):
            df = pd.read_csv(self.input_file, usecols=self.keep_cols)
        console.print(f"  ✓ 載入原始 [bold]{len(df):,}[/] 筆資料（{self.input_file}）")
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.drop_duplicates("open_time")
        df = df.sort_values("open_time").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df.to_csv(self.clean_output, index=False)
        console.print(f"  ✓ 清理完成，已儲存 [bold]{len(df):,}[/] 筆至 {self.clean_output}")
        return df

    def _rolling_iqr_clip(self, series: pd.Series) -> pd.Series:
        min_periods = max(self.window // 4, 10)
        q1 = series.rolling(self.window, min_periods=min_periods).quantile(0.25)
        q3 = series.rolling(self.window, min_periods=min_periods).quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr
        return series.clip(lower=lower, upper=upper)

    def _format_value(self, value: float) -> str:
        return f"{value:.4f}"

    def winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        console.rule("[bold cyan]Winsorize[/]")
        console.print(f"  參數：window=[bold]{self.window}[/]  k=[bold]{self.k}[/]")

        win_table = Table(title="Winsorize 調整結果", box=box.SIMPLE_HEAD)
        win_table.add_column("欄位", style="bold")
        win_table.add_column("調整筆數", justify="right", style="yellow")

        for col in self.check_cols:
            clipped = self._rolling_iqr_clip(df[col])
            changed = (clipped != df[col]).sum()
            win_table.add_row(col, f"{changed:,}")
            df[col] = clipped

        console.print(win_table)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].map(self._format_value)
        df.to_csv(self.outlier_output, index=False)
        console.print(f"  ✓ 已將 Winsorized 結果儲存至 [bold]{self.outlier_output}[/]")
        return df

    def run(self) -> pd.DataFrame:
        df_clean = self.clean()
        return self.winsorize(df_clean)


def main() -> None:
    console.rule("[bold magenta]BTC 資料清理 Pipeline[/]")
    cleaner = BTCDataCleaner()
    cleaner.run()
    console.print("[bold green]✓ Pipeline 完成[/]")


if __name__ == "__main__":
    main()
