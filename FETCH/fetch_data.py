"""
fetch_data.py — 資料抓取與儲存模組

負責從 Binance API 抓取 BTCUSDT K 線資料，並儲存至 SQLite。
只需執行一次，後續測試直接從 DB 讀取即可。
"""
import requests
import pandas as pd
import numpy as np
import time


def fetch_5m_btcusdt_latest_n(total_limit=100000):
    """
    從「現在」開始往回追溯，抓取最新的 n 根 5 分鐘 K 線資料。
    """
    url = "https://api.binance.com/api/v3/klines"
    all_dfs = []
    current_end_ts = None

    print(f"開始抓取最新的 {total_limit} 筆資料...")

    while len(all_dfs) * 1000 < total_limit:
        params = {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "limit": 1000
        }
        if current_end_ts:
            params["endTime"] = current_end_ts

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        batch_df = pd.DataFrame(data, columns=cols)
        all_dfs.append(batch_df)

        current_end_ts = int(data[0][0]) - 1

        print(f"累計已抓取: {len(all_dfs) * 1000} 筆...", end="\r")
        time.sleep(0.1)

        if len(data) < 1000:
            break

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").tail(total_limit)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str = "btc_5m"):
    """將資料儲存至 SQLite，並透過 open_time 避免重複。"""
    import sqlite3

    df_to_store = df.copy()
    for col in ("open_time", "close_time"):
        if col in df_to_store.columns:
            df_to_store[col] = df_to_store[col].astype(str)

    with sqlite3.connect(db_path) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (open_time TEXT PRIMARY KEY)")
        df_to_store.to_sql("temp_table", conn, if_exists="replace", index=False)
        conn.execute(f"INSERT OR IGNORE INTO {table_name} SELECT * FROM temp_table")
        conn.execute("DROP TABLE temp_table")


def load_from_sqlite(db_path: str, table_name: str = "btc_5m",
                     limit: int = 0) -> pd.DataFrame:
    """從 SQLite 讀取資料，回傳 DataFrame。

    Parameters
    ----------
    limit : int  只取最新的 N 筆（0 = 全部）
    """
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        if limit > 0:
            query = (f"SELECT * FROM {table_name} "
                     f"ORDER BY open_time DESC LIMIT {limit}")
        else:
            query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)

    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


if __name__ == "__main__":
    DB_FILE = "btc_data.db"
    target_count = 100000

    df = fetch_5m_btcusdt_latest_n(total_limit=target_count)
    print(f"\n抓取完成！實際抓取 {len(df)} 筆資料。")
    print(f"資料範圍：{df['open_time'].min()} 至 {df['open_time'].max()}")

    save_to_sqlite(df, DB_FILE)
    print(f"資料已儲存至 {DB_FILE}")
