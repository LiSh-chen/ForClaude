"""現貨-期貨基差(basis)測試：TAIEX現貨指數 vs TX近月期貨的價差，跟前一步
測的「近月/遠月」是不同的價差題目（近月/遠月是兩個期貨合約互比，這裡
是期貨對現貨本身）。

**嚴重資料限制（先誠實講清楚）**：TWSE現貨指數歷史資料只抓到
2001-01-02~2004-07-30(765個交易日，約3.5年)，之後的月份重新嘗試抓取
時被TWSE的資安機制直接擋下(HTTP 307資安警示頁，不是查無資料或
rate limit，換了瀏覽器User-Agent/Referer都一樣被擋)。這代表現貨基差
分析目前只能在這3.5年的窗格上做探索性檢視，**遠遠不足以比照這次會話
一貫的IS(2001-2020)/OOS(2021-2023)驗證紀律**，這裡明確標記為探索性
觀察，不會宣稱「驗證出策略」。

用跟近月/遠月價差(validate_calendar_spread_reversion.py)完全同一套
z分數方法（90天滾動窗格、shift(1)避免未來函數），測基差 = 近月期貨
結算價 - TAIEX現貨收盤，是否有均值回歸性質。

用法：
    python scripts/test_spot_futures_basis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SPOT_PATH = REPO_ROOT / "data" / "taiex_spot_daily_partial.parquet"
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"

W = 90
ENTRY_Z = 1.0
EXIT_Z = 0.3
MAX_HOLD = 20
POINT_VALUE = 50.0


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def build_episodes(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    in_position = False
    direction = None
    entry_idx = None
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row["z"]):
            continue
        if not in_position:
            if row["z"] >= ENTRY_Z:
                in_position, direction, entry_idx = True, "short_basis", i
            elif row["z"] <= -ENTRY_Z:
                in_position, direction, entry_idx = True, "long_basis", i
        else:
            held = i - entry_idx
            reverted = abs(row["z"]) <= EXIT_Z
            if reverted or held >= MAX_HOLD or i == len(df) - 1:
                entry_row = df.iloc[entry_idx]
                sign = 1 if direction == "long_basis" else -1
                pnl_points = (row["basis"] - entry_row["basis"]) * sign
                trades.append(dict(
                    entry_date=entry_row["date"], exit_date=row["date"], direction=direction,
                    held_days=held, entry_basis=entry_row["basis"], exit_basis=row["basis"],
                    pnl_points=pnl_points,
                ))
                in_position = False
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    return trades_df.sort_values("entry_date").reset_index(drop=True)


def main() -> None:
    spot = pd.read_parquet(SPOT_PATH)
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]]
    df = spot.merge(tx, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["basis"] = df["near_price"] - df["taiex"]

    print(f"合併後樣本: n={len(df)}，範圍 {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"基差統計: 平均={df['basis'].mean():.2f}點 標準差={df['basis'].std():.2f}點 "
          f"最小={df['basis'].min():.1f} 最大={df['basis'].max():.1f}")

    roll_mean = df["basis"].rolling(W).mean().shift(1)
    roll_std = df["basis"].rolling(W).std(ddof=1).shift(1)
    df["z"] = (df["basis"].shift(1) - roll_mean) / roll_std

    print("\n" + "=" * 70)
    print(f"基差均值回歸(事件週期式，W={W}, entry_z={ENTRY_Z}, exit_z={EXIT_Z}, max_hold={MAX_HOLD})")
    print("=" * 70)
    trades = build_episodes(df)
    if trades.empty:
        print("無交易")
        return
    t, n, m = tstat(trades["pnl_points"])
    gross_twd = (trades["pnl_points"] * POINT_VALUE).sum()
    print(f"n={n} t={t:.2f} mean={m:.2f}點 平均持有{trades['held_days'].mean():.1f}天 "
          f"毛損益合計(未扣成本)={gross_twd:,.0f}元")
    print("\n方向分布:")
    print(trades["direction"].value_counts())
    for d in ["long_basis", "short_basis"]:
        sub = trades[trades["direction"] == d]
        t2, n2, m2 = tstat(sub["pnl_points"])
        print(f"  {d}: n={n2} t={t2:.2f} mean={m2:.2f}點")

    print("\n年度分布(檢查是不是集中在少數年份):")
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    print(trades.groupby("year")["pnl_points"].agg(["count", "sum", "mean"]).round(2))


if __name__ == "__main__":
    main()
