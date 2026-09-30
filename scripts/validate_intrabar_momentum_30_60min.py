"""正式驗證scan_multi_minute_timeframe.py發現的線索：30/60分K在「前一根
K棒大幅波動」後，下一根K棒傾向延續(動能)而非拉回，且30/60分K顆粒度下
毛利量級(1.33~2.86點)已經超過交易成本(3-4點/趟)的量級，值得正式驗證。

**重要提醒（誠實揭露，避免過度樂觀）**：這個發現的核心機制（開盤後動能
延續、用固定水準/門檻確認）跟這次會話更早期已經驗證過的「動能檢查點
策略」(momentum_checkpoint_strategy.py，IS看似有效但OOS轉負)、「VWAP
趨勢日偵測」(近13年拆解後最大回撤嚴重、可實戰性動搖)高度相關，都是
「日內動能延續」這個大類假設的不同操作化版本。這裡是全新的具體操作化
方式（固定30/60分鐘K棒邊界，全天候觸發，不限定特定時間點），跟前兩者
不完全相同，值得正式测，但先誠實列出這個「同類假設先前兩次都撐不住
OOS」的前情，不能因為換了個操作化方式就假裝是全新、獨立的證據。

規則：
- 30分K或60分K收盤時，若這根K棒的|漲跌點數| >= 門檻(用平均K棒振幅的
  1倍)，判定為「大幅波動」
- 下一根K棒開盤價，順著這根大幅波動的方向進場(動能延續假設)
- 下一根K棒收盤價出場(1根K棒的持有期，等同scan_multi_minute_timeframe.py
  測的next_ret_pts，只是這裡改成真正可執行的「進場點=下一根開盤」)
- 全天候都可以觸發，不限定特定時間點；但只在日盤(08:45-13:45)內

先做IS(2001-2020)四個5年子區間穩健性檢查，全部一致才考慮花OOS。

用法：
    python scripts/validate_intrabar_momentum_30_60min.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"

SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
IS_END = "2021-01-01"
OOS_END = "2024-01-01"
THRESHOLD_MULT = 1.0


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def resample_day_session(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    t = df["datetime"].dt.time
    day = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    g = day.sort_values("datetime").set_index("datetime")
    agg = g.resample(f"{minutes}min", origin="epoch", closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
    ).dropna().reset_index()
    agg["trading_date"] = agg["datetime"].dt.date
    agg["bar_time"] = agg["datetime"].dt.time
    first_bar_time = agg.groupby("trading_date")["bar_time"].min()
    agg = agg[agg["bar_time"] != agg["trading_date"].map(first_bar_time)]
    return agg.drop(columns=["bar_time"]).sort_values("datetime").reset_index(drop=True)


def build_trades(bars: pd.DataFrame, threshold_mult: float) -> pd.DataFrame:
    bars = bars.reset_index(drop=True)
    bars["ret_pts"] = bars.groupby("trading_date")["close"].diff()
    avg_abs_move = bars["ret_pts"].dropna().abs().mean()
    threshold = avg_abs_move * threshold_mult

    trades = []
    for i in range(len(bars) - 1):
        cur, nxt = bars.iloc[i], bars.iloc[i + 1]
        if cur["trading_date"] != nxt["trading_date"]:
            continue
        ret = cur["ret_pts"]
        if pd.isna(ret) or abs(ret) < threshold:
            continue
        direction = "long" if ret > 0 else "short"
        sign = 1 if direction == "long" else -1
        entry_price = nxt["open"]
        exit_price = nxt["close"]
        pnl_points = (exit_price - entry_price) * sign
        trades.append(dict(
            trading_date=cur["trading_date"], direction=direction,
            entry_dt=nxt["datetime"], entry_price=entry_price, exit_price=exit_price,
            pnl_points=pnl_points,
        ))
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["trading_date"] = pd.to_datetime(trades_df["trading_date"])
    return trades_df.sort_values("entry_dt").reset_index(drop=True)


def report(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"    {label}: n=0")
        return
    t, n, m = tstat(trades["pnl_points"])
    mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
    print(f"    {label}: n={n} t={t:.2f} mean={m:.2f}點 中檔成本淨損益={mid_net:,.0f}元")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    for minutes in [30, 60]:
        print("=" * 78)
        print(f"時間顆粒度: {minutes}分K，門檻={THRESHOLD_MULT}x平均K棒振幅")
        print("=" * 78)

        print("\n1) IS(2001-2020) 四個5年子區間穩健性")
        for name, start, end in SUB_PERIODS:
            sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
            bars = resample_day_session(sub_df, minutes)
            trades = build_trades(bars, THRESHOLD_MULT)
            report(trades, name)

        print("\n2) IS(2001-2020)整體 + OOS(2021-2023)（僅在子區間一致時才看）")
        is_df = df[df["datetime"] < IS_END]
        oos_df = df[(df["datetime"] >= IS_END) & (df["datetime"] < OOS_END)]
        is_bars = resample_day_session(is_df, minutes)
        oos_bars = resample_day_session(oos_df, minutes)
        is_trades = build_trades(is_bars, THRESHOLD_MULT)
        oos_trades = build_trades(oos_bars, THRESHOLD_MULT)
        report(is_trades, "IS整體(2001-2020)")
        report(oos_trades, "OOS(2021-2023)")
        if not oos_trades.empty:
            print("   OOS三種成本情境:")
            for cost in COST_SCENARIOS:
                net = apply_costs(oos_trades, cost)["net_twd"].sum()
                print(f"     {cost.label}: {net:,.0f}元")
        print()


if __name__ == "__main__":
    main()
