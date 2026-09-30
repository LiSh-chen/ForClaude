"""接續scan_calendar_spread_mean_reversion.py留下的未完成候選（近月/遠月
價差均值回歸）：修正重疊窗格t值虛高問題後，這是目前為止「發現訊號但還
沒完整驗證」的最有希望候選，且對應pending中的task#12（建置近月/遠月/
現貨序列並分析價差）。

延續事項：
1. 前一版用「z分數 -> 接下來N天固定持有」測，重疊窗格需要額外做非重疊
   抽樣校正才能信任t值。這裡改用「事件週期」設計（跟低波動regime週期
   同一套邏輯）：z第一次穿越門檻才進場，一路持有到z回到中性帶或碰到
   最長持有天數才出場，天生不會重複進場、不需要額外做非重疊校正。
2. 2001-2005子區間方向不一致的可能原因：檢查發現該期間遠月合約
   成交量/未平倉量明顯偏低（2001年平均日成交量1215口，2010年已成長
   到6150口），流動性單薄可能讓結算價/價差本身雜訊較大，用「遠月OI
   門檻」排除流動性不足的早期樣本，看能不能讓子區間一致性變好。
3. 換算成真正的損益(點數x2口，因為價差交易同時開兩個部位，成本也要
   雙倍計算)，不是只看「價差本身有沒有收斂」這種抽象統計量。

用法：
    python scripts/validate_calendar_spread_reversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import TradeCost, apply_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
POINT_VALUE = 50.0

W = 90
ENTRY_Z = 1.0
EXIT_Z = 0.3
MAX_HOLD = 20
MIN_FAR_OI = 3000  # 排除遠月流動性過低的樣本

SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
IS_END = "2021-01-01"
OOS_END = "2024-01-01"

# 價差交易的成本：兩腿(近月+遠月)各自都要收手續費+稅，用跟三腿策略一致
# 的三檔情境，但費率乘以2（因為是兩口部位，不是一口）
SPREAD_COST_SCENARIOS = [
    TradeCost("低（折扣網路下單）", commission_round_trip=30.0 * 2, slippage_points_round_trip=1.0 * 2),
    TradeCost("中（一般券商）", commission_round_trip=60.0 * 2, slippage_points_round_trip=1.0 * 2),
    TradeCost("高（傳統營業員）", commission_round_trip=100.0 * 2, slippage_points_round_trip=1.0 * 2),
]


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def build_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["spread"] = df["calendar_spread"].astype(float)
    roll_mean = df["spread"].rolling(W).mean().shift(1)
    roll_std = df["spread"].rolling(W).std(ddof=1).shift(1)
    df["z"] = (df["spread"].shift(1) - roll_mean) / roll_std
    return df


def build_episodes(df: pd.DataFrame, min_far_oi: float | None) -> pd.DataFrame:
    trades = []
    in_position = False
    direction = None
    entry_idx = None
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row["z"]):
            continue
        if min_far_oi is not None and row["far_oi"] < min_far_oi:
            if in_position:
                # 流動性條件本身不強制出場，只是進場濾網；維持原邏輯不中途因濾網出場
                pass
            continue
        if not in_position:
            if row["z"] >= ENTRY_Z:
                in_position, direction, entry_idx = True, "short_spread", i
            elif row["z"] <= -ENTRY_Z:
                in_position, direction, entry_idx = True, "long_spread", i
        else:
            held = i - entry_idx
            reverted = abs(row["z"]) <= EXIT_Z
            if reverted or held >= MAX_HOLD or i == len(df) - 1:
                entry_row = df.iloc[entry_idx]
                exit_row = row
                entry_spread = entry_row["spread"]
                exit_spread = exit_row["spread"]
                sign = 1 if direction == "long_spread" else -1
                pnl_points = (exit_spread - entry_spread) * sign
                trades.append(dict(
                    trading_date=entry_row["date"], direction=direction,
                    entry_date=entry_row["date"], exit_date=exit_row["date"],
                    held_days=held, entry_spread=entry_spread, exit_spread=exit_spread,
                    pnl_points=pnl_points,
                    entry_price=0.0, exit_price=pnl_points,  # 給apply_costs用的佔位欄位
                ))
                in_position = False
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["trading_date"] = pd.to_datetime(trades_df["trading_date"])
    return trades_df.sort_values("entry_date").reset_index(drop=True)


def report(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"    {label}: n=0")
        return
    t, n, m = tstat(trades["pnl_points"])
    mid_cost = SPREAD_COST_SCENARIOS[1]
    net = (trades["pnl_points"] * POINT_VALUE - mid_cost.commission_round_trip
           - mid_cost.slippage_points_round_trip * POINT_VALUE).sum()
    print(f"    {label}: n={n} t={t:.2f} mean={m:.2f}點 中檔成本淨損益={net:,.0f}元 平均持有{trades['held_days'].mean():.1f}天")


def main() -> None:
    raw = pd.read_parquet(DATA_PATH)
    df = build_signal(raw)

    print("=" * 78)
    print(f"事件週期式驗證：W={W}, entry_z={ENTRY_Z}, exit_z={EXIT_Z}, max_hold={MAX_HOLD}天")
    print("=" * 78)

    print("\n1) IS(2001-2020) 四個5年子區間穩健性（無流動性濾網）")
    for name, start, end in SUB_PERIODS:
        sub = df[(df["date"] >= start) & (df["date"] < end)]
        trades = build_episodes(sub, None)
        report(trades, name)

    print("\n2) 同樣四個子區間，加上遠月OI>=3000的流動性濾網")
    for name, start, end in SUB_PERIODS:
        sub = df[(df["date"] >= start) & (df["date"] < end)]
        trades = build_episodes(sub, MIN_FAR_OI)
        report(trades, name)

    print("\n3) IS整體 + OOS（用流動性濾網版本，跨窗格計算z避免子區間切斷暖機期）")
    is_trades_full = build_episodes(df[df["date"] < IS_END], MIN_FAR_OI)
    oos_trades_full = build_episodes(df[(df["date"] >= IS_END) & (df["date"] < OOS_END)], MIN_FAR_OI)
    report(is_trades_full, "IS整體(2001-2020)")
    report(oos_trades_full, "OOS(2021-2023)")
    if not oos_trades_full.empty:
        print("\n   OOS三種成本情境（雙腿，費率x2）:")
        for cost in SPREAD_COST_SCENARIOS:
            net = (oos_trades_full["pnl_points"] * POINT_VALUE - cost.commission_round_trip
                   - cost.slippage_points_round_trip * POINT_VALUE).sum()
            print(f"     {cost.label}: {net:,.0f}元")
        print("\n   方向分布:")
        print(oos_trades_full["direction"].value_counts())
        for d in ["long_spread", "short_spread"]:
            sub = oos_trades_full[oos_trades_full["direction"] == d]
            t, n, m = tstat(sub["pnl_points"])
            print(f"     {d}: n={n} t={t:.2f} mean={m:.2f}點")


if __name__ == "__main__":
    main()
