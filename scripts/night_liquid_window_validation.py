"""夜盤流動性熱區(21:00-23:45)趨勢偵測驗證：兩個預先指定的版本
（frac=1.0嚴格版、frac=0.90移植日盤已驗證值），不做網格搜尋（夜盤資料
只有2017-05-15起約6.5年，樣本太小不適合開網格搜尋做多重比較）。

用法：
    python scripts/night_liquid_window_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402
from tw_quant.night_liquid_window_trend_strategy import NightLiquidWindowConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
MID_COST = COST_SCENARIOS[1]
ZERO_SLIP_COST = TradeCost("零額外成本(僅稅+手續費)", commission_round_trip=60.0)

CONFIGS = {
    "strict(frac=1.0)": NightLiquidWindowConfig(min_dominant_side_fraction=1.0),
    "relaxed(frac=0.90,移植日盤版)": NightLiquidWindowConfig(min_dominant_side_fraction=0.90),
}


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def summarize(trades: pd.DataFrame, cost: TradeCost) -> dict:
    if trades.empty:
        return dict(n=0)
    t = apply_costs(trades, cost)
    net = t["net_twd"]
    gross_win = t.loc[net > 0, "net_twd"].sum()
    gross_loss = -t.loc[net <= 0, "net_twd"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return dict(n=len(t), win_rate=(net > 0).mean(), mean_net=net.mean(), sum_net=net.sum(),
                pf=pf, t_stat=tstat(t["pnl_points"]))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    for label, cfg in CONFIGS.items():
        print("=" * 70)
        print(f"版本: {label}  (window {cfg.window_start}~{cfg.window_end}, decision={cfg.decision_time})")
        print("=" * 70)

        is_trades = backtest(is_df, cfg)
        print(f"IS (2017-05-15~2020-12-31): n={len(is_trades)}")
        if not is_trades.empty:
            for cost in [ZERO_SLIP_COST, MID_COST]:
                print(f"  {cost.label}: {summarize(is_trades, cost)}")
            print(f"  出場原因: {is_trades['exit_reason'].value_counts().to_dict()}")

        oos_trades = backtest(oos_df, cfg)
        print(f"\nOOS (2021-2023): n={len(oos_trades)}")
        if not oos_trades.empty:
            for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
                print(f"  {cost.label}: {summarize(oos_trades, cost)}")
            print(f"  出場原因: {oos_trades['exit_reason'].value_counts().to_dict()}")
        print()

        tag = label.split("(")[0]
        is_trades.to_parquet(OUT_DIR / f"night_liquid_window_{tag}_is_trades.parquet", index=False)
        oos_trades.to_parquet(OUT_DIR / f"night_liquid_window_{tag}_oos_trades.parquet", index=False)

    print("saved: night_liquid_window_*_is/oos_trades.parquet")


if __name__ == "__main__":
    main()
