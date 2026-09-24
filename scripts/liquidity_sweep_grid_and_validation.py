"""流動性掃單反轉（ICT/SMC核心子概念）完整測試：IS參數網格 -> 子區間
穩健性篩選候選 -> 通過的候選各自只驗證一次OOS。維持這次會話一貫的
方法論紀律（跟 comprehensive_channel_strategy_grid.py / validate_sr_
breakout_candidate.py 同一套流程，不是只看IS整體t值挑最好的那個）。

用法：
    python scripts/liquidity_sweep_grid_and_validation.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402
from tw_quant.liquidity_sweep_reversal_strategy import LiquiditySweepConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
MID_COST = COST_SCENARIOS[1]
ZERO_SLIP_COST = TradeCost("零額外成本(僅稅+手續費)", commission_round_trip=60.0)
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty or len(trades) < 10:
        return dict(n=len(trades), t_stat=np.nan, mean_net_twd=np.nan, sum_net_twd=np.nan)
    net = apply_costs(trades, MID_COST)["net_twd"]
    return dict(n=len(trades), t_stat=tstat(trades["pnl_points"]), mean_net_twd=net.mean(), sum_net_twd=net.sum())


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]

    print("=" * 70)
    print("1) IS參數網格")
    print("=" * 70)
    grid = list(itertools.product(
        [10, 20, 30],       # channel_window
        [5.0, 10.0, 20.0],  # sweep_buffer_points
        [5.0, 10.0],        # stop_buffer_points
        [1.5, 2.0, 3.0],    # target_r_multiple
    ))
    rows = []
    for i, (cw, sbp, stbp, trm) in enumerate(grid):
        cfg = LiquiditySweepConfig(channel_window=cw, sweep_buffer_points=sbp,
                                    stop_buffer_points=stbp, target_r_multiple=trm, max_hold_days=10)
        trades = backtest(is_df, cfg)
        row = summarize(trades)
        row.update(channel_window=cw, sweep_buffer_points=sbp, stop_buffer_points=stbp, target_r_multiple=trm)
        rows.append(row)
        print(f"  ... {i+1}/{len(grid)}  n={row['n']}  t={row['t_stat']}")
    grid_df = pd.DataFrame(rows)
    grid_df.to_parquet(OUT_DIR / "liquidity_sweep_grid_is.parquet", index=False)
    print(f"完成，共 {len(grid_df)} 組合")

    print("\n" + "=" * 70)
    print("2) 候選篩選：IS整體顯著(|t|>=2) 且 n>=30，逐一檢查四個子區間")
    print("=" * 70)
    screen = grid_df[(grid_df["t_stat"].abs() >= 2.0) & (grid_df["n"] >= 30)]
    print(f"通過IS初篩: {len(screen)} / {len(grid_df)}")

    candidates = []
    for _, row in screen.iterrows():
        cfg = LiquiditySweepConfig(channel_window=int(row["channel_window"]), sweep_buffer_points=row["sweep_buffer_points"],
                                    stop_buffer_points=row["stop_buffer_points"], target_r_multiple=row["target_r_multiple"],
                                    max_hold_days=10)
        sub_tstats = []
        for name, start, end in SUB_PERIODS:
            sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
            sub_trades = backtest(sub_df, cfg)
            sub_tstats.append(tstat(sub_trades["pnl_points"]) if len(sub_trades) >= 5 else np.nan)
        overall_sign = np.sign(row["t_stat"])
        same_sign_count = sum(1 for t in sub_tstats if not np.isnan(t) and np.sign(t) == overall_sign)
        candidates.append(dict(**row.to_dict(), sub_tstats=sub_tstats, same_sign_count=same_sign_count))

    cand_df = pd.DataFrame(candidates)
    if cand_df.empty:
        print("\n沒有任何組合通過IS初篩，全面否決，不需要往下驗證OOS。")
        return

    cand_df = cand_df.sort_values("same_sign_count", ascending=False)
    pd.set_option("display.width", 200)
    print("\n所有候選（依子區間同方向數排序）：")
    print(cand_df[["channel_window", "sweep_buffer_points", "stop_buffer_points", "target_r_multiple",
                    "n", "t_stat", "same_sign_count", "sub_tstats"]].to_string(index=False))
    cand_df.to_parquet(OUT_DIR / "liquidity_sweep_candidates.parquet", index=False)

    robust = cand_df[cand_df["same_sign_count"] >= 3]
    print(f"\n四個子區間至少3個同方向的候選數: {len(robust)}")
    if robust.empty:
        print("沒有夠穩健的候選（同方向數<3），不驗證OOS。")
        return

    print("\n" + "=" * 70)
    print("3) 穩健候選各自只驗證一次OOS")
    print("=" * 70)
    oos_df = df[df["datetime"] >= IS_CUTOFF]
    for _, row in robust.iterrows():
        cfg = LiquiditySweepConfig(channel_window=int(row["channel_window"]), sweep_buffer_points=row["sweep_buffer_points"],
                                    stop_buffer_points=row["stop_buffer_points"], target_r_multiple=row["target_r_multiple"],
                                    max_hold_days=10)
        print(f"\n--- 候選 channel_window={cfg.channel_window}, sweep_buffer={cfg.sweep_buffer_points}, "
              f"stop_buffer={cfg.stop_buffer_points}, target_r={cfg.target_r_multiple} ---")
        oos_trades = backtest(oos_df, cfg)
        print(f"OOS n={len(oos_trades)}")
        if oos_trades.empty:
            continue
        for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
            t = apply_costs(oos_trades, cost)
            net = t["net_twd"]
            gross_win = t.loc[net > 0, "net_twd"].sum()
            gross_loss = -t.loc[net <= 0, "net_twd"].sum()
            pf = gross_win / gross_loss if gross_loss > 0 else np.inf
            print(f"  {cost.label}: n={len(t)}, win_rate={(net>0).mean():.3f}, mean_net={net.mean():.1f}, "
                  f"sum_net={net.sum():,.0f}, PF={pf:.3f}, t={tstat(t['pnl_points']):.3f}")
        oos_trades.to_parquet(
            OUT_DIR / f"liquidity_sweep_oos_cw{cfg.channel_window}_sb{int(cfg.sweep_buffer_points)}_"
                      f"stb{int(cfg.stop_buffer_points)}_tr{cfg.target_r_multiple}.parquet",
            index=False,
        )


if __name__ == "__main__":
    main()
