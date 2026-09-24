"""順勢/逆勢全面交叉測試：把這次會話討論過的通道類策略參數（支撐壓力
模組的 channel_window/stop_atr_mult/min_range_pct/trend_slope_threshold_pct
x direction_mode=fade/breakout，唐奇安模組的 entry_window/atr_stop_mult
x trend_filter/volume_filter/volatility_squeeze_filter 開關組合）全部
交叉配對跑一次IS，用「子區間穩健性」（不是單看IS整體t值，那個已經證明
會挑到過擬合的假陽性）篩出候選，最後每個候選只驗證一次OOS。

重要方法論提醒：這是一次探索性的大規模參數搜尋（288+72=360組合），
用越多組合去挑「看起來最好」的那個，運氣好隨便挑到一個顯著結果的機率
越高——所以篩選標準刻意設定成「四個子區間方向一致」而不是「IS整體
t值最高」，並且鎖定候選後對OOS只驗證一次、不回頭調整，維持這次會話
一貫的紀律。

用法：
    python scripts/comprehensive_channel_strategy_grid.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.donchian_breakout_strategy import DonchianConfig, backtest as backtest_donchian  # noqa: E402
from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as backtest_sr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
MID_COST = COST_SCENARIOS[1]
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty or len(trades) < 10:
        return dict(n=len(trades), t_stat=np.nan, mean_net_twd=np.nan, sum_net_twd=np.nan)
    net = apply_costs(trades, MID_COST)["net_twd"]
    return dict(n=len(trades), t_stat=tstat(trades["pnl_points"]), mean_net_twd=net.mean(), sum_net_twd=net.sum())


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    # ---------- 1) 支撐壓力模組：順勢/逆勢 x 全參數網格 ----------
    print("=" * 70)
    print("1) 支撐壓力模組 IS 網格（direction_mode x channel_window x stop_atr_mult x min_range_pct x trend_slope）")
    print("=" * 70)
    sr_rows = []
    sr_grid = list(itertools.product(
        ["fade", "breakout"], [10, 15, 20, 25], [0.5, 1.0, 1.5, 2.0], [2.0, 3.0, 5.0], [2.0, 3.0, 5.0],
    ))
    for i, (mode, cw, sam, mrp, tst) in enumerate(sr_grid):
        cfg = SupportResistanceFadeConfig(
            direction_mode=mode, channel_window=cw, stop_atr_mult=sam,
            min_range_pct=mrp, trend_slope_threshold_pct=tst,
        )
        trades = backtest_sr(is_df, cfg)
        row = summarize(trades)
        row.update(module="sr", direction_mode=mode, channel_window=cw, stop_atr_mult=sam,
                    min_range_pct=mrp, trend_slope_threshold_pct=tst)
        sr_rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(sr_grid)}")
    sr_df = pd.DataFrame(sr_rows)
    print(f"完成，共 {len(sr_df)} 組合")

    # ---------- 2) 唐奇安模組：entry_window x atr_stop_mult x 濾網開關 ----------
    print("\n" + "=" * 70)
    print("2) 唐奇安模組 IS 網格（entry_window x atr_stop_mult x trend/volume/squeeze 濾網開關）")
    print("=" * 70)
    donchian_rows = []
    filter_combos = list(itertools.product([False, True], repeat=3))  # (trend, volume, squeeze)
    donchian_grid = list(itertools.product([15, 20, 25], [1.5, 2.0, 2.5], filter_combos))
    for i, (ew, am, (tf, vf, sf)) in enumerate(donchian_grid):
        cfg = DonchianConfig(entry_window=ew, atr_stop_mult=am, trend_filter=tf, volume_filter=vf,
                              volatility_squeeze_filter=sf)
        trades = backtest_donchian(is_df, cfg)
        row = summarize(trades)
        row.update(module="donchian", entry_window=ew, atr_stop_mult=am,
                    trend_filter=tf, volume_filter=vf, volatility_squeeze_filter=sf)
        donchian_rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(donchian_grid)}")
    donchian_df = pd.DataFrame(donchian_rows)
    print(f"完成，共 {len(donchian_df)} 組合")

    sr_df.to_parquet(OUT_DIR / "channel_grid_sr_is.parquet", index=False)
    donchian_df.to_parquet(OUT_DIR / "channel_grid_donchian_is.parquet", index=False)

    # ---------- 3) 挑候選：子區間方向一致性檢查（不是只看IS整體t值） ----------
    print("\n" + "=" * 70)
    print("3) 候選篩選：IS整體顯著(|t|>=2) 且 n>=30 的組合，逐一檢查四個子區間")
    print("=" * 70)

    candidates = []
    sr_screen = sr_df[(sr_df["t_stat"].abs() >= 2.0) & (sr_df["n"] >= 30)]
    print(f"支撐壓力模組通過IS初篩: {len(sr_screen)} / {len(sr_df)}")
    for _, row in sr_screen.iterrows():
        cfg = SupportResistanceFadeConfig(
            direction_mode=row["direction_mode"], channel_window=int(row["channel_window"]),
            stop_atr_mult=row["stop_atr_mult"], min_range_pct=row["min_range_pct"],
            trend_slope_threshold_pct=row["trend_slope_threshold_pct"],
        )
        sub_tstats = []
        for name, start, end in SUB_PERIODS:
            sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
            sub_trades = backtest_sr(sub_df, cfg)
            sub_tstats.append(tstat(sub_trades["pnl_points"]) if len(sub_trades) >= 5 else np.nan)
        overall_sign = np.sign(row["t_stat"])
        same_sign_count = sum(1 for t in sub_tstats if not np.isnan(t) and np.sign(t) == overall_sign)
        candidates.append(dict(**row.to_dict(), sub_tstats=sub_tstats, same_sign_count=same_sign_count))

    donchian_screen = donchian_df[(donchian_df["t_stat"].abs() >= 2.0) & (donchian_df["n"] >= 30)]
    print(f"唐奇安模組通過IS初篩: {len(donchian_screen)} / {len(donchian_df)}")
    for _, row in donchian_screen.iterrows():
        cfg = DonchianConfig(entry_window=int(row["entry_window"]), atr_stop_mult=row["atr_stop_mult"],
                              trend_filter=bool(row["trend_filter"]), volume_filter=bool(row["volume_filter"]),
                              volatility_squeeze_filter=bool(row["volatility_squeeze_filter"]))
        sub_tstats = []
        for name, start, end in SUB_PERIODS:
            sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
            sub_trades = backtest_donchian(sub_df, cfg)
            sub_tstats.append(tstat(sub_trades["pnl_points"]) if len(sub_trades) >= 5 else np.nan)
        overall_sign = np.sign(row["t_stat"])
        same_sign_count = sum(1 for t in sub_tstats if not np.isnan(t) and np.sign(t) == overall_sign)
        candidates.append(dict(**row.to_dict(), sub_tstats=sub_tstats, same_sign_count=same_sign_count))

    cand_df = pd.DataFrame(candidates)
    if cand_df.empty:
        print("\n沒有任何組合通過IS初篩（|t|>=2 且 n>=30），全面否決，不需要再往下驗證OOS。")
        return

    cand_df = cand_df.sort_values("same_sign_count", ascending=False)
    pd.set_option("display.width", 200)
    print("\n所有通過IS初篩的候選（依子區間同方向數排序）：")
    show_cols = ["module", "direction_mode", "channel_window", "stop_atr_mult", "min_range_pct",
                 "trend_slope_threshold_pct", "entry_window", "atr_stop_mult", "trend_filter",
                 "volume_filter", "volatility_squeeze_filter", "n", "t_stat", "same_sign_count", "sub_tstats"]
    show_cols = [c for c in show_cols if c in cand_df.columns]
    print(cand_df[show_cols].to_string(index=False))

    robust = cand_df[cand_df["same_sign_count"] >= 4]
    print(f"\n四個子區間全部同方向的候選數: {len(robust)}")

    cand_df.to_parquet(OUT_DIR / "channel_grid_candidates.parquet", index=False)
    print("\nsaved: channel_grid_sr_is / channel_grid_donchian_is / channel_grid_candidates .parquet")


if __name__ == "__main__":
    main()
