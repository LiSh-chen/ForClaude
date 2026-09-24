"""在唐奇安突破基礎上疊加三個進階濾網（趨勢方向、量能、波動率壓縮），
逐一測試、再測組合，看能不能把之前偏弱的邊際（IS t=0.69、OOS t=0.49）
拉到真正顯著的水準。跟壓力支撐逆勢操作合起來看：那個方向已經證實顯著
虧錢，這裡反過來想順著突破方向、只挑「品質比較好」的突破。

用法：
    python scripts/backtest_donchian_filters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.donchian_breakout_strategy import DonchianConfig, backtest  # noqa: E402
from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]

CANDIDATES = {
    "baseline(無濾網)": DonchianConfig(),
    "trend_filter(MA100)": DonchianConfig(trend_filter=True, trend_ma_window=100),
    "volume_filter(>=1.2x20日均量)": DonchianConfig(volume_filter=True, volume_ma_window=20, volume_min_ratio=1.2),
    "squeeze_filter(ATR比值<60日中位數)": DonchianConfig(volatility_squeeze_filter=True, squeeze_lookback=60, squeeze_percentile=0.5),
    "trend+volume": DonchianConfig(trend_filter=True, trend_ma_window=100, volume_filter=True, volume_ma_window=20, volume_min_ratio=1.2),
    "trend+squeeze": DonchianConfig(trend_filter=True, trend_ma_window=100, volatility_squeeze_filter=True, squeeze_lookback=60, squeeze_percentile=0.5),
    "volume+squeeze": DonchianConfig(volume_filter=True, volume_ma_window=20, volume_min_ratio=1.2, volatility_squeeze_filter=True, squeeze_lookback=60, squeeze_percentile=0.5),
    "trend+volume+squeeze": DonchianConfig(trend_filter=True, trend_ma_window=100, volume_filter=True, volume_ma_window=20,
                                            volume_min_ratio=1.2, volatility_squeeze_filter=True, squeeze_lookback=60, squeeze_percentile=0.5),
}


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return dict(label=label, n=0)
    t = apply_costs(trades, COST_SCENARIOS[1])
    net = t["net_twd"].to_numpy()
    gross_win = t.loc[net > 0, "net_twd"].sum()
    gross_loss = -t.loc[net <= 0, "net_twd"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return dict(label=label, n=len(t), win_rate=(net > 0).mean(), mean_net_twd=net.mean(),
                sum_net_twd=net.sum(), profit_factor=pf, t_stat=tstat(t["pnl_points"]))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    print("=" * 70)
    print("1) IS 整體：baseline vs 三個濾網 vs 組合")
    print("=" * 70)
    rows = []
    for name, cfg in CANDIDATES.items():
        trades = backtest(is_df, cfg)
        row = summarize(trades, name)
        rows.append(row)
        print(row)
    is_summary = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("2) IS 子區間穩健性（只看 IS 整體表現最好的那個候選）")
    print("=" * 70)
    best_name = is_summary.loc[is_summary["n"] > 20, "t_stat"].idxmax()
    best_label = is_summary.loc[best_name, "label"]
    best_cfg = CANDIDATES[best_label]
    print(f"IS 整體 t 值最高的候選（且交易數>20，避免樣本太少的偽最佳）: {best_label}")
    sub_rows = []
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        sub_trades = backtest(sub_df, best_cfg)
        row = summarize(sub_trades, name)
        sub_rows.append(row)
        print(row)

    print("\n" + "=" * 70)
    print(f"3) OOS 唯一一次驗證（{best_label}，沿用 IS 選出的參數，不回頭調參）")
    print("=" * 70)
    oos_trades = backtest(oos_df, best_cfg)
    oos_row = summarize(oos_trades, best_label)
    print(oos_row)
    if not oos_trades.empty:
        print(f"\nOOS 逐筆明細:\n{oos_trades.to_string(index=False)}")

    print("\n" + "=" * 70)
    print("4) 為了對照，也把 baseline（無濾網）拿去跑同樣的 OOS")
    print("=" * 70)
    oos_baseline = backtest(oos_df, CANDIDATES["baseline(無濾網)"])
    print(summarize(oos_baseline, "baseline(無濾網) OOS"))

    is_summary.to_parquet(OUT_DIR / "donchian_filters_is_comparison.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "donchian_filters_best_subperiods.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "donchian_filters_best_oos_trades.parquet", index=False)
    print("\nsaved: donchian_filters_is_comparison / donchian_filters_best_subperiods / donchian_filters_best_oos_trades .parquet")


if __name__ == "__main__":
    main()
