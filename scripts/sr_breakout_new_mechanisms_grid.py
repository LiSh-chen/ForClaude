"""支撐壓力順勢突破候選：測試三個原本沒有的新機制，看能不能在IS找到
比鎖定候選(cw=20,sam=1.0,mrp=2.0,tst=5.0, t=2.08)更穩健的版本。只用
IS(2001-2020)資料+子區間一致性篩選，完全不碰OOS——如果有候選勝出，
要不要對它花掉這個策略唯一一次的OOS驗證機會，留給使用者決定。

三個新機制：
1. target_r_multiple（新增的固定停利目標，見
   tw_quant/support_resistance_fade_strategy.py 的異動）：原始breakout
   版本沒有停利、抱到停損或max_hold_days，測試加上R倍數停利會不會讓
   訊號更穩定（犧牲一些大賺單的上檔，換取更早鎖定獲利）。
2. 突破當天量能確認濾網（沿用 tw_quant.strategy_lab.apply_volume_filter
   同一套邏輯）：假設「有量的突破比較可能是真的」。
3. 200日均線趨勢對齊濾網（沿用 apply_trend_regime_filter）：只在大格局
   同向時才跟突破——這次會話之前對VWAP趨勢日測過沒有幫助，這裡是對
   sr_breakout重新測一次（機制不同，結論不能直接套用）。

用法：
    python scripts/sr_breakout_new_mechanisms_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr  # noqa: E402
from tw_quant.technical_indicators import daily_indicators  # noqa: E402
from tw_quant.strategy_lab import (  # noqa: E402
    VolumeFilterSpec, TrendRegimeFilterSpec, apply_volume_filter, apply_trend_regime_filter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
LOW_COST = COST_SCENARIOS[0]
BASE_CFG_KW = dict(direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
                    min_range_pct=2.0, trend_slope_threshold_pct=5.0)
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


def _to_std_like(trades: pd.DataFrame) -> pd.DataFrame:
    """把sr_breakout的原始交易紀錄轉成strategy_lab濾網函式要的最小欄位。"""
    if trades.empty:
        return trades.assign(trading_date=pd.NaT)
    out = trades.copy()
    out["trading_date"] = pd.to_datetime(out["entry_date"]).dt.date
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.strftime("%Y-%m-%d")
    return out


def screen(label: str, is_df: pd.DataFrame, full_df: pd.DataFrame, trades_fn) -> dict:
    """trades_fn(data_slice) -> trades_df（已經套用完新機制/濾網）。回傳IS整體t值、n，
    跟四個子區間t值+跟整體同號的個數。"""
    is_trades = trades_fn(is_df)
    n = len(is_trades)
    if n < 20:
        print(f"  {label:38s} n={n:4d}（太少，略過子區間檢查）")
        return dict(label=label, n=n, t_stat=np.nan, same_sign_count=0)
    overall_t = tstat(is_trades["pnl_points"])
    overall_sign = np.sign(overall_t)

    sub_tstats = []
    for _, start, end in SUB_PERIODS:
        sub_slice = full_df[(full_df["datetime"] >= start) & (full_df["datetime"] < end)]
        sub_trades = trades_fn(sub_slice)
        sub_tstats.append(tstat(sub_trades["pnl_points"]) if len(sub_trades) >= 5 else np.nan)
    same_sign = sum(1 for t in sub_tstats if not np.isnan(t) and np.sign(t) == overall_sign)

    net = apply_costs(is_trades, LOW_COST)["net_twd"]
    print(f"  {label:38s} n={n:4d}  t={overall_t:+.3f}  淨損益={net.sum():>10,.0f}  "
          f"子區間同號={same_sign}/4  子區間t值=[{', '.join(f'{t:+.2f}' if not np.isnan(t) else 'nan' for t in sub_tstats)}]")
    return dict(label=label, n=n, t_stat=overall_t, sum_net=net.sum(), same_sign_count=same_sign)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < "2021-01-01"]

    print("=" * 70)
    print("0) 基準：鎖定候選（無新機制）")
    print("=" * 70)
    base_cfg = SupportResistanceFadeConfig(**BASE_CFG_KW)
    screen("基準(無新機制)", is_df, df, lambda d: bt_sr(d, base_cfg))

    print("\n" + "=" * 70)
    print("1) 新增固定停利目標 target_r_multiple")
    print("=" * 70)
    for r in [1.0, 1.5, 2.0, 3.0, 4.0]:
        cfg = SupportResistanceFadeConfig(**BASE_CFG_KW, target_r_multiple=r)
        screen(f"target_r_multiple={r}", is_df, df, lambda d, cfg=cfg: bt_sr(d, cfg))

    print("\n" + "=" * 70)
    print("2) 突破當天量能確認濾網（vol_ratio_lag1 >= IS分位數門檻）")
    print("=" * 70)
    vol_ind = daily_indicators(is_df)
    for q in [1 / 3, 0.5, 2 / 3]:
        threshold = vol_ind["vol_ratio_lag1"].quantile(q)

        def fn(d, threshold=threshold):
            raw = bt_sr(d, base_cfg)
            std = _to_std_like(raw)
            filtered = apply_volume_filter(std, d, VolumeFilterSpec(threshold=threshold))
            return filtered

        screen(f"量能濾網(>={q:.0%}分位, thr={threshold:.2f})", is_df, df, fn)

    print("\n" + "=" * 70)
    print("3) 200日均線趨勢對齊濾網")
    print("=" * 70)
    for aligned in [True, False]:
        def fn(d, aligned=aligned):
            raw = bt_sr(d, base_cfg)
            std = _to_std_like(raw)
            filtered = apply_trend_regime_filter(std, d, TrendRegimeFilterSpec(ma_window=200, aligned_only=aligned))
            return filtered

        screen(f"趨勢對齊濾網(aligned_only={aligned})", is_df, df, fn)

    print("\n完成。上面每一行都只用IS(2001-2020)資料，沒有看過OOS(2021-2023)。")


if __name__ == "__main__":
    main()
