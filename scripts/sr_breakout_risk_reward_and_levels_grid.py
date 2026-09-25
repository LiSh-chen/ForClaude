"""支撐壓力順勢突破：風報比（stop_atr_mult x target_r_multiple 聯合網格）
跟壓力支撐點位設定方式（level_quantile，N日高低點 vs 分位數平滑版）兩個
方向的IS特徵化。

**紀律聲明**：這支腳本只用IS(2001-2020)資料+子區間一致性篩選，完全不
產生新的OOS驗證。這個策略分支已經對2021-2023 OOS視窗做過三次驗證
（原始候選 t=0.22、target_r=1.0 t=0.30、target_r=3.0 t=0.28，皆不顯著），
繼續加碼在同一個OOS視窗上測試會讓多重比較問題更嚴重，這裡刻意停在
IS階段——不管這支腳本找到多好看的組合，都不會再回頭驗證OOS。

用法：
    python scripts/sr_breakout_risk_reward_and_levels_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"
LOW_COST = COST_SCENARIOS[0]
BASE_KW = dict(direction_mode="breakout", channel_window=20, min_range_pct=2.0, trend_slope_threshold_pct=5.0)
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


def screen(is_df: pd.DataFrame, full_df: pd.DataFrame, cfg: SupportResistanceFadeConfig) -> dict:
    is_trades = bt_sr(is_df, cfg)
    n = len(is_trades)
    if n < 20:
        return dict(n=n, t_stat=np.nan, sum_net=np.nan, same_sign_count=0, mdd=np.nan)
    overall_t = tstat(is_trades["pnl_points"])
    overall_sign = np.sign(overall_t)

    sub_signs_match = 0
    for _, start, end in SUB_PERIODS:
        sub_slice = full_df[(full_df["datetime"] >= start) & (full_df["datetime"] < end)]
        sub_trades = bt_sr(sub_slice, cfg)
        if len(sub_trades) >= 5:
            t = tstat(sub_trades["pnl_points"])
            if not np.isnan(t) and np.sign(t) == overall_sign:
                sub_signs_match += 1

    net = apply_costs(is_trades, LOW_COST)["net_twd"]
    equity = net.cumsum().reset_index(drop=True)
    mdd = (equity - equity.cummax()).min()
    return dict(n=n, t_stat=overall_t, sum_net=net.sum(), same_sign_count=sub_signs_match, mdd=mdd)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < "2021-01-01"]

    print("=" * 70)
    print("1) 風報比聯合網格：stop_atr_mult x target_r_multiple")
    print("=" * 70)
    rows = []
    stop_mults = [0.5, 1.0, 1.5, 2.0]
    targets = [None, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    print(f"{'stop_atr':>9} {'target_r':>9} {'n':>5} {'t值':>7} {'淨損益':>12} {'MDD':>10} {'子區間一致':>8}")
    for sam in stop_mults:
        for tr in targets:
            cfg = SupportResistanceFadeConfig(**BASE_KW, stop_atr_mult=sam, target_r_multiple=tr)
            r = screen(is_df, df, cfg)
            rows.append(dict(stop_atr_mult=sam, target_r_multiple=tr, **r))
            tr_label = f"{tr:.1f}" if tr is not None else "無停利"
            print(f"{sam:>9.1f} {tr_label:>9} {r['n']:>5} {r['t_stat']:>7.2f} {r['sum_net']:>12,.0f} "
                  f"{r['mdd']:>10,.0f} {r['same_sign_count']:>8}/4")
    grid_df = pd.DataFrame(rows)
    grid_df.to_parquet(OUT_DIR / "sr_breakout_risk_reward_grid.parquet", index=False)

    print("\n最佳前5組合（依t值排序，僅供參考，不代表要拿去驗證OOS）：")
    top5 = grid_df.sort_values("t_stat", ascending=False).head(5)
    print(top5[["stop_atr_mult", "target_r_multiple", "n", "t_stat", "sum_net", "mdd", "same_sign_count"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("2) 壓力支撐點位設定方式：level_quantile（1.0=原始N日極值，<1.0=分位數平滑）")
    print("=" * 70)
    level_rows = []
    print(f"{'level_q':>9} {'n':>5} {'t值':>7} {'淨損益':>12} {'MDD':>10} {'子區間一致':>8}")
    for lq in [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]:
        cfg = SupportResistanceFadeConfig(**BASE_KW, stop_atr_mult=1.0, target_r_multiple=3.0, level_quantile=lq)
        r = screen(is_df, df, cfg)
        level_rows.append(dict(level_quantile=lq, **r))
        print(f"{lq:>9.2f} {r['n']:>5} {r['t_stat']:>7.2f} {r['sum_net']:>12,.0f} {r['mdd']:>10,.0f} {r['same_sign_count']:>8}/4")
    level_df = pd.DataFrame(level_rows)
    level_df.to_parquet(OUT_DIR / "sr_breakout_level_quantile_grid.parquet", index=False)

    print("\n也在原始無停利設定上測一次level_quantile（排除跟target_r交互作用的可能）：")
    for lq in [1.0, 0.9, 0.8]:
        cfg = SupportResistanceFadeConfig(**BASE_KW, stop_atr_mult=1.0, target_r_multiple=None, level_quantile=lq)
        r = screen(is_df, df, cfg)
        print(f"{lq:>9.2f} {r['n']:>5} {r['t_stat']:>7.2f} {r['sum_net']:>12,.0f} {r['mdd']:>10,.0f} {r['same_sign_count']:>8}/4")

    print("\n完成。以上全部只用IS(2001-2020)資料，不會再拿任何組合去看OOS。")


if __name__ == "__main__":
    main()
