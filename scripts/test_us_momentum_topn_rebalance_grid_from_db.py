"""持股檔數（top_n）× 調倉頻率（rebalance_freq_days）交叉網格：兩個維度
合在一起看，有沒有比個別最佳值更好的組合。

背景：10.8 節固定 top_n=20，只測調倉頻率，找到樣本外最佳是 42 天
（Sharpe 0.56）；10.11 節固定 rebalance_freq_days=21，只測持股檔數，
找到樣本外最佳是 top_n=2（Sharpe 0.76）。這兩節都只動一個維度、固定
另一個，沒有測過「top_n=2 搭配 42 天調倉」這種交叉組合會不會比任何
單一維度上找到的最佳值更好——這裡直接做完整的交叉網格，看兩個維度
合在一起有沒有加乘效果，還是各自的最佳值就已經是全局最佳、交叉組合
反而更差。

網格：top_n∈{1,2,3,5,10}（10.11節裡最有意義的範圍，不含已經確認
表現平庸的 20/30/50）× rebalance_freq_days∈{5,10,21,42,63}（10.8節
測過的範圍），共 25 組合，動量窗格固定 momentum_window=126（10.5節
原始網格中段值，不是調參挑選）。

反未來函數：跟前面所有腳本一樣，永遠傳完整 us_prices（不切片），只用
start_date/end_date 限制「哪些日期允許實際調倉」。

**過擬合警語加倍適用**：這是本系列目前網格最大的一次交叉組合搜尋
（25 組合 × 2 期間），「挑出樣本外表現最好的那一格」本身就是在 25 個
選項裡挑最大值，比前面任何單一維度的網格更容易純粹因為選項變多而
剛好挑到運氣好的組合——這裡的樣本外數字比 10.8/10.11 節任何單一維度
的最佳值都更需要用懷疑的角度看待。

2026-09-21 架構修正：改用完整未過濾的 us_prices + membership 參數，
取代先前先用 filter_prices_by_index_membership 預過濾再傳進引擎的舊
寫法（誤傷 MRVL 等 14 檔股票，詳見 tw_quant/us_universe.py 檔頭）；
「樣本外」窗口也改成明確傳 OOS_START，不再依賴 start_date=None 的隱含
語意（資料庫擴充到 2006 年後，None 會混入 2008 危機期間）。

用法：
    python scripts/test_us_momentum_topn_rebalance_grid_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.us_config import build_us_config

MOM_WINDOW = 126
TOP_N_GRID = (1, 2, 3, 5, 10)
REBALANCE_FREQ_GRID = (5, 10, 21, 42, 63)
MIN_TRADES_FOR_RANKING = 10

IN_SAMPLE_START = "2023-09-19"
QQQ_IN_SAMPLE = {"total_return": 0.9693, "cagr": 0.2549, "max_dd": 0.2277, "sharpe": 1.21, "calmar": 1.12}
QQQ_OOS = {"total_return": 1.0782, "cagr": 0.1581, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}
OOS_START = "2018-09-20"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)


def _run(us_prices: pd.DataFrame, base_cfg, top_n: int, rebal: int, start_date, end_date, membership) -> dict:
    factor_cfg = FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=rebal, top_n=top_n, ascending=False)
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs,
        membership=membership,
    )
    return metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)


def _print_period(label: str, us_prices, base_cfg, start_date, end_date, qqq_bench: dict, membership) -> None:
    print(f"\n=== {label} ===")

    rows = []
    for top_n in TOP_N_GRID:
        for rebal in REBALANCE_FREQ_GRID:
            m = _run(us_prices, base_cfg, top_n, rebal, start_date, end_date, membership)
            rows.append({"top_n": top_n, "rebal": rebal, **m})
    df = pd.DataFrame(rows)

    print("\nSharpe 矩陣（列=top_n，欄=調倉頻率天數）：")
    pivot = df.pivot(index="top_n", columns="rebal", values="sharpe")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    print(f"\n（對照：QQQ 買進持有 Sharpe {qqq_bench['sharpe']:.2f}、總報酬 {qqq_bench['total_return']:.2%}）")

    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    print(f"\n前 8 名組合（依 Sharpe 排序，n_trades >= {MIN_TRADES_FOR_RANKING}）：")
    print(
        f"{'top_n':>6} {'rebal':>6} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
        f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"
    )
    for _, r in ranked.head(8).iterrows():
        print(
            f"{int(r['top_n']):>6} {int(r['rebal']):>6} "
            f"{r['total_return']:>9.2%} {r['cagr']:>7.2%} {r['max_dd']:>7.2%} {r['sharpe']:>7.2f} {r['calmar']:>7.2f}  "
            f"{r['n_trades']:>6.0f} {r['win_rate']:>6.1%}"
        )

    best = ranked.iloc[0]
    print(
        f"\n最佳組合：top_n={int(best['top_n'])}、rebalance_freq_days={int(best['rebal'])}，"
        f"Sharpe {best['sharpe']:.2f}（vs QQQ {qqq_bench['sharpe']:.2f}）、"
        f"總報酬 {best['total_return']:.2%}（vs QQQ {qqq_bench['total_return']:.2%}）"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {latest.date()}）")

    base_cfg = build_us_config()
    print(
        f"固定參數：momentum_window={MOM_WINDOW}（10.5 節原始網格中段值，非調參挑選）；"
        f"交叉網格：top_n∈{TOP_N_GRID} × rebalance_freq_days∈{REBALANCE_FREQ_GRID}（共 "
        f"{len(TOP_N_GRID) * len(REBALANCE_FREQ_GRID)} 組合）"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", us_prices, base_cfg, IN_SAMPLE_START, None, QQQ_IN_SAMPLE, membership)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", us_prices, base_cfg, OOS_START, OOS_END, QQQ_OOS, membership)

    print(
        "\n（誠實揭露：25 個組合裡挑「樣本外 Sharpe 最高」的那一格，比 10.8/10.11 節\n"
        "任何單一維度的網格搜尋更容易純粹因為選項變多而挑到運氣好的組合——這裡的\n"
        "「最佳組合」數字要比前面任何一節都更保留地看待，是探索交叉空間的形狀，\n"
        "不是在宣布找到了新的最佳解；如果這裡的最佳組合剛好等於或接近 10.11 節\n"
        "已經找到的 top_n=2/rebalance=21，那至少是一個交叉驗證，比如果换到一個\n"
        "完全不同的新組合更值得信任）"
    )


if __name__ == "__main__":
    main()
