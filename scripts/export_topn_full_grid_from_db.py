"""動量集中度策略的完整三維參數網格：top_n × rebalance_freq_days ×
momentum_window，把每一格的摘要指標（兩個期間）跟樣本外逐日權益曲線
都存下來，供前端儀表板做「所有核心參數都可調整」的即時查表用。

背景：使用者要求把回測系統的「所有參數都可選擇是否使用及調整」搬到
先前發布的儀表板上。瀏覽器沒辦法真的即時重跑 577 檔股票 × 8 年價量
資料的 Python 回測引擎（資料量太大，塞不進網頁），可行的做法是把
使用者最常會想調整的三個核心參數的完整交叉網格，用 GitHub Actions
一次真實跑完存起來，前端用下拉選單/滑桿切換，瞬間換算但背後永遠是
預先算好的真實數字，不是現場運算、也不是插補出來的示意值。

網格：
  - top_n ∈ {1,2,3,5,10,20,30,50}（10.11 節測過的完整範圍）
  - rebalance_freq_days ∈ {5,10,21,42,63,126,252}（10.8 節測過的範圍）
  - momentum_window ∈ {63,126,252}（約季/半年/年，10.5 節前言提過的
    常見天數，之前每一節都固定用 126 沒有另外測試這個維度）
  共 8 × 7 × 3 = 168 組合。

跟 10.15 節的差異：10.15 只測 top_n × rebalance_freq_days 兩維（25
組合、momentum_window 固定 126）；這裡額外加上 momentum_window 這個
維度，範圍也從 10.15 的 top_n∈{1,2,3,5,10} 擴大到跟 10.11 節一致的
完整 8 個值，是目前為止最大的一次網格（168 組合 × 2 期間 = 336 次
完整回測）。**過擬合警語加倍適用**：168 個選項裡挑最佳值，比 10.15
節的 25 個選項更容易純粹因為選項變多而挑到運氣好的組合，這裡的目的
是讓使用者自己探索參數空間的形狀、看清楚哪些區域穩健哪些是孤例，
不是要宣布找到了新的最佳解。

反未來函數：跟前面所有腳本一樣，永遠傳完整 us_prices（不切片），只用
start_date/end_date 限制「哪些日期允許實際調倉」。

輸出：
  - data/topn_grid_metrics.parquet：168 組合 × 2 期間的摘要指標
  - data/topn_grid_equity.parquet：168 組合的樣本外逐日權益曲線
    （只存樣本外，是公允比較基準；樣本內只留指標，不畫逐日曲線，
    控制檔案大小——這點會誠實跟使用者說明）

用法：
    python scripts/export_topn_full_grid_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.curve_export import equity_rows_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

TOP_N_GRID = (1, 2, 3, 5, 10, 20, 30, 50)
REBALANCE_FREQ_GRID = (5, 10, 21, 42, 63, 126, 252)
MOM_WINDOW_GRID = (63, 126, 252)

IN_SAMPLE_START = "2023-09-19"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)
PERIODS = [("樣本內", IN_SAMPLE_START, None), ("樣本外", None, OOS_END)]

METRICS_PATH = Path(__file__).resolve().parents[1] / "data" / "topn_grid_metrics.parquet"
EQUITY_PATH = Path(__file__).resolve().parents[1] / "data" / "topn_grid_equity.parquet"


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()
    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    us_prices = filter_prices_by_index_membership(us_prices, membership)
    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    print(f"讀到 {us_prices['stock_id'].nunique()} 檔美股的資料（{earliest.date()} ~ {latest.date()}）")

    base_cfg = build_us_config()
    total_combos = len(TOP_N_GRID) * len(REBALANCE_FREQ_GRID) * len(MOM_WINDOW_GRID)
    print(f"網格：top_n∈{TOP_N_GRID} × rebalance_freq_days∈{REBALANCE_FREQ_GRID} × momentum_window∈{MOM_WINDOW_GRID}（共 {total_combos} 組合）")

    metrics_rows = []
    equity_frames = []
    done = 0
    for period_label, start_date, end_date in PERIODS:
        print(f"\n=== {period_label} ===")
        for mw in MOM_WINDOW_GRID:
            for top_n in TOP_N_GRID:
                for rebal in REBALANCE_FREQ_GRID:
                    factor_cfg = FactorConfig(momentum_window=mw, rebalance_freq_days=rebal, top_n=top_n, ascending=False)
                    result = run_factor_backtest(
                        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs
                    )
                    m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
                    metrics_rows.append(
                        {
                            "period": period_label, "top_n": top_n, "rebalance_freq_days": rebal, "momentum_window": mw,
                            "total_return": m["total_return"], "cagr": m["cagr"], "max_dd": m["max_dd"],
                            "sharpe": m["sharpe"], "calmar": m["calmar"], "n_trades": m["n_trades"], "win_rate": m["win_rate"],
                        }
                    )
                    if period_label == "樣本外":
                        label = f"top_n={top_n}|rebal={rebal}|mw={mw}"
                        equity_frames.append(equity_rows_from_result(period_label, label, result))
                    done += 1
                    if done % 42 == 0:
                        print(f"  已完成 {done}/{total_combos * 2}...")

    metrics_df = pd.DataFrame(metrics_rows)
    equity_df = pd.concat(equity_frames, ignore_index=True)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_parquet(METRICS_PATH, index=False)
    equity_df.to_parquet(EQUITY_PATH, index=False)

    print(f"\n已匯出 {len(metrics_df)} 列網格指標 -> {METRICS_PATH}")
    print(f"已匯出 {len(equity_df)} 列樣本外權益曲線（{equity_df['strategy'].nunique()} 組合）-> {EQUITY_PATH}")


if __name__ == "__main__":
    main()
