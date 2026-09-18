"""測試「股本越小的股票，同一個策略的表現越好」這個假設。

背景：使用者提供的 Gemini 研究文件核心論點是機構因為資金部位過大、
流動性不足，無法參與中小型股，所以中小型股會有法人研究覆蓋率低、市場
效率較差帶來的超額報酬空間。但目前為止測過的所有策略都是在同一組 150
檔股票 universe 上跑的，而這個 universe 其實不是照市值/流動性選的——
`scripts/generate_stock_universe.py` 只是「全市場清單依股票代號排序取前
150 檔」，台股代號大致依上市時間分配，1000~1999 號段幾乎全是食品/紡織/
水泥/塑膠這類最早上市的傳產龍頭股，完全不是文件講的「中小型/冷門股」。
換句話說，先前所有結果驗證的都是「這些策略在大型老牌股上有沒有效」，
從來沒有真正測過股本大小這個維度。

這裡用 FinMind 的 TaiwanStockShareholding 資料集（見
tw_quant/data_provider.py 的 fetch_shares_issued 說明，這個 dataset 主要
是外資持股用途，已發行股數只是附帶欄位）算出每檔股票的股本，把「現有
150 檔 universe」依股本切成大/小兩半，同一個策略分別在兩半上跑，看
股本小的那一半表現是不是真的比較好。

★ 誠實揭露的限制（這不是真正的「中小型股 vs 大型股」絕對比較）：
  1. Universe 本身就偏大型/老牌股（見上），這裡的「小股本」只是這個
     偏態樣本裡「相對比較小」的一半，不是台股全市場定義下的中小型股。
     真的要驗證文件的論點，需要重新設計 universe 選股邏輯（例如改成依
     成交金額或股本排序取樣、涵蓋真正的中小型股），這裡沒有做，是刻意
     的範圍限制。
  2. 股本 = 已發行股數 * 10（假設面額新台幣 10 元，台股絕大多數股票是
     這個面額，少數特殊面額股票這裡會算錯）/ 100,000,000（換算成「億元」）。
  3. 用每檔股票資料庫裡最早一筆已發行股數當股本分類依據（不是逐日更新
     的股本），因為股本變動很慢（現金增資/減資才會變），用回測期間開始
     時的股本切分，比用最新股本切分更貼近「當時是不是小型股」，但仍然
     是靜態分類，沒有逐日追蹤股本變化。

用法：
    python scripts/test_market_cap_effect_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import run_backtest
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store
from scripts.test_holiday_overlay_from_db import mss_signal_fn
from scripts.test_pead_revenue_drift_from_db import attach_revenue_signal, revenue_signal_fn

PAR_VALUE = 10.0  # 台股絕大多數股票的面額（新台幣元）
SMALL_CAP_THRESHOLD_YI = 50.0  # 文件講的「股本 > 50 億排除」門檻，這裡拿來對照真實分布


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    cfg.sizing.atr_multiplier = 2.5
    cfg.sizing.chandelier_lookback = 10
    return cfg


def compute_share_capital_yi(shares_issued: pd.DataFrame) -> pd.Series:
    """每檔股票的股本（億元），用資料庫裡最早一筆已發行股數當代表值。"""
    earliest = shares_issued.sort_values("date").groupby("stock_id", sort=False).first()
    return earliest["shares_issued"] * PAR_VALUE / 1e8


def split_universe_by_cap(share_capital_yi: pd.Series) -> tuple[list[str], list[str]]:
    """依股本中位數切成大/小兩半（不是用文件的 50 億絕對門檻，因為 universe
    本身偏大型股，用絕對門檻可能只剩幾檔甚至沒有股票，見主程式印出的分布）。
    """
    median = share_capital_yi.median()
    small = share_capital_yi[share_capital_yi <= median].index.tolist()
    large = share_capital_yi[share_capital_yi > median].index.tolist()
    return small, large


HEADER = (
    f"{'strategy':<12} {'universe':<10} {'n_stk':>5}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>5} {'win%':>7} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(strategy: str, universe: str, n_stocks: int, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    return (
        f"{strategy:<12} {universe:<10} {n_stocks:>5}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>5.0f} {m['win_rate']:>6.1%} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin_short = store.load_margin_short()
    revenue = store.load_month_revenue()
    shares_issued = store.load_shares_issued()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)
    if shares_issued.empty:
        print("資料庫裡沒有已發行股數資料，需要先跑過一次 ingest_daily_data.py。", file=sys.stderr)
        sys.exit(1)

    share_capital_yi = compute_share_capital_yi(shares_issued)
    print(f"有股本資料的股票數：{len(share_capital_yi)} / {prices['stock_id'].nunique()}")
    print(share_capital_yi.describe())
    n_below_50yi = (share_capital_yi <= SMALL_CAP_THRESHOLD_YI).sum()
    print(f"\n股本 <= 50 億（文件的中小型股門檻）的股票數：{n_below_50yi} / {len(share_capital_yi)}")

    small_ids, large_ids = split_universe_by_cap(share_capital_yi)
    print(f"\n依股本中位數切分：小股本組 {len(small_ids)} 檔（中位數以下），大股本組 {len(large_ids)} 檔（中位數以上）")
    print(f"小股本組股本範圍：{share_capital_yi[small_ids].min():.1f} ~ {share_capital_yi[small_ids].max():.1f} 億")
    print(f"大股本組股本範圍：{share_capital_yi[large_ids].min():.1f} ~ {share_capital_yi[large_ids].max():.1f} 億\n")

    base_cfg = build_base_config()
    universes = {"全部150檔": None, "小股本組": small_ids, "大股本組": large_ids}

    print("=== 策略 C（結構轉折+量能確認）依股本分組比較 ===")
    print(HEADER)
    for label, ids in universes.items():
        sub_prices = prices if ids is None else prices[prices["stock_id"].isin(ids)]
        sub_margin = margin_short if ids is None else margin_short[margin_short["stock_id"].isin(ids)]
        result = run_backtest(sub_prices, sub_margin, base_cfg, historical_mdd=None, entry_signal_fn=mss_signal_fn)
        m = metrics_from_result(result, base_cfg.initial_capital, prices=sub_prices)
        print(_fmt_row("策略C", label, sub_prices["stock_id"].nunique(), m))

    print("\n=== PEAD 月營收意外漂移（最佳參數：季調倉/top20）依股本分組比較 ===")
    print(HEADER)
    factor_cfg = FactorConfig(rebalance_freq_days=63, top_n=20, ascending=False)
    for label, ids in universes.items():
        sub_prices = prices if ids is None else prices[prices["stock_id"].isin(ids)]
        sub_revenue = revenue if ids is None else revenue[revenue["stock_id"].isin(ids)]
        master_with_signal = attach_revenue_signal(sub_prices, sub_revenue)
        result = run_factor_backtest(master_with_signal, base_cfg, factor_cfg, signal_fn=revenue_signal_fn)
        m = metrics_from_result(result, base_cfg.initial_capital, prices=sub_prices)
        print(_fmt_row("PEAD", label, sub_prices["stock_id"].nunique(), m))

    print(
        "\n（EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "股本切分見檔頭「誠實揭露的限制」——universe 本身偏大型/老牌股，"
        "這裡的「小股本組」不是台股全市場定義下的中小型股，只是現有樣本裡相對較小的一半）"
    )


if __name__ == "__main__":
    main()
