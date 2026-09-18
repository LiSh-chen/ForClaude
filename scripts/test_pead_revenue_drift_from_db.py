"""用真實資料測試「盈餘宣告後漂移」（PEAD）的月營收代理版本。

背景：PEAD 原本是用財報 EPS 超乎市場預期後的股價持續漂移，這裡沒有分析師
共識預期資料，改用台灣月營收公告當代理——每月營收年增率（YoY growth）跟
「近半年平均年增率」的落差當作「意外」（surprise）的代理指標：如果這個月
營收年增率明顯高於近半年的平均水準，視為正向意外，賭市場還沒完全消化這個
資訊、股價會繼續往上漂移數天到數週；買進意外程度排名最高的股票，持有到
下次調倉。

反未來函數（這是這個策略最關鍵的部分，務必看仔細）：
FinMind 的 TaiwanStockMonthRevenue 資料集的 date 欄位是「營收所屬月份」
（例如 2024-01-01 代表 1 月營收本身），**不是公告日期**。台灣上市櫃公司
依規定要在次月 10 日前公告月營收，這裡保守假設要到「次月 15 號」才算
「已公開可用」（比法定期限多留 5 天緩衝，因應少數公司延遲公告或資料
更新延遲），這個 known_date 用 merge_asof「往回找」對齊到每個交易日，
再額外 shift(1) 一天（比照 factor_backtest.py signal_fn 的既有慣例：
回傳值要是「T-1 日已經確定已知」的版本，本函式不會再另外位移）。

用法：
    python scripts/test_pead_revenue_drift_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store

DISCLOSURE_BUFFER_DAY = 15  # 保守假設：次月幾號之後才算已公開可用
TRAILING_MONTHS = 6  # 算「意外」的基準：近幾個月平均年增率
REBALANCE_FREQ_GRID = (21, 42, 63)  # 約 1 / 2 / 3 個月調倉一次
TOP_N_GRID = (10, 20, 30)
MIN_TRADES_FOR_RANKING = 10


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    return cfg


def _revenue_surprise_frame(revenue: pd.DataFrame) -> pd.DataFrame:
    """把逐檔逐月的營收，轉成「年增率意外」+「幾號之後才算公開已知」。"""
    rev = revenue.dropna(subset=["revenue", "revenue_year", "revenue_month"]).copy()
    if rev.empty:
        return pd.DataFrame(columns=["stock_id", "known_date", "surprise"])
    rev["revenue_year"] = rev["revenue_year"].astype(int)
    rev["revenue_month"] = rev["revenue_month"].astype(int)
    rev["period"] = rev["revenue_year"] * 12 + rev["revenue_month"]

    def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
        g = g.drop_duplicates("period").set_index("period").sort_index()
        full_index = pd.RangeIndex(g.index.min(), g.index.max() + 1)
        # 補齊資料裡缺掉的月份成 NaN，這樣 shift(12)/rolling(6) 才不會把
        # 「隔了好幾個月才有資料」誤當成「剛好差一年/半年」
        g = g.reindex(full_index)
        g["yoy_growth"] = g["revenue"] / g["revenue"].shift(12) - 1
        g["trailing_avg_growth"] = g["yoy_growth"].shift(1).rolling(TRAILING_MONTHS, min_periods=3).mean()
        g["surprise"] = g["yoy_growth"] - g["trailing_avg_growth"]
        return g

    out = rev.groupby("stock_id", sort=False, group_keys=True).apply(_per_stock, include_groups=False)
    out = out.reset_index().rename(columns={"level_1": "period"})
    out = out.dropna(subset=["surprise"])
    if out.empty:
        return pd.DataFrame(columns=["stock_id", "known_date", "surprise"])

    known_period = out["period"] + 1  # 次月才算可能公告完成
    known_year = (known_period - 1) // 12
    known_month = (known_period - 1) % 12 + 1
    out["known_date"] = pd.to_datetime(
        {"year": known_year, "month": known_month, "day": DISCLOSURE_BUFFER_DAY}
    )
    return out[["stock_id", "known_date", "surprise"]].sort_values(["stock_id", "known_date"])


def attach_revenue_signal(prices: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    """把「T 日以前最新一筆已公開營收意外」對齊進 master 價量表，並 shift(1)
    符合 factor_backtest signal_fn 「回傳值須為 T-1 已知」的慣例。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    surprise_frame = _revenue_surprise_frame(revenue)
    if surprise_frame.empty:
        master["revenue_surprise_shifted"] = float("nan")
        return master

    left = master[["stock_id", "date"]].reset_index().rename(columns={"index": "_orig_idx"})
    right = surprise_frame.rename(columns={"known_date": "date"})
    merged = pd.merge_asof(
        left.sort_values("date"), right.sort_values("date"),
        on="date", by="stock_id", direction="backward",
    ).sort_values("_orig_idx").reset_index(drop=True)

    master["revenue_surprise_raw"] = merged["surprise"].values
    master["revenue_surprise_shifted"] = ind.shift_by_group(master["revenue_surprise_raw"], master, periods=1)
    return master


def revenue_signal_fn(master: pd.DataFrame) -> pd.Series:
    return master["revenue_surprise_shifted"]


HEADER = (
    f"{'rebalance':>9} {'top_n':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(rebalance: int, top_n: int, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{rebalance:>9} {top_n:>6}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    revenue = store.load_month_revenue()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)
    if revenue.empty:
        print(
            "資料庫裡沒有任何月營收資料——ingest_daily_data.py 剛擴充加入月營收抓取，"
            "需要先跑過至少一次每日排程（或手動跑一次 ingest）才會有資料可用。",
            file=sys.stderr,
        )
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    n_revenue_stocks = revenue["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的價量資料（{prices['date'].min().date()} ~ {prices['date'].max().date()}），"
        f"{n_revenue_stocks} 檔股票有月營收資料（{revenue['date'].min().date()} ~ {revenue['date'].max().date()}）\n"
    )

    master_with_signal = attach_revenue_signal(prices, revenue)
    n_known = master_with_signal["revenue_surprise_shifted"].notna().sum()
    print(f"價量表裡有 {n_known} / {len(master_with_signal)} 列有已知的營收意外訊號可用\n")

    base_cfg = build_base_config()

    print("=== 月營收意外漂移回測（因子式定期調倉，完整交易成本）===")
    print(HEADER)

    rows = []
    for rebalance in REBALANCE_FREQ_GRID:
        for top_n in TOP_N_GRID:
            factor_cfg = FactorConfig(rebalance_freq_days=rebalance, top_n=top_n, ascending=False)
            result = run_factor_backtest(master_with_signal, base_cfg, factor_cfg, signal_fn=revenue_signal_fn)
            m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
            row = {"rebalance": rebalance, "top_n": top_n}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["rebalance"]), int(r["top_n"]), r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名，以下是全部組合：")
        print(df.to_string(index=False))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "訊號 = 當月營收年增率 - 近6個月平均年增率，買排名最高（最正向意外）的股票；"
        f"保守假設營收要到次月 {DISCLOSURE_BUFFER_DAY} 號才算公開已知）"
    )


if __name__ == "__main__":
    main()
