"""針對 IC 掃描找到的短期反轉規律，建一個對應的策略並用完整交易成本回測。

背景（scripts/analyze_return_regularities_from_db.py 的發現）：真實 150 檔股票
3 年資料上，trailing 1~5 日報酬 vs forward 1~5 日報酬的 cross-sectional rank IC
在 -0.05 ~ -0.06 之間、t 值 8~10，而且不是單一格、是整片 L=1~5/F=1~5 的區域
都一致顯著為負——過去幾天漲多的股票，接下來幾天傾向回吐；跌多的，傾向反彈。
這是學術文獻記錄過的「短期反轉」現象，跟前幾輪測的「追突破/動量」邏輯方向
完全相反，也解釋了為什麼超跌反彈（D 策略）是目前唯一 EV 為正的技術面策略。

這支腳本直接把這個訊號做成策略：用 tw_quant/factor_backtest.py 的調倉引擎
（ascending=True，也就是買排名後段/最弱的股票），但持有天數改成貼著訊號
本身的天數（1~5 天）而不是像 D 那樣用吊燈停利拖到平均 28 天。

★ 關鍵風險：持有 1~5 天、頻繁調倉，來回一趟交易成本（證交稅 0.3% +
手續費雙邊 0.285% + 出場滑價）大概 0.6%+，可能跟訊號本身的量級是同一個
數量級。這支腳本用的是系統既有的完整成本模型（tw_quant/costs.py），
所以這裡的報酬數字已經是扣完成本後的淨值，能直接回答「訊號是真的，
但活不活得過交易成本」這個問題。

用法：
    python scripts/explore_reversal_strategy_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store

MIN_TRADES_FOR_RANKING = 20
MOMENTUM_L_GRID = (1, 2, 3, 5)  # 對應 IC 掃描裡訊號最強的觀察窗
REBALANCE_F_GRID = (1, 3, 5)  # 對應 IC 掃描裡訊號最強的預測窗，即持有天數
TOP_N_GRID = (10, 15, 20, 30)


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


HEADER = (
    f"{'param':<20}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>6} {'annual':>7} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<20}  {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
        f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f}  {m['n_trades']:>6.0f} {m['annual_trades']:>7.0f} "
        f"{m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    base_cfg = build_base_config()

    rows = []
    print(f"總共搜尋 {len(MOMENTUM_L_GRID) * len(REBALANCE_F_GRID) * len(TOP_N_GRID)} 組短期反轉參數組合...\n")
    print(HEADER)
    for L in MOMENTUM_L_GRID:
        for F in REBALANCE_F_GRID:
            for top_n in TOP_N_GRID:
                factor_cfg = FactorConfig(
                    momentum_window=L, rebalance_freq_days=F, top_n=top_n, ascending=True
                )
                result = run_factor_backtest(prices, base_cfg, factor_cfg)
                m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
                label = f"L{L}/F{F}/n{top_n}"
                row = {"label": label, "L": L, "F": F, "top_n": top_n}
                row.update(m)
                rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)

    print("\n=== 依 Sharpe 排序前 20 名 ===")
    print(HEADER)
    for _, r in ranked.head(20).iterrows():
        print(_fmt_row(r["label"], r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")
        return

    best = ranked.iloc[0]
    print(f"\n=== 最佳組合逐年拆解：{best['label']} ===")
    best_cfg = FactorConfig(
        momentum_window=int(best["L"]), rebalance_freq_days=int(best["F"]),
        top_n=int(best["top_n"]), ascending=True,
    )

    data_start, data_end = prices["date"].min(), prices["date"].max()
    year_bounds = pd.date_range(data_start, data_end, freq="365D")
    if year_bounds.empty or year_bounds[-1] < data_end:
        year_bounds = year_bounds.append(pd.DatetimeIndex([data_end]))
    for year_start, year_end in zip(year_bounds[:-1], year_bounds[1:]):
        # 指標計算永遠餵完整 prices + start/end_date 門檻，不能先把 prices 切片，
        # 否則魚池篩選會被迫從切片起點重新累積暖身期（見 run_factor_backtest 說明）。
        yr_result = run_factor_backtest(prices, base_cfg, best_cfg, start_date=year_start, end_date=year_end)
        # 這裡的 prices 只用來取「這段期間結束時」的收盤價幫還未平倉部位計算
        # 虛擬平倉損益，切片沒問題（不是用來算指標）。
        yr_prices_for_marking = prices[(prices["date"] >= year_start) & (prices["date"] <= year_end)]
        yr_m = metrics_from_result(yr_result, base_cfg.initial_capital, prices=yr_prices_for_marking)
        print(f"  {year_start.date()} ~ {year_end.date()}:")
        print(f"    {_fmt_row(best['label'], yr_m)}")

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "報酬數字已扣除證交稅/手續費/出場滑價的完整成本模型；n_trades/win%/RR/EV%/PF 已把"
        "回測結束時還未平倉的部位用最後收盤價算進去；逐年拆解用的是「每段區間各自獨立重跑」，"
        "不是連續複利的一部分，且每段開頭都要重新累積魚池的暖身期資料，數字僅供粗略參考）"
    )


if __name__ == "__main__":
    main()
