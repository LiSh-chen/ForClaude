"""美股財報驚喜漂移（PEAD, Post-Earnings-Announcement Drift）策略：用真實
財報公布日 + EPS 預期/實際值，測試「財報意外程度」能不能預測接下來一段
時間的股價漂移。

背景：2026-09-19 用 scripts/probe_us_fundamentals_data.py 探路過，確認
yfinance 的 earnings_dates 資料涵蓋度 100%、平均可回溯約 10~12 年、公布
日期幾乎都是真正的公布日（不是季末代理日期），三個候選的財報基本面方向
（PEAD／營收成長動能／valuation因子）裡，PEAD 是資料最堪用的一個，這裡
接著建立正式回測。

跟 scripts/test_pead_revenue_drift_from_db.py（台股月營收版本）的差異：
  - 這裡有真正的財報公布日期（yfinance 直接給），不需要像月營收那樣
    另外假設「次月幾號才算公開已知」的緩衝日；公布當天盤前/盤後未知的
    模糊性，靠沿用既有慣例的 shift(1) 保守處理，見 attach_earnings_signal
    的說明。
  - 「驚喜幅度」自己用 (實際EPS-預期EPS)/abs(預期EPS) 計算，不直接用
    yfinance 自帶的 Surprise(%) 欄位——避免依賴一個沒辦法從外部稽核公式
    定義的第三方欄位。EPS 預期值接近 0 時這個比例會被放大到不合理的
    量級，這裡沒有做極端值處理（winsorize），是已知的簡化，見腳本結尾
    的誠實揭露。

跟前面所有腳本一樣，重用因子式定期調倉引擎（tw_quant.factor_backtest），
不是逐股在財報公布後立即進場、持有固定天數的「教科書版」事件驅動實作
——這是跟 TW 月營收版本相同的簡化。

反未來函數：排名依據（驚喜幅度，不是動量）永遠用完整 us_prices/
us_earnings（不切片），只用 start_date/end_date 限制交易日期；訊號本身
用 merge_asof 對齊財報公布日、再 shift(1)，T 日的訊號只用到 T-1 為止
已知的財報公布資訊。

2026-09-21 架構修正：改用完整未過濾的 us_prices + membership 參數，
取代先前先用 filter_prices_by_index_membership 預過濾再傳進引擎的舊
寫法（誤傷 MRVL 等 14 檔股票，詳見 tw_quant/us_universe.py 檔頭）；
「樣本外」窗口也改成明確傳 OOS_START，不再依賴 start_date=None 的隱含
語意（資料庫擴充到 2006 年後，None 會混入 2008 危機期間）。

用法：
    python scripts/test_us_pead_earnings_drift_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.data_snapshot import (
    load_us_earnings_snapshot,
    load_us_index_membership_snapshot,
    load_us_prices_snapshot,
)
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.us_config import build_us_config

REBALANCE_FREQ_GRID = (21, 42, 63)
TOP_N_GRID = (5, 10, 20, 30)
MIN_TRADES_FOR_RANKING = 10

IN_SAMPLE_START = "2023-09-19"
OOS_START = "2018-09-20"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}


def _earnings_surprise_frame(earnings: pd.DataFrame) -> pd.DataFrame:
    """把逐檔逐次的財報公布資料，轉成「EPS 驚喜幅度」+「已公開已知日期」。
    未實際公布（eps_actual 是 NaN，代表還沒發生的未來排定財報日）的列
    直接丟掉，這種列沒有意外程度可言。
    """
    e = earnings.dropna(subset=["eps_actual", "eps_estimate"]).copy()
    empty_cols = ["stock_id", "known_date", "surprise"]
    if e.empty:
        return pd.DataFrame(columns=empty_cols)

    safe_estimate = e["eps_estimate"].where(e["eps_estimate"].abs() > 1e-6)
    e["surprise"] = (e["eps_actual"] - e["eps_estimate"]) / safe_estimate.abs()
    e = e.dropna(subset=["surprise"])
    if e.empty:
        return pd.DataFrame(columns=empty_cols)
    e = e.rename(columns={"date": "known_date"})
    return e[["stock_id", "known_date", "surprise"]].sort_values(["stock_id", "known_date"])


def attach_earnings_signal(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """把「T 日以前最新一筆已公開財報驚喜」對齊進 master 價量表，並
    shift(1) 符合 factor_backtest signal_fn「回傳值須為 T-1 已知」的慣例
    ——同時也是對「財報公布當天盤前/盤後未知」這個模糊性的保守處理：
    假設公布當天還不能交易，要等到下一個交易日才反映。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    surprise_frame = _earnings_surprise_frame(earnings)
    if surprise_frame.empty:
        master["eps_surprise_shifted"] = float("nan")
        return master

    left = master[["stock_id", "date"]].reset_index().rename(columns={"index": "_orig_idx"})
    left["date"] = left["date"].astype("datetime64[ns]")
    right = surprise_frame.rename(columns={"known_date": "date"})
    right["date"] = right["date"].astype("datetime64[ns]")
    merged = pd.merge_asof(
        left.sort_values("date"), right.sort_values("date"),
        on="date", by="stock_id", direction="backward",
    ).sort_values("_orig_idx").reset_index(drop=True)

    master["eps_surprise_raw"] = merged["surprise"].values
    master["eps_surprise_shifted"] = ind.shift_by_group(master["eps_surprise_raw"], master, periods=1)
    return master


def earnings_signal_fn(master: pd.DataFrame) -> pd.Series:
    return master["eps_surprise_shifted"]


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


def _run_grid(master_with_signal: pd.DataFrame, base_cfg, start_date, end_date, us_prices: pd.DataFrame, membership) -> pd.DataFrame:
    rows = []
    for rebalance in REBALANCE_FREQ_GRID:
        for top_n in TOP_N_GRID:
            factor_cfg = FactorConfig(rebalance_freq_days=rebalance, top_n=top_n, ascending=False)
            result = run_factor_backtest(
                master_with_signal, base_cfg, factor_cfg,
                start_date=start_date, end_date=end_date,
                signal_fn=earnings_signal_fn, cost_module=us_costs, membership=membership,
            )
            m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
            row = {"rebalance": rebalance, "top_n": top_n}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def _print_period(label: str, df: pd.DataFrame, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)

    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["rebalance"]), int(r["top_n"]), r.to_dict()))
    if ranked.empty:
        print("沒有任何組合達到最低成交筆數門檻，無法排名，以下是全部組合：")
        print(df.to_string(index=False))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")
    print(
        f"（對照：QQQ 買進持有同期間總報酬 {qqq_bench['total_return']:.2%}、"
        f"CAGR {qqq_bench['cagr']:.2%}、MDD {qqq_bench['max_dd']:.2%}、"
        f"Sharpe {qqq_bench['sharpe']:.2f}、Calmar {qqq_bench['calmar']:.2f}）"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()
    earnings = load_us_earnings_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)
    if earnings.empty:
        print(
            "快照裡沒有任何美股財報公布資料（data/us_earnings_snapshot.parquet 是空的）"
            "——需要先跑過 ingest_us_earnings_data.yml。",
            file=sys.stderr,
        )
        sys.exit(1)

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的價量資料（{earliest.date()} ~ {latest.date()}）")

    n_stocks_earnings = earnings["stock_id"].nunique()
    print(
        f"讀到 {n_stocks_earnings} 檔股票的財報公布資料"
        f"（{earnings['date'].min().date()} ~ {earnings['date'].max().date()}）"
    )

    base_cfg = build_us_config()
    master_with_signal = attach_earnings_signal(us_prices, earnings)
    n_known = master_with_signal["eps_surprise_shifted"].notna().sum()
    print(f"價量表裡有 {n_known} / {len(master_with_signal)} 列有已知的財報驚喜訊號可用\n")
    print(f"固定網格：rebalance_freq_days={REBALANCE_FREQ_GRID}、top_n={TOP_N_GRID}\n")

    df_in = _run_grid(master_with_signal, base_cfg, IN_SAMPLE_START, None, us_prices, membership)
    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", df_in, QQQ_IN_SAMPLE)

    df_oos = _run_grid(master_with_signal, base_cfg, OOS_START, OOS_END, us_prices, membership)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", df_oos, QQQ_OOS)

    print(
        "\n（誠實揭露：這裡用因子式定期調倉引擎（跟前面所有腳本共用同一套），\n"
        "不是逐股在財報公布後立即進場、持有固定天數的「教科書版」PEAD 事件驅動\n"
        "實作——調倉日之間發生的財報驚喜，要等到下一個排定的調倉日才會被納入\n"
        "排名考慮，跟真正的「財報公布後幾天內立刻反應」有落差；驚喜幅度自己用\n"
        "(實際EPS-預期EPS)/abs(預期EPS) 計算，不用 yfinance 自帶的 Surprise(%)\n"
        "欄位，EPS 預期值接近 0 時這個比例會被放大到不合理的量級，這裡沒有做\n"
        "極端值處理（winsorize）；財報公布當天盤前/盤後未知，統一保守假設要等\n"
        "下一個交易日才反映，可能低估或高估實際的漂移速度；存活者偏差只部分\n"
        "修正，被剔除指數且 yfinance 查無資料的公司依然完全排除在外。這是第一輪\n"
        "探索結果，不代表已經驗證出可以實際使用的策略。）"
    )


if __name__ == "__main__":
    main()
