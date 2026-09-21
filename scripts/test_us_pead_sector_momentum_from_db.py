"""美股財報驚喜漂移（PEAD）+ 族群資金流向篩選：先用產業（GICS Sector）
相對動能鎖定「最近資金流向最強」的幾個族群，只在這些族群裡的股票，財報
公布後如果驚喜幅度達標，隔天立刻進場，固定持有一段天數後出場。

背景：使用者確認沿用 PEAD 的進場模式（公布後隔天立刻買進、不是提前布局
賭結果），但想知道先用「族群資金流向」鎖定範圍，會不會篩掉雜訊、讓
剩下的訊號品質更好。這裡直接重用 scripts/test_us_pead_event_driven_from_db.py
的事件驅動引擎（tw_quant/event_drift_backtest.py）跟
scripts/test_us_pead_earnings_drift_from_db.py 的驚喜幅度計算（都是
import 重用，不重複定義），只多一層「事件發生當下，這檔股票的產業是不是
當時動能最強的前 K 名」的篩選——確保這次實驗跟前面兩支 PEAD 腳本的差異
只來自「有沒有族群篩選」這一個變數。

族群資金流向的代理指標：每個交易日，把同產業所有股票的當日報酬率取
等權重平均，當作那個產業當天的報酬，再取 T-1 為止 SECTOR_LOOKBACK_DAYS
天的累積報酬排名——正是「資金流向」在沒有真正資金流量資料時最常見的
價量代理做法（見 compute_sector_momentum_ranks）。這不是真正的資金
流向（沒有成交量所有權變化、法人買賣超這些資料），是用價格動能反推
「這個族群最近是不是被錢推著漲」的間接推論，這個簡化在誠實揭露裡有
說明。

同時印出「有族群篩選」跟「沒有族群篩選（跟 test_us_pead_event_driven_from_db.py
完全一樣的網格設定，只是這裡重新算一次方便並排比較）」兩組結果，才能
乾淨地看出族群篩選本身有沒有幫助，而不是跟先前單獨跑的結果隔了一份
log 用記憶比較。

反未來函數：族群動能排名用 shift(1)，T 日的排名只用到 T-1 為止的產業
報酬；跟事件驅動引擎共用的驚喜幅度/進場延遲反未來函數規則見
tw_quant/event_drift_backtest.py 跟 test_us_pead_earnings_drift_from_db.py
開頭的說明。

2026-09-21 架構修正：改用完整未過濾的 us_prices + membership 參數，
取代先前先用 filter_prices_by_index_membership 預過濾再傳進引擎的舊
寫法（誤傷 MRVL 等 14 檔股票，詳見 tw_quant/us_universe.py 檔頭）；
「樣本外」窗口也改成明確傳 OOS_START，不再依賴 start_date=None 的隱含
語意（資料庫擴充到 2006 年後，None 會混入 2008 危機期間）。

用法：
    python scripts/test_us_pead_sector_momentum_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.data_snapshot import (
    load_us_earnings_snapshot,
    load_us_index_membership_snapshot,
    load_us_prices_snapshot,
)
from tw_quant.event_drift_backtest import EventDriftConfig, run_event_drift_backtest
from tw_quant.us_config import build_us_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_us_pead_earnings_drift_from_db import _earnings_surprise_frame  # noqa: E402

ENTRY_LAG_DAYS = 1  # 固定「隔天立刻買入」，不是提前布局
HOLDING_DAYS_GRID = (20, 40, 60)
SURPRISE_THRESHOLD_GRID = (0.0, 0.05, 0.10)  # 驚喜幅度至少要達到多少才算「達標」
MAX_CONCURRENT_POSITIONS = 20
SECTOR_LOOKBACK_DAYS = 60  # 產業動能用 T-1 為止 60 個交易日的累積報酬
TOP_K_SECTORS = 3  # 只取動能最強的前 3 個產業（共 11 個有效 GICS 產業）
MIN_TRADES_FOR_RANKING = 10

IN_SAMPLE_START = "2023-09-19"
OOS_START = "2018-09-20"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}


def compute_sector_momentum_ranks(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """回傳 (date, industry) -> trailing_ret_shifted 長格式表：每個產業當天
    的報酬取同產業所有股票等權重平均，再算 T-1 為止 lookback_days 天的
    累積報酬、shift(1)。industry 是空字串（缺分類）的股票不計入任何產業
    的平均報酬。
    """
    master = prices[prices["industry"] != ""].sort_values(["stock_id", "date"]).copy()
    master["ret"] = master.groupby("stock_id", sort=False)["close"].pct_change()
    daily_sector_ret = master.groupby(["date", "industry"], sort=False)["ret"].mean().reset_index()
    daily_sector_ret = daily_sector_ret.sort_values(["industry", "date"])

    def _trailing_cum_ret(g: pd.Series) -> pd.Series:
        return (1 + g).rolling(lookback_days, min_periods=lookback_days).apply(lambda x: x.prod() - 1, raw=True)

    daily_sector_ret["trailing_ret"] = daily_sector_ret.groupby("industry", sort=False)["ret"].transform(_trailing_cum_ret)
    daily_sector_ret["trailing_ret_shifted"] = daily_sector_ret.groupby("industry", sort=False)["trailing_ret"].shift(1)
    return daily_sector_ret[["date", "industry", "trailing_ret_shifted"]]


def filter_events_by_sector_momentum(
    events: pd.DataFrame, prices: pd.DataFrame, lookback_days: int, top_k_sectors: int
) -> pd.DataFrame:
    """只保留「事件已知日期當下，這檔股票的產業排在動能前 top_k_sectors 名」
    的事件。沒有產業分類的股票（industry 是空字串）一律排除，因為沒辦法
    判斷它們算不算熱門族群。
    """
    stock_industry = prices.drop_duplicates("stock_id").set_index("stock_id")["industry"]
    sector_ranks = compute_sector_momentum_ranks(prices, lookback_days).dropna(subset=["trailing_ret_shifted"])

    hot_sets = {}
    for date, g in sector_ranks.groupby("date"):
        hot_sets[date] = set(g.nlargest(top_k_sectors, "trailing_ret_shifted")["industry"])

    e = events.copy()
    e["industry"] = e["stock_id"].map(stock_industry)
    e = e[e["industry"].notna() & (e["industry"] != "")]

    sector_dates = sorted(hot_sets.keys())
    left = e.sort_values("known_date")
    idx = pd.DatetimeIndex(sector_dates).searchsorted(left["known_date"], side="right") - 1
    valid = idx >= 0
    keep = pd.Series(False, index=left.index)
    matched_dates = pd.Series(pd.NaT, index=left.index, dtype="datetime64[ns]")
    matched_dates.loc[valid] = pd.DatetimeIndex(sector_dates)[idx[valid]]
    for i in left.index[valid]:
        d = matched_dates.loc[i]
        if left.loc[i, "industry"] in hot_sets.get(d, set()):
            keep.loc[i] = True
    return left[keep].drop(columns="industry")


HEADER = (
    f"{'hold_days':>9} {'threshold':>9}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(holding_days: int, threshold: float, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{holding_days:>9} {threshold:>9.0%}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def _run_grid(us_prices: pd.DataFrame, events: pd.DataFrame, base_cfg, start_date, end_date, membership) -> pd.DataFrame:
    rows = []
    for holding_days in HOLDING_DAYS_GRID:
        for threshold in SURPRISE_THRESHOLD_GRID:
            drift_cfg = EventDriftConfig(
                entry_lag_days=ENTRY_LAG_DAYS, holding_days=holding_days,
                signal_threshold=threshold, max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
            )
            result = run_event_drift_backtest(
                us_prices, events, base_cfg, drift_cfg,
                start_date=start_date, end_date=end_date, cost_module=us_costs, membership=membership,
            )
            m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
            row = {"holding_days": holding_days, "threshold": threshold}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def _print_period(label: str, df: pd.DataFrame, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)

    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["holding_days"]), r["threshold"], r.to_dict()))
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
    n_no_industry = us_prices.drop_duplicates("stock_id")["industry"].eq("").sum()
    print(
        f"讀到 {n_stocks} 檔美股的價量資料（{earliest.date()} ~ {latest.date()}），"
        f"其中 {n_no_industry} 檔沒有產業分類（族群篩選版本會完全排除這些股票）"
    )

    events_all = _earnings_surprise_frame(earnings).rename(columns={"surprise": "signal"})
    events_sector = filter_events_by_sector_momentum(events_all, us_prices, SECTOR_LOOKBACK_DAYS, TOP_K_SECTORS)
    print(
        f"全部財報驚喜事件 {len(events_all)} 筆，經族群動能篩選"
        f"（近 {SECTOR_LOOKBACK_DAYS} 個交易日累積報酬前 {TOP_K_SECTORS} 強的產業）"
        f"後剩下 {len(events_sector)} 筆（{len(events_sector) / len(events_all):.1%}）\n"
    )

    base_cfg = build_us_config()
    print(
        f"固定：entry_lag_days={ENTRY_LAG_DAYS}（隔天立刻買入）；"
        f"網格：holding_days={HOLDING_DAYS_GRID}、驚喜幅度門檻={SURPRISE_THRESHOLD_GRID}\n"
    )

    for variant_label, events in (("沒有族群篩選（對照組）", events_all), ("有族群篩選（前3強產業）", events_sector)):
        print(f"\n########## {variant_label} ##########")
        df_in = _run_grid(us_prices, events, base_cfg, IN_SAMPLE_START, None, membership)
        _print_period(f"{variant_label} - 樣本內（2023-09-19 ~ 資料庫最新日期）", df_in, QQQ_IN_SAMPLE)

        df_oos = _run_grid(us_prices, events, base_cfg, OOS_START, OOS_END, membership)
        _print_period(f"{variant_label} - 樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", df_oos, QQQ_OOS)

    print(
        "\n（誠實揭露：族群資金流向是用價格動能反推的代理指標（同產業股票的\n"
        "平均報酬率排名），不是真正的資金流量/法人買賣超資料，「資金流向」\n"
        "只是借用這個說法描述「最近這個產業的股價動能」；產業分類用目前\n"
        "S&P 500 成分股的 GICS 分類套用到歷史，不是逐年重新分類，公司如果\n"
        "曾經改變主要業務／被重新分類，這裡的產業標籤可能跟當時市場認知不同；\n"
        "74 檔沒有產業分類的股票完全排除在族群篩選版本之外；固定持有天數到就\n"
        "出場、不做停損停利；驚喜幅度公式簡化、財報公布當天盤前/盤後未知等\n"
        "限制，跟前面兩支 PEAD 腳本相同，這裡不重複列。這是第一輪探索結果，\n"
        "不代表已經驗證出可以實際使用的策略。）"
    )


if __name__ == "__main__":
    main()
