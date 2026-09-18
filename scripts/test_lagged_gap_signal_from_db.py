"""測試「延遲一天、可執行版本」的跳空缺口規律。

背景：analyze_calendar_effects_from_db.py 發現「今天開盤跳空幅度」跟「今天
盤中報酬（收盤vs開盤）」呈現極強的負相關（mean IC=-0.25, t=-54.82），但這是
同一天之內的事，我們的回測引擎架構是「T 日收盤算訊號、T+1 日開盤才進場」，
天生沒辦法在「今天開盤看到跳空」的當下就操作同一天的走勢，所以這個訊號雖然
最強，但現有架構直接不能用。

這裡改問一個可執行版本的問題：T 日收盤時已經知道的兩個特徵——
  (a) T 日的跳空幅度本身（開盤 vs 前一天收盤）
  (b) T 日的盤中報酬（收盤 vs 開盤，也就是缺口回補了多少）
能不能預測 T+1（甚至更久）的未來報酬？這是完全可執行的：訊號在 T 日收盤已知，
T+1 日開盤進場，符合整個系統一貫的反未來函數規則。

先印 Rank IC 掃描結果（跟 analyze_return_regularities_from_db.py 同一套方法），
再對看起來最有希望的訊號用 tw_quant/factor_backtest.py 的 signal_fn 機制
（本次新加的功能）建一個真正的調倉策略，套用完整交易成本回測。

用法：
    python scripts/test_lagged_gap_signal_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store

MIN_STOCKS_PER_DAY = 30
FORWARD_F_GRID = (1, 3, 5)
MIN_TRADES_FOR_RANKING = 15


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


def _raw_gap_and_intraday(master: pd.DataFrame) -> pd.DataFrame:
    prior_close = master.groupby("stock_id", sort=False)["close"].shift(1)
    gap_pct = ((master["open"] - prior_close) / prior_close).replace([np.inf, -np.inf], np.nan)
    intraday_ret = ((master["close"] - master["open"]) / master["open"]).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"date": master["date"].values, "gap_pct": gap_pct.values, "intraday_ret": intraday_ret.values})


def _rank_ic_series(df: pd.DataFrame, sig_col: str, fwd_col: str) -> pd.Series:
    valid = df[["date", sig_col, fwd_col]].dropna()
    counts = valid.groupby("date").size()
    ok_dates = counts[counts >= MIN_STOCKS_PER_DAY].index
    valid = valid[valid["date"].isin(ok_dates)].copy()
    if valid.empty:
        return pd.Series(dtype=float)
    valid["sig_rank"] = valid.groupby("date")[sig_col].rank()
    valid["fwd_rank"] = valid.groupby("date")[fwd_col].rank()
    return valid.groupby("date").apply(lambda g: g["sig_rank"].corr(g["fwd_rank"]), include_groups=False)


def _ic_summary(ic_series: pd.Series) -> dict:
    ic_series = ic_series.dropna()
    if len(ic_series) < 5:
        return {"mean_ic": np.nan, "std_ic": np.nan, "t_stat": np.nan, "n_days": 0}
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    n = len(ic_series)
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    return {"mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t_stat, "n_days": n}


def run_ic_scan(master: pd.DataFrame) -> pd.DataFrame:
    raw = _raw_gap_and_intraday(master)
    rows = []
    for sig_name in ("gap_pct", "intraday_ret"):
        for F in FORWARD_F_GRID:
            fwd_col = f"_fwd_{F}"
            master[fwd_col] = master.groupby("stock_id", sort=False)["close"].transform(
                lambda s: s.shift(-F) / s - 1
            )
            tmp = raw.copy()
            tmp[fwd_col] = master[fwd_col].values
            ic = _rank_ic_series(tmp, sig_name, fwd_col)
            summary = _ic_summary(ic)
            rows.append({"signal": sig_name, "F": F, **summary})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 可執行版本：用 factor_backtest 的 signal_fn 機制建策略
# ---------------------------------------------------------------------------


def gap_pct_signal_fn(master: pd.DataFrame) -> pd.Series:
    raw = _raw_gap_and_intraday(master)
    raw.index = master.index
    return ind.shift_by_group(raw["gap_pct"], master, periods=1)


def intraday_ret_signal_fn(master: pd.DataFrame) -> pd.Series:
    raw = _raw_gap_and_intraday(master)
    raw.index = master.index
    return ind.shift_by_group(raw["intraday_ret"], master, periods=1)


HEADER = (
    f"{'signal':<14} {'ascending':<10} {'F':>3} {'top_n':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(signal: str, ascending: bool, F: int, top_n: int, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{signal:<14} {str(ascending):<10} {F:>3} {top_n:>6}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
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

    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()

    print("=== Rank IC 掃描：T 日跳空/盤中報酬 -> forward F 日報酬 ===")
    ic_df = run_ic_scan(master)
    print(f"{'signal':<14} {'F':>4}  {'mean_IC':>9} {'std_IC':>8} {'t_stat':>7} {'n_days':>7}")
    for _, r in ic_df.sort_values("mean_ic", key=lambda s: s.abs(), ascending=False).iterrows():
        print(f"{r['signal']:<14} {r['F']:>4}  {r['mean_ic']:>9.4f} {r['std_ic']:>8.4f} {r['t_stat']:>7.2f} {r['n_days']:>7.0f}")

    print("\n=== 可執行版本回測（訊號用 T-1 日已知資訊，T 日開盤進場，套用完整交易成本）===")
    print(HEADER)

    base_cfg = build_base_config()
    signal_fns = {"gap_pct": gap_pct_signal_fn, "intraday_ret": intraday_ret_signal_fn}

    rows = []
    for signal_name, fn in signal_fns.items():
        for ascending in (True, False):
            for F in FORWARD_F_GRID:
                for top_n in (15, 20, 30):
                    factor_cfg = FactorConfig(rebalance_freq_days=F, top_n=top_n, ascending=ascending)
                    result = run_factor_backtest(prices, base_cfg, factor_cfg, signal_fn=fn)
                    m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
                    row = {"signal": signal_name, "ascending": ascending, "F": F, "top_n": top_n}
                    row.update(m)
                    rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(20).iterrows():
        print(_fmt_row(r["signal"], r["ascending"], int(r["F"]), int(r["top_n"]), r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "n_trades/win%/RR/EV%/PF 已把回測結束時還未平倉的部位用最後收盤價算進去；"
        "ascending=True 買排名後段（弱勢/大幅跳空回補後)，False 買排名前段）"
    )


if __name__ == "__main__":
    main()
