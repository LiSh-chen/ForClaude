"""日線資料能測到的「類日內時段」效應。

背景：使用者想測「當沖最佳時段（例如上午9~10點）」這種日內時間窗策略，但
FinMind 目前接的是日線 OHLCV，沒有盤中分鐘級資料，字面上的日內時段濾網
技術上做不到。這裡改測日線解析度下同樣精神的「特定時間點表現比較好/比較差」
規律：

  A. 星期幾效應（修正前一版的 bug：某些股票的收盤價數值讓 shift 計算出現
     inf，沒有過濾掉就直接取平均，導致整欄變成 inf；這裡先濾掉非正值跟
     inf 再統計）
  B. 月初/月底效應（每月前 3 個交易日 vs 後 3 個交易日 vs 月中，文獻上有名的
     turn-of-month effect）
  C. 節前/節後效應（交易日曆出現 3 天以上的間隔，通常代表遇到連假，比較
     間隔前一天、後一天跟平常日的隔日報酬）
  D. 隔夜跳空 -> 當日盤中表現（開盤跳空幅度 vs 當天開盤到收盤的報酬）——
     這是日線資料能做到、最接近「日內時段」精神的分析：跳空缺口大的股票，
     當天盤中是延續（開高走高）還是回補（開高走低）？

A/B/C 是類別型比較（列出各組平均報酬 + t 檢定），D 是連續型訊號，用
Rank IC 方法（跟 analyze_return_regularities_from_db.py 一致）。

用法：
    python scripts/analyze_calendar_effects_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.storage import get_data_store

MIN_STOCKS_PER_DAY = 30


def _clean_forward_return(master: pd.DataFrame, periods: int = 1) -> pd.Series:
    """T 日收盤到 T+periods 日收盤的報酬率，過濾掉非正值/inf 造成的髒資料。"""
    close = master["close"].where(master["close"] > 0)
    fwd = master.groupby("stock_id", sort=False)["close"].transform(
        lambda s: s.where(s > 0).shift(-periods) / s.where(s > 0) - 1
    )
    return fwd.replace([np.inf, -np.inf], np.nan)


def _bucket_stats(df: pd.DataFrame, bucket_col: str, value_col: str) -> pd.DataFrame:
    rows = []
    overall_mean = df[value_col].mean()
    for bucket, g in df.groupby(bucket_col):
        vals = g[value_col].dropna()
        if len(vals) < 30:
            continue
        t_stat, p_val = sp_stats.ttest_1samp(vals, overall_mean, nan_policy="omit")
        rows.append(
            {
                "bucket": bucket, "mean": vals.mean(), "std": vals.std(),
                "n": len(vals), "t_vs_overall_mean": t_stat, "p_value": p_val,
            }
        )
    return pd.DataFrame(rows)


def day_of_week_effect(master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    master["_fwd_1"] = _clean_forward_return(master, 1)
    master["dow"] = master["date"].dt.day_name()
    df = _bucket_stats(master, "dow", "_fwd_1")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    df["_order"] = df["bucket"].apply(lambda d: order.index(d) if d in order else 99)
    return df.sort_values("_order").drop(columns="_order")


def month_position_effect(master: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"date": sorted(master["date"].unique())})
    dates["year_month"] = dates["date"].dt.to_period("M")
    dates["trading_day_of_month"] = dates.groupby("year_month").cumcount() + 1
    days_in_month = dates.groupby("year_month")["date"].transform("count")
    dates["days_from_month_end"] = days_in_month - dates["trading_day_of_month"] + 1

    def _bucket(row):
        if row["trading_day_of_month"] <= 3:
            return "月初(前3個交易日)"
        if row["days_from_month_end"] <= 3:
            return "月底(後3個交易日)"
        return "月中"

    dates["month_bucket"] = dates.apply(_bucket, axis=1)

    master = master.merge(dates[["date", "month_bucket"]], on="date", how="left")
    master["_fwd_1"] = _clean_forward_return(master, 1)
    return _bucket_stats(master, "month_bucket", "_fwd_1")


def holiday_effect(master: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"date": sorted(master["date"].unique())})
    dates["gap_before_days"] = dates["date"].diff().dt.days
    dates["gap_after_days"] = dates["gap_before_days"].shift(-1)
    dates["holiday_bucket"] = "平常日"
    dates.loc[dates["gap_after_days"] > 3, "holiday_bucket"] = "節前最後一天"
    dates.loc[dates["gap_before_days"] > 3, "holiday_bucket"] = "節後第一天"

    master = master.merge(dates[["date", "holiday_bucket"]], on="date", how="left")
    master["_fwd_1"] = _clean_forward_return(master, 1)
    return _bucket_stats(master, "holiday_bucket", "_fwd_1")


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


def overnight_gap_effect(master: pd.DataFrame) -> dict:
    """跳空幅度(隔夜) -> 當天開盤到收盤報酬(盤中)：日線資料能做到、最接近
    「日內時段」精神的分析——大幅跳空後，當天盤中是延續還是回補？
    """
    master = master.copy()
    prior_close = master.groupby("stock_id", sort=False)["close"].shift(1)
    master["gap_pct"] = ((master["open"] - prior_close) / prior_close).replace([np.inf, -np.inf], np.nan)
    master["intraday_ret"] = ((master["close"] - master["open"]) / master["open"]).replace(
        [np.inf, -np.inf], np.nan
    )
    ic = _rank_ic_series(master, "gap_pct", "intraday_ret")
    return _ic_summary(ic)


def _print_bucket_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("  樣本不足，無法統計。")
        return
    print(f"{'bucket':<16} {'mean(隔日報酬)':>14} {'std':>8} {'n':>8} {'t(vs全體均值)':>12} {'p值':>8}")
    for _, r in df.iterrows():
        print(
            f"{r['bucket']:<16} {r['mean']:>13.4%} {r['std']:>8.4f} {r['n']:>8.0f} "
            f"{r['t_vs_overall_mean']:>12.2f} {r['p_value']:>8.3f}"
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
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）"
    )

    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()

    _print_bucket_table("A. 星期幾效應（隔日報酬率）", day_of_week_effect(master))
    _print_bucket_table("B. 月初/月底效應（隔日報酬率）", month_position_effect(master))
    _print_bucket_table("C. 節前/節後效應（隔日報酬率）", holiday_effect(master))

    print("\n=== D. 隔夜跳空 -> 當天盤中報酬（Rank IC，日線能做到最接近日內時段的分析）===")
    gap_ic = overnight_gap_effect(master)
    print(
        f"  mean_IC={gap_ic['mean_ic']:.4f}  std_IC={gap_ic['std_ic']:.4f}  "
        f"t_stat={gap_ic['t_stat']:.2f}  n_days={gap_ic['n_days']}"
    )
    print(
        "  （IC > 0 代表跳空幅度大的股票，當天盤中傾向延續同方向；"
        "IC < 0 代表傾向當天就回補跳空——後者比較接近「開高走低」的常見說法）"
    )

    print(
        "\n（A/B/C 的 t 值是跟「全樣本平均報酬」比較，不是跟 0 比較，目的是看這個"
        "分組是不是顯著偏離平常表現；星期/月曆效應在文獻上很容易因為多重比較冒出"
        "假訊號，即使看到 p<0.05 也要打折扣看待，不能只憑這裡的結果就直接建策略）"
    )


if __name__ == "__main__":
    main()
