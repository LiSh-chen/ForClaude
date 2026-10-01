"""通用的 Rank Information Coefficient（IC）分析引擎，抽出
scripts/analyze_return_regularities_from_db.py（台股版）的核心演算法，
讓美股版（scripts/analyze_us_cross_sectional_ic.py）跟任何未來想做同一種
分析的市場都能共用同一套、已經驗證過的統計邏輯，不用各自重複實作一份、
之後兩邊分別修 bug 卻漏了一邊。

方法（跟台股版完全一致，見該腳本檔頭的完整說明）：
  1. 訊號 = 每檔股票在 T 日為止的某個特徵值（動量、相對量能、財報驚喜幅度…）
  2. 結果 = 每檔股票從 T 日起算未來 F 日的報酬率
  3. 在每一天，計算「訊號」與「結果」在當天所有股票之間的 Spearman 等級
     相關係數（用排名 + Pearson 公式實作，數學上等價，效能較好）
  4. mean IC / std IC / IC_IR（穩定性）/ t 值（粗略顯著性，沒有 Newey-West
     修正，L、F 重疊造成的自相關會讓 t 值偏樂觀，只能當參考）
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_series(df: pd.DataFrame, sig_col: str, fwd_col: str, min_stocks_per_day: int = 30) -> pd.Series:
    """回傳每一天的 cross-sectional Spearman IC。樣本數不足
    min_stocks_per_day 的日期會被排除，避免小樣本雜訊冒充規律。
    """
    valid = df[["date", sig_col, fwd_col]].dropna()
    counts = valid.groupby("date").size()
    ok_dates = counts[counts >= min_stocks_per_day].index
    valid = valid[valid["date"].isin(ok_dates)]
    if valid.empty:
        return pd.Series(dtype=float)

    valid = valid.copy()
    valid["sig_rank"] = valid.groupby("date")[sig_col].rank()
    valid["fwd_rank"] = valid.groupby("date")[fwd_col].rank()

    def _corr(g: pd.DataFrame) -> float:
        return g["sig_rank"].corr(g["fwd_rank"])

    return valid.groupby("date").apply(_corr, include_groups=False)


def ic_summary(ic_series: pd.Series) -> dict:
    """mean_ic / std_ic / ic_ir（= mean/std，穩定性）/ t_stat（= mean/(std/sqrt(n))）
    /n_days。樣本天數不足 5 天時全部回傳 NaN（樣本太少，算出來的統計量沒有
    意義）。
    """
    if ic_series.empty or ic_series.notna().sum() < 5:
        return {"mean_ic": np.nan, "std_ic": np.nan, "ic_ir": np.nan, "t_stat": np.nan, "n_days": 0}
    ic_series = ic_series.dropna()
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    n = len(ic_series)
    ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    return {"mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "t_stat": t_stat, "n_days": n}


def forward_return(master: pd.DataFrame, horizon: int, price_col: str = "close") -> pd.Series:
    """每檔股票從 T 日起算未來 horizon 個交易日的簡單報酬率，
    master 必須已經依 (stock_id, date) 排序。"""
    return master.groupby("stock_id", sort=False)[price_col].transform(lambda s: s.shift(-horizon) / s - 1)


def fmt_ic_table(df: pd.DataFrame, top_n: int = 20) -> str:
    """依 |mean_IC| 排序列出前 top_n 組，df 須有 family/mean_ic/std_ic/ic_ir/t_stat/n_days
    欄位，另外可以有任意數量的參數欄位（例如 L、F，或美股版的 horizon）會原樣顯示。
    """
    df = df.dropna(subset=["mean_ic"]).copy()
    df["abs_ic"] = df["mean_ic"].abs()
    df = df.sort_values("abs_ic", ascending=False)
    param_cols = [c for c in df.columns if c not in {"family", "mean_ic", "std_ic", "ic_ir", "t_stat", "n_days", "abs_ic"}]
    header = f"{'family':<28} " + " ".join(f"{c:>10}" for c in param_cols) + f" {'mean_IC':>9} {'std_IC':>8} {'IC_IR':>7} {'t_stat':>7} {'n_days':>7}"
    lines = [header]
    for _, r in df.head(top_n).iterrows():
        params = " ".join(f"{str(r[c]):>10}" for c in param_cols)
        lines.append(
            f"{r['family']:<28} {params} "
            f"{r['mean_ic']:>9.4f} {r['std_ic']:>8.4f} {r['ic_ir']:>7.3f} {r['t_stat']:>7.2f} {r['n_days']:>7.0f}"
        )
    return "\n".join(lines)
