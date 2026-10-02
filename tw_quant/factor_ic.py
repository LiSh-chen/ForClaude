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

2026-10-02 新增 Newey-West (1987) HAC 修正：forward horizon 是 F 個交易日時，
逐日滾動算出來的 IC 時間序列，相鄰最多 F-1 天的觀測值用的是重疊的
forward return 窗格，天生帶有最多到 F-1 階的自相關——簡單的
std/sqrt(n) 標準誤把這些重疊觀測值當成互相獨立，嚴重低估真正的標準誤、
高估 t 值（n_days 動辄 4000~5000 天看起來「樣本數很大」，但 F=60 時
真正獨立的樣本數量級其實接近 n_days/60，不是 n_days）。ic_summary() 傳入
nw_lags（建議設成 F-1）就會額外算一個用 Bartlett kernel 加權自相關修正過
的 t_stat_nw，兩者對照才看得出「虛高了多少」。
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


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """樣本平均數的 Newey-West (1987) HAC 標準誤，修正序列自相關（Bartlett
    核函數加權，權重隨 lag 線性遞減到 0）。lags=0 時就退化成一般的
    std/sqrt(n)（沒有自相關修正）。

    長期變異數估計值理論上應該非負，但有限樣本 + 不恰當的 lags 設定仍有
    極小機率算出微負值（浮點誤差或 lags 設太大、把雜訊也加總進去），
    這裡截斷在 0 避免對負數開根號。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_demeaned = x - x.mean()
    long_run_var = np.sum(x_demeaned**2) / n
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        gamma = np.sum(x_demeaned[lag:] * x_demeaned[:-lag]) / n
        weight = 1 - lag / (lags + 1)
        long_run_var += 2 * weight * gamma
    long_run_var = max(long_run_var, 0.0)
    return float(np.sqrt(long_run_var / n))


def newey_west_t_stat(x: np.ndarray, lags: int) -> float:
    """用 Newey-West 標準誤算的 t 值：mean(x) / newey_west_se(x, lags)。"""
    x = np.asarray(x, dtype=float)
    se = newey_west_se(x, lags)
    if se == 0 or np.isnan(se):
        return float("nan")
    return float(x.mean() / se)


def ic_summary(ic_series: pd.Series, nw_lags: int | None = None) -> dict:
    """mean_ic / std_ic / ic_ir（= mean/std，穩定性）/ t_stat（= mean/(std/sqrt(n))，
    未修正自相關，重疊窗格會虛高）/ n_days。樣本天數不足 5 天時全部回傳 NaN
    （樣本太少，算出來的統計量沒有意義）。

    傳入 nw_lags（forward horizon F 的話建議設 F-1，見本檔案開頭說明）時，
    額外算一個 Newey-West 修正過的 t_stat_nw——拿兩個 t 值對照，才看得出
    自相關把顯著性虛高了多少。
    """
    if ic_series.empty or ic_series.notna().sum() < 5:
        result = {"mean_ic": np.nan, "std_ic": np.nan, "ic_ir": np.nan, "t_stat": np.nan, "n_days": 0}
        if nw_lags is not None:
            result["t_stat_nw"] = np.nan
        return result
    ic_series = ic_series.dropna()
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    n = len(ic_series)
    ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    result = {"mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "t_stat": t_stat, "n_days": n}
    if nw_lags is not None:
        result["t_stat_nw"] = newey_west_t_stat(ic_series.to_numpy(), nw_lags)
    return result


def forward_return(master: pd.DataFrame, horizon: int, price_col: str = "close") -> pd.Series:
    """每檔股票從 T 日起算未來 horizon 個交易日的簡單報酬率，
    master 必須已經依 (stock_id, date) 排序。"""
    return master.groupby("stock_id", sort=False)[price_col].transform(lambda s: s.shift(-horizon) / s - 1)


def fmt_ic_table(df: pd.DataFrame, top_n: int = 20) -> str:
    """依 |mean_IC| 排序列出前 top_n 組，df 須有 family/mean_ic/std_ic/ic_ir/t_stat/n_days
    欄位，另外可以有任意數量的參數欄位（例如 L、F，或美股版的 horizon）會原樣顯示。
    有 t_stat_nw 欄位（見 ic_summary 的 nw_lags 參數）時會多顯示一欄，方便
    對照自相關修正前後的顯著性差了多少。
    """
    df = df.dropna(subset=["mean_ic"]).copy()
    df["abs_ic"] = df["mean_ic"].abs()
    df = df.sort_values("abs_ic", ascending=False)
    known_cols = {"family", "mean_ic", "std_ic", "ic_ir", "t_stat", "t_stat_nw", "n_days", "abs_ic"}
    param_cols = [c for c in df.columns if c not in known_cols]
    has_nw = "t_stat_nw" in df.columns
    header = (
        f"{'family':<28} "
        + " ".join(f"{c:>10}" for c in param_cols)
        + f" {'mean_IC':>9} {'std_IC':>8} {'IC_IR':>7} {'t_stat':>7}"
        + (f" {'t_stat_NW':>9}" if has_nw else "")
        + f" {'n_days':>7}"
    )
    lines = [header]
    for _, r in df.head(top_n).iterrows():
        params = " ".join(f"{str(r[c]):>10}" for c in param_cols)
        line = (
            f"{r['family']:<28} {params} "
            f"{r['mean_ic']:>9.4f} {r['std_ic']:>8.4f} {r['ic_ir']:>7.3f} {r['t_stat']:>7.2f}"
        )
        if has_nw:
            nw_val = r["t_stat_nw"]
            line += f" {nw_val:>9.2f}" if pd.notna(nw_val) else f" {'n/a':>9}"
        line += f" {r['n_days']:>7.0f}"
        lines.append(line)
    return "\n".join(lines)
