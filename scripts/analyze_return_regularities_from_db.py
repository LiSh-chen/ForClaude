"""全面掃描股價報酬規律性，用實證數據找出真正存在、統計上站得住腳的
動量/反轉規律，取代前幾輪「憑經驗猜一種進場邏輯再回測」的做法。

方法：Rank Information Coefficient（IC）分析——量化研究界找因子的標準做法：

  1. 訊號 = 每檔股票在 T 日為止過去 L 日的表現（例如價格報酬率、相對量能）
  2. 結果 = 每檔股票從 T 日起算未來 F 日的報酬率
  3. 在每一天，計算「訊號」與「結果」在當天所有股票之間的 Spearman 等級相關係數
     （只看排名關係，不受少數離群值影響）
  4. mean IC = IC 的時間序列平均值；IC_IR = mean IC / std(IC)（穩定性指標，
     越高代表這個規律越穩定、不是少數幾天特別准）；
     t 值 = mean IC / (std(IC) / sqrt(n_dates))，粗略判斷是否顯著不為 0
     （注意：L、F 有重疊會讓報酬序列存在自相關，這裡沒有做 Newey-West 修正，
     t 值只能當參考，不是嚴謹統計檢定）

IC > 0：訊號期表現好的股票，之後傾向繼續表現好 -> 動量規律
IC < 0：訊號期表現好的股票，之後傾向回吐 -> 反轉規律
mean IC 接近 0：這個 (L, F) 組合看不出規律，之前幾輪硬是想在這個角落雕出策略，
很可能就是在雜訊裡找訊號。

掃描三種訊號家族（我們手上有的資料就這些，不是憑空編造）：
  A. 價格動量：trailing L 日報酬率
  B. 相對量能：當日成交量 / L 日均量
  C. 籌碼面：券資比（不分 L，因為這是水位型指標不是報酬率）

用法：
    python scripts/analyze_return_regularities_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_provider import compute_margin_short_ratio
from tw_quant.storage import get_data_store

MOMENTUM_L_GRID = (1, 2, 3, 5, 10, 20, 40, 60, 90, 120, 180, 252)
VOLUME_L_GRID = (5, 10, 20, 60)
FORWARD_F_GRID = (1, 3, 5, 10, 20, 60)

MIN_STOCKS_PER_DAY = 30  # 當天可用樣本數太少就跳過，避免小樣本雜訊冒充規律


def _rank_ic_series(df: pd.DataFrame, sig_col: str, fwd_col: str) -> pd.Series:
    """回傳每一天的 cross-sectional Spearman IC（用等級轉換 + Pearson 公式實作，
    數學上等價，效能較好）。樣本數不足 MIN_STOCKS_PER_DAY 的日期會被排除。
    """
    valid = df[[ "date", sig_col, fwd_col]].dropna()
    counts = valid.groupby("date").size()
    ok_dates = counts[counts >= MIN_STOCKS_PER_DAY].index
    valid = valid[valid["date"].isin(ok_dates)]
    if valid.empty:
        return pd.Series(dtype=float)

    valid = valid.copy()
    valid["sig_rank"] = valid.groupby("date")[sig_col].rank()
    valid["fwd_rank"] = valid.groupby("date")[fwd_col].rank()

    def _corr(g: pd.DataFrame) -> float:
        return g["sig_rank"].corr(g["fwd_rank"])

    return valid.groupby("date").apply(_corr, include_groups=False)


def _ic_summary(ic_series: pd.Series) -> dict:
    if ic_series.empty or ic_series.notna().sum() < 5:
        return {"mean_ic": np.nan, "std_ic": np.nan, "ic_ir": np.nan, "t_stat": np.nan, "n_days": 0}
    ic_series = ic_series.dropna()
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    n = len(ic_series)
    ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    return {"mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "t_stat": t_stat, "n_days": n}


def scan_price_momentum(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for F in FORWARD_F_GRID:
        fwd_col = f"_fwd_{F}"
        master[fwd_col] = master.groupby("stock_id", sort=False)["close"].transform(
            lambda s: s.shift(-F) / s - 1
        )
        for L in MOMENTUM_L_GRID:
            sig_col = f"_mom_{L}"
            if sig_col not in master.columns:
                master[sig_col] = master.groupby("stock_id", sort=False)["close"].pct_change(L)
            ic = _rank_ic_series(master, sig_col, fwd_col)
            summary = _ic_summary(ic)
            rows.append({"family": "price_momentum", "L": L, "F": F, **summary})
    return pd.DataFrame(rows)


def scan_relative_volume(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for F in FORWARD_F_GRID:
        fwd_col = f"_fwd_{F}"
        if fwd_col not in master.columns:
            master[fwd_col] = master.groupby("stock_id", sort=False)["close"].transform(
                lambda s: s.shift(-F) / s - 1
            )
        for L in VOLUME_L_GRID:
            sig_col = f"_volratio_{L}"
            if sig_col not in master.columns:
                avg_vol = master.groupby("stock_id", sort=False)["volume"].transform(
                    lambda s: s.rolling(L, min_periods=L).mean()
                )
                master[sig_col] = master["volume"] / avg_vol
            ic = _rank_ic_series(master, sig_col, fwd_col)
            summary = _ic_summary(ic)
            rows.append({"family": "relative_volume", "L": L, "F": F, **summary})
    return pd.DataFrame(rows)


def scan_margin_short(prices: pd.DataFrame, margin_short: pd.DataFrame) -> pd.DataFrame:
    ms = margin_short.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    ms["ratio"] = compute_margin_short_ratio(ms)
    merged = prices[["date", "stock_id", "close"]].merge(
        ms[["date", "stock_id", "ratio"]], on=["date", "stock_id"], how="inner"
    )
    merged = merged.sort_values(["stock_id", "date"]).reset_index(drop=True)

    rows = []
    for F in FORWARD_F_GRID:
        merged[f"_fwd_{F}"] = merged.groupby("stock_id", sort=False)["close"].transform(
            lambda s: s.shift(-F) / s - 1
        )
        ic = _rank_ic_series(merged, "ratio", f"_fwd_{F}")
        summary = _ic_summary(ic)
        rows.append({"family": "margin_short_ratio", "L": "n/a", "F": F, **summary})
    return pd.DataFrame(rows)


def day_of_week_seasonality(prices: pd.DataFrame) -> pd.DataFrame:
    """量粗略的星期幾規律：隔日報酬率依當日星期幾分組平均，僅供參考
    （沒有做多重比較校正，星期效應是文獻裡很容易因為多重測試冒出假訊號的
    典型例子，看到「顯著」也要打折扣）。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    master["_fwd_1"] = master.groupby("stock_id", sort=False)["close"].transform(lambda s: s.shift(-1) / s - 1)
    master["dow"] = master["date"].dt.day_name()
    grouped = master.groupby("dow")["_fwd_1"].agg(["mean", "std", "count"])
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return grouped.reindex([d for d in order if d in grouped.index])


def _fmt_ic_table(df: pd.DataFrame, top_n: int = 20) -> str:
    df = df.dropna(subset=["mean_ic"]).copy()
    df["abs_ic"] = df["mean_ic"].abs()
    df = df.sort_values("abs_ic", ascending=False)
    lines = [f"{'family':<18} {'L':>6} {'F':>4}  {'mean_IC':>9} {'std_IC':>8} {'IC_IR':>7} {'t_stat':>7} {'n_days':>7}"]
    for _, r in df.head(top_n).iterrows():
        lines.append(
            f"{r['family']:<18} {str(r['L']):>6} {r['F']:>4}  "
            f"{r['mean_ic']:>9.4f} {r['std_ic']:>8.4f} {r['ic_ir']:>7.3f} {r['t_stat']:>7.2f} {r['n_days']:>7.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin_short = store.load_margin_short()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()

    print("=== A. 價格動量 IC 掃描（trailing L 日報酬 -> forward F 日報酬）===")
    df_mom = scan_price_momentum(master)
    print(_fmt_ic_table(df_mom, top_n=15))

    print("\n=== B. 相對量能 IC 掃描（今日量/L日均量 -> forward F 日報酬）===")
    df_vol = scan_relative_volume(master)
    print(_fmt_ic_table(df_vol, top_n=10))

    print("\n=== C. 券資比 IC 掃描（券資比水位 -> forward F 日報酬）===")
    df_ms = scan_margin_short(prices, margin_short)
    print(_fmt_ic_table(df_ms, top_n=6))

    print("\n=== D. 星期幾規律（隔日報酬率，僅供參考，容易是多重比較假訊號）===")
    dow = day_of_week_seasonality(prices)
    print(dow.to_string(float_format=lambda x: f"{x:.4%}" if abs(x) < 1 else f"{x:.0f}"))

    all_df = pd.concat([df_mom, df_vol, df_ms], ignore_index=True)
    all_df = all_df.dropna(subset=["mean_ic"])
    strong = all_df[(all_df["mean_ic"].abs() >= 0.03) & (all_df["t_stat"].abs() >= 2.0)]
    print(
        f"\n=== 總結：{len(all_df)} 組 (family, L, F) 中，"
        f"有 {len(strong)} 組 |mean IC| >= 0.03 且 |t| >= 2.0（粗略的「有規律」門檻）==="
    )
    if strong.empty:
        print("  沒有任何組合同時滿足強度與穩定性門檻——目前這 3 種訊號家族，")
        print("  在這段真實資料上都沒有看到清楚站得住腳的規律。")
    else:
        print(_fmt_ic_table(strong, top_n=20))


if __name__ == "__main__":
    main()
