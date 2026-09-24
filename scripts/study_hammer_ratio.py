"""純粹版「長下影線 + 次根紅K」訊號研究：不設任何交易時段、趨勢狀態、前波高低點
限制，只看「下影線/實體」比例門檻如何影響訊號後的遠期報酬。

規則：
1. 任一根K棒 body = |open-close| > 0（排除十字線，比例無意義）。
2. 下影線 = min(open,close) - low，比例 = 下影線 / 實體。
3. 次根若為紅K（收盤 > 開盤，依台股慣例紅=漲），以次根收盤價視為進場價。
4. 用次根之後固定 N 分鐘（5/15/30/60）的遠期報酬（進場價到 N 分鐘後收盤價）
   衡量訊號品質，而不是接一套停損停利——這樣量到的是「訊號本身有沒有預測力」，
   不會跟另一套出場邏輯的假設混在一起。
5. 遠期報酬只在同一段連續盤中（兩根K棒間隔 <= 2 分鐘）才計算，跨盤（日盤收盤
   到夜盤開盤、跨日）一律排除，避免被跳空污染。
6. 用全樣本（2001-2023，日盤+夜盤都算），不分策略一/二的時間窗限制。

輸出：依「下影線/實體 >= 門檻」分組（累積門檻，不是互斥區間，因為實務上策略
參數就是「只做比例 >= X 的訊號」），比較各門檻在各個持有分鐘數下的勝率、平均
遠期報酬（點）、t 統計量（單樣本 t 檢定，H0: 平均報酬=0）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "hammer_ratio_study.parquet"

HORIZONS = [5, 15, 30, 60]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
MAX_GAP_MINUTES = 2  # 超過這個間隔視為跨盤／跨日，不連續


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("datetime").reset_index(drop=True).copy()
    d["body"] = (d["open"] - d["close"]).abs()
    d["lower_shadow"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["ratio"] = np.where(d["body"] > 0, d["lower_shadow"] / d["body"], np.nan)

    gap_min = d["datetime"].diff().dt.total_seconds() / 60
    new_block = (gap_min > MAX_GAP_MINUTES) | gap_min.isna()
    d["block_id"] = new_block.cumsum()

    d["next_open"] = d["open"].shift(-1)
    d["next_close"] = d["close"].shift(-1)
    d["next_datetime"] = d["datetime"].shift(-1)
    d["next_block"] = d["block_id"].shift(-1)
    d["next_bullish"] = d["next_close"] > d["next_open"]
    return d


def build_candidates(d: pd.DataFrame) -> pd.DataFrame:
    contiguous = d["next_block"] == d["block_id"]
    mask = (d["body"] > 0) & d["next_bullish"].fillna(False) & contiguous
    cand = d[mask].copy()
    cand["entry_idx"] = cand.index + 1
    cand["entry_price"] = cand["next_close"]
    cand["entry_datetime"] = cand["next_datetime"]
    cand["entry_block"] = cand["next_block"]

    n = len(d)
    close_arr = d["close"].to_numpy()
    block_arr = d["block_id"].to_numpy()
    for h in HORIZONS:
        fwd_idx = (cand["entry_idx"] + h).to_numpy()
        valid = fwd_idx < n
        fwd_idx_v = fwd_idx[valid].astype(int)
        same_block = block_arr[fwd_idx_v] == cand.loc[valid, "entry_block"].to_numpy()
        fwd_close = np.full(len(cand), np.nan)
        fwd_close[valid] = np.where(same_block, close_arr[fwd_idx_v], np.nan)
        cand[f"fwd_ret_{h}"] = fwd_close - cand["entry_price"].to_numpy()
    return cand


def summarize(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for th in THRESHOLDS:
        sub = cand[cand["ratio"] >= th]
        row = {"ratio_ge": th, "n": len(sub)}
        for h in HORIZONS:
            r = sub[f"fwd_ret_{h}"].dropna()
            n_valid = len(r)
            mean = r.mean() if n_valid else np.nan
            se = r.std(ddof=1) / np.sqrt(n_valid) if n_valid > 1 else np.nan
            row[f"n_valid_{h}"] = n_valid
            row[f"win_rate_{h}"] = (r > 0).mean() if n_valid else np.nan
            row[f"avg_pts_{h}"] = mean
            row[f"tstat_{h}"] = mean / se if se and se > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def baseline(d: pd.DataFrame) -> pd.DataFrame:
    """不設任何訊號條件，從每一根K棒收盤價往前看 N 分鐘的遠期報酬，當對照基準。"""
    n = len(d)
    close_arr = d["close"].to_numpy()
    block_arr = d["block_id"].to_numpy()
    idx = np.arange(n)
    rows = []
    for h in HORIZONS:
        fwd_idx = idx + h
        valid = fwd_idx < n
        same_block = block_arr[fwd_idx[valid]] == block_arr[valid]
        fwd_ret = np.full(n, np.nan)
        fwd_ret[valid] = np.where(same_block, close_arr[fwd_idx[valid]] - close_arr[valid], np.nan)
        r = pd.Series(fwd_ret).dropna()
        se = r.std(ddof=1) / np.sqrt(len(r))
        rows.append({"horizon": h, "n": len(r), "avg_pts": r.mean(), "tstat": r.mean() / se, "win_rate": (r > 0).mean()})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    d = build_indicators(df)
    cand = build_candidates(d)
    summary = summarize(cand)
    base = baseline(d)

    print(f"candidates (body>0, next=紅K, 連續): {len(cand):,}")
    print("\n=== 對照基準：不設訊號，任一根往前看 N 分鐘 ===")
    print(base.to_string(index=False))
    print("\n=== 依比例門檻分組 ===")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
