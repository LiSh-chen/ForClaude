"""探索夜盤（15:00~次日05:00，2017-05-15起才有資料，約6.5年）的趨勢日
現象，把日盤驗證過的「VWAP持續性」概念直接搬過來快速掃一次，看有沒有
訊號再決定要不要投入完整模組開發。夜盤是這次會話幾乎沒碰過的資料維度
——參與者結構跟日盤不同（疊加美股/歐股交易時段），時段長度也長很多
（14小時 vs 日盤5小時），理論上更有機會走出真正的趨勢。

夜盤 session_label 算法：跨夜的整段夜盤（例如2021-05-11 15:00~
2021-05-12 05:00）統一標成它銜接的那個「隔天」（這裡沿用之前修盤前
缺口bug時確認過的正確算法，避免重蹈未來函數覆轍）。

用法：
    python scripts/scan_night_session_trend.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
NIGHT_START = "2017-05-15"


def _night_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(15, 0)) | (t <= time(5, 0))].copy()
    late = d["datetime"].dt.time >= time(15, 0)
    d["session_label"] = d["datetime"].dt.date
    d.loc[late, "session_label"] = d.loc[late, "datetime"].dt.date + pd.Timedelta(days=1)
    return d


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def scan_decision_time(night_df: pd.DataFrame, decision_time: time, min_dominant_fraction: float,
                        min_move_points: float) -> pd.DataFrame:
    rows = []
    for session_label, g in night_df.groupby("session_label"):
        g = g.sort_values("datetime").reset_index(drop=True)
        typical_price = (g["high"] + g["low"] + g["close"]) / 3
        cum_pv = (typical_price * g["volume"]).cumsum()
        cum_v = g["volume"].cumsum().replace(0, np.nan)
        g["vwap"] = cum_pv / cum_v

        t = g["datetime"].dt.time
        # 夜盤跨夜：判斷時點在15:00~23:59（當晚）只框當晚15:00到判斷時點這段；
        # 判斷時點在00:00~04:59（凌晨）則要涵蓋整個當晚+隔天早上到判斷時點這段。
        is_before_decision = ((t >= time(15, 0)) & (t <= decision_time)) if decision_time >= time(15, 0) else \
            ((t >= time(15, 0)) | (t <= decision_time))
        if not is_before_decision.any():
            continue
        decision_idx = is_before_decision[is_before_decision].index[-1]
        if decision_idx + 1 >= len(g):
            continue

        pre = g.loc[: decision_idx]
        side = np.sign(pre["close"] - pre["vwap"])
        side = side.replace(0, np.nan).dropna()
        if side.empty:
            continue
        counts = side.value_counts()
        dominant_sign = counts.idxmax()
        dominant_fraction = counts.max() / len(side)
        if dominant_fraction < min_dominant_fraction:
            continue

        session_open = g["open"].iloc[0]
        decision_close = g["close"].iloc[decision_idx]
        move = (decision_close - session_open) * dominant_sign
        if move < min_move_points:
            continue

        entry_bar = g.iloc[decision_idx + 1]
        entry_price = entry_bar["open"]
        exit_price = g["close"].iloc[-1]  # 簡化：直接用session最後一筆收盤價當出場（探索階段不模擬滑價）
        pnl = (exit_price - entry_price) * dominant_sign
        rows.append(dict(session_label=session_label, direction=int(dominant_sign), move_at_decision=move, pnl_points=pnl))
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    night_df = _night_session_frame(df[df["datetime"] >= NIGHT_START])

    print(f"夜盤 session 數: {night_df['session_label'].nunique()}")

    print("\n=== 掃描不同決策時點（min_dominant_fraction=0.90, min_move=30pt）===")
    for dt_label, dt in [("18:00", time(18, 0)), ("20:00", time(20, 0)), ("22:00", time(22, 0)),
                         ("00:00", time(0, 0)), ("02:00", time(2, 0))]:
        result = scan_decision_time(night_df, dt, 0.90, 30.0)
        if result.empty:
            print(f"  決策時點={dt_label}: n=0")
            continue
        print(f"  決策時點={dt_label}: n={len(result)} mean={result['pnl_points'].mean():.2f} t={tstat(result['pnl_points']):.2f}")

    print("\n=== 決策時點=20:00，min_move門檻掃描 ===")
    for mm in [10, 20, 30, 50, 70]:
        result = scan_decision_time(night_df, time(20, 0), 0.90, mm)
        if result.empty:
            print(f"  min_move={mm}: n=0")
            continue
        print(f"  min_move={mm}: n={len(result)} mean={result['pnl_points'].mean():.2f} t={tstat(result['pnl_points']):.2f}")


if __name__ == "__main__":
    main()
