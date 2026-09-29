"""測試「夜盤美股開盤後動能突破，設定固定停利150-200點、睡前收工」
這個從影片看來的操作方式（使用者原話：日盤振幅538點、下午盤511點，
講者評估美股開盤後(21:30後)夜盤應有400點振幅空間，實戰「去頭去尾」，
目標睡前賺150-200點收工）。

**這不是全新的假設**——這次會話已經測過兩次「夜盤美股開盤後動能」這個
核心概念，都是零信號：
1. scan_night_session_trend.py / night_liquid_window_trend_strategy.py：
   VWAP持續偏一側(=趨勢日)當决策依據，22:30決策，IS t=-0.76、OOS t=0.22，
   三種成本情境OOS淨損益全部是負的。
2. night_session_liquidity_and_momentum.py：前段(15:00-22:00)走勢 vs
   後段(22:00-05:00)走勢的動能/反轉關係，相關係數0.016（近乎零），
   門檻篩選規則的t值在-0.82~1.08之間無規律跳動。

影片講的操作方式在「出場紀律」上跟這兩版不同（固定停利150-200點就收工，
不是VWAP破位出場也不是抱到05:00收盤），所以這裡用第三種操作化方式
（區間突破+固定停利/停損+時間停損)再測一次，把「進場後設定明確目標、
提早獲利了結」這個影片核心精神也測進去，看出場紀律的改變能不能救回
前兩次否決的結論。

進場：用15:00~21:30(美股開盤前，講者用的錨點)區間高低點，21:30後
第一根收盤價突破這個區間(留min_move_points緩衝避免雜訊)的方向進場。
出場：固定停利(150/175/200點) 或 固定停損(100點) 或 01:00時間停損
(「睡前收工」的具體化) 或 04:45(夜盤收盤前)保底出場，四者最先發生者。

不開網格搜尋（沿用這次會話對夜盤資料樣本小的紀律），只測預先指定的
三組停利目標，停損固定100點不隨目標調整。

用法：
    python scripts/test_night_session_target_take.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import TradeCost, apply_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"

NIGHT_COST = TradeCost("夜盤（流動性較差，估3點滑價）", commission_round_trip=60.0, slippage_points_round_trip=3.0)

RANGE_START = time(15, 0)
RANGE_END = time(21, 30)   # 講者用的錨點：美股開盤(21:30)前的區間
TIME_STOP = time(1, 0)     # 「睡前收工」具體化：01:00強制出場
SESSION_FALLBACK = time(4, 45)  # 保底：夜盤收盤前
MIN_MOVE_POINTS = 10.0     # 突破緩衝，避免雜訊
STOP_LOSS_POINTS = 100.0   # 固定停損，不隨停利目標調整
TARGETS = [150.0, 175.0, 200.0]

IS_END = "2022-01-01"
OOS_END = "2024-01-01"


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def night_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    night = df[(t >= RANGE_START) | (t <= time(5, 0))].copy()
    is_evening = night["datetime"].dt.time >= RANGE_START
    night["night_date"] = night["datetime"].dt.date
    night.loc[is_evening, "night_date"] = night.loc[is_evening, "datetime"].dt.date + pd.Timedelta(days=1)
    return night


def simulate(df: pd.DataFrame, target: float) -> pd.DataFrame:
    night = night_session_frame(df)
    trades = []
    for nd, g in night.groupby("night_date"):
        g = g.sort_values("datetime").reset_index(drop=True)
        gt = g["datetime"].dt.time
        is_evening_row = gt >= RANGE_START
        pre_mask = is_evening_row & (gt < RANGE_END)
        # post-21:30：當晚21:30~23:59 的傍晚K棒，加上隔天00:00~05:00的凌晨K棒
        # （凌晨K棒裸時間物件"小於"21:30，但在夜盤時序上其實排在21:30之後，
        # 用is_evening_row排除掉跨午夜比較的陷阱，不能只用 gt >= RANGE_END）
        post_mask = (is_evening_row & (gt >= RANGE_END)) | (~is_evening_row)
        if pre_mask.sum() < 30 or not post_mask.any():
            continue
        pre = g.loc[pre_mask]
        range_high, range_low = pre["high"].max(), pre["low"].min()

        post = g.loc[post_mask].reset_index(drop=True)
        pt = post["datetime"].dt.time
        entry_idx = None
        direction = None
        for i in range(len(post)):
            c = post["close"].iloc[i]
            if c >= range_high + MIN_MOVE_POINTS:
                entry_idx, direction = i, "long"
                break
            if c <= range_low - MIN_MOVE_POINTS:
                entry_idx, direction = i, "short"
                break
        if entry_idx is None or entry_idx + 1 >= len(post):
            continue

        entry_bar = post.iloc[entry_idx + 1]
        sign = 1 if direction == "long" else -1
        entry_price = entry_bar["open"] + sign * (NIGHT_COST.slippage_points_round_trip / 2)
        entry_dt = entry_bar["datetime"]

        exit_price = exit_dt = exit_reason = None
        for j in range(entry_idx + 1, len(post)):
            r = post.iloc[j]
            rt = r["datetime"].time()
            favorable = (r["high"] - entry_price) if direction == "long" else (entry_price - r["low"])
            adverse = (entry_price - r["low"]) if direction == "long" else (r["high"] - entry_price)
            if favorable >= target:
                exit_price = entry_price + sign * target - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "target"
                break
            if adverse >= STOP_LOSS_POINTS:
                exit_price = entry_price - sign * STOP_LOSS_POINTS - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "stop"
                break
            is_evening_bar = rt >= RANGE_START
            crossed_time_stop = (not is_evening_bar) and (rt >= TIME_STOP)
            crossed_fallback = (not is_evening_bar) and (rt >= SESSION_FALLBACK)
            if crossed_time_stop:
                exit_price = r["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "time_stop"
                break
        if exit_price is None:
            last = post.iloc[-1]
            exit_price = last["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
            exit_dt, exit_reason = last["datetime"], "session_end"

        pnl_points = (exit_price - entry_price) * sign
        trades.append(dict(
            night_date=nd, direction=direction, entry_dt=entry_dt, entry_price=entry_price,
            exit_dt=exit_dt, exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["trading_date"] = pd.to_datetime(trades_df["night_date"])
    return trades_df.sort_values("entry_dt").reset_index(drop=True)


def report(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"  {label}: n=0")
        return
    t, n, m = tstat(trades["pnl_points"])
    net = apply_costs(trades, TradeCost("已含夜盤滑價的淨額", commission_round_trip=60.0))["net_twd"].sum()
    win_rate = (trades["pnl_points"] > 0).mean() * 100
    reasons = trades["exit_reason"].value_counts(normalize=True) * 100
    print(f"  {label}: n={n} t={t:.2f} mean={m:.1f}點 勝率={win_rate:.1f}% 淨損益={net:,.0f}元")
    print(f"    出場原因分布: {dict(reasons.round(1))}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    print("=" * 78)
    print(f"區間突破+固定停利/停損 (區間{RANGE_START}~{RANGE_END}, 停損{STOP_LOSS_POINTS:.0f}點, "
          f"時間停損{TIME_STOP})")
    print("=" * 78)

    for target in TARGETS:
        trades = simulate(df, target)
        if trades.empty:
            print(f"\n目標{target:.0f}點: 無交易")
            continue
        is_trades = trades[trades["trading_date"] < IS_END]
        oos_trades = trades[(trades["trading_date"] >= IS_END) & (trades["trading_date"] < OOS_END)]
        print(f"\n--- 停利目標 {target:.0f}點 ---")
        report(is_trades, f"IS(2017-05~2021-12)")
        report(oos_trades, f"OOS(2022-2023)")


if __name__ == "__main__":
    main()
