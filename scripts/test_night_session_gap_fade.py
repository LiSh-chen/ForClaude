"""夜盤全新方向：跳空回補(gap-fill)。日盤收盤(13:45)到夜盤開盤(15:00)
中間有1小時15分鐘空檔，夜盤開盤常常跟日盤收盤價有落差(跳空)。這是
經典的期貨/外匯市場現象，跟先前測過的「方向性動能延續」、「震盪價差
反轉」都不是同一件事——這裡測的是「開盤跳空後，價格有沒有傾向回補
到前一盤收盤價」，機制跟時間點無關，是全新假設。

先做描述性檢查（不牽涉停利/停損/成本，純粹看「有沒有回補」）：
    IS(2017-2021)          OOS(2022-2023)
  0-10點跳空: 90.4%(n=634)   96.5%(n=199)
  10-20點:    82.5%(n=257)   89.9%(n=138)
  20-50點:    63.6%(n=140)   82.9%(n=105)
  50-100點:   52.4%(n=21)    61.1%(n=18)
  100點以上:  12.5%(n=16)     0.0%(n=5，樣本太小僅供參考)

跳空越大、回補率越低，這個型態IS/OOS方向完全一致（單調遞減），不是
巧合——小跳空多半是雜訊，容易被正常夜盤波動蓋過去；大跳空代表真的
有新資訊(消息/美股期貨已經反應)，統計上傾向不會補回去，反而延續。

這裡把「回補」這個描述性觀察轉成實際策略測試（含滑價/手續費/停損/
時間停損），鎖定10-50點這個「跳空有意義、回補率還算高」的區間
（0-10點太小，扣成本後利潤空間有限；50點以上樣本太小，回補率也
明顯下降，不適合做fade）：

進場：夜盤開盤第一根K棒，跳空方向的反方向進場(跳空向上->放空賭回補，
跳空向下->做多賭回補)
停利：拉回到日盤收盤價的某個比例(測50/75/100%回補)
停損：跳空方向繼續延伸(超過night_open再遠離day_close)一段距離
    (用跳空本身的大小當單位，測試延伸50%/100%出場)
時間停損：01:00(睡前收工邏輯，沿用前幾版夜盤測試)，04:45保底出場

不開網格搜尋，只測3個預先指定的停利比例 x 固定停損。

用法：
    python scripts/test_night_session_gap_fade.py
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

DAY_START, DAY_END = time(8, 45), time(13, 45)
NIGHT_START = time(15, 0)
TIME_STOP = time(1, 0)

GAP_MIN, GAP_MAX = 10.0, 50.0  # 鎖定的跳空區間(絕對值)
STOP_FRACTION_OF_GAP = 1.0     # 停損=跳空方向再延伸gap本身的這個倍數
TARGET_FRACTIONS = [0.5, 0.75, 1.0]  # 停利=回補跳空的這個比例

IS_END = "2022-01-01"
OOS_END = "2024-01-01"


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def build_day_close(df: pd.DataFrame) -> pd.Series:
    t = df["datetime"].dt.time
    day = df[(t >= DAY_START) & (t <= DAY_END)].copy()
    day["trading_date"] = day["datetime"].dt.date
    result = day.sort_values("datetime").groupby("trading_date")["close"].last().rename("day_close")
    result.index = pd.to_datetime(result.index)
    return result


def night_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    night = df[(t >= NIGHT_START) | (t <= time(5, 0))].copy()
    is_evening = night["datetime"].dt.time >= NIGHT_START
    night["own_date"] = night["datetime"].dt.date
    night["trading_date"] = np.where(is_evening, night["own_date"],
                                      night["own_date"] - pd.Timedelta(days=1))
    night["trading_date"] = pd.to_datetime(night["trading_date"])
    return night


def simulate(df: pd.DataFrame, day_close: pd.Series, target_fraction: float) -> pd.DataFrame:
    night = night_session_frame(df)
    trades = []
    for td, g in night.groupby("trading_date"):
        if td not in day_close.index:
            continue
        g = g.sort_values("datetime").reset_index(drop=True)
        if len(g) < 30:
            continue
        dclose = day_close.loc[td]
        night_open = g["open"].iloc[0]
        gap = night_open - dclose
        abs_gap = abs(gap)
        if abs_gap < GAP_MIN or abs_gap > GAP_MAX:
            continue

        direction = "short" if gap > 0 else "long"
        sign = 1 if direction == "long" else -1

        entry_bar = g.iloc[0]
        entry_price = entry_bar["open"] + sign * (NIGHT_COST.slippage_points_round_trip / 2)
        entry_dt = entry_bar["datetime"]

        target_dist = abs_gap * target_fraction
        stop_dist = abs_gap * STOP_FRACTION_OF_GAP

        exit_price = exit_dt = exit_reason = None
        for j in range(1, len(g)):
            r = g.iloc[j]
            rt = r["datetime"].time()
            favorable = (r["high"] - entry_price) if direction == "long" else (entry_price - r["low"])
            adverse = (entry_price - r["low"]) if direction == "long" else (r["high"] - entry_price)
            if favorable >= target_dist:
                exit_price = entry_price + sign * target_dist - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "target"
                break
            if adverse >= stop_dist:
                exit_price = entry_price - sign * stop_dist - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "stop"
                break
            is_evening_bar = rt >= NIGHT_START
            if (not is_evening_bar) and (rt >= TIME_STOP):
                exit_price = r["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "time_stop"
                break
        if exit_price is None:
            last = g.iloc[-1]
            exit_price = last["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
            exit_dt, exit_reason = last["datetime"], "session_end"

        pnl_points = (exit_price - entry_price) * sign
        trades.append(dict(
            trading_date=td, direction=direction, gap=gap, abs_gap=abs_gap,
            entry_dt=entry_dt, entry_price=entry_price, exit_dt=exit_dt, exit_price=exit_price,
            exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
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
    day_close = build_day_close(df)

    print("=" * 78)
    print(f"跳空回補策略：鎖定{GAP_MIN:.0f}-{GAP_MAX:.0f}點跳空，反向進場賭回補")
    print("=" * 78)

    for frac in TARGET_FRACTIONS:
        trades = simulate(df, day_close, frac)
        if trades.empty:
            print(f"\n停利比例{frac}: 無交易")
            continue
        is_trades = trades[trades["trading_date"] < IS_END]
        oos_trades = trades[(trades["trading_date"] >= IS_END) & (trades["trading_date"] < OOS_END)]
        print(f"\n--- 停利=回補{frac*100:.0f}% (停損=跳空再延伸{STOP_FRACTION_OF_GAP*100:.0f}%) ---")
        report(is_trades, "IS(2017-05~2021-12)")
        report(oos_trades, "OOS(2022-2023)")


if __name__ == "__main__":
    main()
