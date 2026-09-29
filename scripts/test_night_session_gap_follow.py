"""夜盤跳空的另一半：上一版(test_night_session_gap_fade.py)發現小跳空
(10-50點)回補率高但轉成反向進場策略後R:R全部虧損；描述性統計也顯示
大跳空(50點以上)回補率明顯偏低(IS 52.4%、OOS 61.1%)——這裡反過來測
「跟隨大跳空方向走、賭這次不是雜訊、是真資訊延續」的策略。

用test_night_session_gap_fade.py同樣的3個預先指定停利比例(0.5/0.75/1.0)
搭配停損=跳空延伸100%(跟fade版本一致的停損設計，只是方向相反)，鎖定
50點以上的跳空。

用法：
    python scripts/test_night_session_gap_follow.py

結果與重要警訊：
  target=0.5: IS n=37 t=1.84 勝率59.5% net=23,225 | OOS n=23 t=0.68 勝率65.2% net=6,009
  target=0.75: IS n=37 t=2.14 勝率56.8% net=33,299 | OOS n=23 t=0.41 勝率56.5% net=3,859
  target=1.0: IS n=37 t=2.61 勝率56.8% net=46,675 | OOS n=23 t=0.61 勝率52.2% net=8,234

跟這次會話測過的所有其他夜盤假設不同——這是第一個IS/OOS六格全部同號
(正)的結果，勝率52-65%都在五成以上。但**必須誠實揭露一個嚴重警訊**：
逐筆檢視60筆交易(target=1.0版本)，2020年3月18-26日(COVID崩盤那一週)
就佔了5筆，且包辦了最大的3筆獲利(184.0/142.5/133.5點)。去掉這3筆
之後：t值從2.61掉到1.65，總點數從1204掉到744(幾乎腰斬)。也就是說
這個「邊際」有很大一部分是單一歷史事件(2020年3月流動性危機)貢獻的，
不是穩定可複現的日常型態。n=37(IS)/23(OOS)本身也偏小(平均一年只有
5-6次符合條件的交易機會，因為50點以上的夜盤跳空本來就罕見)。

結論：這是目前這次會話裡最有希望的夜盤線索——經濟邏輯合理(大跳空=
真資訊，統計上不太會補回去)、IS/OOS方向一致、勝率過半——但样本太小、
且明顯被COVID崩盤這種極端事件放大，還不到能拿去實際交易的驗證程度。
建議：先記錄成觀察中的候選，不要現在就當成可用策略；如果之後有更長
的夜盤資料(2024年以後，目前1分鐘資料只到2023-12)，用全新窗格重新
驗證一次，同時可以考慮排除2020年3月這種極端流動性事件單獨檢視穩健性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.test_night_session_gap_fade as gap_fade  # noqa: E402
from tw_quant.hammer_signal_backtest import TradeCost, apply_costs  # noqa: E402

GAP_MIN = 50.0
STOP_FRACTION_OF_GAP = 1.0
TARGET_FRACTIONS = [0.5, 0.75, 1.0]

IS_END = "2022-01-01"
OOS_END = "2024-01-01"


def simulate_follow(df: pd.DataFrame, day_close: pd.Series, target_frac: float) -> pd.DataFrame:
    """跟fade版本的simulate()幾乎一樣，唯一差異：方向跟跳空同向(不是反向)。"""
    night = gap_fade.night_session_frame(df)
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
        if abs_gap < GAP_MIN:
            continue

        direction = "long" if gap > 0 else "short"  # 跟隨跳空方向
        sign = 1 if direction == "long" else -1

        entry_bar = g.iloc[0]
        entry_price = entry_bar["open"] + sign * (gap_fade.NIGHT_COST.slippage_points_round_trip / 2)

        target_dist = abs_gap * target_frac
        stop_dist = abs_gap * STOP_FRACTION_OF_GAP

        exit_price = exit_dt = exit_reason = None
        for j in range(1, len(g)):
            r = g.iloc[j]
            rt = r["datetime"].time()
            favorable = (r["high"] - entry_price) if direction == "long" else (entry_price - r["low"])
            adverse = (entry_price - r["low"]) if direction == "long" else (r["high"] - entry_price)
            if favorable >= target_dist:
                exit_price = entry_price + sign * target_dist - sign * (gap_fade.NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "target"
                break
            if adverse >= stop_dist:
                exit_price = entry_price - sign * stop_dist - sign * (gap_fade.NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "stop"
                break
            is_evening_bar = rt >= gap_fade.NIGHT_START
            if (not is_evening_bar) and (rt >= gap_fade.TIME_STOP):
                exit_price = r["close"] - sign * (gap_fade.NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "time_stop"
                break
        if exit_price is None:
            last = g.iloc[-1]
            exit_price = last["close"] - sign * (gap_fade.NIGHT_COST.slippage_points_round_trip / 2)
            exit_dt, exit_reason = last["datetime"], "session_end"

        pnl_points = (exit_price - entry_price) * sign
        trades.append(dict(
            trading_date=td, direction=direction, gap=gap, abs_gap=abs_gap,
            entry_dt=entry_bar["datetime"], entry_price=entry_price, exit_dt=exit_dt,
            exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    return trades_df.sort_values("entry_dt").reset_index(drop=True)


def report(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"  {label}: n=0")
        return
    t, n, m = gap_fade.tstat(trades["pnl_points"])
    net = apply_costs(trades, TradeCost("已含夜盤滑價的淨額", commission_round_trip=60.0))["net_twd"].sum()
    win_rate = (trades["pnl_points"] > 0).mean() * 100
    print(f"  {label}: n={n} t={t:.2f} mean={m:.1f}點 勝率={win_rate:.1f}% 淨損益={net:,.0f}元")


def main() -> None:
    df = pd.read_parquet(gap_fade.DATA_PATH)
    day_close = gap_fade.build_day_close(df)

    print("=" * 78)
    print(f"跳空延續策略：鎖定{GAP_MIN:.0f}點以上跳空，同向進場賭延續")
    print("=" * 78)

    for target_frac in TARGET_FRACTIONS:
        trades = simulate_follow(df, day_close, target_frac)
        if trades.empty:
            continue
        is_trades = trades[trades["trading_date"] < IS_END]
        oos_trades = trades[(trades["trading_date"] >= IS_END) & (trades["trading_date"] < OOS_END)]
        print(f"\n--- 停利={target_frac*100:.0f}%跳空 (停損=跳空延伸{STOP_FRACTION_OF_GAP*100:.0f}%) ---")
        report(is_trades, "IS(2017-05~2021-12)")
        report(oos_trades, "OOS(2022-2023)")

    print("\n" + "=" * 78)
    print("穩健性警訊：去掉最大3筆獲利（全部集中在2020年3月COVID崩盤那一週）")
    print("=" * 78)
    trades = simulate_follow(df, day_close, 1.0)
    sorted_pnl = trades["pnl_points"].sort_values(ascending=False)
    top3_dates = trades.loc[sorted_pnl.index[:3], "trading_date"].tolist()
    rest = trades[~trades["trading_date"].isin(top3_dates)]
    t, n, m = gap_fade.tstat(rest["pnl_points"])
    print(f"最大3筆日期: {[d.date() for d in top3_dates]}, 點數: {sorted_pnl.head(3).tolist()}")
    print(f"原始(含這3筆): n={len(trades)} sum={trades['pnl_points'].sum():.0f}點")
    print(f"去掉這3筆後: n={n} t={t:.2f} mean={m:.1f} sum={rest['pnl_points'].sum():.0f}點")


if __name__ == "__main__":
    main()
