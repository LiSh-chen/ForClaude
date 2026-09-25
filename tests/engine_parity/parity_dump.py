"""Python vs JS engine 正確性對拍：直接呼叫每個策略的 backtest()（不透過
strategy_lab.py 的 adapter，自己正規化成跟 engine.js 輸出完全同型的欄位），
每個策略跑2-3組參數（含非預設值，確保濾網/分支都被踩到），輸出成JSON
給 Node 測試腳本比對。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from tw_quant.opening_rally_strategy import OpeningRallyConfig, backtest as bt_og
from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest as bt_lu
from tw_quant.trend_day_strategy import TrendDayConfig, backtest as bt_trend
from tw_quant.night_liquid_window_trend_strategy import NightLiquidWindowConfig, backtest as bt_night
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr
from tw_quant.donchian_breakout_strategy import DonchianConfig, backtest as bt_donchian
from tw_quant.rsi2_mean_reversion_strategy import Rsi2Config, backtest as bt_rsi2
from tw_quant.opening_range_breakout_strategy import OpeningRangeBreakoutConfig, backtest as bt_orb
from tw_quant.liquidity_sweep_reversal_strategy import LiquiditySweepConfig, backtest as bt_liqsweep

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "parity_out"
OUT_DIR.mkdir(exist_ok=True)


def dt_to_hhmm(x):
    return pd.Timestamp(x).strftime("%H:%M")


def dt_to_date(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def is_midnight(x):
    ts = pd.Timestamp(x)
    return ts.hour == 0 and ts.minute == 0


def dump(name, records):
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(records, f)
    print(f"  {name}: {len(records)} 筆")


_LEG_DIRECTION = {"opening_rally": "long", "short_lunch_dip": "short", "long_afternoon_rebound": "long"}


def norm_intraday_same_day(df, leg_filter=None):
    """og/lu/re/orb/vwap_trend/night 共用：entry_dt/exit_dt都是同一天真實時刻"""
    out = []
    for _, r in df.iterrows():
        if leg_filter is not None and r.get("leg") != leg_filter:
            continue
        if "direction" in r and pd.notna(r.get("direction")):
            direction = r["direction"]
        elif "leg" in r and r.get("leg") in _LEG_DIRECTION:
            direction = _LEG_DIRECTION[r["leg"]]
        else:
            direction = "long"
        out.append({
            "direction": direction,
            "entryDate": dt_to_date(r["entry_dt"]), "entryTime": dt_to_hhmm(r["entry_dt"]),
            "entryPrice": round(float(r["entry_price"]), 4),
            "exitDate": dt_to_date(r["exit_dt"]), "exitTime": dt_to_hhmm(r["exit_dt"]),
            "exitPrice": round(float(r["exit_price"]), 4),
            "pnlPoints": round(float(r["pnl_points"]), 4), "approxTime": False,
        })
    return out


def norm_daily_only(df):
    """sr_fade/sr_breakout/donchian/rsi2 共用：日K模擬，時間固定08:45/13:30"""
    out = []
    for _, r in df.iterrows():
        out.append({
            "direction": r["direction"],
            "entryDate": dt_to_date(r["entry_date"]), "entryTime": "08:45",
            "entryPrice": round(float(r["entry_price"]), 4),
            "exitDate": dt_to_date(r["exit_date"]), "exitTime": "13:30",
            "exitPrice": round(float(r["exit_price"]), 4),
            "pnlPoints": round(float(r["pnl_points"]), 4), "approxTime": True,
        })
    return out


def norm_liqsweep(df):
    out = []
    for _, r in df.iterrows():
        exit_approx = is_midnight(r["exit_dt"])
        out.append({
            "direction": r["direction"],
            "entryDate": dt_to_date(r["entry_dt"]), "entryTime": dt_to_hhmm(r["entry_dt"]),
            "entryPrice": round(float(r["entry_price"]), 4),
            "exitDate": dt_to_date(r["exit_dt"]),
            "exitTime": "13:30" if exit_approx else dt_to_hhmm(r["exit_dt"]),
            "exitPrice": round(float(r["exit_price"]), 4),
            "pnlPoints": round(float(r["pnl_points"]), 4), "approxTime": bool(exit_approx),
        })
    return out


def main():
    df = pd.read_parquet(REPO_ROOT / "data" / "txf_1min.parquet")
    print(f"loaded {len(df)} rows, {df['datetime'].min()} ~ {df['datetime'].max()}")

    # ---- og ----
    dump("og_default", norm_intraday_same_day(bt_og(df, OpeningRallyConfig())))
    from datetime import time
    dump("og_variant", norm_intraday_same_day(bt_og(df, OpeningRallyConfig(entry=time(9, 0), exit=time(9, 15)))))

    # ---- lu / re (share one config) ----
    lu_default = bt_lu(df, LunchReversalConfig())
    dump("lu_default", norm_intraday_same_day(lu_default, "short_lunch_dip"))
    dump("re_default", norm_intraday_same_day(lu_default, "long_afternoon_rebound"))
    lu_variant = bt_lu(df, LunchReversalConfig(max_loss_points=50.0))
    dump("lu_variant", norm_intraday_same_day(lu_variant, "short_lunch_dip"))
    dump("re_variant", norm_intraday_same_day(lu_variant, "long_afternoon_rebound"))

    # ---- vwap_trend ----
    dump("vwap_trend_default", norm_intraday_same_day(bt_trend(df, TrendDayConfig(min_dominant_side_fraction=0.90))))
    dump("vwap_trend_open_ref", norm_intraday_same_day(
        bt_trend(df, TrendDayConfig(min_dominant_side_fraction=0.90, reference="open"))))
    dump("vwap_trend_trailing_atr", norm_intraday_same_day(
        bt_trend(df, TrendDayConfig(min_dominant_side_fraction=0.90, exit_mode="trailing_atr"))))
    dump("vwap_trend_pct", norm_intraday_same_day(
        bt_trend(df, TrendDayConfig(min_dominant_side_fraction=0.85, min_move_pct=0.005))))

    # ---- night ----
    dump("night_default", norm_intraday_same_day(bt_night(df, NightLiquidWindowConfig(min_dominant_side_fraction=0.90))))
    dump("night_strict", norm_intraday_same_day(bt_night(df, NightLiquidWindowConfig(min_dominant_side_fraction=1.0))))

    # ---- sr_breakout / sr_fade ----
    dump("sr_breakout_default", norm_daily_only(bt_sr(df, SupportResistanceFadeConfig(
        direction_mode="breakout", channel_window=20, stop_atr_mult=1.0, min_range_pct=2.0,
        trend_slope_threshold_pct=5.0))))
    dump("sr_fade_default", norm_daily_only(bt_sr(df, SupportResistanceFadeConfig(direction_mode="fade"))))
    dump("sr_fade_variant", norm_daily_only(bt_sr(df, SupportResistanceFadeConfig(
        direction_mode="fade", channel_window=15, min_range_pct=4.0, stop_atr_mult=1.5, max_hold_days=10))))

    # ---- donchian ----
    dump("donchian_default", norm_daily_only(bt_donchian(df, DonchianConfig())))
    dump("donchian_filters", norm_daily_only(bt_donchian(df, DonchianConfig(
        trend_filter=True, volume_filter=True, volatility_squeeze_filter=True))))
    dump("donchian_variant", norm_daily_only(bt_donchian(df, DonchianConfig(
        entry_window=40, exit_window=20, atr_stop_mult=3.0, max_hold_days=30))))

    # ---- rsi2 ----
    dump("rsi2_default", norm_daily_only(bt_rsi2(df, Rsi2Config())))
    dump("rsi2_variant", norm_daily_only(bt_rsi2(df, Rsi2Config(rsi_period=3, oversold=15, overbought=60, max_hold_days=5))))

    # ---- orb ----
    dump("orb_default", norm_intraday_same_day(bt_orb(df, OpeningRangeBreakoutConfig())))
    dump("orb_target", norm_intraday_same_day(bt_orb(df, OpeningRangeBreakoutConfig(target_r_multiple=2.0))))

    # ---- liqsweep ----
    dump("liqsweep_default", norm_liqsweep(bt_liqsweep(df, LiquiditySweepConfig())))
    dump("liqsweep_variant", norm_liqsweep(bt_liqsweep(df, LiquiditySweepConfig(
        sweep_buffer_points=5.0, stop_buffer_points=8.0, target_r_multiple=3.0, max_hold_days=5))))

    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
