"""策略實驗室：把這次會話探討過的所有策略統一註冊、標準化交易紀錄格式，
支援單獨使用、任意組合、疊加通用濾網，輸出格式跟「三腿策略歷史交易紀錄」
最終版完全相容（給 scripts/export_strategy_lab.py 用來產生前端資料）。

**重要限制，先講清楚**：這是 Python 端的參數化引擎，不是瀏覽器端即時
運算——1分K資料橫跨23年、300萬根K棒，VWAP/ATR/滾動窗格這類運算量
太大，不可能在瀏覽器裡即時重跑。「參數可調整」的意思是：改這個檔案
（或呼叫端傳進來的 config），重新跑 export 腳本，前端頁面重新載入新
結果——不是網頁上拉滑桿即時看到新回測。頁面端能即時做的，是「切換
哪些策略要疊加」跟「開關濾網」這種不需要重新運算原始信號的操作。

**時間精度不一致，誠實揭露**：三腿策略/VWAP趨勢日/夜盤流動性熱區/
流動性掃單（進場）都是1分K精確模擬，entry_time/exit_time 是真實成交
時刻；支撐壓力fade/breakout、唐奇安、RSI2 只用日K模擬（進出場價位是
真的，但精確到分鐘的時刻沒有意義，因為原始模擬就沒有那個資訊）——
這些策略的 entry_time/exit_time 固定填 08:45（開盤）/13:30（收盤前，
沿用「市價單開盤成交、收盤前平倉」的日K模擬慣例），並標記
`approx_time=True`，前端會在tooltip特別註記「僅日K模擬，時間為近似值」。

**已知結論，供選擇策略時參考**（這次會話完整驗證過的結果）：
- 已驗證有正邊際、值得納入組合考慮：三腿(開盤上衝/午盤放空/盤中翻多)、
  VWAP趨勢日(frac=0.90，但近13年含OOS已轉弱，見risk analysis)、
  支撐壓力順勢突破(唯一OOS候選，證據強度弱於三腿)
- 已系統性證明為負或雜訊、不建議實際疊加、僅供對照實驗：支撐壓力
  逆勢fade(288組合27/28顯著負)、唐奇安(含各種濾網)、RSI2均值回歸、
  開盤區間突破ORB(OOS轉負)、ICT流動性掃單(54組合IS全滅)、夜盤各版本
  (全部零信號或負)、跨日反轉(OOS淨損)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

POINT_VALUE = 50.0
STD_COLUMNS = ["strategy_id", "direction", "entry_date", "entry_time", "entry_price",
               "exit_date", "exit_time", "exit_price", "pnl_points", "approx_time"]


@dataclass
class StrategyDef:
    """一個已註冊策略：id、顯示名稱、跑法（df, cfg -> raw trades）、目前鎖定的
    config、驗證結論分類（用來在UI標色/警示），以及把它的原始輸出正規化成
    STD_COLUMNS 的 adapter。"""
    id: str
    label: str
    run: Callable[[pd.DataFrame, object], pd.DataFrame]
    cfg: object
    verdict: str  # "validated" | "weak" | "rejected"
    adapter: Callable[[pd.DataFrame], pd.DataFrame]


def _std(df: pd.DataFrame, strategy_id: str, direction_col, entry_date_col, entry_price_col,
          exit_date_col, exit_price_col, pnl_col, entry_time_col=None, exit_time_col=None,
          approx_time=False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STD_COLUMNS)
    out = pd.DataFrame(index=df.index)  # 先用df的index建立長度，避免對空DataFrame賦純量欄位變成全NaN
    out["strategy_id"] = strategy_id
    out["direction"] = df[direction_col] if direction_col else "long"
    ed = pd.to_datetime(df[entry_date_col])
    xd = pd.to_datetime(df[exit_date_col])
    out["entry_date"] = ed.dt.strftime("%Y-%m-%d")
    out["exit_date"] = xd.dt.strftime("%Y-%m-%d")
    out["entry_time"] = ed.dt.strftime("%H:%M") if entry_time_col is None and not approx_time else (
        df[entry_time_col].dt.strftime("%H:%M") if entry_time_col else "08:45")
    out["exit_time"] = xd.dt.strftime("%H:%M") if exit_time_col is None and not approx_time else (
        df[exit_time_col].dt.strftime("%H:%M") if exit_time_col else "13:30")
    if approx_time:
        out["entry_time"] = "08:45"
        out["exit_time"] = "13:30"
    out["entry_price"] = df[entry_price_col].astype(float)
    out["exit_price"] = df[exit_price_col].astype(float)
    out["pnl_points"] = df[pnl_col].astype(float)
    out["approx_time"] = approx_time
    return out[STD_COLUMNS]


def _adapt_intraday_dt(df, strategy_id, direction_col="direction", entry_dt="entry_dt", exit_dt="exit_dt",
                        entry_price="entry_price", exit_price="exit_price", pnl="pnl_points"):
    if df.empty:
        return pd.DataFrame(columns=STD_COLUMNS)
    ed = pd.to_datetime(df[entry_dt])
    xd = pd.to_datetime(df[exit_dt])
    out = pd.DataFrame(index=df.index)
    out["strategy_id"] = strategy_id
    out["direction"] = df[direction_col]
    out["entry_date"] = ed.dt.strftime("%Y-%m-%d")
    out["exit_date"] = xd.dt.strftime("%Y-%m-%d")
    # entry_dt/exit_dt 有些策略在多日持有、或本來就只用日K模擬時只有到「日」的
    # 精度（時分是00:00），這種情況視同approx，時間補成開盤/收盤前一刻，不假裝
    # 有分鐘級精度
    entry_is_midnight = (ed.dt.hour == 0) & (ed.dt.minute == 0)
    exit_is_midnight = (xd.dt.hour == 0) & (xd.dt.minute == 0)
    out["entry_time"] = np.where(entry_is_midnight, "08:45", ed.dt.strftime("%H:%M"))
    out["exit_time"] = np.where(exit_is_midnight, "13:30", xd.dt.strftime("%H:%M"))
    out["exit_price"] = df[exit_price].astype(float)
    out["entry_price"] = df[entry_price].astype(float)
    out["pnl_points"] = df[pnl].astype(float)
    out["approx_time"] = entry_is_midnight | exit_is_midnight
    return out[STD_COLUMNS]


def build_registry() -> dict[str, StrategyDef]:
    from tw_quant.opening_rally_strategy import OpeningRallyConfig, backtest as bt_opening
    from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest as bt_lunch
    from tw_quant.trend_day_strategy import TrendDayConfig, backtest as bt_trend_day
    from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr
    from tw_quant.donchian_breakout_strategy import DonchianConfig, backtest as bt_donchian
    from tw_quant.rsi2_mean_reversion_strategy import Rsi2Config, backtest as bt_rsi2
    from tw_quant.opening_range_breakout_strategy import OpeningRangeBreakoutConfig, backtest as bt_orb
    from tw_quant.liquidity_sweep_reversal_strategy import LiquiditySweepConfig, backtest as bt_liqsweep
    from tw_quant.night_liquid_window_trend_strategy import NightLiquidWindowConfig, backtest as bt_night

    reg: dict[str, StrategyDef] = {}

    def add(id_, label, run, cfg, verdict, adapter):
        reg[id_] = StrategyDef(id_, label, run, cfg, verdict, adapter)

    add("og", "開盤上衝", bt_opening, OpeningRallyConfig(), "validated",
        lambda df: _std(df, "og", None, "trading_date", "entry_price", "trading_date", "exit_price", "pnl_points",
                         entry_time_col="entry_dt", exit_time_col="exit_dt"))

    lunch_cfg = LunchReversalConfig()

    def run_lunch_short(df, cfg):
        t = bt_lunch(df, cfg)
        return t[t["leg"] == "short_lunch_dip"]

    def run_lunch_long(df, cfg):
        t = bt_lunch(df, cfg)
        return t[t["leg"] == "long_afternoon_rebound"]

    add("lu", "午盤放空", run_lunch_short, lunch_cfg, "validated",
        lambda df: _std(df, "lu", None, "trading_date", "entry_price", "trading_date", "exit_price", "pnl_points",
                         entry_time_col="entry_dt", exit_time_col="exit_dt"))
    add("re", "盤中翻多", run_lunch_long, lunch_cfg, "validated",
        lambda df: _std(df, "re", None, "trading_date", "entry_price", "trading_date", "exit_price", "pnl_points",
                         entry_time_col="entry_dt", exit_time_col="exit_dt"))

    add("vwap_trend", "VWAP趨勢日", bt_trend_day, TrendDayConfig(min_dominant_side_fraction=0.90), "weak",
        lambda df: _std(df, "vwap_trend", "direction", "trading_date", "entry_price", "trading_date", "exit_price",
                         "pnl_points", entry_time_col="entry_dt", exit_time_col="exit_dt"))

    add("sr_breakout", "支撐壓力順勢突破", bt_sr,
        SupportResistanceFadeConfig(direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
                                     min_range_pct=2.0, trend_slope_threshold_pct=5.0), "weak",
        lambda df: _adapt_intraday_dt(df.assign(entry_dt=pd.to_datetime(df["entry_date"]),
                                                  exit_dt=pd.to_datetime(df["exit_date"])) if not df.empty else df,
                                       "sr_breakout") if not df.empty else pd.DataFrame(columns=STD_COLUMNS))

    add("sr_fade", "支撐壓力逆勢fade", bt_sr, SupportResistanceFadeConfig(direction_mode="fade"), "rejected",
        lambda df: _std(df, "sr_fade", "direction", "entry_date", "entry_price", "exit_date", "exit_price",
                         "pnl_points", approx_time=True))

    add("donchian", "唐奇安突破", bt_donchian, DonchianConfig(), "rejected",
        lambda df: _std(df, "donchian", "direction", "entry_date", "entry_price", "exit_date", "exit_price",
                         "pnl_points", approx_time=True))

    add("rsi2", "RSI2均值回歸", bt_rsi2, Rsi2Config(), "rejected",
        lambda df: _std(df, "rsi2", "direction", "entry_date", "entry_price", "exit_date", "exit_price",
                         "pnl_points", approx_time=True))

    add("orb", "開盤區間突破ORB", bt_orb, OpeningRangeBreakoutConfig(), "rejected",
        lambda df: _std(df, "orb", "direction", "trading_date", "entry_price", "trading_date", "exit_price",
                         "pnl_points", entry_time_col="entry_dt", exit_time_col="exit_dt"))

    add("liqsweep", "ICT流動性掃單", bt_liqsweep, LiquiditySweepConfig(), "rejected",
        lambda df: _adapt_intraday_dt(df, "liqsweep", entry_dt="entry_dt", exit_dt="exit_dt")
        if not df.empty else pd.DataFrame(columns=STD_COLUMNS))

    add("night", "夜盤流動性熱區", bt_night, NightLiquidWindowConfig(min_dominant_side_fraction=0.90), "rejected",
        lambda df: _std(df, "night", "direction", "trading_date", "entry_price", "trading_date", "exit_price",
                         "pnl_points", entry_time_col="entry_dt", exit_time_col="exit_dt"))

    return reg


def run_strategy(reg: dict[str, StrategyDef], strategy_id: str, df: pd.DataFrame,
                  cfg_override=None) -> pd.DataFrame:
    """跑單一策略，回傳已正規化成 STD_COLUMNS 的交易紀錄。cfg_override 可傳自訂
    config（例如調整過參數的 dataclass instance），不傳就用註冊時鎖定的預設。"""
    sd = reg[strategy_id]
    cfg = cfg_override if cfg_override is not None else sd.cfg
    raw = sd.run(df, cfg)
    return sd.adapter(raw)


@dataclass
class VolumeFilterSpec:
    """沿用 technical_indicators.filter_trades_by_volume 的邏輯，泛化成可疊加在
    任何策略交易紀錄上的濾網：只保留「昨日成交量比率 >= threshold」的交易日。
    threshold 必須是外部算好傳入（通常用IS資料算三分位數），這裡不重新計算，
    避免用到未來資訊。"""
    threshold: float
    direction: str = "ge"  # "ge" 或 "le"


def apply_volume_filter(std_trades: pd.DataFrame, df: pd.DataFrame, spec: VolumeFilterSpec) -> pd.DataFrame:
    from tw_quant.technical_indicators import daily_indicators
    if std_trades.empty:
        return std_trades
    indicators = daily_indicators(df)
    ind_map = dict(zip(indicators["date"], indicators["vol_ratio_lag1"]))
    entry_dates = pd.to_datetime(std_trades["entry_date"]).dt.date
    vals = entry_dates.map(ind_map)
    mask = (vals >= spec.threshold) if spec.direction == "ge" else (vals <= spec.threshold)
    return std_trades[mask.fillna(False)].reset_index(drop=True)


@dataclass
class TrendRegimeFilterSpec:
    """200日均線趨勢對齊濾網（跟 vwap_trend_day_regime_filter_test.py 同一套邏輯，
    當時測出來對VWAP趨勢日候選沒有幫助，但保留成通用選項讓使用者自行實驗）。
    aligned_only=True 時只保留「方向跟regime對齊」的交易（多單只在多頭regime、
    空單只在空頭regime）。"""
    ma_window: int = 200
    aligned_only: bool = True


def apply_trend_regime_filter(std_trades: pd.DataFrame, df: pd.DataFrame, spec: TrendRegimeFilterSpec) -> pd.DataFrame:
    from tw_quant.technical_indicators import build_daily_bars
    if std_trades.empty:
        return std_trades
    daily = build_daily_bars(df)
    daily["ma"] = daily["close"].shift(1).rolling(spec.ma_window).mean()
    daily["regime_long"] = daily["close"].shift(1) > daily["ma"]
    regime_map = dict(zip(daily["date"].dt.date, daily["regime_long"]))
    entry_dates = pd.to_datetime(std_trades["entry_date"]).dt.date
    regime = entry_dates.map(regime_map)
    aligned = ((std_trades["direction"] == "long") & regime) | ((std_trades["direction"] == "short") & ~regime)
    mask = aligned if spec.aligned_only else ~aligned.fillna(False)
    return std_trades[mask.fillna(False)].reset_index(drop=True)


def combine(trade_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """把多個已正規化的策略交易紀錄疊在一起，回傳合併後的單一交易列表
    （排序、reset_index），用來算合併後的每日/累積損益。"""
    non_empty = [t for t in trade_frames if not t.empty]
    if not non_empty:
        return pd.DataFrame(columns=STD_COLUMNS)
    out = pd.concat(non_empty, ignore_index=True)
    return out.sort_values(["entry_date", "entry_time"]).reset_index(drop=True)
