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
- 已驗證有正邊際、值得納入組合考慮：三腿(開盤上衝/午盤放空/盤中翻多，
  加上高波動體制濾網後最穩健，見 VolatilityRegimeFilterSpec)、
  低波動做多+8%停損(跟三腿互斥的體制互補腿)、近月/遠月價差均值回歸
  (完全不同的多日波段操作頻率，6折walk-forward全正)、VWAP趨勢日
  (frac=0.90，但近13年含OOS已轉弱，見risk analysis)、支撐壓力順勢突破
  (OOS候選，證據強度弱於三腿)
- 已系統性證明為負或雜訊、不建議實際疊加、僅供對照實驗：支撐壓力
  逆勢fade(288組合27/28顯著負)、唐奇安(含各種濾網)、RSI2均值回歸、
  開盤區間突破ORB(OOS轉負)、ICT流動性掃單(54組合IS全滅)、夜盤VWAP版
  (全部零信號或負)、跨日反轉(OOS淨損)
- 證據尚弱、僅供模擬帳戶參考：夜盤大跳空延續(NightGapFollowConfig，
  IS/OOS方向一致但樣本小、部分邊際依賴少數極端年份)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

POINT_VALUE = 50.0
STD_COLUMNS = ["strategy_id", "direction", "entry_date", "entry_time", "entry_price",
               "exit_date", "exit_time", "exit_price", "pnl_points", "approx_time"]
REPO_ROOT = Path(__file__).resolve().parent.parent
MULTI_CONTRACT_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"


def _load_multi_contract() -> pd.DataFrame:
    """近月/遠月合約每日序列（近月結算價/未平倉、遠月結算價/未平倉、價差），
    來源 scripts/build_taifex_multi_contract.py。低波動做多、近月/遠月價差
    這兩個策略都需要這份資料，1分K的txf_1min.parquet不含遠月合約，也不含
    官方結算價（只有日盤逐分鐘成交），所以這兩個策略無法只靠傳進來的df
    自給自足，直接讀這個檔案——跟引擎其餘策略「只靠df」的設計不同，這裡
    誠實揭露這個例外。"""
    return pd.read_parquet(MULTI_CONTRACT_PATH).sort_values("date").reset_index(drop=True)


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
        # direction_col=None預設填"long"，但lu(short_lunch_dip)本身是空單，
        # 之前這裡漏掉了，一直誤標成"long"（pnl_points本身算對，只有這個
        # 顯示/篩選用的metadata欄位錯，這次會話新增的parity測試才抓到）
        lambda df: _std(df.assign(direction="short"), "lu", "direction", "trading_date", "entry_price",
                         "trading_date", "exit_price", "pnl_points",
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

    add("low_vol_buyhold", "低波動做多+8%停損", run_low_vol_buyhold, LowVolBuyHoldConfig(), "validated",
        lambda df: _std(df, "low_vol_buyhold", "direction", "entry_date", "entry_price", "exit_date", "exit_price",
                         "pnl_points", approx_time=True))

    add("calendar_spread", "近月/遠月價差均值回歸", run_calendar_spread, CalendarSpreadConfig(), "validated",
        lambda df: _std(df, "calendar_spread", "direction", "entry_date", "entry_price", "exit_date", "exit_price",
                         "pnl_points", approx_time=True))

    add("night_gap_follow", "夜盤大跳空延續", run_night_gap_follow, NightGapFollowConfig(), "weak",
        lambda df: _adapt_intraday_dt(df, "night_gap_follow")
        if not df.empty else pd.DataFrame(columns=STD_COLUMNS))

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


@dataclass
class VolatilityRegimeFilterSpec:
    """波動度體制濾網(ATR14 vs 252日ATR均值，10%遲滯帶)，跟
    scripts/build_smoothed_regime_classifier.py同一套演算法，只是改用
    df(1分K)聚合出的日盤OHLC重算，不依賴獨立的
    data/regime_classification_daily.parquet——維持這支引擎「單一輸入df」
    的架構一致性（跟raw CSV重建版比對過，體制邊界幾乎一致，差異只在
    極少數轉換日附近的一兩天）。

    high_vol_only=True 時只保留「前一交易日收盤已確定為高波動體制」的
    交易日（三腿策略用，跟Pine Script `isHighVol[1]`同一個「用前一天、
    不用當天」的道理——今天自己的日K要收盤才有ATR，不能拿來決定今天
    要不要交易）；False 時保留低波動體制（低波動做多用）。
    """
    atr_window: int = 14
    ma_window: int = 252
    band: float = 0.10
    high_vol_only: bool = True


def _compute_volatility_regime_map(df: pd.DataFrame, atr_window: int, ma_window: int, band: float) -> dict:
    """回傳 {date: 0.0/1.0/nan}，值＝『前一交易日收盤』已確定的體制
    （對外一律用這個already-lagged版本，呼叫端不需要再shift）。"""
    from tw_quant.technical_indicators import build_daily_bars
    daily = build_daily_bars(df).sort_values("date").reset_index(drop=True)
    close, high, low = daily["close"], daily["high"], daily["low"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_window, adjust=False).mean()
    atr_ma = atr.rolling(ma_window).mean()
    upper = atr_ma * (1 + band)
    lower = atr_ma * (1 - band)

    regime = np.full(len(daily), np.nan)
    current = np.nan
    for i in range(len(daily)):
        if pd.isna(atr_ma.iloc[i]):
            continue
        v = atr.iloc[i]
        if pd.isna(current):
            current = 1.0 if v > atr_ma.iloc[i] else 0.0
        elif v > upper.iloc[i]:
            current = 1.0
        elif v < lower.iloc[i]:
            current = 0.0
        regime[i] = current
    lagged = pd.Series(regime).shift(1)  # 今天能不能交易，看「前一天」已確定的體制
    return dict(zip(daily["date"].dt.date, lagged))


def apply_volatility_regime_filter(std_trades: pd.DataFrame, df: pd.DataFrame,
                                     spec: VolatilityRegimeFilterSpec) -> pd.DataFrame:
    if std_trades.empty:
        return std_trades
    regime_map = _compute_volatility_regime_map(df, spec.atr_window, spec.ma_window, spec.band)
    entry_dates = pd.to_datetime(std_trades["entry_date"]).dt.date
    is_high_vol = entry_dates.map(regime_map)
    mask = (is_high_vol == 1.0) if spec.high_vol_only else (is_high_vol == 0.0)
    return std_trades[mask.fillna(False)].reset_index(drop=True)


@dataclass
class LowVolBuyHoldConfig:
    """低波動做多+移動停損（跟三腿策略互斥的體制互補腿，見
    scripts/test_low_vol_stoploss.py）。用近月期貨(taifex_tx_multi_contract.
    parquet的near_price)在低波動體制期間做多，收盤價從進場後至今最高點
    回落stop_pct即出場，同一段低波動週期內不重新進場。

    體制邊界用同一套VolatilityRegimeFilterSpec邏輯算(atr_window/ma_window/
    band跟三腿濾網共用同一組數字，體制定義本身只能有一套、不能各策略各自
    表述)，這裡的回測維持跟原始驗證腳本一致的「同日規則」（用episode當天
    自己的體制值找週期邊界，不額外lag）——這在統計上跟lag版本幾乎沒有
    差異(轉換日很少見)，但**真實下單一定要按手冊的規則，用前一交易日
    收盤已確定的體制決定隔天要不要進場**，不能用當天還沒收盤的體制。"""
    atr_window: int = 14
    ma_window: int = 252
    band: float = 0.10
    stop_pct: float = 0.08


def run_low_vol_buyhold(df: pd.DataFrame, cfg: LowVolBuyHoldConfig) -> pd.DataFrame:
    # 原本這裡的ATR/體制計算用 build_daily_bars(df)（1分K衍生的日盤OHLC），
    # 但1分K原始資料是靜態歷史檔（只到2023-12-29，來源是網路分享的舊資料，
    # 沒有每日更新管道）。改用_load_multi_contract()裡近月合約的官方
    # open/high/low/近月結算價——這是TAIFEX官方每日行情，可以每天增量更新。
    # 兩者交叉驗證過(2001-2023重疊窗口)：regime分類100%一致(5421天0差異)，
    # 價格差異中位數僅0.008%，可以放心互換，讓這個策略從此不再依賴1分K。
    mc_raw = _load_multi_contract()
    daily = mc_raw.rename(columns={"near_open": "open", "near_high": "high",
                                    "near_low": "low", "near_price": "close"}).sort_values("date").reset_index(drop=True)
    close, high, low = daily["close"], daily["high"], daily["low"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / cfg.atr_window, adjust=False).mean()
    atr_ma = atr.rolling(cfg.ma_window).mean()
    upper, lower = atr_ma * (1 + cfg.band), atr_ma * (1 - cfg.band)

    regime = np.full(len(daily), np.nan)
    current = np.nan
    for i in range(len(daily)):
        if pd.isna(atr_ma.iloc[i]):
            continue
        v = atr.iloc[i]
        if pd.isna(current):
            current = 1.0 if v > atr_ma.iloc[i] else 0.0
        elif v > upper.iloc[i]:
            current = 1.0
        elif v < lower.iloc[i]:
            current = 0.0
        regime[i] = current

    mc = mc_raw.copy()
    mc = mc.merge(pd.DataFrame({"date": daily["date"], "high_vol_regime": regime}), on="date", how="inner")
    mc = mc.sort_values("date").reset_index(drop=True)

    valid = mc["high_vol_regime"].notna()
    r = (mc.loc[valid, "high_vol_regime"] == 0.0)
    d = mc.loc[valid, "date"]
    group_id = r.ne(r.shift()).cumsum()
    tmp = pd.DataFrame({"date": d.values, "is_low_vol": r.values, "group": group_id.values})
    episodes = []
    for _, sub in tmp.groupby("group"):
        if sub["is_low_vol"].iloc[0]:
            episodes.append((sub["date"].min(), sub["date"].max()))

    trades = []
    for start, end in episodes:
        ep = mc[(mc["date"] >= start) & (mc["date"] <= end)].reset_index(drop=True)
        if len(ep) < 2:
            continue
        entry_price = ep["near_price"].iloc[0]
        peak = entry_price
        stopped = False
        exit_price, exit_date = ep["near_price"].iloc[-1], ep["date"].iloc[-1]
        for i in range(1, len(ep)):
            price = ep["near_price"].iloc[i]
            peak = max(peak, price)
            if price <= peak * (1 - cfg.stop_pct):
                exit_price, exit_date, stopped = price, ep["date"].iloc[i], True
                break
        trades.append(dict(
            direction="long", entry_date=ep["date"].iloc[0], exit_date=exit_date,
            entry_price=entry_price, exit_price=exit_price,
            pnl_points=exit_price - entry_price, stopped=stopped,
        ))
    return pd.DataFrame(trades)


@dataclass
class CalendarSpreadConfig:
    """近月/遠月價差均值回歸（見 scripts/validate_calendar_spread_reversion.py）。
    z分數用90天滾動窗格、shift(1)避免未來函數；事件週期式進出場（z第一次
    穿越門檻才進場，回到中性帶或碰到最長持有天數才出場，天生不重疊，
    不需要額外做非重疊窗格校正）。entry_price/exit_price 這裡存的是「價差
    本身」（近月結算價-遠月結算價），不是單一合約價位——這是這支引擎裡
    唯一一個entry_price/exit_price代表價差、不是單一商品價格的策略，
    pnl_points的算法（(exit_spread-entry_spread)*方向）沒有變。"""
    window: int = 90
    entry_z: float = 1.0
    exit_z: float = 0.3
    max_hold_days: int = 20
    min_far_oi: float = 3000.0


def run_calendar_spread(df: pd.DataFrame, cfg: CalendarSpreadConfig) -> pd.DataFrame:
    mc = _load_multi_contract()
    mc["spread"] = mc["calendar_spread"].astype(float)
    roll_mean = mc["spread"].rolling(cfg.window).mean().shift(1)
    roll_std = mc["spread"].rolling(cfg.window).std(ddof=1).shift(1)
    mc["z"] = (mc["spread"].shift(1) - roll_mean) / roll_std

    trades = []
    in_position, direction, entry_idx = False, None, None
    for i in range(len(mc)):
        row = mc.iloc[i]
        if pd.isna(row["z"]) or row["far_oi"] < cfg.min_far_oi:
            continue
        if not in_position:
            if row["z"] >= cfg.entry_z:
                in_position, direction, entry_idx = True, "short", i
            elif row["z"] <= -cfg.entry_z:
                in_position, direction, entry_idx = True, "long", i
        else:
            held = i - entry_idx
            reverted = abs(row["z"]) <= cfg.exit_z
            if reverted or held >= cfg.max_hold_days or i == len(mc) - 1:
                entry_row = mc.iloc[entry_idx]
                sign = 1 if direction == "long" else -1
                entry_spread, exit_spread = entry_row["spread"], row["spread"]
                trades.append(dict(
                    direction=direction, entry_date=entry_row["date"], exit_date=row["date"],
                    entry_price=entry_spread, exit_price=exit_spread,
                    pnl_points=(exit_spread - entry_spread) * sign,
                ))
                in_position = False
    return pd.DataFrame(trades)


@dataclass
class NightGapFollowConfig:
    """夜盤大跳空延續（見 scripts/test_night_session_gap_follow.py）。跳空=
    夜盤開盤價-當天日盤收盤價，|跳空|>=gap_min才進場，跟隨跳空方向（賭
    延續、不回補）。停利=跳空距離的target_frac倍，停損=跳空距離的
    stop_frac倍，01:00時間停損，04:45保底出場。"""
    gap_min: float = 50.0
    target_frac: float = 1.0
    stop_frac: float = 1.0
    slippage_points: float = 3.0


def run_night_gap_follow(df: pd.DataFrame, cfg: NightGapFollowConfig) -> pd.DataFrame:
    from datetime import time as dtime
    t = df["datetime"].dt.time
    day = df[(t >= dtime(8, 45)) & (t <= dtime(13, 45))].copy()
    day["trading_date"] = day["datetime"].dt.date
    day_close = day.sort_values("datetime").groupby("trading_date")["close"].last().rename("day_close")
    day_close.index = pd.to_datetime(day_close.index)

    night = df[(t >= dtime(15, 0)) | (t <= dtime(5, 0))].copy()
    is_evening = night["datetime"].dt.time >= dtime(15, 0)
    night["own_date"] = night["datetime"].dt.date
    night["trading_date"] = np.where(is_evening, night["own_date"], night["own_date"] - pd.Timedelta(days=1))
    night["trading_date"] = pd.to_datetime(night["trading_date"])

    half_slip = cfg.slippage_points / 2
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
        if abs_gap < cfg.gap_min:
            continue

        direction = "long" if gap > 0 else "short"
        sign = 1 if direction == "long" else -1
        entry_bar = g.iloc[0]
        entry_price = entry_bar["open"] + sign * half_slip
        target_dist, stop_dist = abs_gap * cfg.target_frac, abs_gap * cfg.stop_frac

        exit_price = exit_dt = None
        for j in range(1, len(g)):
            r = g.iloc[j]
            rt = r["datetime"].time()
            favorable = (r["high"] - entry_price) if direction == "long" else (entry_price - r["low"])
            adverse = (entry_price - r["low"]) if direction == "long" else (r["high"] - entry_price)
            if favorable >= target_dist:
                exit_price = entry_price + sign * target_dist - sign * half_slip
                exit_dt = r["datetime"]
                break
            if adverse >= stop_dist:
                exit_price = entry_price - sign * stop_dist - sign * half_slip
                exit_dt = r["datetime"]
                break
            is_evening_bar = rt >= dtime(15, 0)
            if (not is_evening_bar) and (rt >= dtime(1, 0)):
                exit_price = r["close"] - sign * half_slip
                exit_dt = r["datetime"]
                break
        if exit_price is None:
            last = g.iloc[-1]
            exit_price, exit_dt = last["close"] - sign * half_slip, last["datetime"]

        trades.append(dict(
            direction=direction, entry_dt=entry_bar["datetime"], exit_dt=exit_dt,
            entry_price=entry_price, exit_price=exit_price,
            pnl_points=(exit_price - entry_price) * sign,
        ))
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    return trades_df.sort_values("entry_dt").reset_index(drop=True)


def combine(trade_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """把多個已正規化的策略交易紀錄疊在一起，回傳合併後的單一交易列表
    （排序、reset_index），用來算合併後的每日/累積損益。"""
    non_empty = [t for t in trade_frames if not t.empty]
    if not non_empty:
        return pd.DataFrame(columns=STD_COLUMNS)
    out = pd.concat(non_empty, ignore_index=True)
    return out.sort_values(["entry_date", "entry_time"]).reset_index(drop=True)
