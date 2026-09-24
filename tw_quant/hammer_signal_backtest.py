"""把「長下影線 + 次根紅K」訊號接上固定風報比停損停利，做正向（做多）跟反向
（做空）的完整交易模擬，並套用真實成本模型。

跟 `scripts/study_hammer_ratio.py`（純遠期報酬版本）的差異：
- 不用固定分鐘數量測遠期報酬，改成每筆訊號真的算一筆交易：進場、停損/停利
  哪個先碰到就出場，都沒碰到則在 `max_hold_minutes` 後以當根收盤價強制出場
  （同一段連續盤中，跨盤直接視為出場，避免持倉跨越大跳空區段）。
- 同一時間只允許一筆未平倉部位（訊號出現時若已在場內就跳過），這樣統計出來
  的才是「一個帳戶真的照規則交易」會發生的筆數，而不是重疊採樣的統計樣本。
- 風險/停損停利定義（跟前一版夜盤策略一致）：
    risk = 進場價 - (訊號K棒最低點 - sl_buffer)
    做多：SL = 進場價 - risk，TP = 進場價 + risk * r_multiple
    做空：SL = 進場價 + risk，TP = 進場價 - risk * r_multiple
  多空共用同一個 risk（點數）算出來的停損距離，只是方向相反，這樣兩個方向的
  部位風險大小才可比。
  若 risk 超過 `max_risk_points`（預設 140，沿用原規格書的小台單筆風控上限）
  就放棄這筆訊號。

成本模型（見 `TradeCost`）：
- 期交稅：對台指期／小型台指期課徵，稅基是「進場與出場當下的名目契約價值」
  （價格 × 點值），雙邊各課一次。稅率 10 萬分之 2（0.00002）——使用者已確認
  這是股價類期貨（大台/小台/股票期貨）目前公告稅率（臺指選擇權是千分之1、
  黃金期貨是十萬分之0.25，跟本模組無關）。
- 手續費：跳動大、券商折扣差異很大，用三個情境（低/中/高，每筆來回固定金額）
  涵蓋常見範圍，不假裝有一個「正確」數字。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POINT_VALUE = 50.0
TAX_RATE_PER_SIDE = 0.00002  # 股價類期貨（大台/小台/股票期貨）稅率，已由使用者確認，見檔頭說明
MAX_GAP_MINUTES = 2


@dataclass
class TradeCost:
    label: str
    commission_round_trip: float  # 每筆來回固定手續費（元）
    tax_rate_per_side: float = TAX_RATE_PER_SIDE
    point_value: float = POINT_VALUE
    slippage_points_round_trip: float = 0.0  # 進場+出場合計的滑價點數（0=沿用舊行為，假設完全用K棒開盤價成交）

    def total_cost(self, entry_price: np.ndarray, exit_price: np.ndarray) -> np.ndarray:
        tax = self.tax_rate_per_side * self.point_value * (np.abs(entry_price) + np.abs(exit_price))
        slippage = self.slippage_points_round_trip * self.point_value
        return tax + self.commission_round_trip + slippage


COST_SCENARIOS = [
    TradeCost("低（折扣網路下單，來回30元）", commission_round_trip=30.0),
    TradeCost("中（一般券商，來回60元）", commission_round_trip=60.0),
    TradeCost("高（傳統營業員，來回100元）", commission_round_trip=100.0),
]


def build_base_arrays(df: pd.DataFrame) -> dict:
    d = df.sort_values("datetime").reset_index(drop=True)
    open_ = d["open"].to_numpy(dtype=float)
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    close = d["close"].to_numpy(dtype=float)

    body = np.abs(open_ - close)
    lower_shadow = np.minimum(open_, close) - low
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(body > 0, lower_shadow / body, np.nan)

    gap_min = d["datetime"].diff().dt.total_seconds().to_numpy() / 60
    new_block = (gap_min > MAX_GAP_MINUTES) | np.isnan(gap_min)
    block_id = np.cumsum(new_block)

    n = len(d)
    next_open = np.append(open_[1:], np.nan)
    next_close = np.append(close[1:], np.nan)
    next_block = np.append(block_id[1:], -1)
    next_bullish = next_close > next_open

    return dict(
        n=n, open=open_, high=high, low=low, close=close,
        body=body, ratio=ratio, block_id=block_id,
        next_close=next_close, next_block=next_block, next_bullish=next_bullish,
    )


def build_htf_trend(df: pd.DataFrame, freq: str = "60min", sma_window: int = 20) -> np.ndarray:
    """用更高時間週期（預設 60 分鐘）K棒收盤 vs SMA(sma_window) 判斷多空趨勢，
    回填成跟 df 等長的陣列：1=多頭（收盤>SMA）、-1=空頭（收盤<SMA）、0=無資料/盤整。

    嚴格避免未來函數：某根高週期K棒收在時刻 T，它的趨勢值只給時刻 > T 的 1 分鐘
    K棒使用（`merge_asof(..., allow_exact_matches=False)`）——不會讓同一根高週期
    K棒還沒收盤就被拿來篩選訊號。
    """
    d = df.sort_values("datetime").reset_index(drop=True)
    htf_close = d.set_index("datetime")["close"].resample(freq, label="right", closed="left").last().dropna()
    htf_sma = htf_close.rolling(sma_window).mean()

    htf_trend = pd.Series(0, index=htf_close.index, dtype=int)
    htf_trend[htf_close > htf_sma] = 1
    htf_trend[htf_close < htf_sma] = -1
    htf_trend = htf_trend[htf_sma.notna()]

    merged = pd.merge_asof(
        d[["datetime"]], htf_trend.rename("htf_trend").reset_index(),
        on="datetime", direction="backward", allow_exact_matches=False,
    )
    return merged["htf_trend"].fillna(0).to_numpy(dtype=int)


def simulate(
    arrays: dict,
    ratio_threshold: float,
    r_multiple: float,
    direction: str,
    sl_buffer: float = 5.0,
    max_risk_points: float = 140.0,
    max_hold_minutes: int = 60,
    htf_trend: np.ndarray | None = None,
    trend_mode: str = "none",
) -> pd.DataFrame:
    """trend_mode: 'none'（不過濾，預設）/ 'with'（順higher-TF趨勢：做多只在高週期
    多頭、做空只在高週期空頭）/ 'against'（逆higher-TF趨勢，跟 'with' 相反）。"""
    assert direction in ("long", "short")
    assert trend_mode in ("none", "with", "against")
    n = arrays["n"]
    high, low, close = arrays["high"], arrays["low"], arrays["close"]
    block_id, next_block = arrays["block_id"], arrays["next_block"]
    next_close, next_bullish = arrays["next_close"], arrays["next_bullish"]
    body, ratio = arrays["body"], arrays["ratio"]

    is_candidate = (body > 0) & (ratio >= ratio_threshold) & next_bullish & (next_block == block_id)

    if trend_mode != "none":
        if htf_trend is None:
            raise ValueError("trend_mode != 'none' 需要提供 htf_trend")
        long_wants_up = trend_mode == "with"
        if direction == "long":
            wanted = 1 if long_wants_up else -1
        else:
            wanted = -1 if long_wants_up else 1
        is_candidate = is_candidate & (htf_trend == wanted)

    cand_idx = np.flatnonzero(is_candidate)

    trades = []
    blocked_until = -1
    for i in cand_idx:
        if i <= blocked_until:
            continue
        entry_idx = i + 1
        entry_price = next_close[i]
        signal_low = low[i]
        risk = entry_price - (signal_low - sl_buffer)
        if not (0 < risk <= max_risk_points):
            continue

        if direction == "long":
            sl = entry_price - risk
            tp = entry_price + risk * r_multiple
        else:
            sl = entry_price + risk
            tp = entry_price - risk * r_multiple

        limit = min(entry_idx + max_hold_minutes, n - 1)
        exit_idx = None
        exit_price = None
        exit_reason = None
        for j in range(entry_idx + 1, limit + 1):
            if block_id[j] != block_id[entry_idx]:
                exit_idx, exit_price, exit_reason = j - 1, close[j - 1], "block_end"
                break
            if direction == "long":
                if low[j] <= sl:
                    exit_idx, exit_price, exit_reason = j, sl, "stop_loss"
                    break
                if high[j] >= tp:
                    exit_idx, exit_price, exit_reason = j, tp, "take_profit"
                    break
            else:
                if high[j] >= sl:
                    exit_idx, exit_price, exit_reason = j, sl, "stop_loss"
                    break
                if low[j] <= tp:
                    exit_idx, exit_price, exit_reason = j, tp, "take_profit"
                    break
        if exit_idx is None:
            exit_idx, exit_price, exit_reason = limit, close[limit], "max_hold"

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            signal_idx=i, entry_idx=entry_idx, exit_idx=exit_idx,
            entry_price=entry_price, exit_price=exit_price, sl=sl, tp=tp,
            risk_points=risk, exit_reason=exit_reason, pnl_points=pnl_points,
        ))
        blocked_until = exit_idx

    return pd.DataFrame(trades)


def apply_costs(trades: pd.DataFrame, cost: TradeCost) -> pd.DataFrame:
    t = trades.copy()
    t["gross_twd"] = t["pnl_points"] * cost.point_value
    t["cost_twd"] = cost.total_cost(t["entry_price"].to_numpy(), t["exit_price"].to_numpy())
    t["net_twd"] = t["gross_twd"] - t["cost_twd"]
    return t
