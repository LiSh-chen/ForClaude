"""肆、台股實務部位計算與流動性限制（單一標的層級，不含跨標的全域風控）。

跨標的的「單日新增曝險上限」「產業集中度」「流動性 tie-breaker」屬於
投資組合層級的全域鎖，見 portfolio_risk.py；本檔只處理單一訊號自己的
進場價、防守價、股數計算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tw_quant.config import PositionSizingConfig


def compute_initial_stop(entry_price: float, prior_10d_high: float, atr_at_signal: float, cfg: PositionSizingConfig) -> float:
    """初始防守價 = max(進場價 * 0.95, 過去10日最高價 - 2.5 * T日ATR)。

    prior_10d_high / atr_at_signal 皆取「訊號日 T」的值（T 日收盤後可得），
    entry_price 則是 T+1 日開盤實際成交價。
    """
    floor_price = entry_price * cfg.initial_stop_pct_of_entry
    atr_stop = prior_10d_high - cfg.atr_multiplier * atr_at_signal
    return max(floor_price, atr_stop)


def compute_chandelier_stop(rolling_10d_high: float, atr_today: float, cfg: PositionSizingConfig) -> float:
    """陸-1、動態吊燈停利：每日防守價 = 過去10日最高價 - 2.5 * 當日 ATR。"""
    return rolling_10d_high - cfg.atr_multiplier * atr_today


@dataclass
class PositionSizeResult:
    shares: int
    position_value: float
    potential_loss: float
    rejected: bool
    reject_reason: str | None = None


def compute_position_size(
    capital_base: float,
    entry_price: float,
    stop_price: float,
    cfg: PositionSizingConfig,
) -> PositionSizeResult:
    """依 2% 風險預算算股數，無條件捨去至張，並套用零股防禦與市值上限防禦。"""
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0 or entry_price <= 0:
        return PositionSizeResult(0, 0.0, 0.0, True, "停損價不低於進場價，無法計算風險")

    risk_budget = capital_base * cfg.risk_pct_per_trade
    estimated_shares = risk_budget / risk_per_share
    shares = math.floor(estimated_shares / cfg.lot_size) * cfg.lot_size

    if shares < cfg.lot_size:
        return PositionSizeResult(0, 0.0, 0.0, True, "股數不足 1 張，放棄執行（禁止零股）")

    position_value = shares * entry_price
    max_value = capital_base * cfg.max_position_value_pct_of_capital
    if position_value > max_value:
        capped_shares = math.floor((max_value / entry_price) / cfg.lot_size) * cfg.lot_size
        if capped_shares < cfg.lot_size:
            return PositionSizeResult(0, 0.0, 0.0, True, "市值上限防禦後不足 1 張，放棄執行")
        shares = capped_shares
        position_value = shares * entry_price

    potential_loss = shares * risk_per_share
    return PositionSizeResult(shares, position_value, potential_loss, False, None)
