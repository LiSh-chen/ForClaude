"""伍、交易摩擦成本：證交稅、手續費、出場跳動點滑價。

台股股票報價跳動單位（tick）依價格級距而不同，出場滑價以「檔」（tick）計算，
故必須先正確算出該價位的跳動單位，2 檔滑價才有意義。
"""

from __future__ import annotations

import numpy as np

from tw_quant.config import CostConfig

# (價格上限（不含）, 該級距的 tick size)，依台灣證交所公告的股票升降單位表
_TICK_TABLE = (
    (10, 0.01),
    (50, 0.05),
    (100, 0.1),
    (500, 0.5),
    (1000, 1.0),
    (np.inf, 5.0),
)


def tick_size(price: float) -> float:
    """回傳單一價格對應的跳動單位（元）。"""
    if price <= 0 or np.isnan(price):
        return 0.0
    for upper, size in _TICK_TABLE:
        if price < upper:
            return size
    return _TICK_TABLE[-1][1]


def apply_exit_slippage(exit_price: float, ticks: int) -> float:
    """出場為市價賣出，不利滑價 = 價格往下跳 N 檔。"""
    price = float(exit_price)
    for _ in range(ticks):
        price -= tick_size(price)
    return max(price, tick_size(price))


def round_to_tick(price: float) -> float:
    size = tick_size(price)
    if size == 0:
        return price
    return round(price / size) * size


def entry_cost(entry_price: float, shares: int, cfg: CostConfig) -> float:
    """進場已用 T+1 開盤價，不額外計滑價，僅收手續費。"""
    notional = entry_price * shares
    return notional * cfg.fee_rate


def exit_proceeds(exit_price: float, shares: int, cfg: CostConfig) -> tuple[float, float]:
    """出場預留 2 檔不利滑價後，扣除證交稅 + 手續費，回傳 (實際滑價後價格, 淨收入)。"""
    slipped_price = apply_exit_slippage(exit_price, cfg.exit_slippage_ticks)
    notional = slipped_price * shares
    net = notional * (1 - cfg.tax_rate - cfg.fee_rate)
    return slipped_price, net
