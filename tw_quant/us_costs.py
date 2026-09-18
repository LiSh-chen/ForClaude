"""美股交易摩擦成本：跟 tw_quant/costs.py 介面完全一致（同樣的 5 個函式
簽名：tick_size / apply_exit_slippage / round_to_tick / entry_cost /
exit_proceeds），這樣 backtest.py / factor_backtest.py / pairs_trading.py
只要把 `cost_module` 參數換成這個模組，就能直接重用同一套回測引擎，不用
另外複製一份事件迴圈邏輯。

跟台股版的關鍵差異：
  - 沒有證交稅：台股對賣出課徵 0.3% 證交稅，美股沒有這種交易稅。SEC
    Section 31 規費（僅賣出方向）確實存在，但費率極小（約
    0.0000278，即每百萬美元賣出金額課約 27.8 美元），這裡用
    US_TAX_RATE 概略帶入，量級遠小於台股的 0.003。
  - 手續費預設 0：美股主要券商（Schwab/Fidelity/Robinhood 等）自 2019
    年前後已全面零手續費交易，這是真實市場結構，不是為了做出好看結果
    刻意調低。
  - Tick size 固定 $0.01：美股自 2001 年全面十進位化後，不像台股依
    價格級距（10/50/100/500/1000 元）有不同跳動單位，所有價位統一
    最小跳動 1 美分。

誠實揭露：這樣建出來的成本模型結構上就是遠低於台股（沒有交易稅、tick
又小），這是市場結構真實的差異。但也代表這裡沒有額外模擬真實下單的
買賣價差、市場衝擊成本——對 S&P 500 這種全球流動性最好的股票，2 檔
$0.01 滑價大致合理，但如果之後要測流動性較差的美股，這個假設可能低估
真實成交成本。
"""

from __future__ import annotations

import numpy as np

from tw_quant.config import CostConfig

US_TAX_RATE = 0.0000278  # SEC Section 31 規費近似值（僅賣出方向）
US_FEE_RATE = 0.0  # 美股主要券商零手續費


def tick_size(price: float) -> float:
    """美股 2001 年十進位化後，所有價位統一最小跳動 $0.01。"""
    if price <= 0 or np.isnan(price):
        return 0.0
    return 0.01


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
    """進場已用 T+1 開盤價，不額外計滑價，僅收手續費（美股預設為 0）。"""
    notional = entry_price * shares
    return notional * cfg.fee_rate


def exit_proceeds(exit_price: float, shares: int, cfg: CostConfig) -> tuple[float, float]:
    """出場預留 N 檔不利滑價後，扣除 SEC 規費 + 手續費，回傳 (實際滑價後價格, 淨收入)。"""
    slipped_price = apply_exit_slippage(exit_price, cfg.exit_slippage_ticks)
    notional = slipped_price * shares
    net = notional * (1 - cfg.tax_rate - cfg.fee_rate)
    return slipped_price, net
