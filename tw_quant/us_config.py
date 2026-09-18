"""美股版策略設定 factory。重用 tw_quant.config.StrategyConfig 整組資料
結構（跟 us_data_provider.py 的作法一致：不重寫資料結構，只覆寫台股跟
美股真正不同的部分），差異只有兩處：

  - sizing.lot_size = 1：美股沒有台股「1 張 = 1000 股」的整股倍數限制，
    可以買賣任意股數。
  - costs：改用 tw_quant/us_costs.py 的美股費率（沒有證交稅、複委託
    每股固定手續費、$0.01 tick）。

其餘參數（濾網天數、風控門檻等）維持跟台股一樣的預設值，當作「同一套
規則直接套用在不同市場」的基準版本；個別測試腳本仍可以再自行覆寫
（例如放寬 regime 門檻、調整 industry exposure 上限，這些跟市場成本模型
無關，屬於策略邏輯本身要不要調整的問題，不在這裡處理）。
"""

from __future__ import annotations

from tw_quant.config import CostConfig, StrategyConfig
from tw_quant.us_costs import US_FEE_RATE, US_PER_SHARE_FEE, US_TAX_RATE


def build_us_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.sizing.lot_size = 1
    cfg.costs = CostConfig(
        tax_rate=US_TAX_RATE, fee_rate=US_FEE_RATE, per_share_fee=US_PER_SHARE_FEE, exit_slippage_ticks=2
    )
    return cfg
