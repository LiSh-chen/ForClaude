"""一次性診斷：查為什麼營收動能+價量突破組合策略在真實資料上 18 組全部 0 筆交易。
用完會刪除。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tw_quant import indicators as ind
from tw_quant.config import StrategyConfig
from tw_quant.signals import build_pool_mask
from tw_quant.storage import get_data_store
from scripts.test_pead_breakout_combo_from_db import attach_revenue_momentum_flag


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    revenue = store.load_month_revenue()

    master = attach_revenue_momentum_flag(prices, revenue)
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9

    pool = build_pool_mask(master, cfg.pool)
    revenue_ok = master["revenue_momentum_ok_shifted"]

    rolling_high = ind.rolling_max(master, "high", 20)
    rolling_high_prior = ind.shift_by_group(rolling_high, master, periods=1)
    price_breakout = (master["close"] > rolling_high_prior).fillna(False)

    avg_vol_5 = ind.sma(master, "volume", 5)
    avg_vol_5_prior = ind.shift_by_group(avg_vol_5, master, periods=1)
    volume_spike = (master["volume"] > avg_vol_5_prior * 1.5).fillna(False)

    facts = [
        f"total rows: {len(master)}",
        f"pool frac: {pool.mean():.4f} ({pool.sum()})",
        f"revenue_ok frac: {revenue_ok.mean():.4f} ({revenue_ok.sum()})",
        f"price_breakout(20d) frac: {price_breakout.mean():.4f} ({price_breakout.sum()})",
        f"volume_spike(1.5x) frac: {volume_spike.mean():.4f} ({volume_spike.sum()})",
        f"pool & revenue_ok: {(pool & revenue_ok).sum()}",
        f"pool & breakout: {(pool & price_breakout).sum()}",
        f"pool & volume_spike: {(pool & volume_spike).sum()}",
        f"pool & revenue_ok & breakout: {(pool & revenue_ok & price_breakout).sum()}",
        f"pool & revenue_ok & volume_spike: {(pool & revenue_ok & volume_spike).sum()}",
        f"pool & breakout & volume_spike: {(pool & price_breakout & volume_spike).sum()}",
        f"all four combined: {(pool & revenue_ok & price_breakout & volume_spike).sum()}",
    ]
    print("\n\n===== DEBUG FACTS =====")
    for line in facts:
        print(line)
    print("===== END DEBUG FACTS =====")


if __name__ == "__main__":
    main()
