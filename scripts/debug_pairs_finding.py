"""一次性診斷腳本：查為什麼 test_pairs_trading_from_db.py 在真實資料上
9 組參數全部 0 筆交易。只做 _find_pairs 本身的診斷，不跑完整回測迴圈，
跑起來快很多。用完會刪除。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from tw_quant.pairs_trading import PairsTradingConfig, _find_pairs
from tw_quant.storage import get_data_store


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()

    facts = []
    facts.append(f"stocks: {prices['stock_id'].nunique()}, dates: {prices['date'].nunique()}")

    zero_or_neg = prices[prices["close"] <= 0]
    facts.append(f"close<=0 rows: {len(zero_or_neg)}")
    facts.append(str(zero_or_neg[["date", "stock_id", "close"]].head(10)))

    n_nan_close = prices["close"].isna().sum()
    facts.append(f"NaN close rows: {n_nan_close}")

    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True)
    close_pivot = master.pivot(index="date", columns="stock_id", values="close").sort_index()
    industry_map = master.drop_duplicates("stock_id", keep="last").set_index("stock_id")["industry"].to_dict()

    facts.append("industry distribution:\n" + str(pd.Series(industry_map).value_counts()))

    dates = close_pivot.index
    formation_window = 252
    window_dates = dates[:formation_window]
    facts.append(f"window: {window_dates[0]} ~ {window_dates[-1]} ({len(window_dates)} days)")

    window = close_pivot.loc[window_dates]
    n_with_nan = window.isna().any().sum()
    n_full = (~window.isna().any()).sum()
    facts.append(f"stocks with any NaN in window: {n_with_nan}, stocks with full data: {n_full}")

    n_zero_in_window = (window == 0).any().sum()
    facts.append(f"stocks with any zero-price in window: {n_zero_in_window}")

    rt_cfg = PairsTradingConfig(formation_window=formation_window, coint_pvalue_threshold=0.05, top_n_pairs=10)
    pairs = _find_pairs(close_pivot, industry_map, window_dates, rt_cfg)
    facts.append(f"pairs found at threshold 0.05: {len(pairs)}")
    facts.append(str(pairs[:10]))

    # 印在最後，蓋掉前面 coint() 洗版的一堆 warning，確保這些關鍵事實一定看得到
    print("\n\n===== DEBUG FACTS (見這裡，上面的 warning 洗版可以忽略) =====")
    for line in facts:
        print(line)
    print("===== END DEBUG FACTS =====")


if __name__ == "__main__":
    main()
