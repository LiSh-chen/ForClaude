"""檢查「三腿策略 + 支撐壓力順勢突破」在策略實驗室(strategy_lab)裡的合併
損益，有沒有正確考慮兩者部位在時間上重疊的情況。

**先講結論**：P&L相加本身沒有錯——TXF/MTX是線性商品（期貨），兩個各自
獨立的部位（不管時間上重不重疊）損益本來就是線性可加，不會因為同時
持有而互相干擾。真正沒有被考慮到、需要另外檢查的是「資金/保證金」：
strategy_lab把每個策略都當成獨立的1口帳戶結算損益，沒有檢查「如果真
的要同時開這些倉位，你同一時間需要準備多少保證金」。這支腳本量化這
兩件事：重疊頻率、跟合併後實際需要的保證金（vs天真加總每個策略各自
的保證金需求）。

用法：
    python scripts/three_legs_sr_breakout_overlap_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.opening_rally_strategy import OpeningRallyConfig, backtest as bt_og  # noqa: E402
from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest as bt_lunch  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"

# 跟 scripts/export_strategy_lab.py 的 ACTIVE 列表完全一致的鎖定參數
SR_CFG = SupportResistanceFadeConfig(
    direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
    min_range_pct=2.0, trend_slope_threshold_pct=5.0,
)

# MTX 保證金（見這次會話稍早查證：原始保證金175,250 / 結算保證金134,500 /
# 當沖保證金88,000，這裡用原始保證金做最保守的估計）
MTX_MARGIN = 175_250


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    og_trades = bt_og(df, OpeningRallyConfig())
    lunch_trades = bt_lunch(df, LunchReversalConfig())
    sr_trades = bt_sr(df, SR_CFG)

    # 三腿：每一筆都是「進場日=出場日」的當沖，用trading_date標記那一天
    three_legs_dates = pd.to_datetime(pd.concat([
        og_trades["trading_date"], lunch_trades["trading_date"],
    ])).dt.normalize()
    three_legs_trade_days = set(three_legs_dates.unique())
    print(f"三腿策略總交易日數（有任何一腿觸發的獨立日期數）: {len(three_legs_trade_days)}")

    # 三腿逐筆交易表（含哪一腿），用來檢查每一筆是否落在sr_breakout持倉期間內
    three_legs_all = pd.concat([
        og_trades.assign(leg="opening_rally")[["trading_date", "leg"]],
        lunch_trades[["trading_date", "leg"]],
    ]).reset_index(drop=True)
    three_legs_all["trading_date"] = pd.to_datetime(three_legs_all["trading_date"]).dt.normalize()

    # sr_breakout：建立「哪些交易日有開放部位」的日期集合（entry_date~exit_date 含頭尾）
    all_days = pd.to_datetime(og_trades["trading_date"]).dt.normalize().sort_values().unique()
    all_days = pd.DatetimeIndex(all_days)

    sr_open_days = set()
    for _, r in sr_trades.iterrows():
        span = all_days[(all_days >= pd.Timestamp(r["entry_date"])) & (all_days <= pd.Timestamp(r["exit_date"]))]
        sr_open_days.update(span)
    print(f"sr_breakout總交易日數(2001-2023全部有效交易日): {len(all_days)}")
    print(f"sr_breakout有開放部位的交易日數: {len(sr_open_days)} "
          f"({len(sr_open_days) / len(all_days):.1%} 的交易日)")

    # 三腿的每一筆交易，檢查當天sr_breakout是否也有開放部位
    three_legs_all["sr_open_same_day"] = three_legs_all["trading_date"].isin(sr_open_days)
    overlap_rate = three_legs_all["sr_open_same_day"].mean()
    print(f"\n三腿策略的交易裡，同一天sr_breakout也有開放部位的比例: {overlap_rate:.1%} "
          f"({three_legs_all['sr_open_same_day'].sum()} / {len(three_legs_all)} 筆)")

    print("\n分腿統計：")
    for leg in ["opening_rally", "short_lunch_dip", "long_afternoon_rebound"]:
        sub = three_legs_all[three_legs_all["leg"] == leg]
        print(f"  {leg}: {sub['sr_open_same_day'].mean():.1%} 的交易日sr_breakout同時有倉位 "
              f"({sub['sr_open_same_day'].sum()}/{len(sub)})")

    # 最大同時併發口數：三腿本身同一時刻最多1口（og/lu/re依時間序列互斥，
    # lu翻多是同一瞬間空單平倉+多單進場，不會真的同時掛兩個方向），
    # 加上sr_breakout自己也是同時最多1口部位 -> 併發上限 = 1(三腿) + 1(sr_breakout) = 2口
    print("\n併發口數：三腿本身任何時刻最多同時1口（og/lu/re彼此依時間序列互斥），"
          "sr_breakout自己任何時刻最多同時1口（一次只允許一筆部位）——"
          "兩者疊加後，實際併發上限＝2口（三腿其中一腿觸發的當下，剛好sr_breakout也持倉中）。")

    # 資金意涵：strategy_lab目前的權益曲線是把每個策略當獨立1口帳戶直接加總損益，
    # 沒有檢查「這一天你真的需要準備多少保證金才擋得住併發部位」。用最大併發口數
    # 重新估算，對比「天真地假設兩策略永遠不會同時要用到保證金」的下限估計。
    naive_min_capital = MTX_MARGIN  # 天真假設：反正不會同時用到，只準備1口的保證金
    actual_required_capital = MTX_MARGIN * 2  # 誠實估計：三腿+sr_breakout同時各1口，要備2口保證金
    print(f"\n保證金意涵（只看原始保證金，不含波動緩衝）：")
    print(f"  如果誤以為兩策略不會同時用到保證金，只準備: {naive_min_capital:,} TWD/組合")
    print(f"  實際上{overlap_rate:.1%}的三腿交易發生時sr_breakout已經持倉中，")
    print(f"  要撐得住兩者同時開倉，需要準備: {actual_required_capital:,} TWD/組合"
          f"（是天真估計的2倍，不含任何回撤緩衝）")


if __name__ == "__main__":
    main()
