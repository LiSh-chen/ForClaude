"""同一天內完成的「開盤買、收盤賣」跳空回補策略。

不需要分鐘級/tick 級盤中資料——日線 OHLC 本身就同時記錄了「當天開盤價」與
「當天收盤價」，所以「開盤觀察到跳空、收盤前決定是否出場」這種同一天內
只用頭尾兩個價位的策略，用現有資料就能正確回測。真正需要盤中序列資料才能
做的是「盤中即時停損」「精確計算當天最高最低價出現的先後順序」這類，跟這裡
的策略無關。

目標訊號：analyze_calendar_effects_from_db.py 量到的隔夜跳空 -> 當天盤中
報酬，mean IC=-0.25、t=-54.82，是整個專案裡最強的統計規律，但同日版本沒辦法
在「T 日收盤算訊號、T+1 日開盤才進場」的逐日引擎裡交易。這裡改成專門的
「當沖」回測引擎：T 日開盤觀察到跳空，立刻在 T 日開盤買進，T 日收盤賣出，
訊號跟執行都在同一天內完成，不依賴任何未來資訊。

★ 重要限制（誠實揭露，不是藏起來的假設）：
  1. 台股現股當沖有資格限制（信用交易資格、注意股/處置股排除等），
     不是所有股票都能做。這裡的回測沒有那份資格清單，等於假設候選股都能
     當沖，實際能交易的標的可能比回測顯示的少。
  2. 進場價用「當天開盤價」，等於假設下單能在開盤時或開盤後極短時間內
     成交，沒有額外的開盤滑價（現有 tw_quant.costs 的出場滑價模型有算，
     但進場沿用系統一貫「進場不額外計滑價、僅收手續費」的慣例）。
  3. 證交稅沿用系統一貫的 0.3%（標準賣出稅率）。台灣當沖交易目前有機會
     適用較低的當沖證交稅率（歷史上曾經階段性調降到 0.15%），但這個優惠
     稅率有沒有生效、生效到什麼時候，屬於會變動的稅法規定，這裡沒有去
     確認現況、也不擅自假設它一定適用，所以保守用標準稅率——如果實際
     當沖稅率更低，真實報酬會比這裡回測出來的更好。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tw_quant import costs as cost_mod
from tw_quant.backtest import BacktestResult, TradeRecord
from tw_quant.config import StrategyConfig
from tw_quant.signals import build_pool_mask


@dataclass
class IntradayRoundtripConfig:
    gap_threshold: float = 0.02  # 跳空幅度至少要達到這個門檻（絕對值）才算候選
    top_n: int = 20  # 每天最多買幾檔（等權重）


def run_intraday_roundtrip_backtest(
    prices: pd.DataFrame, cfg: StrategyConfig, rt_cfg: IntradayRoundtripConfig
) -> BacktestResult:
    """T 日開盤買進跳空幅度最大的下跌股（賭回補），T 日收盤賣出，當天結清。

    魚池沿用 signals.build_pool_mask（流動性 + 站上 60 日均線的多頭排列篩選，
    T-1 日資料，反未來函數）。跳空幅度用 T 日開盤價 vs T-1 日收盤價計算，
    在 T 日開盤當下就已知，不是未來資訊。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    pool = build_pool_mask(master, cfg.pool)

    prior_close = master.groupby("stock_id", sort=False)["close"].shift(1)
    gap_pct = ((master["open"] - prior_close) / prior_close).replace([np.inf, -np.inf], np.nan)

    master["pool"] = pool.values
    master["gap_pct"] = gap_pct.values

    cash = cfg.initial_capital
    trades: list[TradeRecord] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    per_stock_budget = cfg.initial_capital / rt_cfg.top_n

    for date, day_df in master.groupby("date", sort=True):
        eligible = day_df[
            day_df["pool"]
            & day_df["gap_pct"].notna()
            & (day_df["gap_pct"] <= -rt_cfg.gap_threshold)
        ]
        candidates = eligible.sort_values("gap_pct").head(rt_cfg.top_n)

        for row in candidates.itertuples():
            entry_price = row.open
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            shares = math.floor((per_stock_budget / entry_price) / cfg.sizing.lot_size) * cfg.sizing.lot_size
            if shares < cfg.sizing.lot_size:
                continue
            fee = cost_mod.entry_cost(entry_price, shares, cfg.costs)
            cost_basis = shares * entry_price + fee
            if cost_basis > cash:
                continue

            exit_price = row.close
            _, net_proceeds = cost_mod.exit_proceeds(exit_price, shares, cfg.costs)
            pnl = net_proceeds - cost_basis
            cash += pnl

            trades.append(
                TradeRecord(
                    stock_id=row.stock_id,
                    industry=row.industry,
                    strategy="GAP_FADE_INTRADAY",
                    shares=shares,
                    entry_date=date,
                    entry_price=entry_price,
                    exit_date=date,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=pnl / cost_basis if cost_basis else 0.0,
                )
            )

        equity_rows.append((date, cash))

    equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"]).set_index("date")
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return BacktestResult(equity_curve=equity_df, trades=trades_df, open_positions={}, rejected_log=[], mdd_breach_count=0)
