"""結構上不同於逐日事件驅動引擎的「月/季調倉因子組合」回測。

前面測試過的 4 種邏輯（突破/拉回/相對強度確認/超跌反彈）都是同一種範式：
每日掃描個股訊號 -> 個股停損停利出場，本質上都是「用技術面觸發點找進出場
時機」。這裡改用完全不同的典範——傳統因子投資：定期（月/季）對整個股票池
依動能排名，等權重持有前 N 檔，到下次調倉才換股，不用逐日停損停利去挑時機。
這樣可以驗證「換一種完全不同的策略設計哲學」是否比較有 edge，而不是繼續在
同一種「找進場時機點」的思路裡微調參數。

為了讓結果可以直接跟前面幾輪比較，重用 tw_quant.backtest 的 Position /
TradeRecord / BacktestResult 資料結構，這樣 scripts/explore_strategy_from_db.py
系列腳本裡的 trade_stats() / summarize_performance() 可以直接套用，指標定義
完全一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from tw_quant import costs as cost_mod
from tw_quant import indicators as ind
from tw_quant.backtest import BacktestResult, Position, TradeRecord
from tw_quant.config import StrategyConfig
from tw_quant.signals import build_pool_mask


@dataclass
class FactorConfig:
    momentum_window: int = 60  # 排名用的落後報酬率天數
    rebalance_freq_days: int = 21  # 調倉頻率（交易日），21 約等於月調倉，63 約等於季調倉
    top_n: int = 20  # 每次調倉持有的檔數（等權重）
    ascending: bool = False  # False=買排名前段（動量／追強勢）；True=買排名後段（反轉／買弱勢）


def run_factor_backtest(
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    factor_cfg: FactorConfig,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    signal_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
    cost_module=cost_mod,
) -> BacktestResult:
    """月/季調倉動能因子組合：固定頻率對魚池內個股依排名，等權重持有
    前 N 檔，只有在下次調倉日才換股，期間不做個股停損停利。

    魚池沿用 signals.build_pool_mask（T-1 流動性 + 站上 60 日均線的多頭排列
    篩選）。排名依據預設是 T-1 日為止的 momentum_window 日報酬率（shift(1)
    避免用到 T 日未來資訊），T 日開盤價成交。factor_cfg.ascending=True 時排名
    反過來（買排名後段、也就是近期表現最弱的股票），用來測試短期反轉規律
    而非動量。

    signal_fn：選填，用來替換預設的「落後報酬率」排名依據，簽名為
    `fn(master) -> pd.Series`（對齊 master 列順序），回傳值必須是呼叫端自己
    先做好 shift(1) 的排名分數（跟預設的 momentum 欄位一樣，代表「T 日收盤前
    已知」的資訊），本函式不會再另外位移一次。用來測試落後報酬率以外的
    排名依據（例如跳空幅度、當日盤中報酬），見
    scripts/test_lagged_gap_signal_from_db.py。

    start_date/end_date：只限制「哪些日期允許實際調倉/交易」，但魚池篩選與
    排名指標的計算永遠用傳進來的完整 prices 算，不會因為切片而重新計算
    暖身期（這正是先前「簡易切分驗證」踩到的坑：把 prices 切片後 T-1 日均量
    252 日門檻等指標會被迫從切片起點重新累積，導致前一年左右完全沒有股票
    符合資格。要跑分段檢查，永遠傳完整 prices + start_date/end_date，不要自己
    先把 prices 切片）。

    cost_module：選填，同 tw_quant.backtest.run_backtest 的用法，用來替換
    台股成本模型（例如美股，見 tw_quant/us_costs.py），重用同一套定期
    調倉引擎。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    pool = build_pool_mask(master, cfg.pool)

    if signal_fn is not None:
        ranking_signal = signal_fn(master)
    else:
        ret = master.groupby("stock_id", sort=False)["close"].pct_change(factor_cfg.momentum_window)
        ranking_signal = ind.shift_by_group(ret, master, periods=1)

    master["pool"] = pool.values
    master["momentum"] = ranking_signal.values if hasattr(ranking_signal, "values") else ranking_signal

    if start_date is not None:
        master = master[master["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        master = master[master["date"] <= pd.Timestamp(end_date)]

    dates = sorted(master["date"].unique())
    rebalance_dates = set(dates[:: factor_cfg.rebalance_freq_days])

    cash = cfg.initial_capital
    positions: dict[str, Position] = {}
    trades: list[TradeRecord] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []

    for date, day_df in master.groupby("date", sort=True):
        row_by_stock = day_df.set_index("stock_id")

        if date in rebalance_dates:
            eligible = day_df[day_df["pool"] & day_df["momentum"].notna()]
            target = eligible.sort_values("momentum", ascending=factor_cfg.ascending).head(factor_cfg.top_n)["stock_id"].tolist()
            target_set = set(target)

            for stock_id in list(positions.keys()):
                if stock_id in target_set or stock_id not in row_by_stock.index:
                    continue
                pos = positions.pop(stock_id)
                exit_price = row_by_stock.loc[stock_id, "open"]
                _, net_proceeds = cost_module.exit_proceeds(exit_price, pos.shares, cfg.costs)
                cash += net_proceeds
                pnl = net_proceeds - pos.cost_basis
                trades.append(
                    TradeRecord(
                        stock_id=stock_id,
                        industry=pos.industry,
                        strategy="FACTOR",
                        shares=pos.shares,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl / pos.cost_basis if pos.cost_basis else 0.0,
                    )
                )

            n_target = len(target) if target else 1
            per_stock_budget = cfg.initial_capital / n_target
            for stock_id in target:
                if stock_id in positions or stock_id not in row_by_stock.index:
                    continue
                entry_price = row_by_stock.loc[stock_id, "open"]
                if pd.isna(entry_price) or entry_price <= 0:
                    continue
                shares = math.floor((per_stock_budget / entry_price) / cfg.sizing.lot_size) * cfg.sizing.lot_size
                if shares < cfg.sizing.lot_size:
                    continue
                fee = cost_module.entry_cost(entry_price, shares, cfg.costs)
                total_cost = shares * entry_price + fee
                if total_cost > cash:
                    continue
                cash -= total_cost
                positions[stock_id] = Position(
                    stock_id=stock_id,
                    industry=row_by_stock.loc[stock_id, "industry"],
                    strategy="FACTOR",
                    shares=shares,
                    entry_price=entry_price,
                    entry_date=date,
                    stop_price=0.0,
                    cost_basis=total_cost,
                )

        mtm = cash
        for stock_id, pos in positions.items():
            price = row_by_stock.loc[stock_id, "close"] if stock_id in row_by_stock.index else pos.entry_price
            mtm += pos.shares * price
        equity_rows.append((date, mtm))

    equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"]).set_index("date")
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return BacktestResult(
        equity_curve=equity_df, trades=trades_df, open_positions=positions, rejected_log=[], mdd_breach_count=0
    )
