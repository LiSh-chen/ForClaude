"""每日事件迴圈回測引擎：把 regime / signals / risk / portfolio_risk / mdd / costs
串成一個完整的模擬流程。

關鍵時序設計（解決規格書字面上的排序矛盾）：
規格書要求「發送委託前」套用全域鎖（依當日潛在虧損、T 日成交金額排序），
但潛在虧損 = shares * (進場價 - 停損價)，而進場價依規格是 T+1 開盤價——
也就是說，全域鎖要用的「進場價」在鎖定當下（T 日收盤後）根本還沒發生。
本引擎的處理方式：
  1. T 日收盤後，用「T 日收盤價」當作進場價的估計值，估算停損價、股數、
     潛在虧損、市值，供全域鎖排序與額度檢查用（這一步決定「誰被排進隔天」）。
  2. T+1 日開盤，對通過全域鎖的候選，用「真正的開盤價」重新計算最終停損價
     與最終股數（此時風險預算已經確定不需要再排隊，只需要把估計值換成實際值）。
這會讓極端跳空時的實際曝險與 6%/10% 目標值有些微落差，這是用估計值排隊、
真實值成交的必然結果，也是相對貼近真實下單流程的作法。

「總資金 / 總本金」在全程風控公式中一律採用 cfg.initial_capital（原始本金）
乘上 MDD 管理器當下的 capital_scale，而不是逐日盈虧後的浮動淨值——避免部位
規模隨盈虧複利式放大或縮小，這是多數機構型風控偏好的保守做法，也比較貼近
規格書「總資金 6%」「總本金 20%」這種語感（固定基數而非浮動淨值）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from tw_quant import costs as cost_mod
from tw_quant import indicators as ind
from tw_quant import portfolio_risk
from tw_quant import risk
from tw_quant.config import StrategyConfig
from tw_quant.mdd import MDDManager
from tw_quant.portfolio_risk import TradeCandidate
from tw_quant.regime import compute_regime_light
from tw_quant.signals import generate_strategy_a_signals, generate_strategy_b_signals
from tw_quant.us_universe import membership_eligibility_mask


@dataclass
class Position:
    stock_id: str
    industry: str
    strategy: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    stop_price: float
    cost_basis: float  # 進場成交金額 + 手續費，計算已實現損益用


@dataclass
class ApprovedEntry:
    stock_id: str
    industry: str
    strategy: str
    prior_10d_high: float
    atr_at_signal: float


@dataclass
class TradeRecord:
    stock_id: str
    industry: str
    strategy: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame  # columns: date, equity
    trades: pd.DataFrame
    open_positions: dict[str, Position]
    rejected_log: list[dict] = field(default_factory=list)
    mdd_breach_count: int = 0


EntrySignalFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, StrategyConfig], tuple]


def _default_entry_signals(
    master: pd.DataFrame, prices: pd.DataFrame, margin_short: pd.DataFrame, cfg: StrategyConfig
) -> tuple:
    """規格書預設的雙軌訊號（策略 A 動能突破 + 策略 B 軋空異常突破）。"""
    sig_a = generate_strategy_a_signals(master, cfg.pool, cfg.squeeze, cfg.ignition)
    sig_b = generate_strategy_b_signals(master, margin_short, cfg.pool, cfg.ignition, cfg.strategy_b)
    return sig_a["entry_signal"].values, sig_b["entry_signal"].values


def _prepare_master_frame(
    prices: pd.DataFrame,
    margin_short: pd.DataFrame,
    cfg: StrategyConfig,
    entry_signal_fn: EntrySignalFn | None = None,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    master["rolling_high_10"] = ind.rolling_max(master, "high", cfg.sizing.chandelier_lookback)
    master["atr"] = ind.atr(master, cfg.sizing.atr_window)

    entry_signal_fn = entry_signal_fn or _default_entry_signals
    entry_a, entry_b = entry_signal_fn(master, prices, margin_short, cfg)
    master["entry_signal_a"] = entry_a
    master["entry_signal_b"] = entry_b

    # 2026-09-21 架構修正（跟 tw_quant/factor_backtest.py 的 membership 參數
    # 同一套道理）：entry_signal_fn 永遠吃完整、未過濾的 prices 算訊號（均線/
    # ATR/量能這些指標才不會因為股票中途被剔除指數又重新納入而失真），
    # membership 資格判定改在訊號算完之後才疊加——用 T-1 已知的資格遮罩去
    # AND 進兩條進場訊號，不是在算訊號之前就先砍價格序列。membership=None
    # （預設）不做任何資格過濾，保留舊行為（例如台股本來就不需要這個機制）。
    if membership is not None:
        raw_member = membership_eligibility_mask(master, membership)
        member_t_minus_1 = ind.shift_by_group(raw_member.astype(float), master, periods=1).fillna(0).astype(bool)
        master["entry_signal_a"] = master["entry_signal_a"].astype(bool) & member_t_minus_1
        master["entry_signal_b"] = master["entry_signal_b"].astype(bool) & member_t_minus_1

    regime_light = compute_regime_light(master, cfg.regime, cfg.pool.min_history_days)
    master = master.merge(
        regime_light[["is_green"]], left_on="date", right_index=True, how="left"
    )
    master["is_green"] = master["is_green"].fillna(False)
    return master


def run_backtest(
    prices: pd.DataFrame,
    margin_short: pd.DataFrame,
    cfg: StrategyConfig,
    historical_mdd: float | None = None,
    entry_signal_fn: EntrySignalFn | None = None,
    pre_holiday_exit_dates: set[pd.Timestamp] | None = None,
    cost_module=cost_mod,
    membership: pd.DataFrame | None = None,
) -> BacktestResult:
    """執行完整回測。historical_mdd=None 代表不啟用 MDD 熔斷（用於第一次跑出
    基準 MDD），拿到基準值後再傳入做第二次帶熔斷機制的回測。

    entry_signal_fn：選填，用來替換預設的策略 A/B 進場訊號邏輯（出場/風控/
    成本引擎維持不變），簽名為
    `fn(master, prices, margin_short, cfg) -> (entry_signal_a, entry_signal_b)`，
    兩者皆為對齊 master 列順序的布林陣列。用來在同一套出場/風控引擎下比較
    不同進場邏輯（例如拉回買進、相對強度動能、超跌反彈），見
    scripts/explore_alt_strategies_from_db.py。

    membership：選填（`(stock_id, start_date, end_date)` 區間表，見
    tw_quant/us_universe.py），用來在美股上套用存活者偏差修正。`prices`
    永遠要傳完整、未過濾的價格序列（不要用 filter_prices_by_index_membership
    事先砍過）——entry_signal_fn 依賴的均線/ATR/量能這些指標才算得對；
    membership 資格判定改在兩條進場訊號算完之後，才用 T-1 已知的資格遮罩
    AND 進去，跟 tw_quant/factor_backtest.py 的 membership 參數是同一個
    2026-09-21 架構修正。`membership=None`（預設）不套用任何資格過濾，
    台股呼叫端不需要傳這個參數。

    pre_holiday_exit_dates：選填，長假風控疊加層。這裡放的日期是「長假前
    最後一個交易日的前一個交易日」（也就是訊號日 T，執行日 T+1 剛好等於
    長假前最後一天的開盤）。程式在這個日期收盤時會：(1) 把當下所有部位強制
    排入明日出場（清空曝險，避免持倉跨過長假缺口），(2) 當天不產生任何新
    候選（避免當天核准的新單隔天一開盤就進場、還是曝險在缺口裡）。日期集合
    要用呼叫端自己從交易日曆算好再傳進來，本函式不做假期判斷。

    cost_module：選填，用來替換預設的台股成本模型（tw_quant.costs），
    介面須提供跟該模組一致的 `entry_cost(price, shares, cfg.costs)` /
    `exit_proceeds(price, shares, cfg.costs)`。用來在同一套事件迴圈引擎下
    重用在別的市場（例如美股，見 tw_quant/us_costs.py），不用另外複製一份
    出場/風控邏輯。
    """
    master = _prepare_master_frame(prices, margin_short, cfg, entry_signal_fn, membership=membership)
    pre_holiday_exit_dates = pre_holiday_exit_dates or set()

    cash = cfg.initial_capital
    positions: dict[str, Position] = {}
    pending_exits: set[str] = set()
    pending_entries: list[ApprovedEntry] = []
    trades: list[TradeRecord] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    rejected_log: list[dict] = []

    mdd_mgr = MDDManager(historical_mdd=historical_mdd if historical_mdd else float("inf"), cfg=cfg.mdd)
    last_equity = cfg.initial_capital

    for date, day_df in master.groupby("date", sort=True):
        row_by_stock = day_df.set_index("stock_id")

        # 1) 執行昨日收盤觸發、今日開盤平倉的出場
        for stock_id in list(pending_exits):
            if stock_id not in row_by_stock.index or stock_id not in positions:
                pending_exits.discard(stock_id)
                continue
            pos = positions.pop(stock_id)
            exit_open = row_by_stock.loc[stock_id, "open"]
            _, net_proceeds = cost_module.exit_proceeds(exit_open, pos.shares, cfg.costs)
            cash += net_proceeds
            pnl = net_proceeds - pos.cost_basis
            trades.append(
                TradeRecord(
                    stock_id=stock_id,
                    industry=pos.industry,
                    strategy=pos.strategy,
                    shares=pos.shares,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=date,
                    exit_price=exit_open,
                    pnl=pnl,
                    pnl_pct=pnl / pos.cost_basis if pos.cost_basis else 0.0,
                )
            )
            mdd_mgr.on_trade_closed(pnl, last_equity)
            pending_exits.discard(stock_id)

        # 2) 執行昨日核准、今日開盤進場的新部位
        for entry in pending_entries:
            stock_id = entry.stock_id
            if stock_id not in row_by_stock.index or stock_id in positions:
                continue
            entry_price = row_by_stock.loc[stock_id, "open"]
            stop_price = risk.compute_initial_stop(
                entry_price, entry.prior_10d_high, entry.atr_at_signal, cfg.sizing
            )
            capital_base = cfg.initial_capital * mdd_mgr.capital_scale
            sizing = risk.compute_position_size(capital_base, entry_price, stop_price, cfg.sizing)
            if sizing.rejected or sizing.shares <= 0:
                rejected_log.append({"date": date, "stock_id": stock_id, "stage": "execution", "reason": sizing.reject_reason})
                continue
            fee = cost_module.entry_cost(entry_price, sizing.shares, cfg.costs)
            total_cost = sizing.position_value + fee
            if total_cost > cash:
                rejected_log.append({"date": date, "stock_id": stock_id, "stage": "execution", "reason": "現金不足"})
                continue
            cash -= total_cost
            positions[stock_id] = Position(
                stock_id=stock_id,
                industry=entry.industry,
                strategy=entry.strategy,
                shares=sizing.shares,
                entry_price=entry_price,
                entry_date=date,
                stop_price=stop_price,
                cost_basis=total_cost,
            )
        pending_entries = []

        # 3) 以今日收盤更新吊燈停利（只上調不下調），跌破則排入明日出場
        for stock_id, pos in positions.items():
            if stock_id not in row_by_stock.index:
                continue
            r = row_by_stock.loc[stock_id]
            hh10, atr_today = r["rolling_high_10"], r["atr"]
            if pd.notna(hh10) and pd.notna(atr_today):
                new_stop = risk.compute_chandelier_stop(hh10, atr_today, cfg.sizing)
                pos.stop_price = max(pos.stop_price, new_stop)
            if r["close"] < pos.stop_price:
                pending_exits.add(stock_id)

        is_pre_holiday_trigger = date in pre_holiday_exit_dates
        if is_pre_holiday_trigger:
            pending_exits.update(positions.keys())

        # 4) 若大盤綠燈，依今日訊號產生候選並套用全域鎖，核准者排入明日開盤進場
        # （長假風控觸發日當天不產生新候選，避免隔天一開盤就進場又曝險在缺口裡）
        is_green = bool(day_df["is_green"].iloc[0])
        if is_green and not is_pre_holiday_trigger:
            candidates: list[TradeCandidate] = []
            approved_meta: dict[str, ApprovedEntry] = {}
            for r in day_df.itertuples():
                stock_id = r.stock_id
                if stock_id in positions:
                    continue
                if r.entry_signal_a:
                    strategy = "A"
                elif r.entry_signal_b:
                    strategy = "B"
                else:
                    continue
                if pd.isna(r.rolling_high_10) or pd.isna(r.atr):
                    continue
                est_entry_price = r.close
                est_stop = risk.compute_initial_stop(est_entry_price, r.rolling_high_10, r.atr, cfg.sizing)
                capital_base = cfg.initial_capital * mdd_mgr.capital_scale
                sizing = risk.compute_position_size(capital_base, est_entry_price, est_stop, cfg.sizing)
                if sizing.rejected or sizing.shares <= 0:
                    rejected_log.append({"date": date, "stock_id": stock_id, "stage": "sizing", "reason": sizing.reject_reason})
                    continue
                candidates.append(
                    TradeCandidate(
                        stock_id=stock_id,
                        industry=r.industry,
                        strategy=strategy,
                        entry_price=est_entry_price,
                        stop_price=est_stop,
                        shares=sizing.shares,
                        position_value=sizing.position_value,
                        potential_loss=sizing.potential_loss,
                        liquidity_rank_value=r.turnover_value,
                    )
                )
                approved_meta[stock_id] = ApprovedEntry(stock_id, r.industry, strategy, r.rolling_high_10, r.atr)

            if candidates:
                existing_industry_exposure: dict[str, float] = {}
                for pos in positions.values():
                    mark_price = (
                        row_by_stock.loc[pos.stock_id, "close"]
                        if pos.stock_id in row_by_stock.index
                        else pos.entry_price
                    )
                    existing_industry_exposure[pos.industry] = (
                        existing_industry_exposure.get(pos.industry, 0.0) + pos.shares * mark_price
                    )
                lock_result = portfolio_risk.apply_global_lock(
                    candidates,
                    cfg.initial_capital * mdd_mgr.capital_scale,
                    existing_industry_exposure,
                    cfg.global_risk,
                )
                for c in lock_result.accepted:
                    pending_entries.append(approved_meta[c.stock_id])
                for c, reason in lock_result.rejected:
                    rejected_log.append({"date": date, "stock_id": c.stock_id, "stage": "global_lock", "reason": reason})

        # 5) 依今日收盤結算權益，餵給 MDD 管理器
        mtm = cash
        for stock_id, pos in positions.items():
            price = row_by_stock.loc[stock_id, "close"] if stock_id in row_by_stock.index else pos.entry_price
            mtm += pos.shares * price
        equity_rows.append((date, mtm))
        mdd_mgr.on_equity_update(mtm)
        last_equity = mtm

    equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"]).set_index("date")
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return BacktestResult(
        equity_curve=equity_df,
        trades=trades_df,
        open_positions=positions,
        rejected_log=rejected_log,
        mdd_breach_count=mdd_mgr.breach_count,
    )


def summarize_performance(result: BacktestResult, initial_capital: float) -> dict:
    eq = result.equity_curve["equity"]
    if eq.empty:
        return {"total_return": 0.0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0, "n_trades": 0, "win_rate": 0.0}

    total_return = eq.iloc[-1] / initial_capital - 1
    n_days = len(eq)
    years = max(n_days / 252, 1e-9)
    cagr = (eq.iloc[-1] / initial_capital) ** (1 / years) - 1

    running_peak = eq.cummax()
    drawdown = (running_peak - eq) / running_peak
    max_dd = drawdown.max()

    daily_ret = eq.pct_change().dropna()
    sharpe = 0.0
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

    trades = result.trades
    n_trades = len(trades)
    win_rate = (trades["pnl"] > 0).mean() if n_trades else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "annual_trades": n_trades / years,
    }
