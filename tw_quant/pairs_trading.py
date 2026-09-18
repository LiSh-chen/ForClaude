"""統計套利/配對交易（Pairs Trading）策略引擎。

核心邏輯：定期（每 reformation_freq_days 個交易日）用過去 formation_window
天的資料，對「同產業」的股票兩兩做 Engle-Granger 共整合檢定
（statsmodels.tsa.stattools.coint），挑出 p-value 最低的前 top_n_pairs 組
當作接下來這段期間可交易的配對；每組配對用共整合迴歸估出的避險比例
（hedge ratio，形成期內凍結不變）算價差、再用價差的滾動 z-score 當進出場
訊號——z-score 偏離夠多（一腿被高估、一腿被低估）就做多低估的一腿、做空
高估的一腿（金額中性），等價差回歸到接近 0 或超過停損 z-score 或持有太久
就平倉。

反未來函數紀律：
  - 換股（reformation）當天，先用「前一天」收盤為止的資料選配對、估避險
    比例，當天收盤前不產生新的進場訊號（跟 backtest.py 的
    pre_holiday_exit_dates 疊加層同一種「觸發日當天只出場、不進場」設計），
    新配對從隔天開始才能進場。
  - z-score 用滾動窗格（zscore_window）在「T 日收盤」計算，T+1 日開盤才
    進場/出場，跟系統其他每日訊號一致的 T-1 已知資訊 -> T 進場慣例。

★ 誠實揭露的限制：
  1. 沒有台股融資融券/借券資格清單，假設候選股都能做空。
  2. 做空的成本只算了系統既有的證交稅（賣出 0.3%）+ 手續費，**沒有算融券
     手續費/借券費**（台灣借券費率依股票熱門程度浮動，這裡沒有那份資料）。
     如果實際借券成本不低，真實報酬會比這裡回測出來的更差。
  3. 沒有算融資融券的保證金/擔保品利息，做空賣出所得直接視為可用現金
     （學術文獻常見的簡化假設，不是真實券商的保證金機制）。
  4. 共整合檢定只在「同產業」的股票之間做，減少組合數、也讓配對更有經濟
     意義，但這表示不同產業間可能存在的價差關係完全沒被考慮到。
  5. 換股週期內配對的避險比例是凍結的，不會每天重估，價差關係如果在
     窗格內結構性改變（不是均值回歸而是永久脫鉤），策略會持續虧損直到
     停損 z-score 觸發或换股日才被迫停止。
  6. 兩腿部位大小用等金額切分（各佔配對預算一半），不是照共整合迴歸
     估出的避險比例（beta）去加權——嚴格的「市場中性」應該是兩腿的
     曝險金額依 beta 比例配置，這裡用等金額是常見的簡化做法，會讓實際
     組合曝險跟真正的 beta-neutral 有落差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from tw_quant import costs as cost_mod
from tw_quant.backtest import BacktestResult, TradeRecord
from tw_quant.config import StrategyConfig


@dataclass
class PairsTradingConfig:
    formation_window: int = 252  # 用幾天歷史資料做共整合檢定/估避險比例
    reformation_freq_days: int = 63  # 每幾個交易日重新選一次配對（約一季）
    top_n_pairs: int = 10  # 每期最多交易幾組配對
    coint_pvalue_threshold: float = 0.05  # 共整合檢定 p-value 門檻
    zscore_window: int = 20  # 價差 z-score 的滾動窗格
    entry_z: float = 2.0  # |z| 超過這個門檻才進場
    exit_z: float = 0.5  # |z| 回到這個門檻以下就平倉（回歸到位）
    stop_z: float = 4.0  # |z| 超過這個門檻強制停損（判定脫鉤）
    max_holding_days: int = 20  # 最長持有天數，避免無限期套牢


@dataclass
class _OpenPair:
    stock_a: str
    stock_b: str
    beta: float
    direction: int  # +1: 多 A 空 B；-1: 空 A 多 B
    shares_a: int
    shares_b: int
    entry_price_a: float
    entry_price_b: float
    cash_flow_a: float  # 開倉當下對 cash 的影響（多腿為負、空腿為正）
    cash_flow_b: float
    entry_date: pd.Timestamp


def _find_pairs(
    close_pivot: pd.DataFrame,
    industry_map: dict[str, str],
    window_dates: pd.Index,
    rt_cfg: PairsTradingConfig,
) -> list[tuple[str, str, float]]:
    """在 window_dates 這段歷史窗格裡，對同產業配對做共整合檢定，回傳
    [(stock_a, stock_b, beta), ...]，依 p-value 由小到大排序、只取前 top_n_pairs 組。
    beta 是用 log(price_a) ~ beta*log(price_b) 迴歸估出的避險比例（OLS 斜率）。
    """
    window = close_pivot.loc[window_dates]
    by_industry: dict[str, list[str]] = {}
    for stock_id in window.columns:
        col = window[stock_id]
        # 除了 NaN，還要濾掉價格 <= 0 的股票——真實資料裡有些股票在某些日子
        # 價格是 0（缺資料/停牌被記成 0，不是 NaN），log(0) = -inf 會讓
        # np.polyfit 估出來的 beta 變成 NaN，而且 coint() 用被 -inf 汙染的
        # 序列算出來的 p-value 還可能因為數值不穩定而異常地小，讓這種壞資料
        # 配對反而排到最前面、擠掉真正合理的配對（實測就是這樣：真實資料上
        # 選出來的前 10 組配對 beta 全部是 NaN，導致整條價差序列全部 NaN、
        # z-score 永遠算不出來、一筆交易都不會發生）。
        if col.isna().any() or (col <= 0).any():
            continue
        by_industry.setdefault(industry_map.get(stock_id, ""), []).append(stock_id)

    candidates = []
    for stocks in by_industry.values():
        if len(stocks) < 2:
            continue
        for a, b in combinations(sorted(stocks), 2):
            log_a = np.log(window[a].to_numpy())
            log_b = np.log(window[b].to_numpy())
            try:
                _, pvalue, _ = coint(log_a, log_b)
            except Exception:
                continue
            if pvalue < rt_cfg.coint_pvalue_threshold:
                beta = float(np.polyfit(log_b, log_a, 1)[0])
                candidates.append((pvalue, a, b, beta))

    candidates.sort(key=lambda row: row[0])
    return [(a, b, beta) for _, a, b, beta in candidates[: rt_cfg.top_n_pairs]]


def _zscore_series(spread: pd.Series, window: int) -> pd.Series:
    roll = spread.rolling(window)
    return (spread - roll.mean()) / roll.std()


def run_pairs_trading_backtest(
    prices: pd.DataFrame, cfg: StrategyConfig, rt_cfg: PairsTradingConfig
) -> BacktestResult:
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True)
    close_pivot = master.pivot(index="date", columns="stock_id", values="close").sort_index()
    open_pivot = master.pivot(index="date", columns="stock_id", values="open").sort_index()
    industry_map = master.drop_duplicates("stock_id", keep="last").set_index("stock_id")["industry"].to_dict()

    dates = close_pivot.index
    trades: list[TradeRecord] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    cash = cfg.initial_capital
    open_pairs: dict[tuple[str, str], _OpenPair] = {}
    active_pairs: list[tuple[str, str, float]] = []
    spread_cache: dict[tuple[str, str], pd.Series] = {}

    first_idx = rt_cfg.formation_window
    reformation_indices = set(range(first_idx, len(dates), rt_cfg.reformation_freq_days))

    def _close_pair(key: tuple[str, str], date: pd.Timestamp) -> None:
        nonlocal cash
        pos = open_pairs.pop(key)
        price_a = open_pivot.at[date, pos.stock_a] if date in open_pivot.index and pos.stock_a in open_pivot.columns else np.nan
        price_b = open_pivot.at[date, pos.stock_b] if date in open_pivot.index and pos.stock_b in open_pivot.columns else np.nan
        if pd.isna(price_a) or pd.isna(price_b) or price_a <= 0 or price_b <= 0:
            return

        if pos.direction == 1:
            # 多 A：賣出平倉；空 B：買回平倉
            _, proceeds_a = cost_mod.exit_proceeds(price_a, pos.shares_a, cfg.costs)
            fee_b = cost_mod.entry_cost(price_b, pos.shares_b, cfg.costs)
            cover_cost_b = price_b * pos.shares_b + fee_b
            cash += proceeds_a - cover_cost_b
            pnl_a = proceeds_a - (-pos.cash_flow_a)
            pnl_b = pos.cash_flow_b - cover_cost_b
        else:
            # 空 A：買回平倉；多 B：賣出平倉
            fee_a = cost_mod.entry_cost(price_a, pos.shares_a, cfg.costs)
            cover_cost_a = price_a * pos.shares_a + fee_a
            _, proceeds_b = cost_mod.exit_proceeds(price_b, pos.shares_b, cfg.costs)
            cash += proceeds_b - cover_cost_a
            pnl_a = pos.cash_flow_a - cover_cost_a
            pnl_b = proceeds_b - (-pos.cash_flow_b)

        industry = industry_map.get(pos.stock_a, "")
        strategy_a = "PAIRS_LONG" if pos.direction == 1 else "PAIRS_SHORT"
        strategy_b = "PAIRS_SHORT" if pos.direction == 1 else "PAIRS_LONG"
        notional_a = pos.shares_a * pos.entry_price_a
        notional_b = pos.shares_b * pos.entry_price_b
        trades.append(
            TradeRecord(
                stock_id=pos.stock_a, industry=industry, strategy=strategy_a, shares=pos.shares_a,
                entry_date=pos.entry_date, entry_price=pos.entry_price_a, exit_date=date, exit_price=price_a,
                pnl=pnl_a, pnl_pct=pnl_a / notional_a if notional_a else 0.0,
            )
        )
        trades.append(
            TradeRecord(
                stock_id=pos.stock_b, industry=industry_map.get(pos.stock_b, ""), strategy=strategy_b, shares=pos.shares_b,
                entry_date=pos.entry_date, entry_price=pos.entry_price_b, exit_date=date, exit_price=price_b,
                pnl=pnl_b, pnl_pct=pnl_b / notional_b if notional_b else 0.0,
            )
        )

    for i, date in enumerate(dates):
        is_reformation = i in reformation_indices

        if is_reformation:
            for key in list(open_pairs.keys()):
                _close_pair(key, date)
            window_dates = dates[i - rt_cfg.formation_window : i]
            active_pairs = _find_pairs(close_pivot, industry_map, window_dates, rt_cfg)
            spread_cache = {}
            for a, b, beta in active_pairs:
                spread = np.log(close_pivot[a]) - beta * np.log(close_pivot[b])
                spread_cache[(a, b)] = _zscore_series(spread, rt_cfg.zscore_window)

        if i > 0 and not is_reformation:
            prev_date = dates[i - 1]
            per_pair_budget = cfg.initial_capital / max(rt_cfg.top_n_pairs, 1)

            for a, b, beta in active_pairs:
                key = (a, b)
                z = spread_cache[key].get(prev_date, np.nan)
                if pd.isna(z):
                    continue

                if key in open_pairs:
                    pos = open_pairs[key]
                    holding_days = (date - pos.entry_date).days
                    if abs(z) <= rt_cfg.exit_z or abs(z) >= rt_cfg.stop_z or holding_days >= rt_cfg.max_holding_days:
                        _close_pair(key, date)
                    continue

                if abs(z) < rt_cfg.entry_z:
                    continue
                if a not in open_pivot.columns or b not in open_pivot.columns:
                    continue
                price_a = open_pivot.at[date, a] if date in open_pivot.index else np.nan
                price_b = open_pivot.at[date, b] if date in open_pivot.index else np.nan
                if pd.isna(price_a) or pd.isna(price_b) or price_a <= 0 or price_b <= 0:
                    continue

                direction = -1 if z > 0 else 1  # z>0: A 相對偏貴 -> 空A多B；z<0: 多A空B
                leg_budget = per_pair_budget / 2
                shares_a = math.floor((leg_budget / price_a) / cfg.sizing.lot_size) * cfg.sizing.lot_size
                shares_b = math.floor((leg_budget / price_b) / cfg.sizing.lot_size) * cfg.sizing.lot_size
                if shares_a < cfg.sizing.lot_size or shares_b < cfg.sizing.lot_size:
                    continue

                if direction == 1:
                    fee_a = cost_mod.entry_cost(price_a, shares_a, cfg.costs)
                    cost_basis_a = shares_a * price_a + fee_a
                    _, proceeds_b = cost_mod.exit_proceeds(price_b, shares_b, cfg.costs)
                    net_flow = -cost_basis_a + proceeds_b
                    if cost_basis_a > cash:
                        continue
                    cash += net_flow
                    open_pairs[key] = _OpenPair(
                        stock_a=a, stock_b=b, beta=beta, direction=1,
                        shares_a=shares_a, shares_b=shares_b,
                        entry_price_a=price_a, entry_price_b=price_b,
                        cash_flow_a=-cost_basis_a, cash_flow_b=proceeds_b, entry_date=date,
                    )
                else:
                    _, proceeds_a = cost_mod.exit_proceeds(price_a, shares_a, cfg.costs)
                    fee_b = cost_mod.entry_cost(price_b, shares_b, cfg.costs)
                    cost_basis_b = shares_b * price_b + fee_b
                    if cost_basis_b > cash:
                        continue
                    net_flow = proceeds_a - cost_basis_b
                    cash += net_flow
                    open_pairs[key] = _OpenPair(
                        stock_a=a, stock_b=b, beta=beta, direction=-1,
                        shares_a=shares_a, shares_b=shares_b,
                        entry_price_a=price_a, entry_price_b=price_b,
                        cash_flow_a=proceeds_a, cash_flow_b=-cost_basis_b, entry_date=date,
                    )

        mtm = cash
        for pos in open_pairs.values():
            price_a = close_pivot.at[date, pos.stock_a] if pos.stock_a in close_pivot.columns else np.nan
            price_b = close_pivot.at[date, pos.stock_b] if pos.stock_b in close_pivot.columns else np.nan
            price_a = pos.entry_price_a if pd.isna(price_a) else price_a
            price_b = pos.entry_price_b if pd.isna(price_b) else price_b
            if pos.direction == 1:
                mtm += pos.shares_a * price_a - pos.shares_b * price_b
            else:
                mtm += -pos.shares_a * price_a + pos.shares_b * price_b
        equity_rows.append((date, mtm))

    # 資料結束時強制把還沒平倉的配對用最後一天收盤價結清，維持跟
    # intraday_roundtrip.py 一致的「回測結束不留倉」慣例，讓 metrics_from_result
    # 不需要額外處理 dual-leg 的 open_positions。
    if open_pairs and len(dates) > 0:
        last_date = dates[-1]
        for key in list(open_pairs.keys()):
            _close_pair(key, last_date)

    equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"]).set_index("date")
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return BacktestResult(equity_curve=equity_df, trades=trades_df, open_positions={}, rejected_log=[], mdd_breach_count=0)
