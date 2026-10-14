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
from tw_quant.us_universe import membership_eligibility_mask


@dataclass
class FactorConfig:
    momentum_window: int = 60  # 排名用的落後報酬率天數
    rebalance_freq_days: int = 21  # 調倉頻率（交易日），21 約等於月調倉，63 約等於季調倉
    top_n: int = 20  # 每次調倉持有的檔數（等權重）
    ascending: bool = False  # False=買排名前段（動量／追強勢）；True=買排名後段（反轉／買弱勢）
    # 2026-10-14 新增，選填：個股固定停損百分比（例如 0.15 = 從進場價跌 15%
    # 就停損），None = 不停損，完全維持原本「只在調倉日換股、期間不管個股
    # 跌多少」的行為。這是從進場價固定算的停損，不是像 tw_quant/backtest.py
    # 那種「只上調不下調」的吊燈停利（移動停損）——使用者問的是單純的
    # 固定跌幅停損，兩者是不同機制，不要混用。
    stop_loss_pct: float | None = None


def run_factor_backtest(
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    factor_cfg: FactorConfig,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    signal_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
    cost_module=cost_mod,
    membership: pd.DataFrame | None = None,
) -> BacktestResult:
    """月/季調倉動能因子組合：固定頻率對魚池內個股依排名，等權重持有
    前 N 檔，只有在下次調倉日才換股；factor_cfg.stop_loss_pct 給定時，
    期間個股跌破固定停損價會提前出場（見該欄位說明），不給的話（預設）
    維持原本「期間完全不管個股跌多少，只等下次調倉」的行為。

    停損觸發/成交機制跟 tw_quant.backtest.run_backtest 的吊燈停利完全
    一致（刻意保持一樣的反未來函數慣例，不是另外發明一套）：用「今日
    收盤價」判斷是否跌破停損價，觸發後排進「明日開盤」才出場，不假設
    當天盤中就能在停損價那一刻搶先賣掉（tw_quant/intraday_roundtrip.py
    已經說明過，這個資料管線只有日線 OHLC，不假設日內價格順序，用盤中
    低點觸發停損對回測來說過於樂觀）。停損出場跟調倉出場都會真實扣成本
    （cost_module.exit_proceeds），調倉頻率越高、停損越容易被打到，換手
    成本疊加停損成本可能讓淨報酬比「不停損」更差，不是加了風控機制就
    一定更好，要看實際回測數字。

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

    membership：選填，美股用（見 tw_quant/us_universe.py）。傳入時，「是否
    為指數成分股」會用 membership_eligibility_mask 當成跟均線/均量/歷史
    長度同一層級的資格條件（一樣 shift(1) 取 T-1 為止已知資訊）疊加進魚池
    判定，不會去動 prices 本身——2026-09-21 發現先前呼叫端習慣先用
    tw_quant.us_universe.filter_prices_by_index_membership 把 prices
    砍過一輪再傳進來，會連帶砍掉均線/均量/歷史長度這些指標賴以計算的
    完整價格序列，誤傷「公司歷史悠久、但指數成分股身份中途一度中斷又
    重新加入」的股票（詳見 us_universe.py 檔頭的完整說明）。呼叫端應該
    一律傳完整未過濾的 prices，改用這個參數傳 membership。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    pool = build_pool_mask(master, cfg.pool)

    if membership is not None:
        raw_member = membership_eligibility_mask(master, membership)
        member_t_minus_1 = ind.shift_by_group(raw_member.astype(float), master, periods=1).fillna(0).astype(bool)
        pool = pool & member_t_minus_1

    if signal_fn is not None:
        ranking_signal = signal_fn(master)
    else:
        ret = master.groupby("stock_id", sort=False)["close"].pct_change(factor_cfg.momentum_window)
        ranking_signal = ind.shift_by_group(ret, master, periods=1)

    master["pool"] = pool.values
    master["momentum"] = ranking_signal.values if hasattr(ranking_signal, "values") else ranking_signal

    # 調倉日曆的起點永遠錨定在完整歷史的第一天，不能等 start_date/end_date
    # 切片完才算——不然同一組 factor_cfg 只因為傳進來的 start_date 不同，
    # 調倉日期本身就會跟著平移。對月調倉、只押 top_n=3 這種高度集中的組合來說，
    # 調倉日期差個幾天，實際買到/賣掉哪幾檔就可能完全不同，算出來的報酬率可以
    # 差到一倍以上——這正是把同一段歷史切成兩段分別回測、再跟一次連續回測
    # 對比時，總報酬對不起來的根本原因（見 2026-09-19 對話紀錄的
    # regime breakdown 交叉檢查）。先在完整（未切片）日期序列上決定哪些日子
    # 是調倉日，再用 start_date/end_date 篩選要不要「允許」在那天交易，
    # 兩者分開處理，調倉日曆才不會被切片起點污染。
    full_dates = sorted(master["date"].unique())
    rebalance_dates = set(full_dates[:: factor_cfg.rebalance_freq_days])

    if start_date is not None:
        master = master[master["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        master = master[master["date"] <= pd.Timestamp(end_date)]

    cash = cfg.initial_capital
    positions: dict[str, Position] = {}
    trades: list[TradeRecord] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    pending_stop_exits: set[str] = set()  # 昨日收盤觸發停損、今日開盤才出場

    for date, day_df in master.groupby("date", sort=True):
        row_by_stock = day_df.set_index("stock_id")

        # 0) 昨日收盤觸發停損的部位，今天開盤出場（跟 tw_quant.backtest.run_backtest
        # 的 pending_exits 機制同一套反未來函數慣例，見本函式開頭說明）。
        # 永遠在調倉邏輯之前處理，這樣同一天如果恰好也是調倉日，排名用的
        # 現金水位跟持股狀態已經先反映了停損出場。
        for stock_id in list(pending_stop_exits):
            pending_stop_exits.discard(stock_id)
            if stock_id not in positions or stock_id not in row_by_stock.index:
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
                    exit_reason="STOP_LOSS",
                )
            )

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
                        exit_reason="REBALANCE",
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
                # per_stock_budget 可能等於（甚至只略低於）現有 cash（n_target 很小時
                # 尤其明顯，n_target=1 時 budget 直接等於全部資金），floor 完的股數再加上
                # 手續費就可能略微超出 cash——不是「這檔真的買不起」，只是股數還沒扣到
                # 剛好留出手續費的空間，逐張（lot）往下調直到真的塞得進 cash 為止，而不是
                # 整筆直接放棄進場
                while total_cost > cash and shares >= cfg.sizing.lot_size:
                    shares -= cfg.sizing.lot_size
                    if shares < cfg.sizing.lot_size:
                        break
                    fee = cost_module.entry_cost(entry_price, shares, cfg.costs)
                    total_cost = shares * entry_price + fee
                if shares < cfg.sizing.lot_size or total_cost > cash:
                    continue
                cash -= total_cost
                stop_price = (
                    entry_price * (1 - factor_cfg.stop_loss_pct) if factor_cfg.stop_loss_pct is not None else 0.0
                )
                positions[stock_id] = Position(
                    stock_id=stock_id,
                    industry=row_by_stock.loc[stock_id, "industry"],
                    strategy="FACTOR",
                    shares=shares,
                    entry_price=entry_price,
                    entry_date=date,
                    stop_price=stop_price,
                    cost_basis=total_cost,
                )

        # 以今日收盤判斷是否跌破固定停損價（見本函式開頭「停損觸發/成交
        # 機制」說明：今日收盤觸發，排進明日開盤才出場，不是今天立刻賣）。
        # 刻意放在調倉邏輯之後，這樣今天如果剛好也是調倉日、剛買進的新
        # 部位，一樣會被檢查到（極端情況：買進當天就跌破停損價，明天
        # 開盤出場，不會因為是「今天剛買」而被放過）。
        if factor_cfg.stop_loss_pct is not None:
            for stock_id, pos in positions.items():
                if stock_id not in row_by_stock.index:
                    continue
                close_price = row_by_stock.loc[stock_id, "close"]
                if pd.notna(close_price) and close_price <= pos.stop_price:
                    pending_stop_exits.add(stock_id)

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
