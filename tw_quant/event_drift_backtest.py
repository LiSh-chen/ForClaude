"""事件驅動的固定持有期回測引擎：每個事件（例如財報驚喜）各自獨立管理
「延遲進場 -> 固定天數後出場」的生命週期，不受任何全域調倉日曆牽制。

背景：使用者問「如果要在財報公布的幾天內進場，PEAD 績效會不會比較好」。
scripts/test_us_pead_earnings_drift_from_db.py 用的
tw_quant/factor_backtest.py 是定期調倉引擎——訊號在兩次調倉之間發生的
變化要等到下次調倉才會被採用，最壞情況（例如季調倉）代表財報公布後
訊號可能要放將近一整季才真正被拿來決定要不要買，天生稀釋掉 PEAD 假設
「公布後最初幾週到幾個月漂移最明顯」這個時間尺度。

tw_quant/backtest.py 雖然是逐日事件驅動，但出場邏輯是停損/ATR移動停損，
沒有「固定持有 N 個交易日後出場」這種機制，也不適合直接拿來用。

這裡是專門處理「事件觸發 -> 延遲進場 -> 固定天數後出場」這種模式的第三種
引擎，刻意簡化：不做停損/停利，也不做複雜的全域風控排隊（只有簡單的
「同時最多持有幾個部位」上限），先驗證「進場時機貼近事件」這個變數本身
有沒有差，再考慮要不要疊加更多真實限制。跟 factor_backtest.py／
backtest.py 完全獨立，不影響、也不取代既有的任何策略腳本。

反未來函數：entry_lag_days（進場延遲，預設至少 1 天）就是「事件已知日期
當天還不能交易」的保守處理機制本身——呼叫端傳進來的 events 的 known_date
必須已經是「當時市場就知道的日期」（例如財報真正公布日），不用再自己
shift，這裡的 lag 就是負責這個緩衝的地方。

用法：
    events = _earnings_surprise_frame(earnings)  # columns: stock_id, known_date, surprise
    result = run_event_drift_backtest(prices, events.rename(columns={"surprise": "signal"}), cfg, drift_cfg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from tw_quant import costs as cost_mod
from tw_quant.backtest import BacktestResult, Position, TradeRecord
from tw_quant.config import StrategyConfig
from tw_quant.us_universe import membership_eligibility_mask

EVENT_COLS = ["stock_id", "known_date", "signal"]

# events 傳進來的 known_date 是「市場當時就知道的日期」。這裡的 by_stock
# 永遠用完整、未過濾的價格序列建構（呼叫端不該再用
# tw_quant.us_universe.filter_prices_by_index_membership 事先砍過），
# 所以 searchsorted 找到的「known_date 之後第一個有資料的交易日」正常
# 情況下就是緊接在事件之後的下一個交易日。10 天只是擋掉真正的資料尾端
# /資料缺口（例如事件發生在整個快照涵蓋範圍的最後幾天，之後這檔股票
# 完全沒有更多資料），不再是用來吸收存活者偏差過濾造成的假空窗期——
# 見下面 run_event_drift_backtest 的 membership 參數說明。
#
# ★ 2026-09-21 架構修正前的舊版本：呼叫端會先用
# filter_prices_by_index_membership 把 by_stock 的價格序列砍過一輪，
# 讓「該股票被剔除又重新納入指數」的空窗期在價格序列裡完全消失。事件
# 發生在空窗期時，searchsorted 會跳過空窗、找到「重新納入之後」的第一筆
# 資料——可能是好幾年後，等於把舊事件的訊號套用到多年後的進場。
# 2026-09-20 發現：不設這個容忍值時，全部 38126 筆候選事件裡有 25823 筆
# （68%）進場日期跟事件日期相差超過 30 天，最誇張的相差超過 20 年。
# 當時的修正只是加這個容忍值去「擋掉」明顯異常的配對，沒有解決根源——
# 根源跟 tw_quant/factor_backtest.py 的 membership bug 完全一樣：不該
# 在計算任何東西（這裡是「事件後最近的交易日」）之前就先砍價格序列。
MAX_KNOWN_DATE_GAP_DAYS = 10


@dataclass
class EventDriftConfig:
    entry_lag_days: int = 1  # 事件已知日期後幾個交易日進場（>=1，事件當天不能交易）
    holding_days: int = 40  # 進場後固定持有幾個交易日，不管賺賠
    signal_threshold: float = 0.0  # 訊號值（例如驚喜幅度）至少要 >= 這個門檻才觸發，用來只做多正向意外
    max_concurrent_positions: int = 20  # 同時最多持有幾個部位
    capital_per_position: float | None = None  # 每個部位固定分配多少資金；None 時用 initial_capital / max_concurrent_positions


def _build_entry_exit_candidates(
    events: pd.DataFrame, by_stock: dict[str, pd.DataFrame], drift_cfg: EventDriftConfig
) -> list[dict]:
    """把每個事件轉成一筆候選進出場紀錄（進場日期/價格、出場日期/是否因資料
    結束被迫提前出場）。獨立成純函式方便測試，不用真的跑整個回測迴圈。

    找不到對應股票價量資料、或事件之後已經沒有足夠交易日可以進場的事件會
    被跳過（不是錯誤，只是這筆事件沒有可用的執行時機）。
    """
    candidates = []
    filtered = events[events["signal"] >= drift_cfg.signal_threshold]
    for row in filtered.itertuples(index=False):
        stock_df = by_stock.get(row.stock_id)
        if stock_df is None or stock_df.empty:
            continue
        dates = stock_df.index
        base_idx = dates.searchsorted(row.known_date, side="left")
        if base_idx >= len(dates):
            continue  # 事件之後這檔股票完全沒有資料（例如已被移出指數快照）
        if (dates[base_idx] - row.known_date).days > MAX_KNOWN_DATE_GAP_DAYS:
            continue  # 事件當下這檔股票不在存活者偏差過濾後的資料裡，不是真的可交易時機
        entry_idx = base_idx + drift_cfg.entry_lag_days
        if entry_idx >= len(dates):
            continue  # 事件發生在資料尾端附近，之後沒有足夠交易日可以進場

        exit_idx_target = entry_idx + drift_cfg.holding_days
        exit_idx = min(exit_idx_target, len(dates) - 1)
        forced_exit_at_data_end = bool(exit_idx < exit_idx_target)

        candidates.append(
            {
                "stock_id": row.stock_id,
                "known_date": row.known_date,
                "signal": row.signal,
                "entry_date": dates[entry_idx],
                "entry_price": float(stock_df["open"].iloc[entry_idx]),
                "exit_date": dates[exit_idx],
                "exit_price": float(stock_df["open"].iloc[exit_idx]),
                "forced_exit_at_data_end": forced_exit_at_data_end,
            }
        )
    return candidates


def run_event_drift_backtest(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    cfg: StrategyConfig,
    drift_cfg: EventDriftConfig,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cost_module=cost_mod,
    membership: pd.DataFrame | None = None,
) -> BacktestResult:
    """events 須有 stock_id / known_date / signal 三欄。start_date/end_date
    只限制「哪些事件的 known_date 落在這個窗格內才可能觸發新進場」，價量
    資料永遠用完整 prices（不切片）——已經觸發的部位可以持有到窗格結束
    之後才出場，這是預期行為，不是 bug。

    `prices` 須是完整、未過濾的價格序列（呼叫端不要再用
    filter_prices_by_index_membership 事先砍過一輪）——2026-09-21 架構
    修正：`membership` 參數改在「決定哪些事件是有效候選」這一步才介入
    （用 membership_eligibility_mask 判斷這檔股票在 known_date 當天是不是
    已知的指數成分股，不是的事件直接丟掉），而不是先砍價格序列本身。
    這樣「找事件後最近的交易日」永遠用真實連續的股價資料，不會因為股票
    曾經被剔除指數又重新納入，而把事件跟好幾年後的交易日錯誤配對（見
    上面 MAX_KNOWN_DATE_GAP_DAYS 的說明）。`membership=None`（預設）保留
    舊行為，不做任何資格過濾。

    回傳的 equity_curve 會裁到 [start_date, 實際需要的結束日] 這個區間
    （結束日取 end_date 跟「最後一筆交易/未平倉部位的日期」兩者較晚的
    一個，讓跨過窗格結束才出場的部位能完整反映），不是完整價量資料的
    整個日期範圍——2026-09-20 發現這裡原本沒有裁切，回傳的 equity_curve
    橫跨全部歷史（含窗格外一長串沒有任何部位、報酬率永遠是 0 的日子），
    導致 metrics_from_result 算 CAGR 時的年數分母用整段歷史長度（例如
    8年）而不是真正的交易窗格長度（例如樣本內只有3年），CAGR 被嚴重
    低估、Sharpe 也被摻進大量 0 報酬率的日子所扭曲——這是已經送進
    GitHub Actions 跑過三次的真實 bug，見 2026-09-20 對話紀錄的重新
    驗證結果。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True)
    by_stock = {sid: g.set_index("date")[["open", "close"]] for sid, g in master.groupby("stock_id", sort=False)}

    filtered_events = events
    if start_date is not None:
        filtered_events = filtered_events[filtered_events["known_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered_events = filtered_events[filtered_events["known_date"] <= pd.Timestamp(end_date)]

    if membership is not None and not filtered_events.empty:
        event_points = filtered_events.rename(columns={"known_date": "date"})[["stock_id", "date"]]
        eligible = membership_eligibility_mask(event_points, membership)
        filtered_events = filtered_events[eligible.values]

    candidates = _build_entry_exit_candidates(filtered_events, by_stock, drift_cfg)
    entries_by_date: dict[pd.Timestamp, list[dict]] = {}
    for c in candidates:
        entries_by_date.setdefault(c["entry_date"], []).append(c)
    for lst in entries_by_date.values():
        lst.sort(key=lambda c: c["signal"], reverse=True)  # 同一天多個候選時，驚喜幅度高的優先取得部位額度

    capital_per_position = drift_cfg.capital_per_position or (cfg.initial_capital / drift_cfg.max_concurrent_positions)
    calendar = sorted(master["date"].unique())

    cash = cfg.initial_capital
    open_by_stock: dict[str, dict] = {}  # stock_id -> candidate dict (含 shares/cost_basis，執行中才補上)
    trades: list[TradeRecord] = []
    rejected_log: list[dict] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []

    close_lookup = master.set_index(["stock_id", "date"])["close"]

    for date in calendar:
        # 1) 先處理今天排定要出場的部位，釋放額度跟現金
        exiting = [sid for sid, pos in open_by_stock.items() if pos["exit_date"] == date]
        for stock_id in exiting:
            pos = open_by_stock.pop(stock_id)
            _, net_proceeds = cost_module.exit_proceeds(pos["exit_price"], pos["shares"], cfg.costs)
            cash += net_proceeds
            pnl = net_proceeds - pos["cost_basis"]
            trades.append(
                TradeRecord(
                    stock_id=stock_id,
                    industry=pos.get("industry", ""),
                    strategy="EVENT_DRIFT",
                    shares=pos["shares"],
                    entry_date=pos["entry_date"],
                    entry_price=pos["entry_price"],
                    exit_date=date,
                    exit_price=pos["exit_price"],
                    pnl=pnl,
                    pnl_pct=pnl / pos["cost_basis"] if pos["cost_basis"] else 0.0,
                )
            )

        # 2) 再處理今天排定要進場的候選（先出場釋放額度，同一天騰出來的位置可以立刻被新事件用掉）
        for cand in entries_by_date.get(date, []):
            stock_id = cand["stock_id"]
            if stock_id in open_by_stock:
                rejected_log.append({"date": date, "stock_id": stock_id, "stage": "duplicate_position", "reason": "該股票已經持有中"})
                continue
            if len(open_by_stock) >= drift_cfg.max_concurrent_positions:
                rejected_log.append({"date": date, "stock_id": stock_id, "stage": "capacity", "reason": "同時持有部位數已達上限"})
                continue

            entry_price = cand["entry_price"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            shares = math.floor((capital_per_position / entry_price) / cfg.sizing.lot_size) * cfg.sizing.lot_size
            if shares < cfg.sizing.lot_size:
                rejected_log.append({"date": date, "stock_id": stock_id, "stage": "sizing", "reason": "分配資金買不到最小交易單位"})
                continue
            fee = cost_module.entry_cost(entry_price, shares, cfg.costs)
            total_cost = shares * entry_price + fee
            if total_cost > cash:
                rejected_log.append({"date": date, "stock_id": stock_id, "stage": "execution", "reason": "現金不足"})
                continue

            cash -= total_cost
            open_by_stock[stock_id] = {
                "entry_date": cand["entry_date"],
                "entry_price": entry_price,
                "exit_date": cand["exit_date"],
                "exit_price": cand["exit_price"],
                "shares": shares,
                "cost_basis": total_cost,
            }

        # 3) 記錄當天權益（現金 + 持倉市值，用當天收盤價估）
        mtm = cash
        for stock_id, pos in open_by_stock.items():
            price = close_lookup.get((stock_id, date), pos["entry_price"])
            mtm += pos["shares"] * price
        equity_rows.append((date, mtm))

    equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"]).set_index("date")

    window_start = pd.Timestamp(start_date) if start_date is not None else calendar[0]
    window_end = pd.Timestamp(end_date) if end_date is not None else calendar[-1]
    relevant_dates = [t.entry_date for t in trades] + [t.exit_date for t in trades] + [p["entry_date"] for p in open_by_stock.values()]
    if relevant_dates:
        window_end = max(window_end, max(relevant_dates))
    equity_df = equity_df.loc[(equity_df.index >= window_start) & (equity_df.index <= window_end)]

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    open_positions = {
        sid: Position(
            stock_id=sid, industry="", strategy="EVENT_DRIFT", shares=pos["shares"],
            entry_price=pos["entry_price"], entry_date=pos["entry_date"], stop_price=0.0, cost_basis=pos["cost_basis"],
        )
        for sid, pos in open_by_stock.items()
    }
    return BacktestResult(
        equity_curve=equity_df, trades=trades_df, open_positions=open_positions,
        rejected_log=rejected_log, mdd_breach_count=0,
    )
