"""每日模擬倉（紙上交易）前瞻追蹤更新腳本。

追蹤這次會話「已驗證」的策略：
- low_vol_buyhold（低波動做多+8%停損）——跨日持有型
- calendar_spread（近月/遠月價差均值回歸）——跨日持有型
- og/lu/re（開盤上衝/午盤放空/盤中翻多，三腳組合）——當日內完成型，需要
  當天1分K才能算實際進出場價，資料源是TAIFEX官方「前30個交易日逐筆成交」
  （見 fetch_taifex_tick_1min.py，重新聚合成1分K，跟舊的靜態1分K檔案
  交叉驗證過open/high/low完全一致）。

**核心原則，誠實揭露**：
1. 這是模擬倉（紙上交易），不涉及真錢下單，不構成投資建議。
2. 「模擬權益曲線」只從系統第一次執行的那天開始算（state.json的start_date
   ，只設一次），把系統啟用前的歷史回測損益也算進來會製造「這套系統一直
   都在賺錢」的假象——過去的回測結果已經在策略手冊/策略實驗室Artifact裡
   誠實揭露過了，這裡只記錄「從啟用那天起，如果真的照這個規則模擬操作，
   會發生什麼」。
3. 若系統啟用時跨日型策略剛好處於「進行中的部位」，會把這個部位視為
   「啟用當天新進場」記錄（用當天價格模擬進場，不回溯用策略邏輯上真正
   的進場日期/價格），確保模擬績效只反映「從今天跟著做」會發生的損益。
4. **高波動體制濾網一定要用lag1（前一交易日收盤已確定）版本，不能用
   strategy_lab.py歷史回測用的「同日」版本**——同日版本是為了讓
   low_vol_buyhold的歷史回測數字跟原始驗證腳本一致才刻意保留的（見
   LowVolBuyHoldConfig docstring：「真實下單一定要按手冊的規則，用前一
   交易日收盤已確定的體制決定隔天要不要進場，不能用當天還沒收盤的體制」
   ）。模擬倉的目標就是「假裝真的在照規則操作」，用同日版本等於用到了
   未來資訊（今天的體制要等今天收盤才確定，那時已經來不及決定「今天」
   要不要進場）。這裡的run_low_vol_buyhold_lag1()跟三腳的高波動濾網都
   用compute_regime_lag1()，內建lag1，不會有這個問題；calendar_spread
   的z分數本身在strategy_lab.py裡就已經是shift(1)安全的，直接沿用。
5. 每次更新都會計算「未實現」浮動損益（部位還沒出場時，用最新收盤價
   估算），跟「已實現」損益分開記錄，前端會分別標示。

用法：
    python scripts/daily_paper_trading_update.py
"""

from __future__ import annotations

import json
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tw_quant.strategy_lab import run_calendar_spread, CalendarSpreadConfig
from tw_quant.opening_rally_strategy import OpeningRallyConfig, backtest as bt_opening_rally
from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest as bt_lunch_reversal
import fetch_taifex_tx_raw
import build_taifex_multi_contract
import fetch_taifex_tick_1min

STATE_PATH = REPO_ROOT / "data" / "paper_trading_state.json"
LOG_PATH = REPO_ROOT / "data" / "paper_trading_log.parquet"
EQUITY_PATH = REPO_ROOT / "data" / "paper_trading_equity.parquet"
MULTI_CONTRACT_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"

POINT_VALUE = 50.0  # MTX點值（這裡先用小台，跟這次會話其餘策略手冊一致的口數假設）

# 跟export_strategy_lab.py用IS(2001-2020)算出來的固定門檻一致，不重算
# （不能用之後才知道的資料回頭調整門檻，這是「量能濾網」原本設計的精神）
RE_VOLUME_FILTER_THRESHOLD = 1.099965

LOW_VOL_CFG = dict(atr_window=14, ma_window=252, band=0.10, stop_pct=0.08)

TRACKED_STRATEGIES = {
    "calendar_spread": dict(label="近月/遠月價差均值回歸", run=run_calendar_spread, cfg=CalendarSpreadConfig()),
}


def compute_regime_lag1(mc: pd.DataFrame, atr_window: int = 14, ma_window: int = 252, band: float = 0.10) -> pd.Series:
    """用近月合約官方日K(open/high/low/近月結算價)算高波動體制，回傳跟mc同
    長度、index對齊的Series，值＝「前一交易日收盤」已確定的體制(0.0/1.0/nan)
    ，已經內建lag(1)——今天能不能交易，看的是「昨天」就已經確定的值，不是
    今天自己的值（今天的體制要今天收盤才確定，用來決定「今天」要不要交易
    就是未來函數）。邏輯跟strategy_lab.py的_compute_volatility_regime_map
    一樣，只是資料源換成近月合約日K（已交叉驗證兩者regime分類100%一致）。"""
    close, high, low = mc["near_price"], mc["near_high"], mc["near_low"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_window, adjust=False).mean()
    atr_ma = atr.rolling(ma_window).mean()
    upper, lower = atr_ma * (1 + band), atr_ma * (1 - band)

    regime = np.full(len(mc), np.nan)
    current = np.nan
    for i in range(len(mc)):
        if pd.isna(atr_ma.iloc[i]):
            continue
        v = atr.iloc[i]
        if pd.isna(current):
            current = 1.0 if v > atr_ma.iloc[i] else 0.0
        elif v > upper.iloc[i]:
            current = 1.0
        elif v < lower.iloc[i]:
            current = 0.0
        regime[i] = current
    return pd.Series(regime, index=mc.index).shift(1)


def compute_vol_ratio_lag1(mc: pd.DataFrame, window: int = 20) -> pd.Series:
    """re腿量能濾網用：近月合約每日成交量相對20日滾動均量的比率，lag(1)
    （跟export_strategy_lab.py算vol_ratio_lag1的做法一致，只是資料源換成
    near_volume——兩者交叉驗證過中位數比率0.9995，幾乎完全一致，見這次
    會話的驗證記錄）。"""
    ratio = mc["near_volume"] / mc["near_volume"].rolling(window).mean()
    return ratio.shift(1)


def run_low_vol_buyhold_lag1(mc: pd.DataFrame) -> pd.DataFrame:
    """模擬倉專用版本：跟strategy_lab.run_low_vol_buyhold邏輯一樣(近月價格
    做多+8%移動停損、同一段低波動週期內不重新進場)，唯一差異是體制用
    compute_regime_lag1()的lag1版本，不是「同日」版本——見檔案頂端docstring
    第4點，這裡是「真實可執行」的規則，不是為了複現歷史驗證數字。"""
    cfg = LOW_VOL_CFG
    regime_lag1 = compute_regime_lag1(mc, cfg["atr_window"], cfg["ma_window"], cfg["band"])
    mc = mc.copy()
    mc["high_vol_regime"] = regime_lag1.values

    valid = mc["high_vol_regime"].notna()
    r = (mc.loc[valid, "high_vol_regime"] == 0.0)
    d = mc.loc[valid, "date"]
    group_id = r.ne(r.shift()).cumsum()
    tmp = pd.DataFrame({"date": d.values, "is_low_vol": r.values, "group": group_id.values})
    episodes = []
    for _, sub in tmp.groupby("group"):
        if sub["is_low_vol"].iloc[0]:
            episodes.append((sub["date"].min(), sub["date"].max()))

    trades = []
    for start, end in episodes:
        ep = mc[(mc["date"] >= start) & (mc["date"] <= end)].reset_index(drop=True)
        if len(ep) < 2:
            continue
        entry_price = ep["near_price"].iloc[0]
        peak = entry_price
        stopped = False
        exit_price, exit_date = ep["near_price"].iloc[-1], ep["date"].iloc[-1]
        for i in range(1, len(ep)):
            price = ep["near_price"].iloc[i]
            peak = max(peak, price)
            if price <= peak * (1 - cfg["stop_pct"]):
                exit_price, exit_date, stopped = price, ep["date"].iloc[i], True
                break
        trades.append(dict(
            direction="long", entry_date=ep["date"].iloc[0], exit_date=exit_date,
            entry_price=entry_price, exit_price=exit_price,
            pnl_points=exit_price - entry_price, stopped=stopped,
        ))
    return pd.DataFrame(trades)


def _run_low_vol_wrapper(df: pd.DataFrame | None, cfg: object) -> pd.DataFrame:
    """讓run_low_vol_buyhold_lag1(吃mc)套進process_strategy統一的(df, cfg)
    呼叫介面，df/cfg參數都不用（内部自己重新讀取multi_contract.parquet）。"""
    mc = pd.read_parquet(MULTI_CONTRACT_PATH).sort_values("date").reset_index(drop=True)
    return run_low_vol_buyhold_lag1(mc)


TRACKED_STRATEGIES["low_vol_buyhold"] = dict(
    label="低波動做多+8%停損", run=_run_low_vol_wrapper, cfg=None,
)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"start_date": None, "strategies": {}, "last_run": None}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))


def append_log(rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if LOG_PATH.exists():
        existing = pd.read_parquet(LOG_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(LOG_PATH, index=False)


def today_str() -> str:
    mc = pd.read_parquet(MULTI_CONTRACT_PATH)
    return mc["date"].max().strftime("%Y-%m-%d")


def refresh_raw_data() -> None:
    """重抓「本月」的TAIFEX期貨原始行情（fetch_taifex_tx_raw.py預設「檔案已
    存在就跳過」，本月資料每天在累積，要先砍掉本月快取才會真的重抓最新一
    天），過去月份不動（已經是定案資料，不需要重抓）。抓完後重建
    multi_contract.parquet。"""
    from datetime import date
    ym = date.today().strftime("%Y-%m")
    out_path = fetch_taifex_tx_raw.OUT_DIR / f"TX_{ym}.csv"
    if out_path.exists():
        out_path.unlink()
    fetch_taifex_tx_raw.OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_taifex_tx_raw.fetch_month(ym)
    if raw is None:
        raise RuntimeError(f"抓取本月({ym})期貨原始行情失敗，模擬倉更新中止")
    text = raw.decode("big5", errors="replace")
    if "交易日期" not in text:
        raise RuntimeError(f"本月({ym})期貨原始行情回傳內容異常（可能是錯誤頁），模擬倉更新中止")
    out_path.write_text(text, encoding="utf-8")
    print(f"已更新本月原始行情: {out_path}")

    raw_df = build_taifex_multi_contract.load_all_raw()
    cleaned = build_taifex_multi_contract.clean(raw_df)
    nearfar = build_taifex_multi_contract.build_nearfar(cleaned)
    nearfar.to_parquet(MULTI_CONTRACT_PATH, index=False)
    print(f"已重建 multi_contract.parquet，最新交易日: {nearfar['date'].max().strftime('%Y-%m-%d')}")


def process_strategy(sid: str, meta: dict, state: dict, run_date: str) -> list[dict]:
    """跑一次策略全歷史邏輯，跟上次記錄的狀態比較，回傳這次新增的事件（進場/
    出場/浮動損益更新），並就地更新state[sid]。"""
    trades = meta["run"](None, meta["cfg"])
    if trades.empty:
        return []
    trades = trades.sort_values("entry_date").reset_index(drop=True)
    last = trades.iloc[-1]

    mc = pd.read_parquet(MULTI_CONTRACT_PATH)
    data_last_date = mc["date"].max().strftime("%Y-%m-%d")
    last_entry_date = pd.Timestamp(last["entry_date"]).strftime("%Y-%m-%d")
    last_exit_date = pd.Timestamp(last["exit_date"]).strftime("%Y-%m-%d")

    # 若最後一筆的出場日剛好等於資料最後一天，這筆是「尚未真正出場、用
    # 今天價格估浮動損益」的進行中部位；否則代表最新的策略邏輯下，這筆
    # 已經真正結束了(出場日 < 資料最後一天)，此刻沒有持倉。
    is_open_position = (last_exit_date == data_last_date)

    prev = state["strategies"].get(sid)
    events = []

    if is_open_position:
        if prev is None or prev.get("status") != "open" or prev.get("real_entry_date") != last_entry_date:
            # 新偵測到一筆進行中的部位（系統剛啟用、或前一筆才平倉這筆剛進場）。
            # 模擬進場價用「今天」的價格/價差，不回溯用策略邏輯上的真實進場價，
            # 理由見檔案頂端docstring第3點。last["exit_price"]此刻就是用資料
            # 最後一天(=今天)的價格/價差算出來的(策略邏輯裡進行中部位的
            # exit_price固定等於當天price/spread)，兩個策略共用同一個欄位。
            sim_entry_price = float(last["exit_price"])
            events.append(dict(
                sim_date=run_date, strategy_id=sid, event="open",
                direction=last["direction"], sim_entry_price=sim_entry_price,
                real_entry_date=last_entry_date, note="系統偵測到進行中部位，以當天價格模擬進場",
            ))
            state["strategies"][sid] = dict(
                status="open", real_entry_date=last_entry_date,
                sim_entry_date=run_date, sim_entry_price=sim_entry_price,
                direction=last["direction"],
            )
        else:
            # 部位延續中，更新浮動損益（不產生新事件，只更新state給前端讀）。
            # last["exit_price"]此刻就是「用資料最後一天的價格/價差算出來的」
            # （策略邏輯裡進行中部位的exit_price固定等於當天price），兩個策略
            # 共用同一個欄位，不需要另外查near_price表。
            cur_price = float(last["exit_price"])
            sign = 1 if prev["direction"] == "long" else -1
            unrealized = (cur_price - prev["sim_entry_price"]) * sign
            state["strategies"][sid]["unrealized_pnl_points"] = unrealized
            state["strategies"][sid]["as_of"] = run_date
    else:
        if prev is not None and prev.get("status") == "open":
            # 上次還在持倉，這次策略邏輯顯示已經真正出場了——用sim_entry_price
            # (模擬進場價，不是策略邏輯上的真實進場價)算模擬已實現損益，跟
            # exit_price(策略邏輯算出的真實出場價/價差)同一個量綱可以直接相減。
            sign = 1 if prev["direction"] == "long" else -1
            exit_price = float(last["exit_price"])
            realized_pnl = (exit_price - prev["sim_entry_price"]) * sign
            events.append(dict(
                sim_date=run_date, strategy_id=sid, event="close",
                direction=prev["direction"], sim_entry_price=prev["sim_entry_price"],
                sim_exit_price=exit_price, sim_entry_date=prev["sim_entry_date"],
                realized_pnl_points=realized_pnl,
                real_exit_date=last_exit_date, note="策略邏輯判定已出場",
            ))
            state["strategies"][sid] = dict(status="flat", last_realized_pnl_points=realized_pnl, as_of=run_date)
        elif prev is None:
            state["strategies"][sid] = dict(status="flat", as_of=run_date)
        else:
            state["strategies"][sid]["as_of"] = run_date

    return events


INTRADAY_LABELS = {"og": "開盤上衝", "lu": "午盤放空", "re": "盤中翻多"}


def process_intraday_strategies(state: dict, run_date: str) -> list[dict]:
    """og/lu/re是「當日內完成」策略（同一天進場+出場，不跨日持倉），跟
    process_strategy()的跨日部位追蹤模型不一樣，不需要「進行中部位」的
    概念：每次執行判斷「今天」是否符合高波動體制(lag1，三腳的必要條件)，
    符合就抓今天1分K實際跑一次策略，觸發了就是一筆完整的模擬交易。用
    state記錄「上次處理到哪一天」(last_checked_date)避免同一天重跑重複
    記錄——tick資料只保留最近30天，重跑不會拿到不同的「今天」，但仍然
    保留這個檢查，跟其餘部分的冪等設計一致。"""
    events = []
    for sid in INTRADAY_LABELS:
        prev = state["strategies"].get(sid, {})
        if prev.get("last_checked_date") == run_date:
            return []  # 三腳共用同一個「今天」判斷，任一個已處理過代表今天都處理過了

    mc = pd.read_parquet(MULTI_CONTRACT_PATH).sort_values("date").reset_index(drop=True)
    mc["regime_lag1"] = compute_regime_lag1(mc).values
    mc["vol_ratio_lag1"] = compute_vol_ratio_lag1(mc).values
    today_row = mc.iloc[-1]
    is_high_vol = today_row["regime_lag1"] == 1.0
    re_vol_ok = pd.notna(today_row["vol_ratio_lag1"]) and today_row["vol_ratio_lag1"] >= RE_VOLUME_FILTER_THRESHOLD

    def mark_flat(sid: str, note: str) -> None:
        state["strategies"][sid] = dict(status="flat", last_checked_date=run_date, as_of=run_date,
                                          last_trade_date=state["strategies"].get(sid, {}).get("last_trade_date"),
                                          last_realized_pnl_points=state["strategies"].get(sid, {}).get("last_realized_pnl_points"),
                                          note=note)

    if not is_high_vol:
        for sid in INTRADAY_LABELS:
            mark_flat(sid, "今天不是高波動體制，三腳組合的必要條件不成立，不交易")
        return events

    onemin = fetch_taifex_tick_1min.fetch_and_save(run_date)
    if onemin is None:
        # 抓不到今天的1分K（TAIFEX還沒公告、或今天非交易日），跳過今天，
        # 不要用「昨天」的1分K冒充今天——寧可漏記一天，不要記錯資料。
        for sid in INTRADAY_LABELS:
            mark_flat(sid, "今天符合高波動體制，但抓不到當天1分K資料（可能TAIFEX還沒公告），暫緩，下次排程重試")
        return events

    # og
    og_trades = bt_opening_rally(onemin, OpeningRallyConfig())
    if not og_trades.empty:
        t = og_trades.iloc[-1]
        events.append(_intraday_trade_event("og", run_date, "long", t["entry_price"], t["exit_price"],
                                              t["pnl_points"], "開盤上衝觸發"))
        state["strategies"]["og"] = dict(status="flat", last_checked_date=run_date, as_of=run_date,
                                           last_trade_date=run_date, last_realized_pnl_points=float(t["pnl_points"]))
    else:
        mark_flat("og", "今天符合高波動體制，但08:45-09:00沒有偵測到進出場K棒（資料缺漏）")

    # lu / re
    lr_trades = bt_lunch_reversal(onemin, LunchReversalConfig())
    lu_trades = lr_trades[lr_trades["leg"] == "short_lunch_dip"] if not lr_trades.empty else lr_trades
    re_trades = lr_trades[lr_trades["leg"] == "long_afternoon_rebound"] if not lr_trades.empty else lr_trades

    if not lu_trades.empty:
        t = lu_trades.iloc[-1]
        events.append(_intraday_trade_event("lu", run_date, "short", t["entry_price"], t["exit_price"],
                                              t["pnl_points"], "午盤放空觸發"))
        state["strategies"]["lu"] = dict(status="flat", last_checked_date=run_date, as_of=run_date,
                                           last_trade_date=run_date, last_realized_pnl_points=float(t["pnl_points"]))
    else:
        mark_flat("lu", "今天符合高波動體制，但12:00-12:30沒有偵測到進出場K棒（資料缺漏）")

    if not re_trades.empty and re_vol_ok:
        t = re_trades.iloc[-1]
        events.append(_intraday_trade_event("re", run_date, "long", t["entry_price"], t["exit_price"],
                                              t["pnl_points"], "盤中翻多觸發"))
        state["strategies"]["re"] = dict(status="flat", last_checked_date=run_date, as_of=run_date,
                                           last_trade_date=run_date, last_realized_pnl_points=float(t["pnl_points"]))
    elif not re_trades.empty and not re_vol_ok:
        mark_flat("re", "今天符合高波動體制且12:30-13:00有K棒訊號，但量能濾網未通過(vol_ratio_lag1<1.099965)，不進場")
    else:
        mark_flat("re", "今天符合高波動體制，但12:30-13:00沒有偵測到進出場K棒（資料缺漏）")

    return events


def _intraday_trade_event(sid: str, run_date: str, direction: str, entry_price: float, exit_price: float,
                           pnl_points: float, note: str) -> dict:
    return dict(
        sim_date=run_date, strategy_id=sid, event="trade", direction=direction,
        sim_entry_price=float(entry_price), sim_exit_price=float(exit_price),
        sim_entry_date=run_date, real_entry_date=run_date, real_exit_date=run_date,
        realized_pnl_points=float(pnl_points), note=note,
    )


def compute_equity_row(state: dict, run_date: str) -> dict:
    total_realized = 0.0
    total_unrealized = 0.0
    if LOG_PATH.exists():
        log = pd.read_parquet(LOG_PATH)
        realized_events = log[log["event"].isin(["close", "trade"])]
        total_realized = realized_events["realized_pnl_points"].fillna(0).sum() * POINT_VALUE
    for sid, s in state["strategies"].items():
        if s.get("status") == "open":
            total_unrealized += s.get("unrealized_pnl_points", 0.0) * POINT_VALUE
    return dict(date=run_date, realized_pnl_twd=total_realized, unrealized_pnl_twd=total_unrealized,
                total_pnl_twd=total_realized + total_unrealized)


def append_equity(row: dict) -> None:
    new_df = pd.DataFrame([row])
    if EQUITY_PATH.exists():
        existing = pd.read_parquet(EQUITY_PATH)
        existing = existing[existing["date"] != row["date"]]  # 同一天重跑時覆蓋，不重複累加
        combined = pd.concat([existing, new_df], ignore_index=True).sort_values("date")
    else:
        combined = new_df
    combined.to_parquet(EQUITY_PATH, index=False)


def main() -> None:
    refresh_raw_data()
    state = load_state()
    run_date = today_str()
    if state["start_date"] is None:
        state["start_date"] = run_date
        print(f"首次執行，模擬倉追蹤起始日設為 {run_date}")

    all_events = []
    for sid, meta in TRACKED_STRATEGIES.items():
        events = process_strategy(sid, meta, state, run_date)
        all_events.extend(events)
        for e in events:
            print(f"[{sid}] {e['event']}: {e}")

    intraday_events = process_intraday_strategies(state, run_date)
    all_events.extend(intraday_events)
    for e in intraday_events:
        print(f"[{e['strategy_id']}] {e['event']}: {e}")

    append_log(all_events)
    state["last_run"] = run_date
    save_state(state)

    equity_row = compute_equity_row(state, run_date)
    append_equity(equity_row)
    print(f"\n{run_date} 模擬倉權益快照: {equity_row}")
    print(f"\n目前狀態:")
    for sid, s in state["strategies"].items():
        print(f"  {sid}: {s}")


if __name__ == "__main__":
    main()
