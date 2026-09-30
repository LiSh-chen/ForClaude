"""每日模擬倉（紙上交易）前瞻追蹤更新腳本。

只追蹤這次會話「已驗證」且資料源能每天更新的兩個日頻策略：
- low_vol_buyhold（低波動做多+8%停損）
- calendar_spread（近月/遠月價差均值回歸）

三腳組合（開盤上衝/午盤放空/盤中翻多）需要當天1分K才能算進出場，目前
沒有每日更新的1分K資料源，不在這個追蹤範圍內（見這次會話討論）。

**核心原則，誠實揭露**：
1. 這是模擬倉（紙上交易），不涉及真錢下單，不構成投資建議。
2. 「模擬權益曲線」只從系統第一次執行的那天開始算（見PAPER_TRADING_START_DATE
   ，寫在state檔的meta裡，只設一次），把系統啟用前的歷史回測損益也算進來會
   製造「這套系統一直都在賺錢」的假象——過去的回測結果已經在策略手冊/
   策略實驗室Artifact裡誠實揭露過了，這裡只記錄「從啟用那天起，如果真的
   照這個規則模擬操作，會發生什麼」。
3. 若系統啟用時策略剛好處於「進行中的部位」（例如低波動體制已經持續一段
   時間），會把這個部位視為「啟用當天新進場」記錄（用當天的近月價格當
   模擬進場價，不回溯用策略邏輯上真正的進場日期/價格）——這樣「模擬
   權益曲線」才會忠實反映「從今天開始跟著做，會有多少損益」，而不是把
   進場前就已經發生的價格變動也算成這套系統的績效。
4. 每次更新都會計算「未實現」浮動損益（部位還沒出場時，用最新收盤價
   估算），跟「已實現」損益分開記錄，前端會分別標示。

用法：
    python scripts/daily_paper_trading_update.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tw_quant.strategy_lab import (
    run_low_vol_buyhold, LowVolBuyHoldConfig,
    run_calendar_spread, CalendarSpreadConfig,
)
import fetch_taifex_tx_raw
import build_taifex_multi_contract

STATE_PATH = REPO_ROOT / "data" / "paper_trading_state.json"
LOG_PATH = REPO_ROOT / "data" / "paper_trading_log.parquet"
EQUITY_PATH = REPO_ROOT / "data" / "paper_trading_equity.parquet"
MULTI_CONTRACT_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"

POINT_VALUE = 50.0  # MTX點值（這裡先用小台，跟這次會話其餘策略手冊一致的口數假設）

TRACKED_STRATEGIES = {
    "low_vol_buyhold": dict(label="低波動做多+8%停損", run=run_low_vol_buyhold, cfg=LowVolBuyHoldConfig()),
    "calendar_spread": dict(label="近月/遠月價差均值回歸", run=run_calendar_spread, cfg=CalendarSpreadConfig()),
}


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


def compute_equity_row(state: dict, run_date: str) -> dict:
    total_realized = 0.0
    total_unrealized = 0.0
    if LOG_PATH.exists():
        log = pd.read_parquet(LOG_PATH)
        closes = log[log["event"] == "close"]
        total_realized = closes["realized_pnl_points"].fillna(0).sum() * POINT_VALUE
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
