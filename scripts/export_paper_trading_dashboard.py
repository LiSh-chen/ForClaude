"""把模擬倉追蹤系統的state/log/equity parquet轉成前端讀的單一json，供
web/paper_trading/index.html使用。

用法：
    python scripts/export_paper_trading_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "data" / "paper_trading_state.json"
LOG_PATH = REPO_ROOT / "data" / "paper_trading_log.parquet"
EQUITY_PATH = REPO_ROOT / "data" / "paper_trading_equity.parquet"
OUT_PATH = REPO_ROOT / "web" / "paper_trading" / "data.json"

STRATEGY_META = {
    "low_vol_buyhold": dict(
        kind="swing", label="低波動做多+8%停損",
        rule="用近月台指期，在ATR波動體制判定為「低波動」的期間做多；收盤價從進場後至今最高點回落8%即出場，同一段低波動週期內不重新進場。體制用前一交易日收盤已確定的版本，不是歷史回測用的同日版本。",
        historical_note="2001-2023歷史回測25筆交易，淨損益約+52.9萬元（單口小台，低成本情境）——這是過去的回測結果，不是這套模擬系統本身的績效。",
    ),
    "calendar_spread": dict(
        kind="swing", label="近月/遠月價差均值回歸",
        rule="近月/遠月期貨價差相對90天滾動平均的z分數，偏離超過1個標準差進場（押注價差回歸），回到0.3個標準差以內或最長持有20個交易日即出場；遠月未平倉量低於3000口時不進場（流動性濾網）。",
        historical_note="2001-2023歷史回測199筆交易，淨損益約+10.6萬元（低成本情境）——這是過去的回測結果，不是這套模擬系統本身的績效。",
    ),
    "og": dict(
        kind="intraday", label="開盤上衝",
        rule="08:45那根K棒開盤價做多，09:00那根K棒開盤價出場，只在高波動體制成立的交易日進場。",
        historical_note="2001-2023歷史回測（高波動體制篩選後）淨損益轉強，t值顯著——這是過去的回測結果，不是這套模擬系統本身的績效。",
    ),
    "lu": dict(
        kind="intraday", label="午盤放空",
        rule="12:00那根K棒開盤價放空，12:30那根K棒開盤價回補，只在高波動體制成立的交易日進場。",
        historical_note="2001-2023歷史回測（高波動體制篩選後）淨損益轉強——這是過去的回測結果，不是這套模擬系統本身的績效。",
    ),
    "re": dict(
        kind="intraday", label="盤中翻多",
        rule="12:30那根K棒開盤價做多，13:00那根K棒開盤價出場，只在高波動體制成立、且量能濾網(前一交易日成交量/20日均量>=1.10)通過的交易日進場。",
        historical_note="2001-2023歷史回測（高波動體制篩選後）淨損益轉強——這是過去的回測結果，不是這套模擬系統本身的績效。",
    ),
}


def main() -> None:
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"start_date": None, "strategies": {}}
    log = pd.read_parquet(LOG_PATH).sort_values("sim_date") if LOG_PATH.exists() else pd.DataFrame()
    equity = pd.read_parquet(EQUITY_PATH).sort_values("date") if EQUITY_PATH.exists() else pd.DataFrame()

    strategies_out = {}
    for sid, meta in STRATEGY_META.items():
        s = state["strategies"].get(sid, {"status": "flat"})
        strategies_out[sid] = {
            **meta,
            "status": s.get("status", "flat"),
            "direction": s.get("direction"),
            "sim_entry_date": s.get("sim_entry_date"),
            "sim_entry_price": s.get("sim_entry_price"),
            "unrealized_pnl_points": s.get("unrealized_pnl_points"),
            "last_realized_pnl_points": s.get("last_realized_pnl_points"),
            "last_trade_date": s.get("last_trade_date"),
            "note": s.get("note"),
            "as_of": s.get("as_of"),
        }

    log_out = []
    for _, r in log.iterrows():
        log_out.append({k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()})

    equity_out = []
    for _, r in equity.iterrows():
        equity_out.append(dict(date=r["date"], realized=round(float(r["realized_pnl_twd"]), 1),
                                unrealized=round(float(r["unrealized_pnl_twd"]), 1),
                                total=round(float(r["total_pnl_twd"]), 1)))

    payload = dict(
        start_date=state.get("start_date"), last_run=state.get("last_run"),
        strategies=strategies_out, log=log_out, equity=equity_out,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"saved: {OUT_PATH}")
    print(f"策略數: {len(strategies_out)}, 事件數: {len(log_out)}, 權益天數: {len(equity_out)}")


if __name__ == "__main__":
    main()
