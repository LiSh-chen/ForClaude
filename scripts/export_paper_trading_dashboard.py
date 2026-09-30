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
MULTI_CONTRACT_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
OUT_PATH = REPO_ROOT / "web" / "paper_trading" / "data.json"

POINT_VALUE = 50.0

# TAIFEX官方保證金一覽表(股價指數類)查證：https://www.taifex.com.tw/cht/5/indexMarging
# 頁面標示「更新日期：2026/08/12」，這裡用MTX(小型臺指期貨)。保證金會隨市場
# 波動不定期調整，這裡的常數只反映查證當下的公告，不是每天自動更新——之後
# 若TAIFEX調整，這裡要手動跟著改，暫時沒有每日自動抓保證金公告的管道。
MTX_MARGIN_ORIGINAL = 175250
MTX_MARGIN_MAINTENANCE = 134500
MTX_MARGIN_SETTLEMENT = 129750
MARGIN_BUFFER_POINTS = (MTX_MARGIN_ORIGINAL - MTX_MARGIN_MAINTENANCE) / POINT_VALUE  # 815點
LOW_VOL_STOP_PCT = 0.08

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


def compute_margin_info(state: dict) -> dict:
    """真實資金操作會用到的保證金資訊：現在的官方保證金水準、目前(收盤後)
    實際佔用幾口/多少保證金、以及低波動做多8%停損 vs 保證金緩衝的結構性
    風險提示(見test_margin_aware_stoploss.py的發現：現在指數已經上漲到
    2001年進場時的好幾倍，8%停損換算的點數距離遠超過815點的標準保證金
    緩衝，照最低保證金操作會先被追繳/砍倉，根本撐不到8%停損真正觸發)。"""
    open_swing = [sid for sid in ("low_vol_buyhold", "calendar_spread")
                  if state["strategies"].get(sid, {}).get("status") == "open"]
    # intraday(og/lu/re)三腳彼此時段不重疊(08:45-09:00/12:00-12:30/12:30-13:00)
    # ，同一時刻最多1口在動；收盤後(這個資訊產生的時間點)必然已經平倉，
    # 不計入「目前」佔用，但納入「理論最大併發」的估計。
    current_lots = len(open_swing)
    current_margin_original = current_lots * MTX_MARGIN_ORIGINAL
    current_margin_maintenance = current_lots * MTX_MARGIN_MAINTENANCE
    theoretical_max_lots = 2 + 1  # 2個swing都open + 1個intraday同時觸發

    low_vol_state = state["strategies"].get("low_vol_buyhold", {})
    low_vol_risk = None
    if low_vol_state.get("status") == "open" and low_vol_state.get("sim_entry_price"):
        entry_price = float(low_vol_state["sim_entry_price"])
    else:
        mc = pd.read_parquet(MULTI_CONTRACT_PATH)
        entry_price = float(mc.sort_values("date").iloc[-1]["near_price"])
    stop_points = entry_price * LOW_VOL_STOP_PCT
    stop_amount = stop_points * POINT_VALUE
    buffer_ratio = stop_points / MARGIN_BUFFER_POINTS
    # 要撐到8%停損真正觸發都不被追繳，帳戶權益必須在價格反轉8%時仍然
    # >= 維持保證金，所以每口要準備：維持保證金 + 8%停損對應的虧損金額
    recommended_capital_per_lot = MTX_MARGIN_MAINTENANCE + stop_amount
    low_vol_risk = dict(
        reference_price=round(entry_price, 1), stop_points=round(stop_points, 1),
        stop_amount=round(stop_amount, 1), buffer_ratio=round(buffer_ratio, 2),
        recommended_capital_per_lot=round(recommended_capital_per_lot, 1),
        is_live_position=(low_vol_state.get("status") == "open"),
    )

    return dict(
        margin_original=MTX_MARGIN_ORIGINAL, margin_maintenance=MTX_MARGIN_MAINTENANCE,
        margin_settlement=MTX_MARGIN_SETTLEMENT, margin_buffer_points=MARGIN_BUFFER_POINTS,
        current_lots=current_lots, current_margin_original=current_margin_original,
        current_margin_maintenance=current_margin_maintenance,
        theoretical_max_lots=theoretical_max_lots,
        theoretical_max_margin=theoretical_max_lots * MTX_MARGIN_ORIGINAL,
        low_vol_risk=low_vol_risk,
    )


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

    margin_info = compute_margin_info(state)

    payload = dict(
        start_date=state.get("start_date"), last_run=state.get("last_run"),
        strategies=strategies_out, log=log_out, equity=equity_out, margin=margin_info,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"saved: {OUT_PATH}")
    print(f"策略數: {len(strategies_out)}, 事件數: {len(log_out)}, 權益天數: {len(equity_out)}")


if __name__ == "__main__":
    main()
