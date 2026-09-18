"""用真實資料測試「開盤跳空、收盤前評估出場」的同日回補策略。

背景：analyze_calendar_effects_from_db.py 量到「當天開盤跳空 -> 當天盤中報酬
（收盤vs開盤）」是整個專案裡最強的統計規律（mean IC=-0.25, t=-54.82），但因為
訊號跟可交易時機都在同一天之內，原本「T 日收盤算訊號、T+1 日開盤進場」的逐日
引擎架構上無法交易它。tw_quant/intraday_roundtrip.py 是專門為這個訊號寫的
「當沖」回測引擎：T 日開盤看到跳空幅度夠大就買進，T 日收盤全部平倉，訊號跟
執行都在同一天完成，不依賴未來資訊、也不需要盤中 tick 資料（日線的開盤價/
收盤價就夠了）。

這裡對 gap_threshold（跳空幅度門檻）跟 top_n（每天最多買幾檔）做網格搜尋，
套用完整交易成本（含當沖沿用系統一貫的標準 0.3% 證交稅——保守假設，實際
若有當沖優惠稅率，真實報酬會更好，見 intraday_roundtrip.py 檔頭說明的
三個誠實揭露的限制），印出完整報酬/風險指標排名。

用法：
    python scripts/test_intraday_gap_fade_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.intraday_roundtrip import IntradayRoundtripConfig, run_intraday_roundtrip_backtest
from tw_quant.storage import get_data_store

GAP_THRESHOLD_GRID = (0.01, 0.02, 0.03, 0.05, 0.07)
TOP_N_GRID = (5, 10, 15, 20, 30)
MIN_TRADES_FOR_RANKING = 15


def build_base_config() -> StrategyConfig:
    # 沿用目前找到最有效的魚池設定（流動性 + 站上 60 日均線多頭排列，
    # 放寬的 regime 篩選門檻），跟其他 explore/test 腳本一致
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    return cfg


HEADER = (
    f"{'gap_thr':>8} {'top_n':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(gap_threshold: float, top_n: int, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{gap_threshold:>7.1%} {top_n:>6}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    base_cfg = build_base_config()

    print("=== 同日跳空回補回測（T 日開盤買進、T 日收盤賣出，完整交易成本）===")
    print(HEADER)

    rows = []
    for gap_threshold in GAP_THRESHOLD_GRID:
        for top_n in TOP_N_GRID:
            rt_cfg = IntradayRoundtripConfig(gap_threshold=gap_threshold, top_n=top_n)
            result = run_intraday_roundtrip_backtest(prices, base_cfg, rt_cfg)
            m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
            row = {"gap_threshold": gap_threshold, "top_n": top_n}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(r["gap_threshold"], int(r["top_n"]), r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "每筆交易當天開盤進、當天收盤出，不留倉，所以沒有「還未平倉部位」需要額外處理；"
        "證交稅沿用標準 0.3% 保守假設，若當沖優惠稅率生效中，實際報酬會更好；"
        "回測沒有台股當沖資格清單，假設候選股都能當沖）"
    )


if __name__ == "__main__":
    main()
