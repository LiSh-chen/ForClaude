"""把配對交易/統計套利策略（tw_quant/pairs_trading.py）重新套用在美股
S&P 500 資料上，跟台股版（scripts/test_pairs_trading_from_db.py，結果是
否證：9 組參數全部負報酬，最佳 Sharpe -0.73）做同樣的網格搜尋，驗證
「換一個流動性/微結構都不同的市場，這個策略類型是不是還是不管用」。

跟台股版唯一的差異：資料來源換成 us_prices，成本模型/整股限制換成美股版
（tw_quant/us_config.py + tw_quant/us_costs.py）。其餘網格、配對邏輯、
z-score 進出場規則完全相同。

沿用台股版檔頭誠實揭露的限制（做空只算了名目上的稅/手續費，沒有算真實
借券費/保證金利息；配對只在同產業內找；避險比例形成期內凍結不重估）——
這些限制對美股版一樣成立。

用法：
    python scripts/test_us_pairs_trading_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.pairs_trading import PairsTradingConfig, run_pairs_trading_backtest
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config

FORMATION_WINDOW_GRID = (120, 180, 252)
ENTRY_Z_GRID = (1.5, 2.0, 2.5)
MIN_TRADES_FOR_RANKING = 6

HEADER = (
    f"{'form_win':>9} {'entry_z':>8}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(formation_window: int, entry_z: float, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{formation_window:>9} {entry_z:>8.1f}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    store = get_data_store()
    us_prices = store.load_us_prices()

    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = us_prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔美股的資料"
        f"（{us_prices['date'].min().date()} ~ {us_prices['date'].max().date()}）\n"
    )

    base_cfg = build_us_config()

    print("=== 配對交易回測（美股 S&P 500、同產業共整合配對、z-score 進出場、完整交易成本）===")
    print(HEADER)

    rows = []
    for formation_window in FORMATION_WINDOW_GRID:
        for entry_z in ENTRY_Z_GRID:
            rt_cfg = PairsTradingConfig(formation_window=formation_window, entry_z=entry_z)
            result = run_pairs_trading_backtest(us_prices, base_cfg, rt_cfg, cost_module=us_costs)
            m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
            row = {"formation_window": formation_window, "entry_z": entry_z}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["formation_window"]), r["entry_z"], r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")
        print(df.to_string(index=False))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "n_trades 每組配對進出場都算 2 筆（A腿+B腿）；做空沒有算真實借券費/保證金利息；"
        "每季（63個交易日）重新選一次配對；台股版對照：9 組全部負報酬，最佳 Sharpe -0.73）"
    )


if __name__ == "__main__":
    main()
