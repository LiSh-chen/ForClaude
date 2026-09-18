"""用真實資料測試統計套利/配對交易策略（tw_quant/pairs_trading.py）。

對 formation_window（共整合檢定用的歷史窗格長度）跟 entry_z（進場門檻）
做網格搜尋，其餘參數固定在常見的實務值（每季換股一次、z-score 用 20 日
滾動窗格、|z|<0.5 平倉、|z|>4 停損、最長持有 20 個交易日）。套用完整交易
成本（含賣出證交稅，兩腿的做多/做空都算，見 pairs_trading.py 檔頭的
誠實揭露限制——尤其是「沒有算融券/借券費」這一點，真實報酬可能比這裡
回測出來的更差）。

用法：
    python scripts/test_pairs_trading_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.pairs_trading import PairsTradingConfig, run_pairs_trading_backtest
from tw_quant.storage import get_data_store

FORMATION_WINDOW_GRID = (120, 180, 252)
ENTRY_Z_GRID = (1.5, 2.0, 2.5)
MIN_TRADES_FOR_RANKING = 6  # 配對交易本來樣本就少（每季換股、又要通過共整合門檻），門檻降低一些


def build_base_config() -> StrategyConfig:
    return StrategyConfig()


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

    print("=== 配對交易回測（同產業共整合配對、z-score 進出場、完整交易成本）===")
    print(HEADER)

    rows = []
    for formation_window in FORMATION_WINDOW_GRID:
        for entry_z in ENTRY_Z_GRID:
            rt_cfg = PairsTradingConfig(formation_window=formation_window, entry_z=entry_z)
            result = run_pairs_trading_backtest(prices, base_cfg, rt_cfg)
            m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
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
        "n_trades 每組配對進出場都算 2 筆（A腿+B腿）；做空沒有算融券/借券費，也沒有台股融資融券"
        "資格清單，真實報酬可能比這裡回測出來的更差；每季（63個交易日）重新選一次配對）"
    )


if __name__ == "__main__":
    main()
