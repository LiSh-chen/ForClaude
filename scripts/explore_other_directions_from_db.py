"""跳出「找進場時機點」的思路，測試兩個結構上不同的方向：

  E. 月/季調倉動能因子組合（tw_quant/factor_backtest.py）
     不逐日找訊號、不用個股停損停利，改成定期對魚池依落後報酬排名，
     等權重持有前 N 檔到下次調倉才換股。傳統因子投資範式，跟前面測試過的
     4 種「技術面觸發點」邏輯完全不同的典範。

  F. 規格書策略 B（軋空異常突破）單獨測試
     前幾輪的比較都只換了策略 A 的進場邏輯、策略 B 一直是關閉的（entry_signal_b
     全設 False）。策略 B 用的是籌碼面資料（券資比異常），經濟邏輯跟純價量
     動能完全不同（軋空機制），值得單獨測試，而不是隱藏在從未被測過的角落。

用法：
    python scripts/explore_other_directions_from_db.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.signals import generate_strategy_b_signals
from tw_quant.storage import get_data_store

MIN_TRADES_FOR_RANKING = 15
ATR_MULT_GRID = (2.0, 2.5, 3.0)
LOOKBACK_GRID = (10, 15)


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "risk_reward_ratio": 0.0,
            "ev_pct": 0.0, "profit_factor": 0.0, "avg_holding_days": 0.0,
        }
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades)
    avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0.0
    avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0.0
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    risk_reward_ratio = (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct < 0 else float("inf")
    ev_pct = win_rate * avg_win_pct + (1 - win_rate) * avg_loss_pct
    holding_days = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days
    return {
        "avg_win_pct": avg_win_pct, "avg_loss_pct": avg_loss_pct,
        "risk_reward_ratio": risk_reward_ratio, "ev_pct": ev_pct,
        "profit_factor": profit_factor, "avg_holding_days": holding_days.mean(),
    }


def metrics_from_result(result, initial_capital) -> dict:
    m = summarize_performance(result, initial_capital)
    m["calmar"] = (m["cagr"] / m["max_dd"]) if m["max_dd"] > 0 else 0.0
    m.update(trade_stats(result.trades))
    return m


def _fmt_row(name: str, extra_label: str, extra_val, exit_label: str, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{name:<12} {extra_label}={str(extra_val):<8} {exit_label:<14}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>5.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


HEADER = (
    f"{'strategy':<12} {'param':<13} {'exit':<14}  "
    f"{'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def explore_factor(prices: pd.DataFrame, base_cfg: StrategyConfig) -> pd.DataFrame:
    print("\n=== E. 月/季調倉動能因子組合 ===")
    print(HEADER)
    rows = []
    for momentum_window in (40, 60, 90, 120):
        for rebalance_freq_days in (21, 63):
            for top_n in (15, 20, 30):
                factor_cfg = FactorConfig(
                    momentum_window=momentum_window,
                    rebalance_freq_days=rebalance_freq_days,
                    top_n=top_n,
                )
                result = run_factor_backtest(prices, base_cfg, factor_cfg)
                m = metrics_from_result(result, base_cfg.initial_capital)
                label = f"mw{momentum_window}/rb{rebalance_freq_days}/n{top_n}"
                row = {"strategy": "E_factor", "param": label, "exit": "n/a(調倉)"}
                row.update(m)
                rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(8).iterrows():
        print(_fmt_row("E_factor", "param", r["param"], "n/a(調倉)", r.to_dict()))
    if ranked.empty:
        print(f"  （沒有組合達到最低 {MIN_TRADES_FOR_RANKING} 筆交易門檻）")
    return df


def strategy_b_alone_signal_fn(margin_short_ratio_pr_threshold: float):
    def _fn(master, prices, margin_short, cfg):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.strategy_b.margin_short_ratio_pr_threshold = margin_short_ratio_pr_threshold
        sig_b = generate_strategy_b_signals(master, margin_short, local_cfg.pool, local_cfg.ignition, local_cfg.strategy_b)
        return np.zeros(len(master), dtype=bool), sig_b["entry_signal"].values

    return _fn


def explore_strategy_b(prices: pd.DataFrame, margin_short: pd.DataFrame, base_cfg: StrategyConfig) -> pd.DataFrame:
    print("\n=== F. 規格書策略 B（軋空異常突破）單獨測試 ===")
    print(HEADER)
    rows = []
    for pr_threshold in (80, 85, 90, 95):
        entry_fn = strategy_b_alone_signal_fn(pr_threshold)
        for atr_mult in ATR_MULT_GRID:
            for lookback in LOOKBACK_GRID:
                cfg = copy.deepcopy(base_cfg)
                cfg.sizing.atr_multiplier = atr_mult
                cfg.sizing.chandelier_lookback = lookback
                result = run_backtest(prices, margin_short, cfg, historical_mdd=None, entry_signal_fn=entry_fn)
                m = metrics_from_result(result, cfg.initial_capital)
                label = f"pr{pr_threshold}"
                exit_label = f"atr={atr_mult}/lb={lookback}"
                row = {"strategy": "F_strategy_b", "param": label, "exit": exit_label}
                row.update(m)
                rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(8).iterrows():
        print(_fmt_row("F_strategy_b", "param", r["param"], r["exit"], r.to_dict()))
    if ranked.empty:
        print(f"  （沒有組合達到最低 {MIN_TRADES_FOR_RANKING} 筆交易門檻）")
    return df


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin_short = store.load_margin_short()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）"
    )

    base_cfg = build_base_config()

    df_e = explore_factor(prices, base_cfg)
    df_f = explore_strategy_b(prices, margin_short, base_cfg)

    all_df = pd.concat([df_e, df_f], ignore_index=True)
    n_total = len(all_df)
    n_profitable = (all_df["total_return"] > 0).sum()
    print(f"\n兩個方向合計 {n_total} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / n_total:.1%}）")

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；"
        "calmar = CAGR / MDD；E 策略沒有個股停損停利，出場只在下次調倉時發生）"
    )


if __name__ == "__main__":
    main()
