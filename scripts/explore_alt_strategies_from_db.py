"""在真實資料上比較多種進場邏輯（同一套出場/風控/成本引擎），找正期望值策略。

背景：`explore_strategy_from_db.py` 對原始「壓縮濾網 + 60 日新高突破 + 爆量」
進場邏輯做了 180 組進出場參數搜尋，結果 0/180 組總報酬為正——勝率太低
（多落在 20%~40%）配不上風報比，EV 全部是負的。這代表問題不是某個參數沒
調好，而是「追突破」這個進場邏輯本身在這段真實資料上沒有 edge。

這支腳本改成比較 4 種不同的進場邏輯設計（出場仍統一用吊燈停利 + ATR 停損，
成本/風控引擎完全相同，只換進場觸發條件，做公平比較）：

  A. breakout   （基準對照組）原始「壓縮 + 60 日新高突破 + 爆量」
  B. pullback   拉回買進：多頭排列中價格拉回碰到短均線，隔日收紅站回均線才進場
                （目標：比追突破更早、更便宜的價位進場，拉高勝率）
  C. relative_strength  相對強度動能：橫斷面排名，只買 N 日報酬排名前段的股票，
                再要求價格創短期新高做確認（目標：只挑真正強勢股，減少假訊號）
  D. mean_reversion     超跌反彈：多頭排列的股票短線跌到近期低點，隔日收紅反彈
                就進場（目標：用更好的價位換取更高勝率，風報比會犧牲一些）

每種邏輯都做小網格搜尋（出場參數 atr_multiplier x chandelier_lookback，
加上各自邏輯特有的 1 個參數），列出每種邏輯的最佳結果，最後做總表比較。

用法：
    python scripts/explore_alt_strategies_from_db.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import StrategyConfig
from tw_quant.signals import build_pool_mask, generate_strategy_a_signals
from tw_quant.storage import get_data_store

MIN_TRADES_FOR_RANKING = 15

# 出場參數網格（所有策略共用，保持出場邏輯完全一致，公平比較進場邏輯本身）
ATR_MULT_GRID = (2.0, 2.5, 3.0)
LOOKBACK_GRID = (10, 15)


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


# ---------------------------------------------------------------------------
# 進場邏輯 A：原始突破（對照組，直接沿用 signals.py 既有實作）
# ---------------------------------------------------------------------------


def breakout_signal_fn(squeeze_pr: float):
    def _fn(master, prices, margin_short, cfg):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.squeeze.pr_threshold = squeeze_pr
        sig_a = generate_strategy_a_signals(master, local_cfg.pool, local_cfg.squeeze, local_cfg.ignition)
        return sig_a["entry_signal"].values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# 進場邏輯 B：拉回買進——多頭排列中拉回碰短均線，隔日收紅站回才進場
# ---------------------------------------------------------------------------


def pullback_signal_fn(short_ma_window: int):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)  # T-1：流動性 + 站上 60 日均線（多頭排列）

        ma_short = ind.sma(master, "close", short_ma_window)
        ma_short_prior = ind.shift_by_group(ma_short, master, periods=1)
        close_prior = ind.shift_by_group(master["close"], master, periods=1)

        # T-1 日：價格拉回到短均線之下（進入「拉回口袋」）
        pulled_back = (close_prior <= ma_short_prior).fillna(False)
        # T 日：收紅（比昨天高）且收盤重新站上短均線 -> 確認反彈
        confirm = (master["close"] > close_prior) & (master["close"] > ma_short)

        entry = pool & pulled_back & confirm.fillna(False)
        return entry.fillna(False).values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# 進場邏輯 C：相對強度動能——橫斷面排名前段 + 短期新高確認
# ---------------------------------------------------------------------------


def relative_strength_signal_fn(top_quantile: float, momentum_window: int = 60, confirm_window: int = 5):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)

        ret_n = master.groupby("stock_id", sort=False)["close"].pct_change(momentum_window)
        ret_n_prior = ind.shift_by_group(ret_n, master, periods=1)

        tmp = pd.DataFrame(
            {"date": master["date"].values, "ret": ret_n_prior.values}, index=master.index
        )
        pct_rank = tmp.groupby("date")["ret"].rank(pct=True, ascending=True)
        top_mask = (pct_rank >= (1 - top_quantile)).fillna(False)

        hh_confirm = ind.rolling_max(master, "high", confirm_window)
        hh_confirm_prior = ind.shift_by_group(hh_confirm, master, periods=1)
        confirm = master["close"] > hh_confirm_prior

        entry = pool & top_mask.values & confirm.fillna(False)
        return entry.fillna(False).values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# 進場邏輯 D：超跌反彈——多頭排列中跌到近期低點，隔日收紅反彈就進場
# ---------------------------------------------------------------------------


def mean_reversion_signal_fn(lookback: int):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)

        ll_n = ind.rolling_min(master, "low", lookback)
        ll_n_prior = ind.shift_by_group(ll_n, master, periods=1)
        close_prior = ind.shift_by_group(master["close"], master, periods=1)

        # T-1 日：收盤價貼近（2% 內）過去 N 日低點 -> 超跌口袋
        near_low = (close_prior <= ll_n_prior * 1.02).fillna(False)
        # T 日：收紅反彈確認
        confirm = master["close"] > close_prior

        entry = pool & near_low & confirm.fillna(False)
        return entry.fillna(False).values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# 共用：跑一次、算完整指標
# ---------------------------------------------------------------------------


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


def run_one(prices, margin_short, cfg, entry_signal_fn) -> dict:
    result = run_backtest(prices, margin_short, cfg, historical_mdd=None, entry_signal_fn=entry_signal_fn)
    metrics = summarize_performance(result, cfg.initial_capital)
    metrics["calmar"] = (metrics["cagr"] / metrics["max_dd"]) if metrics["max_dd"] > 0 else 0.0
    metrics.update(trade_stats(result.trades))
    return metrics


def _fmt_row(name: str, extra_label: str, extra_val, atr_mult, lookback, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{name:<17} {extra_label}={extra_val:<6} atr={atr_mult:<4.1f} lb={lookback:<3.0f}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>5.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


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
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    base_cfg = build_base_config()

    strategy_specs = [
        ("A_breakout", "sq_pr", (10, 30, 50), breakout_signal_fn),
        ("B_pullback", "ma", (5, 10, 20), pullback_signal_fn),
        ("C_rel_strength", "top_q", (0.10, 0.20, 0.30), relative_strength_signal_fn),
        ("D_mean_reversion", "near_low_win", (5, 10, 20), mean_reversion_signal_fn),
    ]

    header = (
        f"{'strategy':<17} {'param':<15} {'exit':<9}  "
        f"{'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
        f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
    )

    all_rows = []
    best_per_strategy = {}

    for name, param_label, param_grid, fn_builder in strategy_specs:
        print(f"\n=== 策略 {name}：搜尋 {len(param_grid) * len(ATR_MULT_GRID) * len(LOOKBACK_GRID)} 組合 ===")
        print(header)
        strat_rows = []
        for p in param_grid:
            entry_fn = fn_builder(p)
            for atr_mult in ATR_MULT_GRID:
                for lookback in LOOKBACK_GRID:
                    cfg = copy.deepcopy(base_cfg)
                    cfg.sizing.atr_multiplier = atr_mult
                    cfg.sizing.chandelier_lookback = lookback
                    m = run_one(prices, margin_short, cfg, entry_fn)
                    row = {
                        "strategy": name, "param_label": param_label, "param": p,
                        "atr_mult": atr_mult, "lookback": lookback,
                    }
                    row.update(m)
                    strat_rows.append(row)
                    all_rows.append(row)

        strat_df = pd.DataFrame(strat_rows)
        ranked = strat_df[strat_df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
        for _, r in ranked.head(5).iterrows():
            print(_fmt_row(name, param_label, r["param"], r["atr_mult"], r["lookback"], r.to_dict()))
        if ranked.empty:
            print(f"  （沒有組合達到最低 {MIN_TRADES_FOR_RANKING} 筆交易門檻）")
        else:
            best_per_strategy[name] = ranked.iloc[0].to_dict()

    df = pd.DataFrame(all_rows)
    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合（4 種策略總計）中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    print("\n\n=== 總表：各策略最佳組合比較 ===")
    print(header)
    for name, _, _, _ in strategy_specs:
        if name in best_per_strategy:
            r = best_per_strategy[name]
            print(_fmt_row(name, r["param_label"], r["param"], r["atr_mult"], r["lookback"], r))
        else:
            print(f"{name:<17} (無達門檻組合)")

    print(
        "\n（RR = 風報比 = 平均獲利% / 平均虧損%；EV% = 勝率加權後單筆期望報酬率；"
        "PF = 獲利因子 = 總獲利 / 總虧損；calmar = CAGR / MDD；出場統一用吊燈停利 + ATR 停損）"
    )


if __name__ == "__main__":
    main()
