"""改用「獨立資金池」的方式做策略混合，取代上一輪失敗的「共用資金池+全域風控」
做法。

上一輪教訓：C（結構轉折）+ D（超跌反彈）共用同一個資金池、同一個全域風控
（6% 單日新增曝險上限、產業曝險上限）一起跑，Sharpe 從 0.43 衝到 0.60，
但穩健度檢查一戳就破——D 的 near_low_win 只要從 5 移到 10，Sharpe 就崩到
0.13，再移到 15/20 直接變負的。懷疑原因：兩個訊號家族共用同一個全域風控，
會互相搶「同一份」單日風險預算跟部位額度，兩者的候選誰先出現在候選清單裡
（依當日成交金額排序）、誰被擠掉，會產生跟訊號本身好壞無關、卻高度影響
結果的路徑依賴雜訊。

這裡改成機構常見的「獨立資金池（sleeve）」做法：每個策略拿自己那一份本金
獨立跑，互不干擾、互不搶額度，事後只是把兩條權益曲線加總起來看整體組合
的波動度有沒有被分散掉。這樣測出來的「分散化效果」才是乾淨的、不會被
資金排隊的雜訊污染。

測兩組：
  C + D 獨立資金池（重測上一輪失敗的組合，看換個混合方式結果會不會不同）
  C + E 獨立資金池（結構轉折 vs 動能因子組合，兩種完全不同引擎的策略）

用法：
    python scripts/blend_independent_capital_from_db.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant.backtest import BacktestResult, run_backtest
from tw_quant.backtest_stats import build_combined_trades, trade_stats
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.signals import build_pool_mask
from tw_quant.storage import get_data_store

BEST_DT_WIN = 20
BEST_SW_WIN = 15
BEST_VOL_X = 1.3
BEST_ATR_MULT = 2.5
BEST_LOOKBACK = 10

BEST_E_MOMENTUM_WINDOW = 120
BEST_E_REBALANCE_FREQ = 63
BEST_E_TOP_N = 15

MIN_TRADES_FOR_RANKING = 15


def build_base_config(initial_capital: float) -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    cfg.sizing.atr_multiplier = BEST_ATR_MULT
    cfg.sizing.chandelier_lookback = BEST_LOOKBACK
    cfg.initial_capital = initial_capital
    return cfg


def mss_mask(master: pd.DataFrame) -> pd.Series:
    low_min_now = ind.rolling_min(master, "low", BEST_DT_WIN)
    low_min_earlier = ind.shift_by_group(low_min_now, master, periods=BEST_DT_WIN)
    low_min_now_prior = ind.shift_by_group(low_min_now, master, periods=1)
    low_min_earlier_prior = ind.shift_by_group(low_min_earlier, master, periods=1)
    downtrend_context = (low_min_now_prior < low_min_earlier_prior).fillna(False)

    swing_high = ind.rolling_max(master, "high", BEST_SW_WIN)
    swing_high_prior = ind.shift_by_group(swing_high, master, periods=1)
    breaks_structure = (master["close"] > swing_high_prior).fillna(False)

    avg_vol = ind.sma(master, "volume", 20)
    avg_vol_prior = ind.shift_by_group(avg_vol, master, periods=1)
    volume_ok = (master["volume"] > avg_vol_prior * BEST_VOL_X).fillna(False)

    return downtrend_context & breaks_structure & volume_ok


def mean_reversion_mask(master: pd.DataFrame, lookback: int) -> pd.Series:
    ll_n = ind.rolling_min(master, "low", lookback)
    ll_n_prior = ind.shift_by_group(ll_n, master, periods=1)
    close_prior = ind.shift_by_group(master["close"], master, periods=1)
    near_low = (close_prior <= ll_n_prior * 1.02).fillna(False)
    confirm = (master["close"] > close_prior).fillna(False)
    return near_low & confirm


def mss_signal_fn(master, prices, margin_short, cfg):
    pool = build_pool_mask(master, cfg.pool)
    entry = pool & mss_mask(master)
    return entry.values, np.zeros(len(master), dtype=bool)


def mean_reversion_signal_fn(near_low_win: int):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        entry = pool & mean_reversion_mask(master, near_low_win)
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# 獨立資金池混合：分開跑、加總權益曲線
# ---------------------------------------------------------------------------


def metrics_from_equity(equity: pd.Series, initial_capital_total: float) -> dict:
    total_return = equity.iloc[-1] / initial_capital_total - 1
    n_days = len(equity)
    years = max(n_days / 252, 1e-9)
    cagr = (equity.iloc[-1] / initial_capital_total) ** (1 / years) - 1

    running_peak = equity.cummax()
    drawdown = (running_peak - equity) / running_peak
    max_dd = drawdown.max()

    daily_ret = equity.pct_change().dropna()
    sharpe = 0.0
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    return {"total_return": total_return, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar}


def blend_results(results: list[tuple[BacktestResult, float]], prices: pd.DataFrame) -> dict:
    """results: [(單一策略的 BacktestResult, 該策略分到的本金), ...]，
    彼此獨立跑（不同 cfg.initial_capital，各自互不干擾），這裡只是加總權益曲線
    跟交易紀錄，不重跑回測。
    """
    combined_equity = None
    all_trades = []
    total_capital = 0.0

    for result, capital in results:
        eq = result.equity_curve["equity"]
        combined_equity = eq if combined_equity is None else combined_equity.add(eq, fill_value=eq.iloc[0])
        total_capital += capital
        all_trades.append(build_combined_trades(result, prices))

    m = metrics_from_equity(combined_equity, total_capital)
    combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    m.update(trade_stats(combined_trades))
    m["n_trades"] = len(combined_trades)
    return m


HEADER = (
    f"{'label':<40}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, m: dict) -> str:
    pf = m.get("profit_factor", 0.0)
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m.get("risk_reward_ratio", 0.0)
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<40}  {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
        f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f}  {m.get('n_trades', 0):>5.0f} "
        f"{m.get('win_rate', 0.0):>6.1%} {rr_str} {m.get('ev_pct', 0.0):>7.2%} {pf_str}"
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

    total_capital = 10_000_000.0
    half_capital = total_capital / 2

    # C 單獨跑（全額本金，當對照組）
    cfg_c_full = build_base_config(total_capital)
    result_c_full = run_backtest(prices, margin_short, cfg_c_full, historical_mdd=None, entry_signal_fn=mss_signal_fn)
    m_c_full = blend_results([(result_c_full, total_capital)], prices)
    print("=== 對照組：C 單獨（全額本金）===")
    print(HEADER)
    print(_fmt_row("C 單獨", m_c_full))

    # C 半額 + D 半額（獨立資金池），D 的 near_low_win 做穩健度檢查
    print("\n=== C + D 獨立資金池（各半本金），D 的 near_low_win 穩健度檢查 ===")
    print(HEADER)
    cfg_c_half = build_base_config(half_capital)
    result_c_half = run_backtest(prices, margin_short, cfg_c_half, historical_mdd=None, entry_signal_fn=mss_signal_fn)

    for near_low_win in (5, 10, 15, 20):
        cfg_d_half = build_base_config(half_capital)
        result_d_half = run_backtest(
            prices, margin_short, cfg_d_half, historical_mdd=None,
            entry_signal_fn=mean_reversion_signal_fn(near_low_win),
        )
        m = blend_results([(result_c_half, half_capital), (result_d_half, half_capital)], prices)
        print(_fmt_row(f"C+D 獨立資金池 (D_win={near_low_win})", m))

    # C 半額 + E 半額（獨立資金池，跨引擎）
    print("\n=== C + E 獨立資金池（各半本金，跨引擎：逐日事件 + 調倉組合）===")
    print(HEADER)
    cfg_e_half = build_base_config(half_capital)
    factor_cfg = FactorConfig(
        momentum_window=BEST_E_MOMENTUM_WINDOW, rebalance_freq_days=BEST_E_REBALANCE_FREQ, top_n=BEST_E_TOP_N
    )
    result_e_half = run_factor_backtest(prices, cfg_e_half, factor_cfg)
    m_e_only = blend_results([(result_e_half, half_capital)], prices)
    print(_fmt_row("E 單獨（半額本金，對照）", m_e_only))
    m_cd_blend = blend_results([(result_c_half, half_capital), (result_e_half, half_capital)], prices)
    print(_fmt_row("C+E 獨立資金池", m_cd_blend))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "n_trades/win%/RR/EV%/PF 已把回測結束時還未平倉的部位用最後收盤價算進去；"
        "獨立資金池 = 每個策略拿自己的本金獨立跑，不共用全域風控，事後只加總權益曲線）"
    )


if __name__ == "__main__":
    main()
