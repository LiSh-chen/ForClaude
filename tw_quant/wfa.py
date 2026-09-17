"""伍、WFA 滾動驗證與交易摩擦成本 —— 滾動視窗切分、參數穩定度檢查、大數檢驗。

每個訓練/測試視窗都是「獨立重跑」的全新回測（資金重置為 initial_capital），
避免跨視窗共用部位狀態污染樣本外測試的獨立性。切窗時額外往前多抓一段
「暖身期」資料餵給指標（min_history_days + PR 兩層 lookback），確保視窗一開始
魚池/濾網就有效，而不是要等視窗內部自己累積 250 天才開始出訊號。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import StrategyConfig


def _with_breakout_window(cfg: StrategyConfig, n: int) -> StrategyConfig:
    new_cfg = copy.deepcopy(cfg)
    new_cfg.ignition.breakout_window = n
    return new_cfg


def _relax_squeeze(cfg: StrategyConfig) -> StrategyConfig:
    """放寬 = 提高 PR 門檻（越高越多天數符合「震幅夠低」）。用 max() 而非直接覆蓋，
    避免使用者已自訂比預設更寬鬆的門檻時，被這一步意外收緊回 PR15。
    """
    new_cfg = copy.deepcopy(cfg)
    new_cfg.squeeze.pr_threshold = max(cfg.squeeze.pr_threshold, cfg.wfa.relaxed_squeeze_pr_threshold)
    return new_cfg


def _relax_breakout(cfg: StrategyConfig) -> StrategyConfig:
    """放寬 = 縮短突破窗口（越短越容易觸發突破）。同樣用 min() 避免意外收緊。"""
    new_cfg = copy.deepcopy(cfg)
    new_cfg.ignition.breakout_window = min(cfg.ignition.breakout_window, cfg.wfa.relaxed_breakout_window)
    return new_cfg


def generate_rolling_windows(all_dates: pd.Series, train_years: int, test_years: int) -> list[dict]:
    """依交易日曆切出 (train_start, train_end, test_start, test_end) 滾動視窗。

    以 252 個交易日近似一年（台股每年約 240~250 個交易日）。每完成一個視窗，
    往前步進一個測試窗長度，是標準 walk-forward 的滾動方式。
    """
    dates = pd.DatetimeIndex(sorted(pd.unique(all_dates)))
    trading_days_per_year = 252
    train_len = train_years * trading_days_per_year
    test_len = test_years * trading_days_per_year

    windows = []
    start_idx = 0
    while True:
        train_end_idx = start_idx + train_len
        test_end_idx = train_end_idx + test_len
        if test_end_idx > len(dates):
            break
        windows.append(
            {
                "train_start": dates[start_idx],
                "train_end": dates[train_end_idx - 1],
                "test_start": dates[train_end_idx],
                "test_end": dates[test_end_idx - 1],
            }
        )
        start_idx += test_len
    return windows


def _slice_with_warmup(
    prices: pd.DataFrame, margin_short: pd.DataFrame, window_start, window_end, warmup_days: int
):
    prior_dates = prices.loc[prices["date"] < window_start, "date"].sort_values().unique()
    warmup_start = prior_dates[-warmup_days] if len(prior_dates) >= warmup_days else prices["date"].min()
    p = prices[(prices["date"] >= warmup_start) & (prices["date"] <= window_end)]
    m = margin_short[(margin_short["date"] >= warmup_start) & (margin_short["date"] <= window_end)]
    return p, m


@dataclass
class WFAWindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_param: int
    train_metrics_by_param: dict
    test_metrics: dict


def run_wfa(
    prices: pd.DataFrame,
    margin_short: pd.DataFrame,
    base_cfg: StrategyConfig,
    metric_key: str = "sharpe",
) -> tuple[list[WFAWindowResult], list[str]]:
    """3 年訓練 + 1 年盲測滾動分析，回傳每一窗結果與過度擬合警報訊息列表。"""
    windows = generate_rolling_windows(prices["date"], base_cfg.wfa.train_years, base_cfg.wfa.test_years)
    warmup_days = base_cfg.pool.min_history_days + base_cfg.squeeze.pr_lookback + base_cfg.squeeze.range_window

    results: list[WFAWindowResult] = []
    for w in windows:
        train_metrics_by_param = {}
        for n in base_cfg.wfa.breakout_window_grid:
            cfg_n = _with_breakout_window(base_cfg, n)
            p_train, m_train = _slice_with_warmup(prices, margin_short, w["train_start"], w["train_end"], warmup_days)
            bt = run_backtest(p_train, m_train, cfg_n)
            train_metrics_by_param[n] = summarize_performance(bt, cfg_n.initial_capital)

        best_param = max(train_metrics_by_param, key=lambda n: train_metrics_by_param[n].get(metric_key, -np.inf))

        cfg_best = _with_breakout_window(base_cfg, best_param)
        p_test, m_test = _slice_with_warmup(prices, margin_short, w["test_start"], w["test_end"], warmup_days)
        bt_test = run_backtest(p_test, m_test, cfg_best)
        test_metrics = summarize_performance(bt_test, cfg_best.initial_capital)

        results.append(
            WFAWindowResult(
                train_start=w["train_start"],
                train_end=w["train_end"],
                test_start=w["test_start"],
                test_end=w["test_end"],
                best_param=best_param,
                train_metrics_by_param=train_metrics_by_param,
                test_metrics=test_metrics,
            )
        )

    alerts = detect_param_instability(results, base_cfg.wfa.param_variation_alert_pct)
    return results, alerts


def detect_param_instability(results: list[WFAWindowResult], tolerance_pct: float) -> list[str]:
    """相鄰兩個訓練窗選出的最佳參數變異幅度 > tolerance_pct，亮起過度擬合警報。"""
    alerts = []
    for i in range(len(results) - 1):
        p0, p1 = results[i].best_param, results[i + 1].best_param
        if p0 == 0:
            continue
        change = abs(p1 - p0) / abs(p0)
        if change > tolerance_pct:
            alerts.append(
                f"[過度擬合警報] 訓練窗 {results[i].train_start.date()}~{results[i].train_end.date()} "
                f"最佳 N={p0} -> 下一窗 {results[i + 1].train_start.date()}~{results[i + 1].train_end.date()} "
                f"最佳 N={p1}，變異 {change:.1%} > 容忍度 {tolerance_pct:.0%}"
            )
    return alerts


@dataclass
class LargeNumberCheckOutcome:
    applied_relaxations: list[str]
    final_cfg: StrategyConfig
    final_results: list[WFAWindowResult]
    final_alerts: list[str]
    avg_annual_trades: float


def run_wfa_with_large_number_check(
    prices: pd.DataFrame,
    margin_short: pd.DataFrame,
    base_cfg: StrategyConfig,
    metric_key: str = "sharpe",
) -> LargeNumberCheckOutcome:
    """大數檢驗：WFA 盲測階段平均年交易次數 < 門檻時，依序放寬
    (1) PR10 -> PR15  (2) 60 日突破 -> 40 日突破，直到滿足門檻或兩個方案都用盡。
    """
    applied: list[str] = []
    cfg = base_cfg
    results, alerts = run_wfa(prices, margin_short, cfg, metric_key)

    def avg_trades(rs: list[WFAWindowResult]) -> float:
        return float(np.mean([r.test_metrics.get("annual_trades", 0.0) for r in rs])) if rs else 0.0

    trades_avg = avg_trades(results)

    if trades_avg < base_cfg.wfa.min_annual_trades:
        before = cfg.squeeze.pr_threshold
        cfg = _relax_squeeze(cfg)
        note = f"PR{before:.0f} 放寬至 PR{cfg.squeeze.pr_threshold:.0f}" if cfg.squeeze.pr_threshold != before else f"PR{before:.0f} 已比建議值寬鬆，維持不變"
        applied.append(note)
        results, alerts = run_wfa(prices, margin_short, cfg, metric_key)
        trades_avg = avg_trades(results)

    if trades_avg < base_cfg.wfa.min_annual_trades:
        before = cfg.ignition.breakout_window
        cfg = _relax_breakout(cfg)
        note = f"突破窗口 {before} 日降為 {cfg.ignition.breakout_window} 日" if cfg.ignition.breakout_window != before else f"突破窗口 {before} 日已比建議值短，維持不變"
        applied.append(note)
        results, alerts = run_wfa(prices, margin_short, cfg, metric_key)
        trades_avg = avg_trades(results)

    return LargeNumberCheckOutcome(applied, cfg, results, alerts, trades_avg)
