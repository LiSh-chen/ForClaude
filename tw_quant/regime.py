"""壹、大盤環境判定與參數敏感度檢驗 (Regime Switch)

規格書的兩個條件本身有解讀空間，此處明確記錄採用的定義（README 亦有摘要）：

1. 市場寬度「20 日均線多頭排列家數比例」
   採：收盤價站上 20 日均線，且 20 日均線本身向上（5 日前的 MA20 更低）。
   只有兩者同時成立才算該股「多頭排列」。只計入資料已滿 min_history_days 的股票，
   避免剛上市、均線尚未穩定的個股扭曲比例。

2. 「大盤盤中預估總量」
   歷史回測沒有盤中資料，改用「當日全市場總成交金額」做為代理變數（較成交股數更能
   反映真實資金動能，且不受面額、拆股影響）。實盤上線時應改用真正的盤中預估總量
   （見 project_intraday_turnover），此處以 hook 方式預留。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from tw_quant.config import RegimeConfig
from tw_quant import indicators as ind


def compute_bullish_alignment(
    prices: pd.DataFrame,
    ma_window: int = 20,
    slope_lookback: int = 5,
    min_history_days: int = 252,
) -> pd.Series:
    """逐檔逐日判斷是否「20 日均線多頭排列」：close > MA20 且 MA20 向上。"""
    ma = ind.sma(prices, "close", ma_window)
    ma_prior = ind.shift_by_group(ma, prices, periods=slope_lookback)
    bars_seen = prices.groupby("stock_id", sort=False).cumcount() + 1
    eligible = bars_seen >= min_history_days
    aligned = (prices["close"] > ma) & (ma > ma_prior) & eligible
    return aligned.where(eligible, other=np.nan)


def compute_breadth_ratio(prices: pd.DataFrame, cfg: RegimeConfig, min_history_days: int) -> pd.Series:
    """回傳依日期彙總的市場寬度比例（0~1），index 為 date。"""
    aligned = compute_bullish_alignment(
        prices, ma_window=cfg.breadth_ma_window, min_history_days=min_history_days
    )
    tmp = pd.DataFrame({"date": prices["date"].values, "aligned": aligned.values})
    grouped = tmp.groupby("date")["aligned"]
    ratio = grouped.mean()  # NaN 自動被 pandas mean 排除在分母外
    return ratio.sort_index()


def compute_market_turnover_series(prices: pd.DataFrame) -> pd.Series:
    """全市場每日總成交金額，index 為 date。"""
    return prices.groupby("date")["turnover_value"].sum().sort_index()


def compute_volume_ratio(prices: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    """當日大盤總成交金額 / 過去 N 日均值，index 為 date。"""
    total = compute_market_turnover_series(prices)
    avg = total.rolling(cfg.volume_avg_window, min_periods=cfg.volume_avg_window).mean()
    return total / avg


def compute_regime_light(
    prices: pd.DataFrame,
    cfg: RegimeConfig,
    min_history_days: int,
    breadth_threshold: float | None = None,
    volume_ratio_threshold: float | None = None,
) -> pd.DataFrame:
    """回傳每日的綠燈/紅燈判定表：breadth_ratio, volume_ratio, is_green。

    兩條件皆須達標才亮綠燈；任一不達標，強制暫停新增部位（紅燈）。
    """
    breadth_threshold = cfg.breadth_threshold if breadth_threshold is None else breadth_threshold
    volume_ratio_threshold = (
        cfg.volume_ratio_threshold if volume_ratio_threshold is None else volume_ratio_threshold
    )

    breadth = compute_breadth_ratio(prices, cfg, min_history_days)
    vol_ratio = compute_volume_ratio(prices, cfg)

    out = pd.DataFrame({"breadth_ratio": breadth, "volume_ratio": vol_ratio}).sort_index()
    out["is_green"] = (out["breadth_ratio"] > breadth_threshold) & (
        out["volume_ratio"] > volume_ratio_threshold
    )
    out["is_green"] = out["is_green"].fillna(False)
    return out


def project_intraday_turnover(cumulative_turnover_so_far: float, elapsed_minutes: float, session_minutes: float = 270.0) -> float:
    """實盤用：以線性外推法，將盤中至今累計成交金額推估為全日預估總量。

    台股一般交易時段為 09:00-13:30，共 270 分鐘。這是最單純的線性外推，
    實務上可再用日內量能曲線（U 型分佈）做加權校正，此處僅提供介面掛鉤。
    """
    if elapsed_minutes <= 0:
        return 0.0
    return cumulative_turnover_so_far * (session_minutes / elapsed_minutes)


# ---------------------------------------------------------------------------
# 參數敏感度檢驗
# ---------------------------------------------------------------------------


@dataclass
class SensitivityResult:
    breadth_threshold: float
    volume_threshold: float
    metrics: dict


def run_regime_sensitivity_grid(
    cfg: RegimeConfig,
    backtest_fn: Callable[[float, float], dict],
    breadth_grid: tuple[float, ...] | None = None,
    volume_grid: tuple[float, ...] | None = None,
) -> list[SensitivityResult]:
    """對 breadth/volume 門檻做網格測試。

    backtest_fn(breadth_threshold, volume_threshold) -> 績效指標 dict
    （例如 {'total_return':..., 'sharpe':..., 'max_dd':..., 'n_trades':...}）
    由呼叫端注入完整回測流程，本函式只負責跑網格與蒐集結果，
    不預設任何回測實作，避免模組間循環相依。
    """
    breadth_grid = breadth_grid or cfg.breadth_sensitivity_grid
    volume_grid = volume_grid or cfg.volume_sensitivity_grid

    results = []
    for b in breadth_grid:
        for v in volume_grid:
            metrics = backtest_fn(b, v)
            results.append(SensitivityResult(b, v, metrics))
    return results


def detect_performance_cliff(
    results: list[SensitivityResult], metric: str = "total_return", tolerance_pct: float = 0.30
) -> list[str]:
    """比較網格中相鄰參數點的績效落差；若鄰近點差異超過 tolerance_pct，
    視為「踩在績效懸崖上」，回傳警示訊息列表。
    """
    warnings: list[str] = []
    by_breadth: dict[float, list[SensitivityResult]] = {}
    for r in results:
        by_breadth.setdefault(r.breadth_threshold, []).append(r)

    breadth_values = sorted(by_breadth)
    for i in range(len(breadth_values) - 1):
        b0, b1 = breadth_values[i], breadth_values[i + 1]
        rows0 = {r.volume_threshold: r.metrics.get(metric, np.nan) for r in by_breadth[b0]}
        rows1 = {r.volume_threshold: r.metrics.get(metric, np.nan) for r in by_breadth[b1]}
        for v in set(rows0) & set(rows1):
            m0, m1 = rows0[v], rows1[v]
            if m0 in (0, None) or np.isnan(m0):
                continue
            change = abs(m1 - m0) / abs(m0)
            if change > tolerance_pct:
                warnings.append(
                    f"[懸崖警報] breadth {b0}->{b1} (volume={v}): {metric} 變化 {change:.1%} > 容忍度 {tolerance_pct:.0%}"
                )
    return warnings
