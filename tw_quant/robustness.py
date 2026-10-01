"""績效穩健性檢定工具：用移動區塊拔靴法（moving-block bootstrap）估計
Sharpe ratio 的抽樣分布，而不是只看單一點估計。

背景（2026-10-01）：168 組參數網格顯示 top_n=2/momentum_window=126 這個
組合的樣本外 Sharpe 對調倉頻率（rebalance_freq_days）極度敏感——21 天
調倉只有 Sharpe 0.57（略輸 QQQ 的 0.69），126 天調倉卻有 0.83（看似贏
QQQ）；但 126 天調倉在約 5 年的樣本外窗格裡只交易了 20 次。20 筆交易
算出來的 Sharpe，抽樣誤差可能大到讓「0.83 贏過 QQQ」這個結論根本站不
住腳——必須量化這個不確定性，不能只看點估計就下結論。

移動區塊拔靴法：把每日報酬序列切成長度 block_size 的重疊區塊，隨機抽取
區塊拼回原長度的新序列，重複很多次、每次都重算年化 Sharpe，得到 Sharpe
估計值本身的抽樣分布——比直接假設報酬為常態分布、用解析公式算標準誤更
穩健，因為交易策略的每日報酬有自相關（同一組持股會連續好幾天、調倉前後
報酬不獨立）跟肥尾，獨立抽樣（iid bootstrap）會低估真正的不確定性。
block_size 預設抓跟調倉頻率同量級的天數，讓每個區塊內部盡量包含完整的
「持倉到調倉」週期。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_sharpe(daily_returns: np.ndarray, periods_per_year: int = 252) -> float:
    """年化 Sharpe（無風險利率設為 0，跟 tw_quant/backtest_stats.py 的
    metrics_from_result 一致）。報酬序列是常數（標準差為 0）時回傳 NaN，
    避免除以零產生 inf 污染後續的拔靴分布統計。
    """
    std = daily_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(daily_returns.mean() / std * np.sqrt(periods_per_year))


def moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """產生一組長度 n 的重抽樣索引：用重疊區塊（block_size 天為一段，起點
    在 [0, n-block_size] 間均勻隨機抽取）接龍拼到至少長度 n，保留區塊內
    的自相關結構，最後截斷成剛好 n 筆。block_size 必須 <= n。
    """
    if block_size > n:
        raise ValueError(f"block_size（{block_size}）不能大於序列長度（{n}）")
    n_blocks = -(-n // block_size)  # ceil div
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n]


def bootstrap_sharpe_ci(
    daily_returns: pd.Series,
    block_size: int = 21,
    n_boot: int = 5000,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict:
    """回傳點估計 Sharpe、拔靴分布的 95% 信賴區間（2.5%/97.5% 分位數），
    跟拔靴分布本身（方便呼叫端自己畫圖或做進一步檢定）。
    """
    rng = np.random.default_rng(seed)
    values = daily_returns.to_numpy()
    n = len(values)
    point_estimate = annualized_sharpe(values, periods_per_year)

    boot_sharpes = np.empty(n_boot)
    for i in range(n_boot):
        idx = moving_block_bootstrap_indices(n, block_size, rng)
        boot_sharpes[i] = annualized_sharpe(values[idx], periods_per_year)

    return {
        "point_estimate": point_estimate,
        "ci_low": float(np.nanpercentile(boot_sharpes, 2.5)),
        "ci_high": float(np.nanpercentile(boot_sharpes, 97.5)),
        "boot_sharpes": boot_sharpes,
    }


def paired_bootstrap_sharpe_diff(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    block_size: int = 21,
    n_boot: int = 5000,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict:
    """同一組隨機區塊同時套用在策略跟對照組（例如 QQQ）上，保留兩者在
    同一天的相關性（共同的市場風險、同漲同跌），算 Sharpe 差值
    （策略 - 對照組）的拔靴分布。用配對（paired）而非各自獨立拔靴，是因為
    我們真正關心的是「策略有沒有在相同市場環境下真的贏過對照組」，獨立
    拔靴兩條序列會把共同的市場波動誤當成兩者各自的雜訊，低估差值的顯著性。

    回傳差值點估計、95% CI，以及「拔靴分布裡差值 <= 0 的比例」（類似
    單尾 p-value：這個比例越接近甚至超過 0.5，代表策略贏過對照組這件事
    越可能只是這段樣本期間的運氣，不是穩定的優勢）。
    """
    aligned = pd.concat({"strategy": strategy_returns, "benchmark": benchmark_returns}, axis=1).dropna()
    s = aligned["strategy"].to_numpy()
    b = aligned["benchmark"].to_numpy()
    n = len(aligned)

    point_diff = annualized_sharpe(s, periods_per_year) - annualized_sharpe(b, periods_per_year)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = moving_block_bootstrap_indices(n, block_size, rng)
        diffs[i] = annualized_sharpe(s[idx], periods_per_year) - annualized_sharpe(b[idx], periods_per_year)

    return {
        "point_diff": point_diff,
        "ci_low": float(np.nanpercentile(diffs, 2.5)),
        "ci_high": float(np.nanpercentile(diffs, 97.5)),
        "prob_strategy_not_better": float((diffs <= 0).mean()),
        "diffs": diffs,
    }


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """從資產淨值序列（index 是日期、已排序）算簡單日報酬，丟掉第一筆
    （沒有前一天可以算報酬）的 NaN。"""
    return equity.pct_change().dropna()
