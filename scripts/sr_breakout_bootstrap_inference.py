"""支撐壓力順勢突破候選：用穩健的區塊自助法（stationary block bootstrap）
重新評估信賴區間，取代原本假設交易彼此獨立同分布(iid)的簡單t檢定。

**為什麼原本的t檢定可能過度樂觀**：308筆交易裡，同一時間只允許一筆部位、
平均持有15天（最長20天）——連續幾筆交易之間很可能共享同一段市場regime
（例如同一波趨勢延伸出的兩三次突破），報酬不是真正互相獨立。t檢定假設
iid會低估變異數、高估t值，變相讓顯著性看起來比實際更強。

**做法**：stationary bootstrap（Politis & Romano 1994）用幾何分布決定的
隨機區塊長度重複抽樣交易序列（不是逐筆獨立抽樣），保留區塊內的序列相關
結構，藉此得到更誠實的信賴區間跟p值。同時比較不同區塊長度的穩健性、
檢查原始序列本身的自相關係數（驗證區塊自助法是不是真的有必要）。

**不做的事**：不重新看/重新算OOS的樣本外表現、不調整策略參數——只是用更
嚴謹的統計方法重新評估「已經有的」IS(2001-2020)跟OOS(2021-2023)交易紀錄，
維持這次會話的驗證紀律。

用法：
    python scripts/sr_breakout_bootstrap_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
LOW_COST = COST_SCENARIOS[0]
IS_CUTOFF = "2021-01-01"
LOCKED_CFG = SupportResistanceFadeConfig(
    direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
    min_range_pct=2.0, trend_slope_threshold_pct=5.0,
)
N_BOOT = 20000


def naive_tstat_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """回傳 (t_stat, p_value, ci_lo, ci_hi)，假設iid常態。環境沒裝scipy，改用
    常態分布近似（erf），n=44~262時跟精確t分布的差異在小數點後第2~3位，
    不影響這裡要呈現的「iid假設 vs block bootstrap」對比結論。"""
    import math
    n = len(x)
    mean, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    t = mean / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    zcrit = 1.959963985  # 95% 常態分位數
    return t, p, mean - zcrit * se, mean + zcrit * se


def stationary_bootstrap_indices(n: int, block_len_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap：區塊長度服從幾何分布
    （平均長度 block_len_mean），環狀wrap-around避免邊界效應。block_len_mean=1
    時退化成一般iid bootstrap（每個位置獨立重抽）。"""
    idx = np.empty(n, dtype=int)
    p = 1.0 / block_len_mean
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i % n
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i += 1
    return idx


def block_bootstrap(x: np.ndarray, block_len_mean: float, n_boot: int, alpha: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, block_len_mean, rng)
        boot_means[b] = x[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # 雙尾p值：bootstrap均值分布裡，跟0同號或異號哪邊比例小，乘2
    p_left = (boot_means <= 0).mean()
    p_right = (boot_means >= 0).mean()
    p_value = 2 * min(p_left, p_right)
    return dict(block_len=block_len_mean, boot_mean=boot_means.mean(), ci_lo=lo, ci_hi=hi,
                p_value=min(p_value, 1.0), boot_std=boot_means.std(ddof=1))


def autocorr(x: np.ndarray, max_lag: int = 5) -> list[float]:
    x = x - x.mean()
    n = len(x)
    var = (x ** 2).sum()
    out = []
    for lag in range(1, max_lag + 1):
        out.append((x[:-lag] * x[lag:]).sum() / var)
    return out


def analyze(label: str, trades: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"{label}  (n={len(trades)})")
    print("=" * 70)
    if len(trades) < 10:
        print("樣本太小，略過")
        return
    net = apply_costs(trades, LOW_COST)["net_twd"].to_numpy()

    t, p, lo, hi = naive_tstat_ci(net)
    print(f"[原本的t檢定，假設iid]  mean={net.mean():,.0f}  t={t:.3f}  p={p:.4f}  95%CI=[{lo:,.0f}, {hi:,.0f}]")

    ac = autocorr(net, max_lag=5)
    print(f"[交易序列自相關 lag1-5]  " + ", ".join(f"{a:+.3f}" for a in ac))

    print(f"\n[Stationary block bootstrap，{N_BOOT}次重抽，比較不同區塊長度]")
    print(f"{'區塊長度':>10} {'bootstrap均值':>14} {'95%CI':>28} {'p值':>8}")
    for block_len in [1, 5, 10, 20, 40]:
        r = block_bootstrap(net, block_len, N_BOOT, 0.05, seed=42)
        print(f"{block_len:>10} {r['boot_mean']:>14,.0f} "
              f"[{r['ci_lo']:>10,.0f}, {r['ci_hi']:>10,.0f}] {r['p_value']:>8.4f}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    trades = bt_sr(df, LOCKED_CFG)

    is_trades = trades[pd.to_datetime(trades["entry_date"]) < IS_CUTOFF].reset_index(drop=True)
    oos_trades = trades[pd.to_datetime(trades["entry_date"]) >= IS_CUTOFF].reset_index(drop=True)

    analyze("IS (2001-2020)", is_trades)
    analyze("OOS (2021-2023，僅供參考，不作為新驗證)", oos_trades)
    analyze("全歷史 (2001-2023，IS+OOS合併，僅供參考——不能取代IS/OOS分離的驗證紀律)", trades)


if __name__ == "__main__":
    main()
