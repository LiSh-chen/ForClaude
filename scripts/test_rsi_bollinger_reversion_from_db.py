"""用真實資料測試 RSI 超賣 / 布林通道下軌反彈這兩種短期均值回歸訊號。

背景：docs/research_findings.md 4.1 節已經測過並否證「短期反轉策略」（用
IC 掃描最強的反轉訊號設計，統計上顯著但扣成本後轉負）。這裡改用業界最
常見的兩個具體均值回歸指標重新驗證一次——RSI 極度超賣、跌破布林通道
下軌（可選加成交量爆增確認）——預期會重現同樣的失敗模式，但用具體、
常見的參數重新確認，排除「換個指標會不會不一樣」的疑慮。

兩種訊號都用既有的每日回測引擎（tw_quant/backtest.py 的 entry_signal_fn
掛鉤），出場沿用系統既有的 2.5×ATR 吊燈停利機制，跟策略 C 的出場邏輯
一致，只換進場邏輯。

反未來函數：RSI/布林通道/均量都只在「訊號當天 T 的收盤」已知（RSI 用
T 日以前的漲跌幅平滑, 布林通道上下軌跟均量都額外 shift(1) 只用 T-1 資料），
T 日自己的收盤價/成交量可以拿來跟這些已經 shift 過的門檻比較（跟
signals.py/refine_mss_strategy_from_db.py 既有的「T 日觸發、T-1 已知的
指標當防禦線」慣例一致），T+1 日開盤才進場。

用法：
    python scripts/test_rsi_bollinger_reversion_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant.backtest import run_backtest
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.signals import build_pool_mask
from tw_quant.storage import get_data_store

RSI_WINDOW = 14
BB_WINDOW = 20
RSI_THRESHOLD_GRID = (20, 25, 30)
BB_STD_MULT_GRID = (1.5, 2.0, 2.5)
VOLUME_SPIKE_MULT = 1.5
MIN_TRADES_FOR_RANKING = 15


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    return cfg


def _compute_rsi(master: pd.DataFrame, window: int) -> pd.Series:
    """Wilder's RSI，逐檔股票獨立算。RSI(T) 用到 T 日自己的收盤價（今天漲跌），
    屬於「訊號當天自己的收盤價」這個既有允許的例外，不用再 shift。
    """
    delta = master.groupby("stock_id", sort=False)["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(master["stock_id"], sort=False).transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    avg_loss = loss.groupby(master["stock_id"], sort=False).transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    both_zero = (avg_gain == 0) & (avg_loss == 0)  # 完全平盤，沒漲也沒跌
    only_loss_zero = (avg_loss == 0) & ~both_zero  # 窗格內只漲不跌，RS 理論上是無限大
    rsi = rsi.mask(only_loss_zero, 100.0)
    rsi = rsi.mask(both_zero, 50.0)
    return rsi.fillna(50.0)  # 暖身期不足時的 NaN，保守當中性值處理，不會被誤判成超賣訊號


def make_rsi_signal_fn(rsi_threshold: float):
    def rsi_signal_fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        rsi = _compute_rsi(master, RSI_WINDOW)
        entry = pool & (rsi < rsi_threshold)
        return entry.fillna(False).values, np.zeros(len(master), dtype=bool)

    return rsi_signal_fn


def make_bollinger_signal_fn(std_mult: float, require_volume_spike: bool):
    def bollinger_signal_fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        sma = ind.sma(master, "close", BB_WINDOW)
        std = ind.rolling_std(master, "close", BB_WINDOW)
        lower_band = sma - std_mult * std
        lower_band_prior = ind.shift_by_group(lower_band, master, periods=1)
        below_band = (master["close"] < lower_band_prior).fillna(False)

        entry = pool & below_band
        if require_volume_spike:
            avg_vol = ind.sma(master, "volume", BB_WINDOW)
            avg_vol_prior = ind.shift_by_group(avg_vol, master, periods=1)
            volume_ok = (master["volume"] > avg_vol_prior * VOLUME_SPIKE_MULT).fillna(False)
            entry = entry & volume_ok

        return entry.values, np.zeros(len(master), dtype=bool)

    return bollinger_signal_fn


HEADER = (
    f"{'signal':<24}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<24}  {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
        f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f}  {m['n_trades']:>5.0f} {m['win_rate']:>6.1%} "
        f"{rr_str} {m['ev_pct']:>7.2%} {pf_str}"
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

    print("=== RSI 超賣 / 布林通道下軌反彈回測（完整交易成本，出場沿用 2.5xATR 吊燈停利）===")
    print(HEADER)

    rows = []
    for rsi_threshold in RSI_THRESHOLD_GRID:
        label = f"RSI<{rsi_threshold}"
        result = run_backtest(prices, margin_short, base_cfg, historical_mdd=None, entry_signal_fn=make_rsi_signal_fn(rsi_threshold))
        m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
        rows.append({"label": label, **m})

    for std_mult in BB_STD_MULT_GRID:
        for require_vol in (False, True):
            label = f"BB(std={std_mult}{',vol' if require_vol else ''})"
            result = run_backtest(
                prices, margin_short, base_cfg, historical_mdd=None,
                entry_signal_fn=make_bollinger_signal_fn(std_mult, require_vol),
            )
            m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
            rows.append({"label": label, **m})

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(r["label"], r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名，以下是全部組合：")
        print(df.to_string(index=False))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "出場沿用系統既有 2.5xATR 吊燈停利，跟策略 C 一致；RSI(14)/布林通道(20日)為業界常見參數）"
    )


if __name__ == "__main__":
    main()
