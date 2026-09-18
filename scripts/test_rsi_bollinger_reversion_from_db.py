"""用真實資料測試 RSI 超賣 / 布林通道下軌反彈這兩種短期均值回歸訊號。

背景：docs/research_findings.md 4.1 節已經測過並否證「短期反轉策略」（用
IC 掃描最強的反轉訊號設計，統計上顯著但扣成本後轉負）。這裡改用業界最
常見的兩個具體均值回歸指標重新驗證一次——RSI 極度超賣、跌破布林通道
下軌的乖離程度（可選加成交量爆增確認）——預期會重現同樣的失敗模式，
但用具體、常見的參數重新確認，排除「換個指標會不會不一樣」的疑慮。

★ 開發過程中一個重要教訓（沒有藏起來）：第一版直接把這兩個訊號接到
tw_quant/backtest.py 的 entry_signal_fn（日事件迴圈引擎），結果 9 組參數
組合全部 0 筆交易。追查後發現原因不是訊號太少見，而是**結構性不相容**：
那個引擎的吊燈停利公式 `compute_initial_stop` 用「過去 10 日高點 - 2.5x
ATR」當初始停損價，這是為了「進場點貼近近期高點」的突破式策略設計的
（`rolling_high_10 ≈ entry_price`，算出來的停損自然會落在進場價之下）。
均值回歸策略的進場點恰好相反——貼近近期低點，此時「過去 10 日高點」離
現價很遠，算出來的停損價常常高於進場價，被
`compute_position_size` 的「停損價不低於進場價，無法計算風險」規則直接
擋掉，導致幾乎所有候選都在下單前就被拒絕。實測（合成資料）235 筆拒單
裡有 215 筆是這個原因。這跟 scripts/explore_reversal_strategy_from_db.py
（已驗證過的短期反轉策略）採用的解法一致：均值回歸/反轉類訊號改用
tw_quant/factor_backtest.py 的定期調倉引擎（不含逐筆吊燈停利，用「排名
+ 固定持有期」取代），才是真正公平的測試方式，見下面 main() 的實作。

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
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store

RSI_WINDOW = 14
BB_WINDOW = 20
VOLUME_SPIKE_MULT = 1.5
REBALANCE_FREQ_GRID = (3, 5, 10, 21)  # 進場後持有幾個交易日（近似短期均值回歸的合理持有期）
TOP_N_GRID = (10, 20, 30)
MIN_TRADES_FOR_RANKING = 15


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    return cfg


def _compute_rsi(master: pd.DataFrame, window: int) -> pd.Series:
    """Wilder's RSI，逐檔股票獨立算。RSI(T) 用到 T 日自己的收盤價（今天漲跌），
    這裡當成排名依據的原始值，呼叫端要自己 shift(1) 才能餵給 factor_backtest。
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


def rsi_ranking_signal_fn(master: pd.DataFrame) -> pd.Series:
    """排名依據 = RSI 本身，ascending=True 時買 RSI 最低（最超賣）的股票。"""
    rsi = _compute_rsi(master, RSI_WINDOW)
    return ind.shift_by_group(rsi, master, periods=1)


def make_bollinger_ranking_signal_fn(require_volume_spike: bool):
    """排名依據 = 布林通道 z-score（(close-均值)/標準差），越負代表跌越深、
    離布林通道下軌越遠。ascending=True 時買 z-score 最低的股票。
    """

    def _fn(master: pd.DataFrame) -> pd.Series:
        sma = ind.sma(master, "close", BB_WINDOW)
        std = ind.rolling_std(master, "close", BB_WINDOW)
        z = (master["close"] - sma) / std.replace(0, np.nan)
        if require_volume_spike:
            avg_vol = ind.sma(master, "volume", BB_WINDOW)
            volume_ok = master["volume"] > avg_vol * VOLUME_SPIKE_MULT
            z = z.where(volume_ok.fillna(False), np.nan)  # 沒有量能確認的候選直接排除排名（NaN 會被 factor_backtest 濾掉）
        return ind.shift_by_group(z, master, periods=1)

    return _fn


HEADER = (
    f"{'signal':<20} {'hold_d':>7} {'top_n':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, hold_days: int, top_n: int, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<20} {hold_days:>7} {top_n:>6}  "
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

    print("=== RSI 超賣 / 布林通道乖離反彈回測（因子式定期調倉，完整交易成本）===")
    print(HEADER)

    signal_fns = {
        "RSI": rsi_ranking_signal_fn,
        "BB_zscore": make_bollinger_ranking_signal_fn(False),
        "BB_zscore+vol": make_bollinger_ranking_signal_fn(True),
    }

    rows = []
    for signal_name, fn in signal_fns.items():
        for hold_days in REBALANCE_FREQ_GRID:
            for top_n in TOP_N_GRID:
                factor_cfg = FactorConfig(rebalance_freq_days=hold_days, top_n=top_n, ascending=True)
                result = run_factor_backtest(prices, base_cfg, factor_cfg, signal_fn=fn)
                m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
                rows.append({"signal": signal_name, "hold_days": hold_days, "top_n": top_n, **m})

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(25).iterrows():
        print(_fmt_row(r["signal"], int(r["hold_days"]), int(r["top_n"]), r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "hold_d = 調倉週期（近似持有天數）；ascending=True 買排名最低（RSI最低/z-score最負，"
        "也就是最超賣）的股票；RSI(14)/布林通道(20日)為業界常見參數）"
    )


if __name__ == "__main__":
    main()
