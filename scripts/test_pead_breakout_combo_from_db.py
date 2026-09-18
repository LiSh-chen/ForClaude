"""測試「月營收動能 + 價量突破確認」組合策略——對應使用者提供的第二份
Gemini 研究文件裡，針對散戶量身規劃的具體 PEAD 策略設計。

原始設計（文件裡的「台股中小型股：PEAD 盈餘動能突破與順勢波段策略」）：
  基本面濾網：
    條件 A：單月營收創近 12 個月新高，或 YoY 成長 > 15%
    條件 B：最新一季毛利率/營益率較前一季成長
    條件 C：排除股本 > 50 億，鎖定中小型股
  技術面動能：
    當日收盤價突破近 20 日最高價
    當日成交量突破過去 5 日均量 2 倍以上
  出場：進場點 - 2xATR（或帶量突破紅K棒低點）當初始停損；獲利後改用
    10 日均線防守，跌破且隔天不站回即平倉
  部位規模：固定風險模型，單筆風險為總資金 1.5%

★ 這裡能測、不能測的部分（誠實列出，不是假裝做得到）：
  - 能測：條件 A（月營收 YoY>15% 或創 12 個月新高）+ 技術面動能（20 日新高
    +成交量 2 倍），因為這兩項用現有資料（tw_quant.storage 的 prices/
    month_revenue）就算得出來。
  - 不能測：條件 B（毛利率/營益率 QoQ 成長）——資料庫目前只抓月營收，
    沒有抓季報財報資料，不會為了這次測試臨時假造。
  - 不能測：條件 C（股本 < 50 億的中小型股濾網）——沒有股本/市值資料，
    這裡的 150 檔股票universe本身也不是刻意篩過中小型股，維持原樣。
  - 出場沿用系統既有的吊燈停利機制（rolling_10d_high - ATR倍數 * ATR），
    這在方向上跟文件描述的「10 日均線防守」類似（都是用 10 日窗格算防守
    線），但公式不同，不特地另外實作一套只為了貼合文件寫法；ATR 倍數
    改成可調參數，加入 2.0（文件原始建議值）當網格選項之一，跟系統原本
    最佳化出來的 2.5 一起比較。
  - 台股處置股/當沖占比等法規防禦機制——現有資料沒有注意股/處置股名單、
    也沒有逐股當沖占比，這部分完全無法測試，回測結果沒有反映這些真實
    世界的額外風險（真實報酬可能因為踩到處置股流動性折價而比這裡差）。

用法：
    python scripts/test_pead_breakout_combo_from_db.py
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
from scripts.test_pead_revenue_drift_from_db import _revenue_surprise_frame

BREAKOUT_WINDOW_GRID = (10, 20, 30)
VOLUME_MULT_GRID = (1.5, 2.0, 2.5)
ATR_MULT_GRID = (2.0, 2.5)  # 2.0 是文件原始建議值，2.5 是系統之前找到的最佳值
YOY_THRESHOLD = 0.15
MIN_TRADES_FOR_RANKING = 10


def attach_revenue_momentum_flag(prices: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    """條件 A：月營收 YoY > 15% 或創近 12 個月新高，比照 test_pead_revenue_drift_from_db.py
    的 known_date 反未來函數規則對齊進價量表，並 shift(1)。
    """
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    rev_frame = _revenue_surprise_frame(revenue)
    if rev_frame.empty:
        master["revenue_momentum_ok_shifted"] = False
        return master

    rev_frame = rev_frame.copy()
    rev_frame["momentum_ok"] = (rev_frame["yoy_growth"] > YOY_THRESHOLD) | rev_frame["revenue_12m_high"].fillna(False)

    left = master[["stock_id", "date"]].reset_index().rename(columns={"index": "_orig_idx"})
    left["date"] = left["date"].astype("datetime64[ns]")
    right = rev_frame.rename(columns={"known_date": "date"})[["stock_id", "date", "momentum_ok"]]
    right["date"] = right["date"].astype("datetime64[ns]")
    # 同一個 dtype 精度不一致的問題，見 test_pead_revenue_drift_from_db.py 的註解
    merged = pd.merge_asof(
        left.sort_values("date"), right.sort_values("date"),
        on="date", by="stock_id", direction="backward",
    ).sort_values("_orig_idx").reset_index(drop=True)

    master["revenue_momentum_ok_raw"] = merged["momentum_ok"].fillna(False).values
    shifted = ind.shift_by_group(master["revenue_momentum_ok_raw"].astype(float), master, periods=1)
    master["revenue_momentum_ok_shifted"] = shifted.fillna(0).astype(bool)
    return master


def build_base_config(atr_mult: float) -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.sizing.atr_multiplier = atr_mult
    return cfg


def make_combo_signal_fn(breakout_window: int, volume_mult: float, master_with_revenue: pd.DataFrame):
    """條件 A（已預先算好、shift 過的月營收動能旗標）+ 技術面動能
    （T 日收盤突破 T-1 日為止的 breakout_window 日高點、T 日成交量突破
    T-1 日為止的 5 日均量 volume_mult 倍）。
    """
    revenue_flag_lookup = master_with_revenue.set_index(["stock_id", "date"])["revenue_momentum_ok_shifted"]

    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)

        idx = pd.MultiIndex.from_arrays([master["stock_id"], master["date"]])
        revenue_ok = pd.Series(revenue_flag_lookup.reindex(idx).fillna(False).values, index=master.index)

        rolling_high = ind.rolling_max(master, "high", breakout_window)
        rolling_high_prior = ind.shift_by_group(rolling_high, master, periods=1)
        price_breakout = (master["close"] > rolling_high_prior).fillna(False)

        avg_vol_5 = ind.sma(master, "volume", 5)
        avg_vol_5_prior = ind.shift_by_group(avg_vol_5, master, periods=1)
        volume_spike = (master["volume"] > avg_vol_5_prior * volume_mult).fillna(False)

        entry = pool & revenue_ok & price_breakout & volume_spike
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn


HEADER = (
    f"{'brk_win':>8} {'vol_x':>6} {'atr_x':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(breakout_window: int, volume_mult: float, atr_mult: float, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{breakout_window:>8} {volume_mult:>6.1f} {atr_mult:>6.1f}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>5.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin_short = store.load_margin_short()
    revenue = store.load_month_revenue()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)
    if revenue.empty:
        print("資料庫裡沒有任何月營收資料，需要先跑過至少一次每日排程 ingest。", file=sys.stderr)
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的價量資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    master_with_revenue = attach_revenue_momentum_flag(prices, revenue)
    n_ok = master_with_revenue["revenue_momentum_ok_shifted"].sum()
    print(f"營收動能條件成立的列數：{n_ok} / {len(master_with_revenue)}\n")

    print("=== 月營收動能 + 價量突破組合策略回測（完整交易成本）===")
    print(HEADER)

    rows = []
    for atr_mult in ATR_MULT_GRID:
        base_cfg = build_base_config(atr_mult)
        for breakout_window in BREAKOUT_WINDOW_GRID:
            for volume_mult in VOLUME_MULT_GRID:
                signal_fn = make_combo_signal_fn(breakout_window, volume_mult, master_with_revenue)
                result = run_backtest(prices, margin_short, base_cfg, historical_mdd=None, entry_signal_fn=signal_fn)
                m = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
                rows.append({"breakout_window": breakout_window, "volume_mult": volume_mult, "atr_mult": atr_mult, **m})

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["breakout_window"]), r["volume_mult"], r["atr_mult"], r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名，以下是全部組合：")
        print(df.to_string(index=False))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "brk_win = 突破窗格天數；vol_x = 成交量倍數門檻；atr_x = 吊燈停利 ATR 倍數；"
        "已排除毛利率條件、股本濾網、處置股/當沖占比法規防禦——見檔頭說明的資料限制）"
    )


if __name__ == "__main__":
    main()
