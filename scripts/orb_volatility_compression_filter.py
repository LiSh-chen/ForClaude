"""波動度壓縮濾網 + 開盤區間突破(ORB)：ORB 已經測過且被否決（見
scripts/backtest_opening_range_breakout.py 的結果，commit 59629bd——
「IS顯著但逐年衰退，OOS明確轉負」），但那個版本是無條件版本：不管前一天
是什麼波動度狀態，只要今天開盤區間被突破就進場。

這裡測試的是真正沒測過的新訊號：**波動度本身**（不是價格通道，是用
ATR相對自己近期水準算出來的「近期波動度壓縮」狀態）能不能篩出更乾淨的
突破——古典「盤勢壓縮後的突破，續航力比隨機時刻的突破更可靠」假說
（vol squeeze -> breakout）。跟已經否決的唐奇安通道突破（那個訊號來源是
「價格」突破N日高低點）跟ORB無條件版本（訊號來源是「今天開盤區間」被
突破，不看前一天的狀態）都不一樣：這裡的濾網訊號來源是「前一天收盤時
已知的近期已實現波動度水準」。

具體做法：
- 沿用 tw_quant.technical_indicators.daily_indicators 已經算好的
  atr_ratio_lag1（= 昨天ATR14 / 昨天為止60日ATR14均值的比值，衡量「昨天
  收盤時的近期波動度相對自己60日均值是壓縮還是擴張」，已經shift(1)避免
  用到未來資訊）。
- 壓縮狀態定義：atr_ratio_lag1 落在IS期間全樣本最低1/3分位（相對定義，
  適應23年間TXF點位水準/絕對波動度大幅變遷）。
- 交易本體完全沿用 tw_quant.opening_range_breakout_strategy 預設參數
  （開盤區間08:45-09:15、最小區間5點、突破後抱到收盤），只是額外疊加
  「進場當天屬於壓縮狀態」這個濾網，用 filter_trades_by_indicator 既有
  基礎設施（跟re腿的量能濾網用同一套函式，direction='le'）。

流程：IS整體(濾網 vs 無條件對照) -> 四個子區間穩健性 -> 如果通過，才考慮
花OOS驗證機會；不因為某個子集看起來好看就回頭調整門檻/參數。

用法：
    python scripts/orb_volatility_compression_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.opening_range_breakout_strategy import OpeningRangeBreakoutConfig, backtest  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_indicator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
DEFAULT_CFG = OpeningRangeBreakoutConfig()
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def report(trades: pd.DataFrame, label: str) -> dict:
    n = len(trades)
    if n == 0:
        print(f"  {label}: n=0")
        return dict(label=label, n=0)
    t = tstat(trades["pnl_points"])
    mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
    print(f"  {label}: n={n} t={t:.2f} 中檔成本淨損益={mid_net:,.0f}")
    return dict(label=label, n=n, t_stat=t, mid_net_twd=mid_net)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    is_indicators = daily_indicators(is_df)
    compression_threshold = is_indicators["atr_ratio_lag1"].quantile(1 / 3)
    print(f"波動度壓縮門檻（IS全期間atr_ratio_lag1最低1/3分位）: {compression_threshold:.4f}\n")

    print("=" * 70)
    print("1) IS（2001-2020）整體：無條件 vs 壓縮濾網")
    print("=" * 70)
    is_trades = backtest(is_df, DEFAULT_CFG)
    report(is_trades, "無條件ORB(對照組)")
    is_compressed = filter_trades_by_indicator(is_trades, is_df, "atr_ratio_lag1", compression_threshold, "le")
    report(is_compressed, "壓縮濾網ORB(候選)")

    print("\n" + "=" * 70)
    print("2) IS 子區間穩健性（壓縮濾網版本，各段分開；用全IS門檻，不逐段重估）")
    print("=" * 70)
    sub_rows = []
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        sub_trades = backtest(sub_df, DEFAULT_CFG)
        sub_compressed = filter_trades_by_indicator(sub_trades, sub_df, "atr_ratio_lag1", compression_threshold, "le")
        print(f"\n--- {name} ---")
        row_uncond = report(sub_trades, "無條件(對照)")
        row_uncond["period"] = name
        row_uncond["variant"] = "unconditional"
        row = report(sub_compressed, "壓縮濾網")
        row["period"] = name
        row["variant"] = "compressed"
        sub_rows.append(row_uncond)
        sub_rows.append(row)

    result = pd.DataFrame(sub_rows)
    compressed_rows = result[result["variant"] == "compressed"]
    n_sig = (compressed_rows["t_stat"].abs() >= 2).sum()
    print(f"\n四個子區間中，壓縮濾網版本 |t|>=2 的窗格數: {n_sig} / 4")
    same_sign = (compressed_rows["t_stat"].dropna() > 0).nunique() <= 1
    print(f"四個子區間方向是否一致: {'是' if same_sign else '否'}")

    result.to_parquet(OUT_DIR / "orb_volatility_compression_subperiods.parquet", index=False)
    print("\nsaved: orb_volatility_compression_subperiods.parquet")

    if n_sig >= 3 and same_sign:
        print("\n通過子區間穩健性門檻，可考慮進入OOS驗證階段（另外討論是否花掉這個候選的OOS機會）。")
    else:
        print("\n未通過子區間穩健性門檻，不進入OOS驗證階段。")


if __name__ == "__main__":
    main()
