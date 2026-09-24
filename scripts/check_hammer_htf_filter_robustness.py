"""對 backtest_hammer_ratio_htf_filter.py 找到的「疊加 60 分鐘趨勢濾網、做空、
高比例門檻」候選組合做穩健性檢查，而不是直接把單一好看的數字當結論。

依循 docs/research_findings.md 既有方法論：「任何好看的單一回測數字，一定要
在鄰近參數值上重跑，確認是平滑高原還是一碰就崩的懸崖；後者視為過擬合/巧合」。

兩項檢查：
1. 鄰近參數網格（ratio_threshold × r_multiple × SMA窗口）：好結果的鄰居也該
   是好結果，不然就是雜訊剛好落在那個組合上。
2. 切半樣本外檢查（2001-2012 vs 2013-2023）：真正的邊際應該兩段都survive，
   只有其中一段賺錢、另一段虧錢，代表那不是穩健訊號。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import (  # noqa: E402
    COST_SCENARIOS, apply_costs, build_base_arrays, build_htf_trend, simulate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
GRID_OUT = REPO_ROOT / "data" / "hammer_htf_robustness_grid.parquet"
SPLIT_OUT = REPO_ROOT / "data" / "hammer_htf_split_sample.parquet"

RATIOS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
R_MULTIPLES = [2.0, 2.5, 3.0, 3.5]
SMA_WINDOWS = [30, 50, 70]
MID_COST = COST_SCENARIOS[1]  # 「中」情境：來回 60 元

SPLIT_CONFIGS = [(2.5, 2.0), (3.0, 2.0), (7.0, 3.0)]


def net_mid_twd(arrays: dict, htf: np.ndarray, ratio_th: float, r_mult: float) -> tuple[int, float]:
    trades = simulate(arrays, ratio_threshold=ratio_th, r_multiple=r_mult, direction="short",
                       htf_trend=htf, trend_mode="with")
    if trades.empty:
        return 0, np.nan
    priced = apply_costs(trades, MID_COST)
    return len(trades), priced["net_twd"].sum()


def run_neighborhood_grid(df: pd.DataFrame) -> pd.DataFrame:
    arrays = build_base_arrays(df)
    rows = []
    for sma_w in SMA_WINDOWS:
        htf = build_htf_trend(df, freq="60min", sma_window=sma_w)
        for ratio_th in RATIOS:
            for r_mult in R_MULTIPLES:
                n, net_mid = net_mid_twd(arrays, htf, ratio_th, r_mult)
                rows.append(dict(sma_w=sma_w, ratio_ge=ratio_th, r_multiple=r_mult, trades=n, net_mid_twd=net_mid))
    return pd.DataFrame(rows)


def run_split_sample(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {
        "2001-2012": df[df["datetime"] < "2013-01-01"],
        "2013-2023": df[df["datetime"] >= "2013-01-01"],
    }
    for period_label, sub in periods.items():
        arrays = build_base_arrays(sub)
        htf50 = build_htf_trend(sub, freq="60min", sma_window=50)
        for ratio_th, r_mult in SPLIT_CONFIGS:
            n, net_mid = net_mid_twd(arrays, htf50, ratio_th, r_mult)
            rows.append(dict(period=period_label, ratio_ge=ratio_th, r_multiple=r_mult, trades=n, net_mid_twd=net_mid))
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    print("=== 鄰近參數網格（60min SMA{30,50,70} x ratio 5-10 x R 2-3.5，中成本情境）===")
    grid = run_neighborhood_grid(df)
    grid.to_parquet(GRID_OUT, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    for sma_w in SMA_WINDOWS:
        print(f"\n-- SMA{sma_w} --")
        piv = grid[grid["sma_w"] == sma_w].pivot(index="ratio_ge", columns="r_multiple", values="net_mid_twd")
        print(piv.to_string())

    print("\n\n=== 切半樣本外檢查（2001-2012 vs 2013-2023，60min SMA50，中成本情境）===")
    split = run_split_sample(df)
    split.to_parquet(SPLIT_OUT, index=False)
    print(split.to_string(index=False))

    print("\n結論：只有兩段時間都是正值，才算是禁得起檢驗的邊際；只有一段正、另一段負的組合視為過擬合/巧合，不採用。")


if __name__ == "__main__":
    main()
