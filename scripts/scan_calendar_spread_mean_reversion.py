"""近月/遠月價差(calendar spread)均值回歸探索：價差異常偏離近期水準時，
接下來N天價差本身會不會往回收斂？

背景：data/taifex_tx_multi_contract.parquet（見
scripts/build_taifex_multi_contract.py）算出每個交易日的
calendar_spread = 近月結算價 - 次近月(遠月)結算價。這是「近月/遠月合約
價差」題目下，第一個要測的具體、可交易的假說：價差本身有沒有均值回歸
性質，能不能靠這個賺錢（做法：價差異常寬時放空價差=空近月多遠月，價差
異常窄時做多價差=多近月空遠月，等價差回到正常水準附近平倉）。

方法：
1. 用 z_t = (spread_t - rolling_mean(spread, W)) / rolling_std(spread, W)，
   W天滾動窗格，都用 shift(1) 避免用到當天自己的值（今天訊號只能看
   「昨天收盤後」已知的z分數）。
2. 測試「今天訊號」對「接下來N天價差變化」(spread_{t+N} - spread_t) 的
   預測力：如果均值回歸成立，z越正（價差異常寬），接下來spread變化應該
   越負（價差收斂）——也就是z跟接下來的spread變化應該是負相關。
3. IS(2001-2020)先做完整的(W, N)網格掃描，篩出|t|>=3且方向一致(負相關)
   的候選，再檢查四個子區間穩健性，通過才考慮OOS。

用法：
    python scripts/scan_calendar_spread_mean_reversion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
OUT_PATH = REPO_ROOT / "data" / "calendar_spread_mean_reversion_grid.parquet"

IS_CUTOFF = "2021-01-01"
W_GRID = [10, 20, 30, 60, 90]
N_GRID = [1, 3, 5, 10, 20]
MIN_N = 200


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    df = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)
    df["spread"] = df["calendar_spread"].astype(float)

    is_df = df[df["date"] < IS_CUTOFF].copy()
    print(f"IS交易日數: {len(is_df)} ({is_df['date'].min()} ~ {is_df['date'].max()})")

    rows = []
    for W in W_GRID:
        roll_mean = df["spread"].rolling(W).mean().shift(1)
        roll_std = df["spread"].rolling(W).std(ddof=1).shift(1)
        z = (df["spread"].shift(1) - roll_mean) / roll_std
        for N in N_GRID:
            fwd_change = df["spread"].shift(-N) - df["spread"]
            is_mask = df["date"] < IS_CUTOFF
            zz = z[is_mask]
            fc = fwd_change[is_mask]
            valid = zz.notna() & fc.notna()
            if valid.sum() < MIN_N:
                continue
            corr = np.corrcoef(zz[valid], fc[valid])[0, 1]
            # 用z高分位(z>=1)跟低分位(z<=-1)分組看фwd_change的t值，
            # 比單純相關係数更貼近「訊號觸發才進場」的交易邏輯
            high = fc[valid][zz[valid] >= 1.0]
            low = fc[valid][zz[valid] <= -1.0]
            t_high, n_high, m_high = tstat(high)
            t_low, n_low, m_low = tstat(low)
            rows.append(dict(
                W=W, N=N, corr=corr, n_total=int(valid.sum()),
                n_high=n_high, mean_high=m_high, t_high=t_high,
                n_low=n_low, mean_low=m_low, t_low=t_low,
            ))

    res = pd.DataFrame(rows)
    res.to_parquet(OUT_PATH, index=False)

    print(f"\n組合數: {len(res)}")
    print("\n依 corr 排序（負相關=均值回歸方向），前10個最負：")
    print(res.reindex(res["corr"].sort_values().index).head(10)
          [["W", "N", "corr", "n_total", "t_high", "mean_high", "t_low", "mean_low"]]
          .to_string(index=False))

    print("\n依 |t_high| 排序（z高分位分組的t值），前10個：")
    print(res.reindex(res["t_high"].abs().sort_values(ascending=False).index).head(10)
          [["W", "N", "corr", "t_high", "n_high", "mean_high", "t_low", "n_low", "mean_low"]]
          .to_string(index=False))

    print(f"\nsaved: {OUT_PATH}")
    print("\n以上全部只用IS(2001-2020)，尚未看OOS。")


if __name__ == "__main__":
    main()
