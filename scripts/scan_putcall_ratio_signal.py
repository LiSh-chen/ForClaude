"""選擇權put/call比率訊號探索：測試「市場情緒」代理指標能不能預測近月
期貨接下來的報酬。

**已知限制**：taifex_txo_daily_putcall.parquet 裡的 atm_straddle_price
（粗略IV代理）發現資料品質問題——換月當天偶爾回傳literal 0（跟期貨結算價
=0是同一種陷阱），且2008-2009金融海嘯期間有幾筆超過70-100%現貨價的離譜
值，尚未完整清理，這裡先不用它。改用 put_call_volume_ratio /
put_call_oi_ratio 這兩個「當天全部到期月份成交量/未平倉量加總」算出來的
比率——是單純加總，不像單一履約價的價格那樣容易被單一筆異常委託污染。

假說（選擇權市場分析的經典框架）：
- put/call比率異常偏高 -> 市場過度悲觀/避險需求高 -> 反向指標，接下來
  可能反彈（逆勢操作邏輯，類似VIX過高買進的思路）
- 或者：put/call比率上升本身反映真實下跌預期，應該順勢（動能邏輯）
兩個方向都測，不預設答案。

方法：跟這次會話一貫的z分數 + shift(1)手法，比率相對自己60日滾動水準
的z分數，測試對接下來N天近月期貨報酬的預測力，IS(2001-2020)。

用法：
    python scripts/scan_putcall_ratio_signal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TXO_PATH = REPO_ROOT / "data" / "taifex_txo_daily_putcall.parquet"
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
OUT_PATH = REPO_ROOT / "data" / "putcall_ratio_signal_grid.parquet"

IS_CUTOFF = "2021-01-01"
W = 60
N_GRID = [1, 3, 5, 10, 20]
MIN_N = 200


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    txo = pd.read_parquet(TXO_PATH)
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]]
    df = txo.merge(tx, on="date", how="inner").sort_values("date").reset_index(drop=True)
    print(f"合併後交易日數: {len(df)} ({df['date'].min()} ~ {df['date'].max()})")

    df["pc_vol_ratio"] = df["put_volume"] / df["call_volume"]
    df["pc_oi_ratio"] = df["put_oi"] / df["call_oi"]

    for col in ["pc_vol_ratio", "pc_oi_ratio"]:
        roll_mean = df[col].rolling(W).mean().shift(1)
        roll_std = df[col].rolling(W).std(ddof=1).shift(1)
        df[f"{col}_z"] = (df[col].shift(1) - roll_mean) / roll_std

    is_mask = df["date"] < IS_CUTOFF
    print(f"IS交易日數: {is_mask.sum()}")

    rows = []
    for sig_col in ["pc_vol_ratio_z", "pc_oi_ratio_z"]:
        for N in N_GRID:
            fwd_ret = (df["near_price"].shift(-N) / df["near_price"] - 1) * 100
            z = df[sig_col]
            valid = is_mask & z.notna() & fwd_ret.notna()
            high = fwd_ret[valid & (z >= 1.0)]
            low = fwd_ret[valid & (z <= -1.0)]
            t_high, n_high, m_high = tstat(high)
            t_low, n_low, m_low = tstat(low)
            corr = np.corrcoef(z[valid], fwd_ret[valid])[0, 1] if valid.sum() > MIN_N else np.nan
            rows.append(dict(
                signal=sig_col, N=N, corr=corr,
                n_high=n_high, mean_high=m_high, t_high=t_high,
                n_low=n_low, mean_low=m_low, t_low=t_low,
            ))

    res = pd.DataFrame(rows)
    res.to_parquet(OUT_PATH, index=False)

    for sig_col in ["pc_vol_ratio_z", "pc_oi_ratio_z"]:
        print(f"\n=== {sig_col} ===")
        sub = res[res["signal"] == sig_col]
        print(sub[["N", "corr", "n_high", "mean_high", "t_high", "n_low", "mean_low", "t_low"]]
              .to_string(index=False))

    print(f"\nsaved: {OUT_PATH}")
    print("以上全部只用IS(2001-2020)，尚未看OOS，也尚未做不重疊抽樣校正。")


if __name__ == "__main__":
    main()
