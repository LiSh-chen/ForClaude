"""未平倉量(open interest, OI)訊號探索：期貨市場「新資金流入/部位解除」的
經典代理指標，這次會話先前判定拿不到的資料，實際上已經包含在
scripts/fetch_taifex_tx_raw.py 抓到的原始檔案裡（未沖銷契約數欄位）。

**重要**：不能直接用 taifex_tx_multi_contract.parquet 裡的 near_oi/far_oi
——那是「近月/次近月」這兩個標籤本身的OI，每次月結算換月時，「近月」這個
標籤會從舊合約換成新合約，OI會在換月瞬間從（舊合約到期前的高點）斷崖式
掉到（新合約剛變成近月時的低點），是人為的斷點，不是真實的市場OI變化。

這裡改用「當天所有到期月份合約OI加總」(total_oi)，這是連續、沒有換月
斷點的整體市場曝險水位指標。

測試假說（期貨市場分析裡最經典的OI解讀框架，價格與OI變化的四象限）：
- 價漲 + OI增加：新多單進場（動能確認，看漲）
- 價漲 + OI減少：空單回補（軋空式上漲，動能較不紮實）
- 價跌 + OI增加：新空單進場（動能確認，看跌）
- 價跌 + OI減少：多單停損/獲利了結（下跌動能較不紮實）

具體測試：OI日變化率（正規化，因為23年間OI水位本身有長期成長趨勢）能不能
預測接下來N天的近月價格報酬，尤其是「OI變化 x 價格變化」交乘項是否比
單獨看OI或單獨看價格更有預測力。

用法：
    python scripts/scan_open_interest_signal.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "taifex_tx"
NEARFAR_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
OUT_PATH = REPO_ROOT / "data" / "open_interest_signal_grid.parquet"

IS_CUTOFF = "2021-01-01"
EXPIRY_RE = re.compile(r"^\d{6}$")


def build_total_oi() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("TX_*.csv"))
    frames = [pd.read_csv(fp, dtype=str, index_col=False) for fp in files]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.columns = [c.strip() for c in all_df.columns]

    d = all_df.copy()
    d["到期月份(週別)"] = d["到期月份(週別)"].str.strip()
    d["交易時段"] = d["交易時段"].str.strip()
    d = d[d["交易時段"] == "一般"]
    d = d[d["到期月份(週別)"].str.match(EXPIRY_RE, na=False)]
    d["date"] = pd.to_datetime(d["交易日期"], format="%Y/%m/%d")
    d["open_interest"] = pd.to_numeric(d["未沖銷契約數"].str.replace(",", "", regex=False), errors="coerce")

    total_oi = d.groupby("date")["open_interest"].sum().reset_index()
    total_oi.columns = ["date", "total_oi"]
    return total_oi.sort_values("date").reset_index(drop=True)


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    print("計算每日全合約OI加總（避免近月/遠月換月造成的人為斷點）...")
    total_oi = build_total_oi()
    print(f"交易日數: {len(total_oi)}, 日期範圍: {total_oi['date'].min()} ~ {total_oi['date'].max()}")

    nearfar = pd.read_parquet(NEARFAR_PATH)[["date", "near_price"]]
    df = nearfar.merge(total_oi, on="date", how="inner").sort_values("date").reset_index(drop=True)
    print(f"合併後交易日數: {len(df)}")

    # OI變化率用20日滾動平均正規化，因為23年間OI水位本身有長期成長趨勢
    # （市場規模擴大），直接看差分會被長期趨勢污染
    df["oi_chg_pct"] = df["total_oi"].pct_change() * 100
    df["oi_ma20"] = df["total_oi"].rolling(20).mean()
    df["oi_rel_chg"] = (df["total_oi"] - df["oi_ma20"].shift(1)) / df["oi_ma20"].shift(1) * 100
    df["price_ret_1d"] = df["near_price"].pct_change() * 100

    is_df = df[df["date"] < IS_CUTOFF].copy()
    print(f"\nIS交易日數: {len(is_df)}")

    N_GRID = [1, 3, 5, 10, 20]
    rows = []
    for N in N_GRID:
        fwd_ret = (df["near_price"].shift(-N) / df["near_price"] - 1) * 100
        is_mask = df["date"] < IS_CUTOFF

        # 四象限測試：oi_rel_chg(昨天已知,shift(1)) x price_ret_1d(昨天已知,shift(1))
        oi_sig = df["oi_rel_chg"].shift(1)
        price_sig = df["price_ret_1d"].shift(1)

        quad_up_oi_up = (price_sig > 0) & (oi_sig > 0)
        quad_up_oi_down = (price_sig > 0) & (oi_sig < 0)
        quad_down_oi_up = (price_sig < 0) & (oi_sig > 0)
        quad_down_oi_down = (price_sig < 0) & (oi_sig < 0)

        for label, mask in [
            ("漲+OI增(新多單)", quad_up_oi_up), ("漲+OI減(空單回補)", quad_up_oi_down),
            ("跌+OI增(新空單)", quad_down_oi_up), ("跌+OI減(多單停損)", quad_down_oi_down),
        ]:
            sub_mask = mask & is_mask
            t, n, m = tstat(fwd_ret[sub_mask])
            rows.append(dict(N=N, quadrant=label, n=n, mean_fwd_ret=m, tstat=t))

    res = pd.DataFrame(rows)
    res.to_parquet(OUT_PATH, index=False)

    for N in N_GRID:
        print(f"\n--- 預測未來{N}天報酬 ---")
        sub = res[res["N"] == N]
        print(sub[["quadrant", "n", "mean_fwd_ret", "tstat"]].to_string(index=False))

    print(f"\nsaved: {OUT_PATH}")
    print("\n以上全部只用IS(2001-2020)，尚未看OOS。")


if __name__ == "__main__":
    main()
