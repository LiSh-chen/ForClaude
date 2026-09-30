"""「最大OI履約價=支撐/壓力」訊號探索。

使用者轉述Gemini提的外資跨市場作戰SOP第三層：選擇權市場最大OI的履約價，
莊家(賣方)為了避免結算日賠付鉅額現金，會在現貨市場護盤——最大Put OI
所在價位=地板(支撐)，最大Call OI所在價位=天花板(壓力)。

跟這次會話已經測過的put/call比率訊號不一樣：put/call比率是「當天全部
到期月份成交量/未平倉量的加總比率」，一個抽象的情緒指標；這裡測的是
「哪一個具體履約價OI最大」，一個價位，可以直接跟近月期貨價格算距離，
是這次會話第一次測試「選擇權部位當價位訊號」的角度。

同時測兩個版本：
1. 素樸版（使用者原話）：單一履約價OI最大值（max_call_oi_strike /
   max_put_oi_strike）——已知風險：深度價外的履約價常年掛著大量「結構型
   商品避險/長期尾部避險」的靜態OI，不代表真的有人會去那個價位防守，
   可能污染訊號（fetch腳本docstring已提過這個疑慮）。
2. 嚴謹版：正式max pain計算（讓所有賣方結算總賠付最小的履約價）——對
   單一履約價異常值比較不敏感，因為是對整條OI曲線加權，理論上更貼近
   莊家實際的避險誘因。

假說跟測法：
- 壓力：當近月期貨收盤價低於max_call_oi_strike、且距離夠近時，接下來
  N天報酬應該被壓抑（距離跟報酬應該正相關：離牆越遠越有上漲空間）。
- 支撐：當收盤價高於max_put_oi_strike、且距離夠近時，接下來N天報酬
  應該被撐住（距離跟報酬應該負相關：離牆越遠[越高]，越沒有下檔支撐可言
  ，這裡的「距離」定義是price-put_strike，越大代表越高，所以理論上
  距離大->還沒跌到支撐->下跌風險較高->跟後續報酬應該負相關）。
- max pain磁吸：距離(pain_strike - price)跟後續報酬應該正相關（pain在
  上方->價格被往上吸；pain在下方->價格被往下吸），且結算日前應該更強
  （pin risk理論）。

跟這次會話一貫的方法論一致：shift(1)避免用到當天才知道的OI、只用
IS(2001-2020)先篩、之後才決定要不要花OOS驗證機會、只用標準月合約(近月)
避免多個到期月份OI混在一起。

用法：
    python scripts/scan_max_oi_strike_signal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MAXOI_PATH = REPO_ROOT / "data" / "taifex_txo_max_oi_strike.parquet"
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
OUT_PATH = REPO_ROOT / "data" / "max_oi_strike_signal_grid.parquet"

IS_CUTOFF = "2021-01-01"
N_GRID = [1, 3, 5, 10, 20]
MIN_N = 100
# 「夠近」的距離門檻：素樸版用歷史分位數(收盤價相對履約價步階常態化)，
# 這裡固定用價格的2%當作門檻（TX目前約4-5萬點，2%約800-1000點，
# 大致等於1-2個履約價步階，跟"貼著牆"的直覺一致）
NEAR_THRESHOLD_PCT = 2.0


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def days_to_expiry(near_expiry: pd.Series) -> pd.Series:
    """同一個near_expiry(標準月合約標籤)區間內，從結算日往回數的交易日
    倒數(結算日當天=0，前一天=1...)。用near_expiry變化的邊界抓結算日，
    不用曆法天數（跟這次會話一貫的做法一致：用交易日不用日曆日）。"""
    out = np.zeros(len(near_expiry), dtype=int)
    n = len(near_expiry)
    i = n - 1
    while i >= 0:
        j = i
        while j >= 0 and near_expiry.iloc[j] == near_expiry.iloc[i]:
            j -= 1
        # (j, i] 是同一個到期月份的區間，i是這個區間最後一天(=結算日)
        group_len = i - j
        for k, idx in enumerate(range(j + 1, i + 1)):
            out[idx] = group_len - k
        i = j
    return pd.Series(out, index=near_expiry.index)


def main() -> None:
    maxoi = pd.read_parquet(MAXOI_PATH)
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]]
    df = maxoi.merge(tx, on="date", how="inner").sort_values("date").reset_index(drop=True)
    print(f"合併後交易日數: {len(df)} ({df['date'].min()} ~ {df['date'].max()})")

    df["dte"] = days_to_expiry(df["nearest_expiry"])

    # shift(1)：今天的訊號只能用「昨天收盤後才知道」的OI資料，不能用今天
    # 當天盤中/盤後才公布的OI去預測今天接下來的報酬（look-ahead陷阱，
    # 這次會話從三腿策略開始就一直很注意的事）
    price_lag = df["near_price"].shift(1)
    df["dist_call_pct"] = (df["max_call_oi_strike"].shift(1) - price_lag) / price_lag * 100
    df["dist_put_pct"] = (price_lag - df["max_put_oi_strike"].shift(1)) / price_lag * 100
    df["dist_pain_pct"] = (df["max_pain_strike"].shift(1) - price_lag) / price_lag * 100
    df["dte_lag"] = df["dte"].shift(1)

    is_mask = df["date"] < IS_CUTOFF
    print(f"IS交易日數: {is_mask.sum()}")

    rows = []

    for N in N_GRID:
        fwd_ret = (df["near_price"].shift(-N) / df["near_price"] - 1) * 100

        # ---- 壓力：只看價格在牆下方(dist_call_pct>0)的日子 ----
        valid = is_mask & df["dist_call_pct"].notna() & (df["dist_call_pct"] > 0) & fwd_ret.notna()
        near = fwd_ret[valid & (df["dist_call_pct"] <= NEAR_THRESHOLD_PCT)]
        far = fwd_ret[valid & (df["dist_call_pct"] > NEAR_THRESHOLD_PCT)]
        t_near, n_near, m_near = tstat(near)
        t_far, n_far, m_far = tstat(far)
        corr = np.corrcoef(df.loc[valid, "dist_call_pct"], fwd_ret[valid])[0, 1] if valid.sum() > MIN_N else np.nan
        rows.append(dict(signal="max_call_oi_wall(壓力)", N=N, corr=corr,
                          n_near=n_near, mean_near=m_near, t_near=t_near,
                          n_far=n_far, mean_far=m_far, t_far=t_far))

        # ---- 支撐：只看價格在牆上方(dist_put_pct>0)的日子 ----
        valid = is_mask & df["dist_put_pct"].notna() & (df["dist_put_pct"] > 0) & fwd_ret.notna()
        near = fwd_ret[valid & (df["dist_put_pct"] <= NEAR_THRESHOLD_PCT)]
        far = fwd_ret[valid & (df["dist_put_pct"] > NEAR_THRESHOLD_PCT)]
        t_near, n_near, m_near = tstat(near)
        t_far, n_far, m_far = tstat(far)
        corr = np.corrcoef(df.loc[valid, "dist_put_pct"], fwd_ret[valid])[0, 1] if valid.sum() > MIN_N else np.nan
        rows.append(dict(signal="max_put_oi_wall(支撐)", N=N, corr=corr,
                          n_near=n_near, mean_near=m_near, t_near=t_near,
                          n_far=n_far, mean_far=m_far, t_far=t_far))

        # ---- max pain磁吸：全樣本 ----
        valid = is_mask & df["dist_pain_pct"].notna() & fwd_ret.notna()
        above = fwd_ret[valid & (df["dist_pain_pct"] >= 1.0)]   # pain在上方1%以上
        below = fwd_ret[valid & (df["dist_pain_pct"] <= -1.0)]  # pain在下方1%以上
        t_above, n_above, m_above = tstat(above)
        t_below, n_below, m_below = tstat(below)
        corr = np.corrcoef(df.loc[valid, "dist_pain_pct"], fwd_ret[valid])[0, 1] if valid.sum() > MIN_N else np.nan
        rows.append(dict(signal="max_pain磁吸(全樣本)", N=N, corr=corr,
                          n_near=n_above, mean_near=m_above, t_near=t_above,
                          n_far=n_below, mean_far=m_below, t_far=t_below))

        # ---- max pain磁吸：只看結算日前5個交易日內(pin risk理論最強的時候) ----
        valid = is_mask & df["dist_pain_pct"].notna() & fwd_ret.notna() & (df["dte_lag"] <= 5)
        above = fwd_ret[valid & (df["dist_pain_pct"] >= 1.0)]
        below = fwd_ret[valid & (df["dist_pain_pct"] <= -1.0)]
        t_above, n_above, m_above = tstat(above)
        t_below, n_below, m_below = tstat(below)
        corr = np.corrcoef(df.loc[valid, "dist_pain_pct"], fwd_ret[valid])[0, 1] if valid.sum() > MIN_N else np.nan
        rows.append(dict(signal="max_pain磁吸(結算前5日)", N=N, corr=corr,
                          n_near=n_above, mean_near=m_above, t_near=t_above,
                          n_far=n_below, mean_far=m_below, t_far=t_below))

    res = pd.DataFrame(rows)
    res.to_parquet(OUT_PATH, index=False)

    for sig in res["signal"].unique():
        print(f"\n=== {sig} ===")
        sub = res[res["signal"] == sig]
        print(sub[["N", "corr", "n_near", "mean_near", "t_near", "n_far", "mean_far", "t_far"]]
              .to_string(index=False))

    print(f"\nsaved: {OUT_PATH}")
    print("以上全部只用IS(2001-2020)，尚未看OOS，也尚未做不重疊抽樣校正/子區間穩健性檢查。")


if __name__ == "__main__":
    main()
