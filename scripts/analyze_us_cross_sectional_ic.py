"""美股版的全面 Rank IC 掃描：涵蓋技術面、財報面兩大類候選因子，找出哪些
參數對未來報酬最有預測力——方法跟台股版
scripts/analyze_return_regularities_from_db.py 完全一致（見
tw_quant/factor_ic.py 的方法說明），只是資料源換成美股、因子清單也依美股
實際能拿到的資料重新設計。

★★★ 重要：「籌碼面」在美股目前無法做 ★★★
使用者要求財務、籌碼、技術三個面向都做。台股版的籌碼因子是融資融券餘額
（FinMind 每日更新的水位指標），美股完全沒有對應資料：yfinance 的
`.info`/`institutional_holders`/`major_holders`/`insider_transactions` 只有
「現在」的快照，沒有歷史時間序列，没辦法拿來做需要逐日對齊的 IC 分析
（見 2026-10-01 的調查紀錄）。這裡誠實跳過籌碼面，不拿量能指標冒充——
相對量能被歸類在技術面（量能本來就是技術分析的一部分，不是持股結構）。
如果之後要真的做美股籌碼面，需要額外付費資料源（例如 13F 持股申報、
FINRA 空單回報）才可能有逐日/逐週的歷史資料可用。

技術面候選（見 build_technical_factors）：
  - 價格動量（trailing L 日報酬率）
  - 相對量能（當日量 / L 日均量）
  - 波動度（L 日日報酬標準差）
  - 均線乖離（收盤價相對 L 日均線的乖離率，趨勢強度代理指標）
  - RSI（相對強弱指標）
  - 布林通道 z-score
  - 距 52 週高點的百分比

財務面候選：
  - 財報驚喜幅度（surprise_pct，用 merge_asof 往回對齊到「當時已公告、
    尚未過期」的最近一次財報，不會用到未來才公布的財報資料）——這是
    唯一一個美股資料管線裡有、而且有完整歷史時間序列的「財務」因子
    （P/E、P/B、負債比等財報比率 yfinance 只保留約 5 季深度，無法做
    20 年的歷史 IC 分析，見 2026-10-01 調查紀錄）。

全部計算完全讀本機已經 commit 的 Parquet 快照（價量、財報、成分股區間），
不連資料庫、不用網路，可以在這個沙盒環境直接跑。

用法：
    python scripts/analyze_us_cross_sectional_ic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tw_quant.data_snapshot import load_us_earnings_snapshot, load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.factor_ic import fmt_ic_table, forward_return, ic_summary, rank_ic_series
from tw_quant.indicators import avg_volume, rolling_max, rolling_std, sma
from tw_quant.us_universe import membership_eligibility_mask

MOMENTUM_L_GRID = (5, 10, 20, 40, 60, 90, 120, 180, 252)
VOLUME_L_GRID = (5, 10, 20, 60)
VOLATILITY_L_GRID = (20, 60, 120)
MA_TREND_L_GRID = (50, 100, 150, 200)
RSI_L_GRID = (14, 21)
BOLLINGER_L = 20
FORWARD_F_GRID = (5, 10, 20, 60)

# 美股股票池比台股大（約570檔 vs 150檔），同樣的「避免小樣本雜訊冒充規律」
# 門檻拉高一點，門檻本身的選擇不影響結論方向，只影響要排除掉多少早期
# 股票池還沒湊齊的日子。
MIN_STOCKS_PER_DAY = 50

# 財報驚喜幅度超過這麼多天還沒有新一季財報，就不再使用這筆舊資料當訊號——
# 避免拿掉上一季、甚至上上季的驚喜幅度硬套在今天身上，那已經不是「財報
# 驚喜」而是陳舊資訊。一季約 63 個交易日/91 個日曆日，抓 95 天留一點緩衝。
EARNINGS_SURPRISE_MAX_STALE_DAYS = 95


def _rsi(master: pd.DataFrame, window: int) -> pd.Series:
    """Wilder's RSI：用 EWM(alpha=1/window) 平滑漲跌幅，跟
    tw_quant/indicators.py 的 atr() 用同一種平滑方式，風格一致。"""
    delta = master.groupby("stock_id", sort=False)["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    tmp = pd.DataFrame(
        {"stock_id": master["stock_id"].to_numpy(), "gain": gain.to_numpy(), "loss": loss.to_numpy()},
        index=master.index,
    )
    avg_gain = tmp.groupby("stock_id", sort=False)["gain"].transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    avg_loss = tmp.groupby("stock_id", sort=False)["loss"].transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_technical_factors(master: pd.DataFrame) -> dict[str, pd.Series]:
    """master 須已依 (stock_id, date) 排序。回傳 {因子名稱: 訊號序列}，
    序列跟 master 同長度、同順序、同 index。"""
    factors: dict[str, pd.Series] = {}

    for L in MOMENTUM_L_GRID:
        factors[f"price_momentum_L{L}"] = master.groupby("stock_id", sort=False)["close"].pct_change(L)

    for L in VOLUME_L_GRID:
        factors[f"relative_volume_L{L}"] = master["volume"] / avg_volume(master, L)

    daily_return = master.groupby("stock_id", sort=False)["close"].pct_change()
    ret_df = master[["stock_id"]].copy()
    ret_df["_ret"] = daily_return
    for L in VOLATILITY_L_GRID:
        factors[f"volatility_L{L}"] = rolling_std(ret_df, "_ret", L)

    for L in MA_TREND_L_GRID:
        factors[f"ma_trend_L{L}"] = master["close"] / sma(master, "close", L) - 1

    for L in RSI_L_GRID:
        factors[f"rsi_L{L}"] = _rsi(master, L)

    boll_mean = sma(master, "close", BOLLINGER_L)
    boll_std = rolling_std(master, "close", BOLLINGER_L)
    factors[f"bollinger_zscore_L{BOLLINGER_L}"] = (master["close"] - boll_mean) / boll_std

    factors["pct_from_52w_high"] = master["close"] / rolling_max(master, "close", 252) - 1

    return factors


def build_earnings_surprise_factor(master: pd.DataFrame, earnings: pd.DataFrame) -> pd.Series:
    """用 merge_asof 把「當日已知、最近一次公告」的財報驚喜幅度對齊到
    master 每一列——只往回看（direction="backward"），不會用到未來才
    公布的財報，加上 tolerance 避免用太陳舊的數字。

    merge_asof 要求整體依 "on" 欄位排序（不能只依群組排序），這裡另外
    開一個依 date 排序的副本做 merge，再用暫存的列號 mapping 回 master
    原本的列順序，master 本身的 (stock_id, date) 排序不受影響。
    """
    # us_prices_snapshot 跟 us_earnings_snapshot 這兩份 Parquet 快照各自獨立
    # 寫入，datetime 的內部精度（us vs ms）不保證一致，merge_asof 要求兩邊
    # join 欄位 dtype 完全相同，這裡先統一轉成 datetime64[ns] 再合併。
    e = earnings.dropna(subset=["surprise_pct"])[["date", "stock_id", "surprise_pct"]].sort_values("date")
    e["date"] = e["date"].astype("datetime64[ns]")

    left = master[["date", "stock_id"]].copy()
    left["date"] = left["date"].astype("datetime64[ns]")
    left["_row"] = np.arange(len(left))
    left_sorted = left.sort_values("date")

    merged = pd.merge_asof(
        left_sorted,
        e,
        on="date",
        by="stock_id",
        direction="backward",
        tolerance=pd.Timedelta(days=EARNINGS_SURPRISE_MAX_STALE_DAYS),
    )
    merged = merged.sort_values("_row")
    return pd.Series(merged["surprise_pct"].to_numpy(), index=master.index)


def scan_all_factors(master: pd.DataFrame, factors: dict[str, pd.Series], eligible: pd.Series) -> pd.DataFrame:
    """對每個 (因子, forward horizon) 組合算 IC 摘要。eligible 是跟 master
    同長度的布林遮罩（見 tw_quant/us_universe.py 的 membership_eligibility_mask），
    把「當天還不是/已經不是成分股」的列從訊號裡挖掉（設成 NaN），避免
    用倖存者偏差污染過的股票池算出虛高的 IC。
    """
    rows = []
    for F in FORWARD_F_GRID:
        fwd = forward_return(master, F)
        for name, sig in factors.items():
            sig_masked = sig.where(eligible)
            df = pd.DataFrame(
                {"date": master["date"].to_numpy(), "_sig": sig_masked.to_numpy(), "_fwd": fwd.to_numpy()}
            )
            ic = rank_ic_series(df, "_sig", "_fwd", min_stocks_per_day=MIN_STOCKS_PER_DAY)
            summary = ic_summary(ic)
            rows.append({"family": name, "horizon": F, **summary})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()
    earnings = load_us_earnings_snapshot()

    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True)
    eligible = membership_eligibility_mask(master, membership)

    print(
        f"讀到 {master['stock_id'].nunique()} 檔股票"
        f"（{master['date'].min().date()} ~ {master['date'].max().date()}），"
        f"membership 篩選後符合資格的列數佔比 {eligible.mean():.1%}\n"
    )

    print("建構技術面因子...")
    factors = build_technical_factors(master)
    print(f"建構財報驚喜幅度因子（merge_asof 往回對齊，容許 {EARNINGS_SURPRISE_MAX_STALE_DAYS} 天內的舊資料）...")
    factors["earnings_surprise_pct"] = build_earnings_surprise_factor(master, earnings)
    print(f"共 {len(factors)} 個候選因子，{len(FORWARD_F_GRID)} 種 forward horizon，開始算 IC...\n")

    result = scan_all_factors(master, factors, eligible)
    result = result.dropna(subset=["mean_ic"])

    print("=== 全部組合（依 |mean_IC| 排序，前 30 組）===")
    print(fmt_ic_table(result, top_n=30))

    # 每個因子家族取「跨 4 種 horizon 裡 |mean_IC| 最大」的那一組代表這個因子，
    # 避免同一個因子在不同 horizon 出現好幾次把排行榜洗版。
    best_per_family = result.loc[result.groupby("family")["mean_ic"].apply(lambda s: s.abs().idxmax())]
    best_per_family = best_per_family.sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)

    print("\n=== 每個因子取最佳 horizon 代表，前 5 名（這就是「影響股價前五名的參數」）===")
    print(fmt_ic_table(best_per_family, top_n=5))

    strong = result[(result["mean_ic"].abs() >= 0.02) & (result["t_stat"].abs() >= 2.0)]
    print(
        f"\n=== 總結：{len(result)} 組 (因子, horizon) 中，"
        f"{len(strong)} 組 |mean IC| >= 0.02 且 |t| >= 2.0（粗略的「有規律」門檻，"
        f"注意 t 值沒有 Newey-West 修正，重疊窗格會讓 t 值偏樂觀）==="
    )

    print(f"\n（全部計算耗時 {time.time() - t0:.1f} 秒）")


if __name__ == "__main__":
    main()
