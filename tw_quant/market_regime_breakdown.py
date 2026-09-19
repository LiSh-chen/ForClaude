"""把一段連續歷史依市場狀態（動能強勢期／盤整期／修正下跌期）分組，供事後
歸因分析用：策略在不同市況下的表現是不是真的有條件式的規律，還是整段期間
單純打平/超額只是不同市況互相抵銷的結果。

跟 tw_quant/regime.py（台股大盤綠燈/紅燈判定與參數敏感度檢驗）是不同用途
的模組，刻意用不同檔名避免混淆：那邊是拿來當「要不要允許新增部位」的即時
交易門檻，這裡純粹是回測結束後對已經算好的逐日報酬率做歸因分組，不參與
任何交易決策。

背景：使用者觀察到 top_n=3 動量策略在樣本內（2023下半年起的科技動能噴出期）
大幅超越 QQQ，但在樣本外（2018~2023，混合了 COVID 崩盤、2022 熊市、多次
多頭段的 5 年期間）只小幅打平，因而假設「動能強勢期有超額報酬、平穩期不輸
大盤」。這個假設本身沒有被驗證過（樣本外整段結果沒辦法分辨內部是「全程小贏」
還是「強勢期贏、下跌期輸，加總打平」），這裡提供分組工具來拆解驗證。

反未來函數：trailing_return_regime_labels 用 shift(1)，T 日的市場狀態標籤
只會用到 T-1 為止已經發生的報酬率，不會偷看 T 日當天的漲跌。門檻本身（三分
位數）用全樣本分位數決定，這是事後歸因分組用的統計手法，不是即時交易訊號，
不會影響、也沒有用到策略本身逐日報酬率的計算（策略的動能排名跟進出場邏輯
完全獨立算好之後，這裡只是把已經算出來的逐日報酬率依日期分組）。
"""

from __future__ import annotations

import pandas as pd

REGIME_LOW = "修正／下跌期"
REGIME_MID = "盤整期"
REGIME_HIGH = "動能強勢期"


def trailing_return_regime_labels(close: pd.Series, lookback_days: int) -> pd.Series:
    """依 T-1 為止 lookback_days 天的已實現報酬率，把 close 的每個日期分類成
    REGIME_LOW / REGIME_MID / REGIME_HIGH 三分位其中一種（分位數用全樣本決定，
    見模組開頭說明），回傳跟 close 同索引的標籤 Series（NaN 代表資訊不足，
    通常是序列最前面 lookback_days+1 天）。
    """
    trailing_ret = close.pct_change(lookback_days).shift(1)
    valid = trailing_ret.dropna()
    if valid.empty:
        return trailing_ret.map(lambda _: None)
    q_low, q_high = valid.quantile([1 / 3, 2 / 3])

    def _label(x: float) -> str | None:
        if pd.isna(x):
            return None
        if x <= q_low:
            return REGIME_LOW
        if x >= q_high:
            return REGIME_HIGH
        return REGIME_MID

    return trailing_ret.map(_label)


def regime_episode_stats(labels: pd.Series, label: str) -> tuple[int, int]:
    """回傳 (獨立連續區段數, 最長連續區段的交易日數)，用來說明某個市場狀態
    的天數是不是集中在少數幾段連續期間（例如整個 COVID 崩盤算一段），而不是
    分散、獨立的許多次事件——這會影響能不能把「這個狀態下策略表現比較好/差」
    當作有統計意義的規律，還是只是一兩段特殊期間主導的結果。
    """
    mask = labels == label
    if not mask.any():
        return 0, 0
    groups = (mask != mask.shift()).cumsum()
    run_lengths = mask[mask].groupby(groups[mask]).size()
    return int(run_lengths.shape[0]), int(run_lengths.max())
