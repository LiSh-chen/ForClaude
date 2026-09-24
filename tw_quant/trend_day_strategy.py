"""趨勢日偵測（Trend Day Detection），日盤當沖、用 1 分鐘資料判斷「今天是
不是趨勢盤」再進場，跟開盤區間突破（ORB，已經驗證失敗——OOS轉負）根本
不同的篩選邏輯：

ORB 的問題：只要價格「摸到」開盤區間邊界一次就進場，抓到的是很多「早盤
噴出一下馬上被巴」的假突破日，這種日子在越來越有效率的市場裡越來越常見
（這次會話已經證實 ORB 的邊際逐年衰退到轉負）。

這裡改用更嚴格的篩選：不是「有沒有突破」，是「有沒有持續維持在盤中VWAP
（成交量加權均價，從當天開盤累積算）同一側，且已經走出有意義的幅度」。
邏輯是：真正的趨勢日，價格會早早站上（或跌破）VWAP 之後就不再回頭測試，
盤整/假突破日則會在VWAP兩側來回穿梭——用「到判斷時點為止有沒有回頭
穿越過 VWAP」當篩選器，比單純的區間邊界突破更嚴格，理論上能濾掉大部分
的假訊號日，只留下真正單邊控盤的那些日子。

下單機制：
- decision_time（預設11:00）之前，逐分鐘算「當天累積 VWAP」跟收盤價的
  相對位置；如果整段期間收盤價都在 VWAP 同一側（沒有反向穿越過），且
  從開盤到 decision_time 的累積移動幅度 >= min_move_points，判定為
  「趨勢日候選」，方向＝VWAP的那一側。
- 進場：decision_time 那根K棒收盤後，用下一根K棒的開盤價成交（市價單
  概念，讓決策跟成交分開，避免用同一根K棒的資訊進出場），扣滑價。
- 出場（趨勢失效停損）：進場後任何一分鐘，如果收盤價穿越回 VWAP 的
  反向那一側，視為趨勢假設失效，當根K棒收盤價成交出場（市價單，扣
  滑價）——VWAP 本身每分鐘都在變動，這是「移動中的停損水準」，不是
  固定停損單，是系統化策略常見的做法（跟固定停損單機制不同，這裡簡化
  用收盤價判斷觸發跟成交，不逐檔模擬停損單掛單）。
- 沒有觸發停損就抱到 session_end（收盤前強制平倉），市價出場、扣滑價。
- 同一天只交易一次（決策點只評估一次），沒有趨勢日候選的日子完全不交易。

簡化與已知限制：
- 只做日盤（08:45-13:45），不含夜盤。
- 參考水準（VWAP或開盤價）的停損用收盤價判斷+同根K棒收盤價成交，是簡化
  （沒有模擬停損單的跳空/滑價機制）；正常滑價仍然扣在出場成交價上。
- 「有沒有回頭穿越參考水準」用收盤價逐分鐘檢查，不用高低點（用高低點會
  太敏感，稍微碰一下就判定失效，不符合「趨勢日」通常允許小幅拉回但
  不破均價的現實）。

VWAP vs open（reference參數）：
VWAP 版本（reference="vwap"，預設、已經完整驗證過）在實務上有個麻煩：
出場水準是「移動中的」，每分鐘都要重新累積運算，手動或用一般看盤軟體
不容易即時追蹤，需要額外寫程式。open 版本改用「當天開盤價」當參考水準
──整天固定不變，判斷「有沒有跨過」只要記住開盤價這一個數字，出場水準
也能直接掛一張固定價位的真實停損單（不像 VWAP 停損需要每分鐘重新判斷、
沒辦法真的掛在市場上）。這是用「可能稍微犧牲一點篩選品質」換「大幅
降低實務執行門檻」，實際效果需要重新完整驗證，不能假設兩者等價。

已知結論更新（重要）：把 reference 換成 "open" 或改用固定 ATR 停損停利
（見 momentum_checkpoint_strategy.py）都在完整驗證中明確失敗、OOS轉負。
進一步拆解發現關鍵不在「進場濾網用什麼當參考」，而在「出場機制夠不夠
積極」——VWAP版本進場後約25%的交易會提早被「趨勢失效」出場，固定水準
版本（開盤價/ATR）幾乎99%以上都是硬抱到收盤才出場，等於移除了「趨勢
真的走掉就提早停損、不要硬拗」這個核心風控動作。這指向：真正該保留的
不是「VWAP」這個特定指標本身，是它提供的「隨價格動態調整、進場後仍會
頻繁重新評估」這個機制。

另一個一直沒控制的變數（重要）：TXF 價格水準 23 年間漲了約 3.3 倍
（2001年均價約4900點，2023年約16400點）。min_move_points 這個進場
幅度門檻整段會話都是用「絕對點數」設定（預設20點），代表同一個門檻在
2001年約當0.4%的價格波動，2023年卻只剩約0.12%——濾網隨著指數點位
上升變得越來越寬鬆，越到後面越容易放行雜訊訊號。這很可能是這次會話
反覆看到「早期強、近十年弱」型態的部分成因（不只這個策略，前面測過的
其他好幾個策略都用了類似的絕對點數門檻）。

min_move_pct 參數（用來控制這個變數）：設定後會取代 min_move_points，
改用「決策時點累積幅度 / 當天開盤價」的百分比當門檻，隨價格水準自動
調整，理論上能讓濾網在 23 年間維持一致的相對嚴格程度，不會隨指數上漲
變得越來越寬鬆。預設 None（維持原本用絕對點數的行為不變）。

exit_mode 參數（獨立於 reference，用來測試上面這個推論）：
- "reference_break"（預設，原始版本）：出場水準＝進場濾網用的同一個
  參考水準（VWAP或開盤價），持續判斷有沒有穿越。
- "trailing_atr"：改用「移動停損單」（多數券商下單軟體/API都有這個
  現成的委託類型，掛一次就會自動跟著價格走，不需要自己逐分鐘運算）——
  停損水準＝進場後至今最有利價位（多單看最高收盤、空單看最低收盤）
  減/加 atr_stop_mult 倍的「前一天為止」日線ATR，只會往有利方向收緊，
  不會鬆動。進場濾網（entry_reference）不變，只是把「持續盯盤判斷出場」
  換成「掛一張會自動追蹤的停損單」，兩者理論上都能提供類似的「趨勢
  走掉就提早出場」效果，但trailing_atr完全不需要交易人自己即時運算
  任何東西。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from tw_quant.donchian_breakout_strategy import _atr

POINT_VALUE = 50.0


@dataclass
class TrendDayConfig:
    decision_time: time = time(11, 0)
    min_move_points: float = 20.0
    min_move_pct: float | None = None  # 設定後取代 min_move_points，見檔頭說明
    session_end: time = time(13, 25)
    slippage_points: float = 1.0
    min_dominant_side_fraction: float = 1.0  # 1.0=原始版本「決策時點前完全沒穿越過參考水準」
    # <1.0＝放寬：允許決策時點前有一部分分鐘K棒收在參考水準反向側（雜訊型短暫拉回），
    # 只要「多數方向」那一側的比例達到這個門檻就算趨勢日候選，方向＝多數方向。
    reference: str = "vwap"  # 進場濾網用的參考水準："vwap"（預設）或 "open"
    exit_mode: str = "reference_break"  # "reference_break"（預設）或 "trailing_atr"（見檔頭說明）
    atr_window: int = 14
    atr_stop_mult: float = 1.5


def _day_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d


def _prior_day_atr_map(df: pd.DataFrame, atr_window: int) -> dict:
    from tw_quant.technical_indicators import build_daily_bars

    daily = build_daily_bars(df)
    daily["atr_prior"] = _atr(daily, atr_window).shift(1)
    return dict(zip(daily["date"].dt.date, daily["atr_prior"]))


def backtest(df: pd.DataFrame, cfg: TrendDayConfig | None = None) -> pd.DataFrame:
    cfg = cfg or TrendDayConfig()
    day_df = _day_session_frame(df)

    exit_reason_break = "vwap_break" if cfg.reference == "vwap" else "open_break"
    atr_map = _prior_day_atr_map(df, cfg.atr_window) if cfg.exit_mode == "trailing_atr" else {}

    trades = []
    for trading_date, g in day_df.groupby("trading_date"):
        g = g.sort_values("datetime").reset_index(drop=True)
        if cfg.reference == "vwap":
            typical_price = (g["high"] + g["low"] + g["close"]) / 3
            cum_pv = (typical_price * g["volume"]).cumsum()
            cum_v = g["volume"].cumsum().replace(0, np.nan)
            g["ref_price"] = cum_pv / cum_v
        else:
            g["ref_price"] = g["open"].iloc[0]  # 整天固定＝開盤價，不需要逐分鐘重算

        t = g["datetime"].dt.time
        decision_mask = t <= cfg.decision_time
        if not decision_mask.any():
            continue
        decision_idx = decision_mask[decision_mask].index[-1]
        if decision_idx + 1 >= len(g):
            continue  # 判斷時點之後沒有下一根K棒可以進場，跳過

        pre = g.loc[: decision_idx]
        side = np.sign(pre["close"] - pre["ref_price"])
        side = side.replace(0, np.nan).dropna()
        if side.empty:
            continue

        counts = side.value_counts()
        dominant_sign = counts.idxmax()
        dominant_fraction = counts.max() / len(side)
        if dominant_fraction < cfg.min_dominant_side_fraction:
            continue  # 反向側的比例超過容許範圍，不是趨勢日候選

        direction_sign = dominant_sign
        day_open = g["open"].iloc[0]
        decision_close = g["close"].iloc[decision_idx]
        move = (decision_close - day_open) * direction_sign
        move_threshold = cfg.min_move_pct * day_open if cfg.min_move_pct is not None else cfg.min_move_points
        if move < move_threshold:
            continue

        if cfg.exit_mode == "trailing_atr":
            atr_prior = atr_map.get(trading_date)
            if atr_prior is None or pd.isna(atr_prior):
                continue  # 沒有前一天ATR（例如資料第一天），跳過

        direction = "long" if direction_sign > 0 else "short"
        entry_bar = g.iloc[decision_idx + 1]
        entry_price = entry_bar["open"] + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)
        entry_dt = entry_bar["datetime"]

        exit_price = exit_dt = exit_reason = None
        if cfg.exit_mode == "trailing_atr":
            trail_distance = cfg.atr_stop_mult * atr_prior
            extreme = entry_price
            for j in range(decision_idx + 1, len(g)):
                r = g.iloc[j]
                if t.iloc[j] >= cfg.session_end:
                    break
                if direction == "long":
                    extreme = max(extreme, r["close"])
                    stop = extreme - trail_distance
                    if r["close"] <= stop:
                        exit_price = r["close"] - cfg.slippage_points
                        exit_dt, exit_reason = r["datetime"], "trailing_stop"
                        break
                else:
                    extreme = min(extreme, r["close"])
                    stop = extreme + trail_distance
                    if r["close"] >= stop:
                        exit_price = r["close"] + cfg.slippage_points
                        exit_dt, exit_reason = r["datetime"], "trailing_stop"
                        break
        else:
            for j in range(decision_idx + 1, len(g)):
                r = g.iloc[j]
                if t.iloc[j] >= cfg.session_end:
                    break
                broke = (r["close"] < r["ref_price"]) if direction == "long" else (r["close"] > r["ref_price"])
                if broke:
                    exit_price = r["close"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
                    exit_dt, exit_reason = r["datetime"], exit_reason_break
                    break

        if exit_price is None:
            close_mask = t >= cfg.session_end
            last_bar = g.loc[close_mask].iloc[0] if close_mask.any() else g.iloc[-1]
            exit_price = last_bar["open"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
            exit_dt, exit_reason = last_bar["datetime"], "session_close"

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            trading_date=trading_date, direction=direction, decision_close=decision_close,
            move_at_decision=move, entry_dt=entry_dt, entry_price=entry_price,
            exit_dt=exit_dt, exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_dt").reset_index(drop=True)
