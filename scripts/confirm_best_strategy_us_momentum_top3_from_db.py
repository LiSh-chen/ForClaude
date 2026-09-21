"""最終確認腳本：用「目前」資料庫的最新狀態（20 年歷史回填 + 52+2 檔
剔除股回補後，共 631 檔股票的 us_index_membership），重新驗證報告主要
引用的策略（top_n=3、momentum_window=126、rebalance_freq_days=21）在
樣本內/樣本外兩段的完整績效指標，並跟 QQQ 買進持有同期間做逐項對照。

背景：2026-09-21 對話紀錄——這個 session 為了測 2008 金融風暴，陸續把
US 股價資料從 577 檔（2018-09-21~2026-09-18）擴充到 631 檔
（2006-09-25~2026-09-18，含 52+2 檔 2019 年後被剔除指數的回補股票）。
資料庫底層股票池組成改變了，代表先前報告反覆引用的「OOS 116.67% vs QQQ
107.60%，五項指標全勝」這個數字，有可能已經因為資料更新而過時——
2026-09-21 稍早跑 test_us_momentum_top3_trend_filter_2008_crisis_from_db.py
的「無濾網」對照組時，「既有樣本外 2018-09-20~2023-09-18」印出的數字是
total_ret=106.48%（QQQ 同期 107.82%），已經跟舊數字不一致，需要一支
專門的腳本把完整 5 項指標（含 QQQ 的 sharpe/calmar）重新算一次、白紙黑字
確認，不能繼續引用可能過時的舊數字。

反未來函數：跟其他所有美股策略腳本一致——永遠傳完整 us_prices（不切片），
只用 start_date/end_date 限制「哪些日期允許實際調倉」。

2026-09-21 架構修正：改用完整未過濾的 us_prices + run_factor_backtest 的
membership 參數，取代先前「先用 filter_prices_by_index_membership 砍過
一輪再傳進引擎」的舊寫法——後者會連帶砍掉均線/均量/歷史長度指標賴以
計算的完整價格序列，誤傷「公司歷史悠久、但指數成分股身份中途一度中斷
又重新加入」的股票（例如 MRVL、FLEX、CASY、COHR、CIEN 等 14 檔，詳見
tw_quant/us_universe.py 檔頭說明）。這裡是第一支改用新架構重新驗證的
腳本，數字如果跟先前（含這個 session 稍早）用舊架構跑出來的版本不同，
差異就是這個 bug 修正的直接影響。

用法：
    python scripts/confirm_best_strategy_us_momentum_top3_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 3

IN_SAMPLE_START = pd.Timestamp("2023-09-19")
OOS_START = pd.Timestamp("2018-09-20")
OOS_END = IN_SAMPLE_START - pd.Timedelta(days=1)


def _qqq_metrics(qqq_close: pd.Series, start, end) -> dict:
    s = pd.Timestamp(start) if start else qqq_close.index.min()
    e = pd.Timestamp(end) if end else qqq_close.index.max()
    window = qqq_close[(qqq_close.index >= s) & (qqq_close.index <= e)]
    daily_ret = window.pct_change().dropna()
    total_ret = float(window.iloc[-1] / window.iloc[0] - 1)
    n_years = (window.index[-1] - window.index[0]).days / 365.25
    cagr = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    running_max = window.cummax()
    max_dd = float(-((window - running_max) / running_max).min())
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0
    return {"total_return": total_ret, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar}


def _fmt(label: str, m: dict) -> str:
    return (
        f"{label:<28} total_ret={m['total_return']:>8.2%}  cagr={m['cagr']:>7.2%}  "
        f"max_dd={m['max_dd']:>7.2%}  sharpe={m['sharpe']:>6.2f}  calmar={m['calmar']:>6.2f}"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()  # 完整、未過濾——membership 資格判定交給 run_factor_backtest 處理
    membership = load_us_index_membership_snapshot()

    n_stocks = us_prices["stock_id"].nunique()
    earliest, latest = us_prices["date"].min(), us_prices["date"].max()

    print(f"目前資料庫狀態：{n_stocks} 檔股票，{earliest.date()} ~ {latest.date()}（完整未過濾，membership 資格判定交給引擎處理）\n")

    base_cfg = build_us_config()
    factor_cfg = FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False)

    print(f"固定參數：momentum_window={MOM_WINDOW}、rebalance_freq_days={REBALANCE_FREQ_DAYS}、top_n={TOP_N}\n")

    provider = YFinanceUSDataProvider()
    qqq_df = provider.fetch_price("QQQ", start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF")
    qqq_close = qqq_df.sort_values("date").set_index("date")["close"]

    print("=== 樣本外 OOS（2018-09-20 ~ 2023-09-18，跟舊資料的 116.67% 對照）===")
    # 注意：start_date 必須明確指定 OOS_START，不能用 None——None 的語意是
    # 「不限制起點」，資料庫還只到 2018-09-21 時 None 剛好等於 2018-09-20，
    # 但現在資料庫擴充到 2006-09-25 了，None 會變成從 2006 年開始交易、
    # 混入 2008 危機期間，不是原本定義的 OOS 段（這裡第一次跑就踩到這個
    # 坑：QQQ total_ret 印出離譜的 951%，回撤也跟「延伸樣本外 2006-2018」
    # 的 79.04% 一模一樣，才發現 start_date=None 的語意已經隨資料庫擴充
    # 跟著變了）。動量排名計算依然用完整未切片的 us_prices（2006 年起的
    # 完整歷史），只是「允許交易」的窗口明確限制在 OOS_START~OOS_END。
    oos_result = run_factor_backtest(us_prices, base_cfg, factor_cfg, start_date=OOS_START, end_date=OOS_END, cost_module=us_costs, membership=membership)
    oos_m = metrics_from_result(oos_result, base_cfg.initial_capital, prices=us_prices)
    print(_fmt("策略（最新資料）", oos_m))
    print(_fmt("QQQ 買進持有", _qqq_metrics(qqq_close, OOS_START, OOS_END)))
    print(f"策略交易筆數：{oos_m['n_trades']:.0f}、勝率：{oos_m['win_rate']:.1%}")
    print(f"起始權益 $10,000,000，結束權益 ${base_cfg.initial_capital * (1 + oos_m['total_return']):,.0f}\n")

    print("=== 樣本內 IS（2023-09-19 ~ 資料庫最新日期）===")
    is_result = run_factor_backtest(us_prices, base_cfg, factor_cfg, start_date=IN_SAMPLE_START, end_date=None, cost_module=us_costs, membership=membership)
    is_m = metrics_from_result(is_result, base_cfg.initial_capital, prices=us_prices)
    print(_fmt("策略（最新資料）", is_m))
    print(_fmt("QQQ 買進持有", _qqq_metrics(qqq_close, IN_SAMPLE_START, None)))
    print(f"策略交易筆數：{is_m['n_trades']:.0f}、勝率：{is_m['win_rate']:.1%}")
    print(f"起始權益 $10,000,000，結束權益 ${base_cfg.initial_capital * (1 + is_m['total_return']):,.0f}\n")

    print(
        "（誠實提醒：這裡的數字如果跟這個 session 稍早（架構修正前）跑出來的版本不同，\n"
        "差異來自 membership 架構修正本身——先前先過濾再算指標的舊寫法，會誤傷\n"
        "MRVL、FLEX、CASY、COHR、CIEN 等 14 檔歷史悠久、但指數成分股身份中途一度\n"
        "中斷又重新加入的股票，這裡改用完整未過濾的價格序列 + membership 參數，\n"
        "讓這些股票的均線/均量/歷史長度指標算對，同一組參數、同一顆引擎，純粹是\n"
        "股票池資格判定的架構不同）"
    )


if __name__ == "__main__":
    main()
