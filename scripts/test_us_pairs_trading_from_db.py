"""把配對交易/統計套利策略（tw_quant/pairs_trading.py）重新套用在美股
S&P 500 資料上，跟台股版（scripts/test_pairs_trading_from_db.py，結果是
否證：9 組參數全部負報酬，最佳 Sharpe -0.73）做同樣的網格搜尋，驗證
「換一個流動性/微結構都不同的市場，這個策略類型是不是還是不管用」。

跟台股版的差異：
  1. 資料來源換成 us_prices，成本模型/整股限制換成美股版
     （tw_quant/us_config.py + tw_quant/us_costs.py）。網格、配對邏輯、
     z-score 進出場規則完全相同。
  2. 效能：台股 150 檔股票、單一產業最多幾十檔，_find_pairs 逐對呼叫
     coint()（O(股票數^2)）還算得動。S&P 500 有 503 檔，GICS 單一產業動輒
     60~80 檔，同樣的窮舉法第一次真的接上真實資料跑，"Test pairs trading
     on US data" 這個 step 在 30 分鐘 timeout 內連第一組參數組合都沒跑完
     就被強制取消——這不是訊號邏輯的問題，是純粹的計算量問題。因此這裡
     用兩個新的效能參數（tw_quant/pairs_trading.py 的
     pre_filter_min_abs_corr / max_stocks_per_industry，兩者預設值都是
     「關閉」，不影響台股版任何已發表結果）：
       - pre_filter_min_abs_corr=0.6：呼叫昂貴的 coint() 之前，先用便宜
         很多、向量化算好的相關係數矩陣篩一輪，濾掉明顯不相關的候選。
       - max_stocks_per_industry=25：每個產業最多只保留成交金額最高的
         25 檔進候選（其餘直接不參與共整合檢定）。
     這兩個參數都是「效能取捨」，不是「訊號設計」的一部分——會讓實際搜尋
     到的候選配對比窮舉法少（尤其排除了低流動性但可能真的共整合的配對），
     這點誠實揭露：美股版的配對搜尋不是完全窮舉。

沿用台股版檔頭誠實揭露的限制（做空只算了名目上的稅/手續費，沒有算真實
借券費/保證金利息；配對只在同產業內找；避險比例形成期內凍結不重估）——
這些限制對美股版一樣成立。

2026-09-21 架構修正：這支腳本先前直接連 get_data_store()/store.
load_us_prices()（本機空的 SQLite DB，這個沙盒環境完全連不到資料、
根本跑不動），也完全沒有處理 S&P 500 存活者偏差。改成跟其他美股腳本
一致，讀本機快照（load_us_prices_snapshot()/
load_us_index_membership_snapshot()），並把 membership 參數傳進
run_pairs_trading_backtest——候選股票是否合格改在 _find_pairs 內部、
用配對形成窗格「最後一天」的 membership 資格判定，不再靠預先過濾價格
樞紐表（詳見 tw_quant/pairs_trading.py 開頭說明）。

用法：
    python scripts/test_us_pairs_trading_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.pairs_trading import PairsTradingConfig, run_pairs_trading_backtest
from tw_quant.us_config import build_us_config

FORMATION_WINDOW_GRID = (120, 180, 252)
ENTRY_Z_GRID = (1.5, 2.0, 2.5)
MIN_TRADES_FOR_RANKING = 6

HEADER = (
    f"{'form_win':>9} {'entry_z':>8}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(formation_window: int, entry_z: float, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{formation_window:>9} {entry_z:>8.1f}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    n_stocks = us_prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔美股的資料"
        f"（{us_prices['date'].min().date()} ~ {us_prices['date'].max().date()}）\n"
    )

    base_cfg = build_us_config()

    print("=== 配對交易回測（美股 S&P 500、同產業共整合配對、z-score 進出場、完整交易成本）===")
    print(HEADER)

    rows = []
    for formation_window in FORMATION_WINDOW_GRID:
        for entry_z in ENTRY_Z_GRID:
            rt_cfg = PairsTradingConfig(
                formation_window=formation_window,
                entry_z=entry_z,
                pre_filter_min_abs_corr=0.6,
                max_stocks_per_industry=25,
            )
            result = run_pairs_trading_backtest(us_prices, base_cfg, rt_cfg, cost_module=us_costs, membership=membership)
            m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
            row = {"formation_window": formation_window, "entry_z": entry_z}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["formation_window"]), r["entry_z"], r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")
        print(df.to_string(index=False))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "n_trades 每組配對進出場都算 2 筆（A腿+B腿）；做空沒有算真實借券費/保證金利息；"
        "每季（63個交易日）重新選一次配對；為了讓 O(股票數^2) 的共整合檢定在合理時間內跑完，"
        "候選配對先用相關係數 >= 0.6 篩過、每個產業只保留成交金額最高的前 25 檔，不是完全窮舉"
        "（見檔頭說明）；台股版對照：9 組全部負報酬，最佳 Sharpe -0.73）"
    )


if __name__ == "__main__":
    main()
