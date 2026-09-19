"""驗證「QQQ 搭配其他 ETF 的組合配比、以及要不要定期再平衡」會不會比
單押 QQQ 更好——使用者在看過 10.14 節「四檔ETF期初等額不再平衡」輸給
QQQ 的結果後，明確要求驗證不同組合/配比、以及再平衡與否的差異，不要
只看等額這一種配法。

背景：10.14 節只測過一種混合方式（QQQ/VOO/VTI/VT 各 25%、期初買進後
永遠不再平衡），結論是輸給單押 QQQ（樣本外 CAGR 11.05% vs 15.95%）。
但那只是「等額」+「不再平衡」這一種組合，沒有回答：
  (a) 如果 QQQ 占大部分權重、只搭配少量其他 ETF，會不會比單押 QQQ好？
  (b) 定期再平衡（把飄掉的權重拉回目標值）會不會改善結果，還是反而
      因為交易成本、或是「賣掉表現最好的資產去買落後的」而更差？

這裡把 QQQ 跟 VOO（美股大盤，跟 QQQ 相關性較高）、VT（含國際股市，
跟 QQQ 相關性較低）分別配成不同比例，加上一個三檔混合跟原本的四檔
等額當對照，每一種配比都測「不再平衡」「每季再平衡」「每年再平衡」
三種模式，用 tw_quant/etf_combo.py 的加權組合模擬（含真實複委託成本）
算出來。

反未來函數：組合模擬本身逐日往前跑，任何一天的再平衡決策只用當天
（及之前）的收盤價，見 tw_quant/etf_combo.py 說明；跟前面所有腳本
一樣只用 start_date/end_date 限制期間，不影響這一點。

用法：
    python scripts/test_us_etf_weighted_combo_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import load_us_prices_snapshot
from tw_quant.etf_combo import metrics_from_equity_curve, simulate_weighted_portfolio
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider

ETF_TICKERS = ["QQQ", "VOO", "VTI", "VT"]

IN_SAMPLE_START = "2023-09-19"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

REBALANCE_GRID = [("不再平衡", None), ("每季再平衡（63交易日）", 63), ("每年再平衡（252交易日）", 252)]

COMBOS: list[tuple[str, dict[str, float]]] = [
    ("QQQ 90% / VOO 10%", {"QQQ": 0.9, "VOO": 0.1}),
    ("QQQ 80% / VOO 20%", {"QQQ": 0.8, "VOO": 0.2}),
    ("QQQ 70% / VOO 30%", {"QQQ": 0.7, "VOO": 0.3}),
    ("QQQ 50% / VOO 50%", {"QQQ": 0.5, "VOO": 0.5}),
    ("QQQ 90% / VT 10%", {"QQQ": 0.9, "VT": 0.1}),
    ("QQQ 80% / VT 20%", {"QQQ": 0.8, "VT": 0.2}),
    ("QQQ 70% / VT 30%", {"QQQ": 0.7, "VT": 0.3}),
    ("QQQ 50% / VT 50%", {"QQQ": 0.5, "VT": 0.5}),
    ("QQQ 70% / VOO 10% / VTI 10% / VT 10%", {"QQQ": 0.7, "VOO": 0.1, "VTI": 0.1, "VT": 0.1}),
    ("四檔等額 25/25/25/25", {"QQQ": 0.25, "VOO": 0.25, "VTI": 0.25, "VT": 0.25}),
]

HEADER = f"{'組合':>42} {'再平衡':>20} {'總報酬':>10} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'再平衡次數':>8}"


def _fmt_row(label: str, rebal_label: str, m: dict, n_rebal: int) -> str:
    return (
        f"{label:>42} {rebal_label:>20} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f} "
        f"{n_rebal:>8}"
    )


def _clip(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    lo = pd.Timestamp(start_date) if start_date is not None else df["date"].min()
    hi = pd.Timestamp(end_date) if end_date is not None else df["date"].max()
    s = df[(df["date"] >= lo) & (df["date"] <= hi)].sort_values("date")
    return s.set_index("date")["close"]


def _print_period(label: str, etf_closes: dict[str, pd.DataFrame], base_cfg, start_date, end_date) -> None:
    print(f"\n=== {label} ===")

    clipped = {t: _clip(df, start_date, end_date) for t, df in etf_closes.items()}
    aligned = pd.concat(clipped, axis=1).dropna()

    qqq_m = metrics_from_equity_curve(aligned["QQQ"])
    print(HEADER)
    print(_fmt_row("QQQ 100%（對照組，單押）", "—", qqq_m, 0))
    print()

    for combo_label, weights in COMBOS:
        cols = list(weights.keys())
        prices = aligned[cols]
        for rebal_label, freq in REBALANCE_GRID:
            values, n_rebal = simulate_weighted_portfolio(
                prices, weights, base_cfg.costs, base_cfg.initial_capital, freq
            )
            m = metrics_from_equity_curve(values)
            print(_fmt_row(combo_label, rebal_label, m, n_rebal))
        print()


def main() -> None:
    us_prices = load_us_prices_snapshot()
    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)
    earliest, latest = us_prices["date"].min(), us_prices["date"].max()

    provider = YFinanceUSDataProvider()
    etf_closes = {}
    for ticker in ETF_TICKERS:
        print(f"抓取 {ticker} 價格歷史...")
        df = provider.fetch_price(
            ticker, start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF"
        )
        if df.empty:
            print(f"{ticker} 資料抓取失敗，中止。", file=sys.stderr)
            sys.exit(1)
        etf_closes[ticker] = df.sort_values("date").reset_index(drop=True)
    print()

    base_cfg = build_us_config()
    print(
        "每個組合都測三種再平衡模式：不再平衡（期初買進後永遠不動，權重隨漲跌\n"
        "自然漂移）、每季（63 交易日）、每年（252 交易日）再平衡回原始目標權重；\n"
        "所有買進/賣出都用真實複委託成本模型（每股固定手續費 + 賣出方向 SEC\n"
        "規費，跟本專案其他美股腳本一致），不是零成本假設。"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", etf_closes, base_cfg, IN_SAMPLE_START, None)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", etf_closes, base_cfg, None, OOS_END)

    print(
        "\n（誠實揭露：這裡只測了 QQQ 搭配 VOO／VT／三檔混合／四檔等額這幾種\n"
        "組合，不是窮舉所有可能配比，權重間距也只抓了 10/20/30/50% 幾個代表點，\n"
        "不是連續掃描；VTI 因為跟 VOO 持股高度重疊（見 10.14 節），這裡沒有\n"
        "單獨測 QQQ+VTI 的配對，只出現在三檔/四檔混合裡；每次再平衡都是把\n"
        "全部持股一次調回目標權重，不是只調整偏離最大的那一檔，實務上更貼近\n"
        "定期定額再平衡的做法，但也代表交易次數/成本會比「只調整偏離較大者」\n"
        "的做法更高，這裡沒有測試更省成本的門檻式再平衡）"
    )


if __name__ == "__main__":
    main()
