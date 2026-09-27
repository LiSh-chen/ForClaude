"""跨市場訊號探索：美股（S&P500/VIX）前一交易日收盤到收盤的變化，能不能
預測台指期次一交易日盤中表現？

方法跟 scan_full_time_window_grid.py（挖出三腿策略08:45-09:00等時段的
同一套方法）完全一致——完整(start,end)時段網格掃描，只是這裡先把交易日
依美股訊號的正負分組，再對每一組分別做網格掃描，比較「有沒有比無條件
（全部交易日）版本更強、更一致」的窗口。只用IS(2001-2020)，不看OOS。

測試的訊號分組（都是最自然的正負二分法，不做門檻擬合，降低事後挑值的
過擬合風險）：
- sp500_sign：前一美股交易日 S&P500 收盤對收盤報酬 是正是負
- vix_sign：前一美股交易日 VIX 較前一天是漲是跌（VIX上漲=恐慌升溫）

用法：
    python scripts/scan_us_signal_time_window_grid.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.scan_full_time_window_grid import build_price_grid, full_grid_scan  # noqa: E402
from tw_quant.us_market_signal import align_to_txf_dates  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"
IS_CUTOFF = "2021-01-01"
# full_grid_scan()（從scan_full_time_window_grid.py匯入）內部本來就有
# n>=200的門檻，這裡分組後每組仍有約2000+交易日，該門檻幾乎都能滿足，
# 不需要另外調整；MIN_N只是這支腳本自己輸出時的顯示門檻，不是放寬條件。
MIN_N = 200


def top_n(res: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return res.reindex(res["tstat"].abs().sort_values(ascending=False).index).head(n)[
        ["start_label", "end_label", "duration", "n", "mean_ret", "tstat"]
    ]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF].copy()

    t = is_df["datetime"].dt.time
    day_df = is_df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["minute_of_day"] = day_df["datetime"].dt.hour * 60 + day_df["datetime"].dt.minute
    day_df["session_id"] = day_df["datetime"].dt.date
    day_anchors = list(range(8 * 60 + 45, 13 * 60 + 46, 15))

    trading_dates = pd.Series(sorted(day_df["session_id"].unique()))
    aligned = align_to_txf_dates(pd.to_datetime(trading_dates))
    aligned["session_id"] = aligned["txf_date"].dt.date
    sig = aligned.set_index("session_id")[["sp500_ret", "vix_chg"]]

    print("=" * 70)
    print("0) 基準：無條件（全部交易日）網格掃描，跟原始腳本結果應該一致")
    print("=" * 70)
    base_grid = build_price_grid(day_df, "session_id", day_anchors)
    base_res = full_grid_scan(base_grid)
    print(f"組合數: {len(base_res)}")
    print(top_n(base_res).to_string(index=False))

    signals = {
        "sp500_sign": ("sp500_ret", lambda s: s > 0, lambda s: s < 0, "美股上漲日", "美股下跌日"),
        "vix_sign": ("vix_chg", lambda s: s > 0, lambda s: s < 0, "VIX上升日(恐慌升溫)", "VIX下降日(恐慌降溫)"),
    }

    all_results = {}
    for sig_name, (col, pos_mask, neg_mask, pos_label, neg_label) in signals.items():
        print("\n" + "=" * 70)
        print(f"訊號: {sig_name}（{col}）")
        print("=" * 70)
        values = day_df["session_id"].map(sig[col])
        for label, mask_fn in [(pos_label, pos_mask), (neg_label, neg_mask)]:
            mask = mask_fn(values)
            sub_df = day_df[mask.fillna(False).to_numpy()]
            n_days = sub_df["session_id"].nunique()
            print(f"\n--- {label}（{n_days}個交易日）---")
            if n_days < MIN_N:
                print("樣本太少，略過")
                continue
            grid = build_price_grid(sub_df, "session_id", day_anchors)
            res = full_grid_scan(grid)
            res_filtered = res[res["n"] >= MIN_N]
            if res_filtered.empty:
                print("沒有組合達到最小樣本數門檻")
                continue
            print(top_n(res_filtered, 5).to_string(index=False))
            all_results[f"{sig_name}__{label}"] = res_filtered

    for key, res in all_results.items():
        safe = key.replace("(", "").replace(")", "").replace(" ", "_")
        res.to_parquet(OUT_DIR / f"us_signal_scan_{safe}.parquet", index=False)

    print("\n完成。以上全部只用IS(2001-2020)資料，尚未看OOS。")


if __name__ == "__main__":
    main()
