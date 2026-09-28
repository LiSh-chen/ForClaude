"""跳空缺口探索：日盤開盤(08:45)對「前一個有交易的session」最後收盤價的跳空，
能不能預測當天日盤後續走勢？測試兩個互斥假設：

- 補缺口（均值回歸）：跳空向上開盤 -> 當天盤中該回跌（負相關）；跳空向下
  開盤 -> 當天該反彈（正相關）。
- 跳空延續（動能）：跳空向上開盤 -> 當天該續漲（正相關）；跳空向下 ->
  該續跌（負相關）。

跟這次會話先前的「跨日日盤反轉」（tw_quant/cross_day_reversal_strategy.py，
訊號是昨天日盤自己的漲跌方向）不一樣：這裡的訊號是「今天開盤價相對昨天
收盤價的跳空方向」，是完全沒測過的訊號來源（前者測「昨天走勢會不會反轉」，
這裡測「隔夜/跨盤跳空本身會不會被回補或延續」）。

只用日盤資料算「前一收盤」，不含夜盤（夜盤資料只從2017年才有，若定義
「前一收盤」要含夜盤，IS 2001-2020大部分年份會直接沒有訊號可用，樣本量
崩潰）——這代表這裡測的其實是「日盤收盤(13:45)到隔天日盤開盤(08:45)之間
的跳空」，涵蓋隔夜+可能的夜盤在內的整段間隔，不特別拆解夜盤內部發生
什麼事。

方法跟 scan_us_signal_time_window_grid.py（挖出美股訊號調節效果的同一套
方法）完全一致——完整(start,end)時段網格掃描，把交易日依跳空正負分組後
各自掃描，只用IS(2001-2020)，不看OOS。

用法：
    python scripts/scan_gap_time_window_grid.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.scan_full_time_window_grid import build_price_grid, full_grid_scan  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"
IS_CUTOFF = "2021-01-01"
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

    daily = day_df.groupby("session_id").agg(open=("open", "first"), close=("close", "last")).reset_index()
    daily = daily.sort_values("session_id").reset_index(drop=True)
    daily["prev_close"] = daily["close"].shift(1)
    daily["gap_points"] = daily["open"] - daily["prev_close"]
    daily["gap_pct"] = daily["gap_points"] / daily["prev_close"] * 100
    gap = daily.set_index("session_id")["gap_pct"]

    print("=" * 70)
    print("0) 基準：無條件（全部交易日）網格掃描，跟原始腳本結果應該一致")
    print("=" * 70)
    base_grid = build_price_grid(day_df, "session_id", day_anchors)
    base_res = full_grid_scan(base_grid)
    print(f"組合數: {len(base_res)}")
    print(top_n(base_res).to_string(index=False))

    print("\n" + "=" * 70)
    print("跳空幅度分布（前一收盤->今天開盤，百分比）")
    print("=" * 70)
    print(gap.describe().to_string())
    print(f"\n跳空>0(向上開)天數: {(gap > 0).sum()}  跳空<0(向下開)天數: {(gap < 0).sum()}  "
          f"跳空=0天數: {(gap == 0).sum()}")

    groups = {
        "跳空向上開盤(gap>0)": gap > 0,
        "跳空向下開盤(gap<0)": gap < 0,
    }

    # 額外測試：只留幅度較大的跳空（|gap|前1/3分位），排除雜訊等級的小跳空
    abs_gap = gap.abs()
    big_threshold = abs_gap.quantile(2 / 3)
    print(f"\n大跳空門檻(|gap|前1/3分位): {big_threshold:.4f}%")
    groups["大跳空向上(gap>0且|gap|前1/3大)"] = (gap > 0) & (abs_gap >= big_threshold)
    groups["大跳空向下(gap<0且|gap|前1/3大)"] = (gap < 0) & (abs_gap >= big_threshold)

    all_results = {}
    for label, cond in groups.items():
        print("\n" + "=" * 70)
        print(f"分組: {label}")
        print("=" * 70)
        mask = day_df["session_id"].map(cond)
        sub_df = day_df[mask.fillna(False).to_numpy()]
        n_days = sub_df["session_id"].nunique()
        print(f"交易日數: {n_days}")
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
        all_results[label] = res_filtered

    for key, res in all_results.items():
        safe = (key.replace("(", "_").replace(")", "").replace(">", "gt").replace("<", "lt")
                .replace("|", "").replace("/", "_").replace(" ", "_"))
        res.to_parquet(OUT_DIR / f"gap_scan_{safe}.parquet", index=False)

    print("\n完成。以上全部只用IS(2001-2020)資料，尚未看OOS。")


if __name__ == "__main__":
    main()
