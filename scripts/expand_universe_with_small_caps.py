"""擴充股票清單，加入真正的中小型股。

背景：現有 data/stock_universe.txt 是 generate_stock_universe.py 依股票
代號排序取前 150 檔產生的，不是依股本/市值選的，導致先前用「依股本切半」
比較策略表現時，兩邊都不是真正意義下的中小型股 vs 大型股（見
docs/research_findings.md 6.5 節）。這裡改成主動找真正股本偏小的股票，
加進既有清單，讓後續的股本效應比較有真正的中小型股樣本可用。

做法：
  1. 抓全市場普通股清單（跟 generate_stock_universe.py 同一套篩選規則），
     排除已經在現有 universe 裡的股票。
  2. 從剩下的候選裡等距抽樣（跨整個代號區間抽，不是只抽前面幾檔），
     避免抽樣本身又重蹈「代號排序=規模排序」的覆轍。
  3. 對每檔候選只抓「最近一筆」已發行股數（不是 3 年全部歷史——算股本
     只需要一個時間點，抓整段歷史像上一輪那樣會很慢，這是特別針對這個
     用途的簡化，之後真的要回測這些股票時，ingest_daily_data.py 本來的
     每日排程還是會抓完整歷史）。
  4. 依股本由小到大排序，取最小的 --add-n 檔（預設 70，跟先前「股本
     切半比較」小股本組的檔數對齊，讓兩輪結果可比），寫回
     data/stock_universe.txt（原本的 150 檔 + 新增的檔）。

這支腳本只負責「決定要加哪些股票進清單」，不會抓這些新股票的完整價量/
營收歷史——寫完清單後還是要跑一次 ingest_daily_data.py（新股票會被
逐股回填邏輯自動偵測成「從來沒同步過」，觸發完整 3 年回填）。

用法：
    python scripts/expand_universe_with_small_caps.py --add-n 70
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tw_quant.data_provider import FinMindDataProvider

_ORDINARY_STOCK_ID = re.compile(r"^(?!00)\d{4}$")
PAR_VALUE = 10.0

T = TypeVar("T")


def _call_with_timeout(func: Callable[[], T], timeout_s: float) -> T | None:
    """跟 ingest_daily_data.py 同一套保護：任何一次呼叫最多等 timeout_s 秒。"""
    result: list[T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(func())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive() or error or not result:
        return None
    return result[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add-n", type=int, default=70, help="要新增的中小型股檔數")
    parser.add_argument("--candidate-pool", type=int, default=350, help="要抽樣檢查股本的候選檔數上限")
    parser.add_argument("--universe-file", default="data/stock_universe.txt")
    parser.add_argument("--sleep", type=float, default=0.4)
    args = parser.parse_args()

    existing = [
        line.strip() for line in Path(args.universe_file).read_text().splitlines() if line.strip()
    ]
    existing_set = set(existing)
    print(f"現有 universe：{len(existing)} 檔")

    provider = FinMindDataProvider(token=os.environ.get("FINMIND_TOKEN"))
    info = provider.fetch_stock_info()
    if info.empty:
        print("FinMind 沒有回傳任何股票基本資料，中止。", file=sys.stderr)
        sys.exit(1)

    filtered = info[
        info["type"].isin(["twse", "tpex"]) & info["stock_id"].astype(str).str.match(_ORDINARY_STOCK_ID)
    ].drop_duplicates(subset=["stock_id"])
    filtered = filtered.sort_values("stock_id")
    name_lookup = dict(zip(filtered["stock_id"], filtered["stock_name"]))

    candidates_all = [sid for sid in filtered["stock_id"].tolist() if sid not in existing_set]
    print(f"全市場普通股 {len(filtered)} 檔，扣掉已經在 universe 裡的，剩 {len(candidates_all)} 檔候選")

    # 等距抽樣，跨整個代號區間抽，不是只抽最前面幾檔（避免抽樣本身又變成
    # 「代號排序=規模排序」的偏差）
    n_candidates = min(args.candidate_pool, len(candidates_all))
    stride = max(1, len(candidates_all) // n_candidates)
    sampled = candidates_all[::stride][:n_candidates]
    print(f"等距抽樣（stride={stride}）取 {len(sampled)} 檔候選，查最近一筆已發行股數\n")

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=20)).strftime("%Y-%m-%d")

    rows = []
    for i, stock_id in enumerate(sampled, start=1):
        shares_df = _call_with_timeout(
            lambda sid=stock_id: provider.fetch_shares_issued(sid, start_date, end_date), timeout_s=20.0
        )
        if shares_df is not None and not shares_df.empty:
            latest = shares_df.sort_values("date").iloc[-1]
            share_capital_yi = latest["shares_issued"] * PAR_VALUE / 1e8
            rows.append({"stock_id": stock_id, "stock_name": name_lookup.get(stock_id, ""), "share_capital_yi": share_capital_yi})
        if i % 50 == 0:
            print(f"[{i}/{len(sampled)}] 已查詢完成，目前取得 {len(rows)} 筆有效股本資料")
        time.sleep(args.sleep)

    if not rows:
        print("沒有任何候選股票查得到股本資料，中止。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("share_capital_yi")
    print(f"\n成功取得股本資料：{len(df)} / {len(sampled)} 檔候選")
    print(f"候選股本分布：\n{df['share_capital_yi'].describe()}\n")

    to_add = df.head(args.add_n)
    print(f"=== 新增的 {len(to_add)} 檔中小型股（股本由小到大） ===")
    print(to_add.to_string(index=False))

    combined = existing + to_add["stock_id"].tolist()
    Path(args.universe_file).write_text("\n".join(combined) + "\n")
    print(f"\n已寫入 {args.universe_file}：原本 {len(existing)} 檔 + 新增 {len(to_add)} 檔 = 共 {len(combined)} 檔")
    print(
        "\n下一步：跑一次 ingest_daily_data.py（daily_data_ingest.yml），"
        "新股票會被逐股回填邏輯自動偵測成「從來沒同步過」，觸發完整 3 年回填。"
    )


if __name__ == "__main__":
    main()
