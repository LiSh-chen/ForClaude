"""跨策略綜合觀察：「毛利t值顯著、扣成本後淨損益轉負」不是分鐘級季節性
掃描的孤例，這次會話最早的起點——三腿策略（開盤上衝/午盤放空/盤中翻多，
revalidate_three_legs_after_data_fix.py 已經產出的資料）——用同樣的角度
重新檢視，同一個模式其實已經存在，只是先前的敘述都停留在「t值逐年變小」
這個較溫和的講法，沒有把「淨損益直接轉負」這個更嚴重的事實明講出來。

這支腳本不重新跑回測，只是讀取已經存在、已經委交過的
three_legs_revalidation_subperiods.parquet / _oos.parquet，換一個角度
（t值 vs 淨損益 並列）重新呈現，把這個跨策略的共通模式明確記錄下來。

用法：
    python scripts/cost_erosion_cross_strategy_synthesis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    sub = pd.read_parquet(DATA_DIR / "three_legs_revalidation_subperiods.parquet")
    oos = pd.read_parquet(DATA_DIR / "three_legs_revalidation_oos.parquet")

    print("=" * 70)
    print("三腿策略（開盤上衝/午盤放空/盤中翻多）子區間：t值 vs 中成本淨損益")
    print("=" * 70)
    combined = sub[sub["leg"] == "combined"]
    pd.set_option("display.width", 200)
    print(combined[["period", "n", "t_stat", "mid_net_twd"]].to_string(index=False))
    negative_net_but_significant = combined[(combined["t_stat"] >= 2.0) & (combined["mid_net_twd"] < 0)]
    print(f"\nt值仍>=2但中成本淨損益已轉負的子區間數: {len(negative_net_but_significant)} / {len(combined)}")

    print("\n" + "=" * 70)
    print("OOS（2021-2023）三腿合併：不同成本情境")
    print("=" * 70)
    print(oos.to_string(index=False))

    print("\n" + "=" * 70)
    print("結論：這個「毛利統計顯著、淨損益被成本吃掉」的模式，不是分鐘級")
    print("季節性掃描獨有的現象——三腿策略在2011-2015、2016-2020兩個子區間")
    print("都是同樣的模式（t值分別4.80、3.36「仍顯著」，但淨損益是")
    print("-12,014、-80,288，已經是虧損）。這次會話最早的起點策略，跟")
    print("最新發現的分鐘季節性，本質上敗在同一個原因：TXF近十幾年的")
    print("點數邊際持續縮小，縮小到不夠付交易成本，不是訊號消失，是")
    print("訊號的『大小』跟不上成本的『固定門檻』。")
    print("=" * 70)


if __name__ == "__main__":
    main()
