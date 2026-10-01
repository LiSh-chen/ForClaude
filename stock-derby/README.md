# Stock Derby 美股賽馬

用美股股價賽馬：選馬群（S&P 500 / Nasdaq-100 / 兩者合併）與賽程（1/3/6 個月），
看「賽馬報」（近績、跑法、能力）選馬，再回放賽事看誰先衝線。

## 賽制
- **賽程**：1 / 3 / 6 個月 = 21 / 63 / 126 個交易日，以區間累積漲幅排名；賽事是「最近一個已完成的區間」。
- **參賽馬**：起跑前 60 日平均成交金額最高的 10 檔（人氣馬）。
- **近績**：同賽程長度、起跑前連續 4 場的名次（在這 10 匹馬之間比），新→舊。
- **跑法**：看前 3 場前 1/3 段的平均排名 — 領放(≤2.5) / 跟前(≤4.5) / 居中(≤7) / 後上。
- **能力**：五大關鍵參數在馬群中的百分位（0–100），依各賽程的 IC 加權成「能力指數」；◎○▲△ 為前 5 名。
- 賽馬報只用起跑日**以前**的資料（有單元測試檢查因子無前視偏差）。

## 五大關鍵參數怎麼來
`factor_study.py` 對 12 個候選指標（技術 7、籌碼 3、財務 2）逐月算與未來 1/3/6 個月報酬的 Rank IC，
挑出相關性最強且彼此相關 ≤0.7 的五項，結果寫在 `data/key_factors.json`（含全部 12 項成績）。
**誠實提醒**：IC 都很弱（約 0.01–0.06），名單有存活者偏差；美股無免費法人籌碼資料，
籌碼面用量價資金流向代替（本次未入選）。這是遊戲化的參考，不是預測，也不是投資建議。

## 檔案
- `data_loader.py` 讀價量（repo 的 S&P 500 parquet 快照 + `data/extra_prices.csv`）
- `factors.py` 候選因子　`factor_study.py` 因子研究　`build_derby.py` 產生 `data/derby.json`
- `fetch_extras.py` 補抓不在 S&P 快照內的 Nasdaq-100 成分股；名單抓 Wikipedia，失敗退回 `universe.json`
- `index.html` 純靜態前端　`.github/workflows/stock_derby_daily.yml` 每日建置＋發佈 Pages

本機：`python stock-derby/build_derby.py && python -m http.server -d stock-derby`
