# Stock Derby 美股賽馬

用美股股價賽馬：每匹馬 = 一檔股票，賽程區間內累積報酬越高，跑得越前面。

- `tickers.json` — 參賽股票/馬名/顏色（改這裡就能換參賽者）
- `fetch_prices.py` — 用 yfinance 抓每日收盤價，增量合併進 `data/prices.csv`
- `data/prices.csv` — 資料本體（寬表，commit 在 repo，GitHub 即資料庫）
- `index.html` — 純靜態前端（無需 build），可選 5/20/60 日或全部賽程，播放/拖曳回放
- `.github/workflows/stock_derby_daily.yml` — 每個交易日收盤後抓價、commit 資料、發佈 GitHub Pages

本機預覽：`python -m http.server -d stock-derby` 然後開 http://localhost:8000
啟用網站：repo Settings → Pages → Source 選 **GitHub Actions**。
