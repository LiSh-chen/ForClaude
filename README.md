# 台股系統化量化交易策略：動能突破與流動性狙擊（全域風控終極版）

依需求規格書實作的完整 Python 量化交易系統，涵蓋大盤環境判定、雙軌進場
訊號（策略 A 籌碼沉澱 / 策略 B 軋空異常）、投資組合全域風控、台股實務部位
計算、WFA 滾動驗證，以及 MDD 熔斷降級恢復矩陣。

**本專案僅供研究與開發使用，不構成投資建議。所有示範腳本使用合成假資料，
在真實資金上線前，務必換成真實市場資料並重新驗證每一項假設。**

---

## 1. 為什麼選這個技術路線

沒有用 `backtrader` / `vectorbt` / `zipline` 這類現成回測框架，改寫一個
輕量的自製「每日事件迴圈」引擎（`tw_quant/backtest.py`），原因：

- 全域鎖（單日 6% 曝險、10% 產業集中度、流動性 tie-breaker）是**跨標的、
  跨天、路徑相依**的邏輯，需要在同一天把所有候選訊號收集起來一起裁決，
  這類框架的「逐標的訊號」模型不好直接表達。
- MDD 熔斷降級矩陣是一個**跨越整個回測期間的有限狀態機**（50%→75%→100%
  三階段、以「已經歷的新訊號筆數」與「EV/資金曲線回穩」為轉移條件），
  這也不是現成框架的內建功能。
- 部位規模計算（無條件捨去至張、零股防禦、市值上限防禦、2.5×ATR 吊燈
  停利）是台股特有規則，直接用 pandas/numpy 手刻最直接、最容易驗證正確性。

代價是：**效能沒有向量化框架好**（純 Python 逐日迴圈），詳見第 6 節「已知
限制」。指標計算（SMA/ATR/百分位排名）本身仍然是向量化的（`indicators.py`），
只有「跨標的的每日風控裁決」這段是逐日迴圈。

## 2. 模組地圖

```
tw_quant/
  config.py          全部參數（規格書每個數字都在這裡，改參數只改這檔）
  data_provider.py   資料 schema、CSV 讀取、合成假資料產生器、FinMind 介接骨架
  indicators.py      SMA / ATR(Wilder) / 滾動百分位排名 / 分組 shift
  regime.py          壹、大盤環境判定 + 參數敏感度網格測試工具
  signals.py         參、策略 A/B 訊號產生（強制 shift(1)、日曆防禦）
  risk.py            肆、單一標的部位計算（進場價/停損價/股數/市值上限）
  portfolio_risk.py  貳、投資組合全域鎖（6% 曝險 / 10% 產業 / 流動性排序）
  costs.py           伍、交易摩擦成本（證交稅/手續費/跳動點滑價）
  mdd.py             陸-2、MDD 熔斷與降級恢復矩陣（狀態機）
  backtest.py         把以上全部串成完整每日回測引擎
  wfa.py             伍、WFA 滾動驗證、過度擬合警報、大數檢驗放寬邏輯
  storage.py         資料落地層：SQLite（預設）/ Postgres（可選）DataStore

scripts/
  run_demo_backtest.py             端到端示範（合成假資料）
  run_sensitivity_test.py          大盤環境參數敏感度網格測試（合成假資料）
  run_wfa_demo.py                  WFA 滾動驗證示範（合成假資料）
  ingest_daily_data.py             每日資料抓取（給 GitHub Actions 排程用，見第 5 節）
  generate_stock_universe.py       從 FinMind 產生股票清單（篩掉 ETF/權證），寫 data/stock_universe.txt
  show_data_status.py              印出資料庫目前實際內容（每檔股票的資料範圍/筆數）
  run_backtest_from_db.py          讀取累積的真實資料跑正式回測
  run_sensitivity_test_from_db.py  讀取累積的真實資料跑敏感度網格測試
  run_wfa_from_db.py               讀取累積的真實資料跑 WFA（資料不夠長會印出提示，不是錯誤）

.github/workflows/
  daily_data_ingest.yml       每日排程抓資料的 GitHub Actions workflow（每次執行後都會印出資料庫現況）
  generate_stock_universe.yml 手動觸發，重新產生股票清單

tests/               pytest 單元測試（62 個，涵蓋每個模組的關鍵行為）
```

## 3. 規格書的解讀與明確假設

規格書多處用詞在工程實作上有解讀空間，以下是本專案採用的定義，**上線前
請務必與你的策略設計者確認是否與原意一致**：

| 規格條文 | 本實作採用的定義 | 位置 |
|---|---|---|
| 「20 日均線多頭排列家數比例」 | 收盤價站上 20 日均線，且 20 日均線本身向上（5 日前更低） | `regime.compute_bullish_alignment` |
| 「大盤盤中預估總量」 | 回測用「全市場當日總成交金額」代理；實盤另提供 `project_intraday_turnover` 線性外推 hook | `regime.py` |
| ATR 天數 | 規格未指定，採業界慣例 14 日（`PositionSizingConfig.atr_window`，可調） | `config.py` |
| 「T 日 ATR」「過去 10 日最高價」在初始停損公式中的基準日 | 一律取**訊號日 T**（點火扣板機當天），因為進場執行在 T+1 開盤 | `risk.compute_initial_stop` |
| 吊燈停利是否單調上調 | 採標準作法：`stop = max(前一日停損, 當日新算停損)`，只上調不下調，避免防線無故鬆動 | `backtest.py` 第 3 步 |
| 全域鎖排隊時「進場價」與潛在虧損怎麼算，但進場價要 T+1 才知道 | 用 **T 日收盤價**當估計進場價做排序與額度裁決；核准後 T+1 真正開盤價出爐時，重新算最終停損價與股數 | `backtest.py` 檔頭註解 |
| 「總資金/總本金」在風控公式中是否隨盈虧變動 | 採用**固定的期初本金**（`cfg.initial_capital`）乘上 MDD 管理器的 capital_scale，不用逐日浮動淨值，避免部位規模隨盈虧複利式放大/縮小 | `backtest.py` |
| MDD 降級矩陣「第 50 筆後恢復 100%」是否需要條件 | 規格書第二段（21~50 筆）明確寫了條件（EV 轉正 + 資金曲線回穩），第三段（50 筆後）沒寫條件——本實作**字面照做**：50%→75% 需要條件通過，75%→100% 在滿 50 筆後無條件觸發並重新校準 MDD 基準。若你更保守，`mdd.py` 檔頭註解說明了怎麼一行改成有條件 | `mdd.py` |
| 「資金曲線回穩」的量化定義 | 規格書沒有明講，本實作定義為「最新權益 ≥ 熔斷以來的最低權益」 | `mdd.MDDManager.equity_curve_stabilized` |

## 4. 怎麼跑

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 單元測試（42 個，全綠）
pytest tests/ -q

# 端到端示範：完整回測 + MDD 兩階段流程
python scripts/run_demo_backtest.py

# 大盤環境參數敏感度網格測試 + 懸崖警報
python scripts/run_sensitivity_test.py

# WFA 3年訓練+1年盲測滾動驗證 + 過度擬合警報 + 大數檢驗
python scripts/run_wfa_demo.py
```

三個 demo 腳本都使用 `generate_synthetic_universe` 產生的合成假資料
（幾何布朗運動 + 隨機插入的動能事件），**純粹用來證明整條 pipeline 能正確
串接執行**，數字不代表任何真實績效。

## 5. 自動資料抓取架構

### 5.1 為什麼不能在這個開發環境直接抓

Claude Code 的沙盒環境對外網路走**政策白名單代理**，只放行固定幾個網域
（PyPI、npm registry、Anthropic API 等），`api.finmindtrade.com` 不在清單
內，任何連線在代理層就被擋掉回 403——這跟「有沒有連結 GitHub」是完全不同
的兩套權限系統：GitHub 授權管的是 `git push`/建 PR，跟這個沙盒對外網路的
白名單完全無關，兩者互不影響。

因此抓資料這件事被設計成**跑在 GitHub Actions 的 runner 上**，那裡有正常
的網際網路存取權，不受這個沙盒限制。

### 5.2 架構總覽

```
GitHub Actions（每個交易日排程觸發，見 .github/workflows/daily_data_ingest.yml）
        │
        ▼
scripts/ingest_daily_data.py  ← 用 FinMindDataProvider 抓價量 + 融資券 + 產業分類
        │
        ▼
tw_quant/storage.get_data_store()
        │
        ├─ 沒設 DATABASE_URL → SQLiteDataStore（data/tw_market.db，
        │                       workflow 執行完自動 git commit 回 repo）
        │
        └─ 設了 DATABASE_URL → PostgresDataStore（雲端資料庫）
        │
        ▼
scripts/run_backtest_from_db.py  ← 讀累積下來的真實資料，直接跑 run_backtest()
```

**現在（還沒申請雲端資料庫）**：workflow 每天抓完資料寫進本機 SQLite 檔案
`data/tw_market.db`，然後自動 commit 回這個 repo——等於用 git 當免費、
零設定的「暫時資料庫」，你今天就能開始每天累積真實資料，不用等雲端資料庫
辦好。

**之後（申請好 Postgres）**：只要在 GitHub repo 的 Settings → Secrets and
variables → Actions 裡加一個 secret `DATABASE_URL`
（例如 `postgresql://user:pass@host:5432/dbname`），下一次排程執行就會
自動改寫進雲端資料庫，**不用改任何程式碼**，也不用手動搬資料
（`get_data_store()` 是全系統唯一判斷用哪個後端的地方）。

### 5.3 設定步驟

1. **（選填）申請 FinMind token**：免費版不用 token 也能用，但速率限制
   更嚴；到 [finmindtrade.com](https://finmind.github.io/) 申請帳號拿
   token 後，到 repo 的 Settings → Secrets → Actions 加一個 secret
   `FINMIND_TOKEN`。
2. **（選填）擴充股票清單**：預設只抓 10 檔示範用權值股
   （`scripts/ingest_daily_data.py` 裡的 `DEFAULT_UNIVERSE`）。要擴大規模，
   兩個方式擇一：
   - 到 repo 的 Actions 分頁選 `Generate Stock Universe` → Run workflow，
     填想要的檔數（`limit`），它會呼叫 FinMind 抓全市場股票清單、篩掉
     ETF/權證等衍生代號，取前 N 檔寫成 `data/stock_universe.txt` 並自動
     commit 回 repo——`ingest_daily_data.py` 會自動把這個檔案當成預設清單，
     不用再額外設定。
   - 或到 repo 的 Settings → Secrets and variables → Actions → Variables
     加一個 `STOCK_UNIVERSE`，值是逗號分隔的股票代號（例如
     `2330,2317,2454,...`），這個設定的優先權比 `data/stock_universe.txt`
     更高。
   
   規模與耗時的抓取結果（`execute_values` 修好之後的實測數字）：
   - 10 檔（預設）：全量回填約 1 分鐘
   - 150 檔：全量回填約 15 分鐘（一次性成本；之後每日只做增量同步，
     幾秒鐘內完成，不受股票數影響太多）
   
   FinMind 免費註冊 token 的額度是每小時 600 次 API 呼叫，每檔股票每次
   同步要打 2 次（價量 + 融資券），所以單次執行建議不要超過約 300 檔。
3. **手動觸發測試**：到 repo 的 Actions 分頁，選
   `Daily TW Market Data Ingest` → Run workflow，可以立刻手動跑一次，
   不用等排程時間到。
4. **確認排程會不會生效**：GitHub 的 `schedule` 觸發**只認 repo 的預設
   分支**——這個 repo 目前只有 `claude/nice-rubin-h6i35f` 一個分支，它
   本身就是預設分支，所以排程不需要額外合併就會生效；如果你之後改了預設
   分支或建了 `main`，記得把這個 workflow 帶過去。
5. 資料累積一段時間後：
   - `python scripts/run_backtest_from_db.py` 直接用真實資料做回測
   - `python scripts/run_sensitivity_test_from_db.py` 用真實資料跑大盤環境
     參數敏感度網格
   - `python scripts/run_wfa_from_db.py` 用真實資料跑 WFA 滾動驗證
     （這個需要至少 4 年多的歷史才會有結果，資料不夠長時會印出提示、
     不是報錯）

### 5.4 之後想換別的資料來源（TEJ / 券商 API）

`FinMindDataProvider` 只是眾多資料來源之一。若你有 TEJ 或永豐 Shioaji 這類
券商 API 帳號，只要照 `data_provider.PRICE_COLUMNS` /
`MARGIN_SHORT_COLUMNS` 定義的 schema 寫一個新的 provider 類別（實作
`fetch_price` / `fetch_margin_short` / `fetch_stock_info` 三個方法），
`ingest_daily_data.py` 只要把 `FinMindDataProvider(...)` 換成你的新類別即可，
`storage.py` 那一層完全不用動。

## 6. 已知限制與建議（請務必閱讀）

### 6.1 10% 產業上限 vs 20% 單檔上限 vs 2% 風險預算的交互作用

這是實測合成資料時發現的**結構性問題**，不是實作 bug，而是規格書三個數字
組合起來的數學後果，強烈建議你在真正上線前想清楚：

依 2% 風險預算公式，`部位市值 = 風險預算 / (停損距離佔進場價的比例)`。
若停損距離 < 20% 進場價（吊燈停利 2.5×ATR、或防呆下限 5%，兩者幾乎必然
比 20% 窄很多），部位市值就會超過 10% 總本金，撞上「單一產業曝險上限
10%」——**即使該產業目前完全沒有既有部位**，這筆全新訊號也會被全域鎖
直接捨棄。實測下來，用規格預設的 10%/20%/2% 三個數字，幾乎所有訊號都會
在這一關被擋下（demo 腳本因此把 `max_industry_exposure_pct` 臨時調到
30% 才能觀察到成交）。

建議三選一：
- 提高 `max_industry_exposure_pct`（例如 15~20%），讓它只在「同產業已有
  多檔部位」時才真正發揮攔截作用；
- 或降低 `max_position_value_pct_of_capital`（例如降到 8~9%，低於產業
  上限），讓 20% 上限先於 10% 上限生效；
- 或修改 `portfolio_risk.apply_global_lock` 的邏輯，讓超過產業額度的候選
  「縮減部位以塞滿剩餘額度」而非整筆捨棄（規格書字面只說「捨棄」，如果
  你要的是縮減而非捨棄，這裡需要改程式）。

### 6.2 效能

`backtest.py` 是逐日 Python 迴圈，示範規模（30-40 檔、2-3 年）在 1 秒內
跑完；WFA 示範（10 檔、10 年、3 組參數網格、6 個滾動視窗）約需數十秒到
幾分鐘。若要跑全市場（1700+ 檔上市櫃股票）× 10+ 年 × WFA 參數網格，
建議：先用 cProfile 找熱點，多半會落在 `groupby('date')` 迭代與
`row_by_stock.loc[...]` 逐檔查找；可考慮改用 numpy 陣列索引取代
`.loc`、或用 `multiprocessing` 平行跑 WFA 的不同訓練窗（各窗互相獨立）。

### 6.3 「盤中預估總量」在回測中的近似

規格要求的是「盤中」即時估算的全日量能，回測資料只有日終資料，所以用
「當日全市場總成交金額」直接代理，等於是「事後諸葛」版本的量能條件。
實盤上線要接即時報價，用 `regime.project_intraday_turnover` 做線性外推
（或更精細的日內量能曲線加權），且務必用實盤 WFA 或前向測試驗證兩者的
差異不會讓策略行為改變太多。

### 6.4 敏感度測試 / WFA 的統計顯著性

`run_regime_sensitivity_grid` 與 `run_wfa` 的框架完全依規格書要求做好了
（網格測試、懸崖偵測、3+1 年滾動窗、參數變異 > 30% 警報、大數檢驗放寬
梯度），但**這些工具的結論只跟你餵進去的歷史資料一樣可信**。合成假資料
測出來的「無懸崖」「無過度擬合」不代表真實市場也是這樣，正式上線前必須
用真實歷史資料重新跑過這整套檢驗。

### 6.5 資料自動化管線的限制

- **FinMind 免費額度速率限制**：`ingest_daily_data.py` 是逐檔股票打 API
  （每檔一次價量請求、一次融資券請求），股票清單一大，同步時間會拉長，
  免費額度也可能被打到限速報錯（腳本會 catch 例外、記錄失敗清單、繼續跑
  下一檔，不會整個任務失敗，但當天那幾檔就會缺資料，下次執行的
  `LOOKBACK_DAYS` 補資料視窗可以補回來）。要拉全市場 1700+ 檔，建議申請
  付費 token 或把清單拆成好幾個 workflow 分批跑。
- **git-as-database 是暫時方案**：把 SQLite 檔案 commit 回 repo 雖然零設定，
  但長期下來 repo 體積會隨著資料量增長（尤其是全市場 × 多年歷史），且
  多個 workflow 同時寫入同一個 SQLite 檔案沒有做鎖定/合併衝突處理
  （目前排程是序列執行、單一 job，还沒有並行寫入的問題，但如果你之後改成
  多個 workflow 平行抓不同批股票，要自己加鎖或改用真正的資料庫）。這是
  刻意的過渡設計，申請好雲端 Postgres 後應盡快切換。
- **排程只認預設分支**：見第 5.3 節第 4 點，`schedule` 觸發不會在
  feature 分支上生效，合併前只能手動 `workflow_dispatch` 測試。
- **`FinMindDataProvider` 沒有實測過**：因為這個沙盒連不到 FinMind，欄位
  對應（如 `Trading_Volume` → `volume`）是照官方文件寫的，**你第一次跑
  workflow 時務必檢查 Actions 的執行紀錄與 `data/tw_market.db` 裡的實際
  數值**，確認欄位對應、單位（股數 vs 張數）都正確，再開始長期累積資料。

### 6.6 未實作／簡化的部分

- 沒有做「新股/下市」的完整生命週期管理（下市直接消失於資料中，回測遇到
  持股標的資料中斷會用進場價估值，不會自動平倉，實務上應該加停牌/下市
  的強制出場邏輯）。
- 沒有做保證金融資、放空、當沖等機制（規格本身也沒要求）。
- `industry` 欄位目前假設固定不變；若某檔股票中途更換產業分類，目前的
  實作不會追蹤變更歷史。
- 台股「零股交易」自 2020 年起已開放盤中零股，規格明確要求「嚴禁掛零股
  交易」，本實作照規格排除零股，若你之後想開放零股要另外設計。

## 7. 測試涵蓋範圍

`tests/` 42 個測試，涵蓋：
- 指標正確性（SMA 不跨標的污染、ATR 隨波動率增加、滾動百分位排名手算驗證）
- 成本模型（跳動單位級距、滑價方向、稅費淨額）
- 部位計算（無條件捨去至張、零股防禦、市值上限防禦、初始停損 max() 邏輯）
- 全域鎖（6% 曝險截斷、流動性 tie-breaker 排序、產業上限攔截/放行邊界）
- MDD 狀態機（觸發熔斷、EV 為負時卡在 50%、EV 轉正且回穩才升到 75%、滿
  50 筆重新校準回 100%）
- 訊號無未來函數（把未來資料放大 5 倍，過去的訊號結果必須完全不變）
- 回測整合（零成交時權益守恆、有成交時股數必為 1000 倍數、市值不超過上限）
- 資料落地層（SQLite upsert 冪等性、日期/股票篩選、latest_date 增量同步邏輯）
- 資料抓取腳本（用假的 provider 驗證寫入流程，不觸碰真實網路）
