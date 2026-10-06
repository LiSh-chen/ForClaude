# 投資研究日誌系統（invest-journal）

每個交易日自動：**讀財經新聞 → 找出市場關注/有潛力的產業 → 找龍頭股與財務績優股 → 選出台股 5 檔、美股 5 檔 → 以多種方法估算 12 個月目標價（含 bull/base/bear 情境）→ 把完整調查與評估過程寫成研究日誌並發佈到網站**。
每週追蹤過往推薦的實際績效，每月檢討（目標價偏差、因子有效性、產業表現），並在樣本足夠時把結論**回饋到下一期的評估參數**。

> 僅供研究與教育用途，不構成投資建議。

## 網站

發佈在 GitHub Pages 的 `/journal/` 子路徑（與 stock-derby 共用同一個 Pages 站台，因為一個 repo 只能有一個 Pages）：

- 最新日誌、日誌列表、績效追蹤（含每筆推薦的目標價/現價/報酬/超額報酬）、檢討報告、方法論

## 運作方式

| Workflow | 時間 | 做什麼 |
|---|---|---|
| `Invest Journal Daily` | 週一至週五 22:30 UTC | 更新追蹤 → 產生當日日誌 → commit |
| `Invest Journal Review` | 每週六 02:00 UTC | 週度檢討；每月第一個週六加月度檢討並嘗試回饋參數 |
| `Stock Derby Daily Build` | 接在上面兩者之後（`workflow_run`） | 把 `invest-journal/site` 併入 Pages 發佈 |

啟用步驟：

1. 合併到預設分支（排程只會在預設分支跑）；repo **Settings → Pages → Source 選 GitHub Actions**（stock-derby 已需要這個設定）。
2. 到 Actions 頁手動執行一次 `Invest Journal Daily`（可立刻看到第一份日誌）。
3. （選用）Settings → Secrets → 新增 `ANTHROPIC_API_KEY`，日誌會加上 AI 撰寫的市場摘要與個股評論；可用環境變數 `IJ_LLM_MODEL` 指定模型。量化選股/目標價不依賴它。

## 本機使用

```bash
cd invest-journal
pip install -r requirements.txt
python -m ijournal daily          # 產生今日日誌（需要網路）
python -m ijournal track          # 追蹤過往推薦績效
python -m ijournal review --period monthly --tune
python -m ijournal site           # 重建網站（輸出到 site/）

# 離線驗證流程（合成資料，輸出導到別處，絕不可當真實行情）
IJ_OUT=/tmp/ij-demo python -m ijournal daily --demo --asof 2026-06-01
python -m pytest tests
```

## 目錄

```
config/universe.json      產業與股票宇宙、產業關鍵字、產業風險描述
config/sources.json       新聞 RSS 來源、情緒詞
config/params.json        所有可被回饋調整的參數（權重、門檻、估值假設）
config/params_history.json 每次自動調參的稽核紀錄
data/picks/               每日推薦（含目標價各方法拆解）
data/candidates/          每日全宇宙評分（用於檢驗因子預測力）
data/performance.json     追蹤結果
journal/  reviews/        研究日誌、檢討報告（Markdown）
site/                     渲染後的靜態網站
docs/methodology.md       方法論與已知限制（也會出現在網站上）
```

## 如何調整

- 新增/移除關注的股票或產業：改 `config/universe.json`（`代號|名稱|別名`；`.TW` 上市、`.TWO` 上櫃）。
- 新增新聞來源：改 `config/sources.json`。
- 調整篩選門檻、估值假設：改 `config/params.json`。
