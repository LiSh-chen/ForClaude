# news-curator

從大量新聞來源中,挑出「新穎、有洞見、影響深遠、有證據」的少數文章,定期寄給訂閱者。寧缺勿濫:沒有達標內容就不寄。

流程:抓 RSS → 去重/排除已寄 → 評分 → 門檻+多樣性篩選 → 寄信 → 記錄已寄。

## 評分
Claude 對每篇文章給 novelty / depth / impact / evidence (0–10) 與 trivia(瑣事程度,最多扣 40%),
總分 0–100,預設 ≥55 才入選,單一來源最多 2 篇。沒有 `ANTHROPIC_API_KEY` 時退回關鍵字啟發式(品質明顯較差)。

## 使用
```bash
python -m curator run --dry-run --out preview.html   # 預覽,不寄信
python -m curator subscribe you@example.com           # 本機名單 subscribers.txt (已 gitignore)
python -m pytest tests
```
可調參數:`--days --top --min-score --per-source --model --sources`。來源在 `sources.json`。

## 部署 (GitHub Actions)
Repo Secrets:`ANTHROPIC_API_KEY`、`NEWS_SUBSCRIBERS`(逗號或換行分隔)、`SMTP_HOST`、`SMTP_PORT`、`SMTP_USER`、`SMTP_PASSWORD`、`MAIL_FROM`。
`.github/workflows/digest.yml` 預設週一、週四寄出。名單放 Secret 而非檔案,因為 email 是個資。

## 已知限制
- 退訂目前是「回信說 unsubscribe」,由你手動改 Secret;訂閱量大時需要另做訂閱/退訂網頁。
- `sources.json` 的 feed 網址未經實際連線驗證(開發環境網路受限),首次執行看日誌中的 ✓/✗ 並修正。
- 評分只看標題+摘要,未讀全文。
