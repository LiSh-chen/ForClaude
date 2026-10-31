# Podcast Studio（網頁 MVP）

使用者錄/傳聲音樣本 + 輸入主題與重點 → Claude 產生帶情緒標記的口語劇本 → 使用者編修確認 → 用克隆聲音逐段合成 → 串接成 mp3。

## 快速開始

```bash
cd podcast-studio
pip install -r requirements.txt     # 另需系統安裝 ffmpeg / ffprobe
uvicorn app.main:app --port 8000    # 開啟 http://localhost:8000
```

**不設任何 key 也能跑**：自動進入「試跑模式」（範本劇本 + 音調代替聲音），可完整驗證流程。要用真實服務：

| 環境變數 | 作用 |
|---|---|
| `ANTHROPIC_API_KEY` | 啟用 Claude 生成劇本（`ANTHROPIC_MODEL` 可換模型，預設 `claude-sonnet-5-5`） |
| `ELEVENLABS_API_KEY` | 啟用聲音克隆與合成（Instant Voice Clone 需付費方案；`ELEVENLABS_MODEL` 預設 `eleven_v3`） |
| `ELEVENLABS_GUEST_VOICE_ID` | 「雙人對談」時 AI 搭檔使用的現成聲音 |
| `PODCAST_DATA_DIR` | 資料目錄，預設 `./storage` |

## 設計重點

- **同意機制**：建立專案必須勾選「這是我本人的聲音」並填姓名；成品 mp3 metadata 標示為 AI 合成。
- **劇本帶表演標記**：每段有 `speaker` 與 `emotion`（平穩/興奮/好奇/沉思/溫暖/認真/驚訝/笑），轉成 TTS 的 audio tag；提示詞要求台灣口語、語助詞、短句、情緒起伏（對應先前調查中 MoonCast 的「腳本口語化」做法）。
- **確認關卡**：任何劇本修改都會重置確認狀態，必須再按一次確認才能渲染。
- **逐段合成 + 快取**：長句依標點切開；片段以 (內容、情緒、聲音) 雜湊快取，改一句只重合成那一句。
- **後製**：停頓依換人/情緒調整，ffmpeg 串接並標準化到 -16 LUFS。
- **Provider 可抽換**：`app/tts.py` 的 `Provider` 介面（`clone_voice` / `synthesize`），要換 MiniMax、Fish Audio 或自架 CosyVoice/F5-TTS 只需新增一個 class。

## 結構

```
app/main.py        API（FastAPI）        app/pipeline.py  背景渲染流程
app/script_gen.py  劇本生成與解析        app/tts.py       TTS / 克隆 provider
app/audio.py       ffmpeg 處理           app/store.py     專案儲存（資料夾 + json）
static/index.html  前端（四步驟精靈）     tests/           pytest（含端到端試跑流程）
```

## 已知限制（MVP）

- 無登入/多使用者；請只在本機或受信任網路使用（預設不要對外開放）。
- 渲染用程序內執行緒，伺服器重啟會中斷進行中的工作（介面會提示重試，已合成片段不會重做）。
- 數字/英文/破音字的文字正規化只做了最基本處理。
- ElevenLabs 與 Claude 的真實 API 呼叫**尚未用真實 key 測過**（只驗證了試跑模式與解析邏輯），第一次接上時請先用短劇本試。
- 還沒做：片頭片尾音樂、RSS 發佈、聲音本人驗證（朗讀隨機句子）、情緒參考樣本檢索。

## 測試

```bash
cd podcast-studio && python -m pytest tests -q
```
