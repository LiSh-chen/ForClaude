"""劇本生成：Claude API（有 key）或範本（試跑模式）。

劇本格式 = 帶「表演標記」的片段清單，供 TTS 依情緒調整語氣：
  {"speaker": "host" | "guest", "emotion": <EMOTIONS 之一>, "text": "..."}
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from . import config

EMOTIONS = ["neutral", "excited", "curious", "thoughtful", "warm", "serious", "surprised", "laughing"]
SPEAKERS = ["host", "guest"]
CHARS_PER_MIN = 270  # 華語口語約 250~300 字/分鐘
MAX_SEGMENT_CHARS = 220


class ScriptError(Exception):
    pass


SYSTEM_PROMPT = """你是資深 Podcast 編劇，專寫「像真人即興聊天」的台灣華語口語稿。

寫作規則：
1. 用台灣用語與口語，不要書面語、不要條列式說話。句子短，一個片段最多 {max_chars} 字。
2. 適度加入口語元素：語助詞（嗯、欸、對啊）、自我修正（我是說…）、反問、停頓用「…」表示、偶爾的猶豫。不要過量。
3. 情緒要有起伏，不要整集同一個語氣。每個片段標一個 emotion，只能用：{emotions}。
4. 數字、英文縮寫盡量改寫成好念的說法（例如「3C」寫成「三C」）。
5. 必須涵蓋使用者提供的所有重點，但不要像唸重點清單；用故事、比喻、舉例帶出。
6. 開場要有鉤子，結尾要有收束與一句行動建議或提問。
7. 總字數約 {total_chars} 字（誤差 15% 內）。
{mode_rule}
只輸出 JSON，不要任何其他文字，格式：
{{"title": "節目標題", "segments": [{{"speaker": "host", "emotion": "warm", "text": "..."}}]}}"""

MODE_SOLO = "8. 單人節目：所有片段 speaker 都是 \"host\"。"
MODE_DUAL = (
    "8. 雙人對談：\"host\" 是節目主持人（會用使用者本人的聲音），\"guest\" 是搭檔。"
    "兩人要有自然的來回、附和（嗯、對）與偶爾的打斷，host 約佔 55~65% 的內容。"
)


def build_prompt(brief: dict[str, Any]) -> tuple[str, str]:
    minutes = brief["minutes"]
    system = SYSTEM_PROMPT.format(
        max_chars=MAX_SEGMENT_CHARS,
        emotions="、".join(EMOTIONS),
        total_chars=minutes * CHARS_PER_MIN,
        mode_rule=MODE_DUAL if brief["mode"] == "dual" else MODE_SOLO,
    )
    points = "\n".join(f"- {p}" for p in brief["points"])
    user = (
        f"主題：{brief['topic']}\n"
        f"重點：\n{points}\n"
        f"風格：{brief.get('style') or '輕鬆、親切、有點幽默'}\n"
        f"目標長度：約 {minutes} 分鐘"
    )
    return system, user


def parse_script(raw: str) -> dict[str, Any]:
    """容忍 ```json 圍欄與前後雜訊，並驗證/修正欄位。"""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ScriptError("模型回傳的內容不是 JSON")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ScriptError(f"JSON 解析失敗：{e}") from e
    return {"title": str(data.get("title", "")).strip(), "segments": clean_segments(data.get("segments"))}


def clean_segments(segs: Any) -> list[dict[str, str]]:
    if not isinstance(segs, list) or not segs:
        raise ScriptError("劇本沒有任何片段")
    out = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        speaker = s.get("speaker") if s.get("speaker") in SPEAKERS else "host"
        emotion = s.get("emotion") if s.get("emotion") in EMOTIONS else "neutral"
        out.append({"speaker": speaker, "emotion": emotion, "text": text})
    if not out:
        raise ScriptError("劇本沒有任何有效片段")
    return out


def _mock_script(brief: dict[str, Any]) -> dict[str, Any]:
    topic, pts = brief["topic"], brief["points"]
    dual = brief["mode"] == "dual"
    segs = [
        {"speaker": "host", "emotion": "warm", "text": f"哈囉大家好，歡迎收聽。今天想跟你聊一個我最近一直在想的事…{topic}。"},
        {"speaker": "guest" if dual else "host", "emotion": "curious", "text": "欸，這個主題聽起來很有趣，你怎麼會想到要聊這個？"},
    ]
    for i, p in enumerate(pts):
        segs.append({"speaker": "host", "emotion": "excited" if i % 2 == 0 else "thoughtful", "text": f"第{i + 1}個重點，{p}。這件事其實比大家想的還要重要。"})
        if dual:
            segs.append({"speaker": "guest", "emotion": "surprised", "text": "嗯，對，我也有類似的感覺。"})
    segs.append({"speaker": "host", "emotion": "warm", "text": "好，今天就聊到這邊。如果你有想法，歡迎留言跟我分享，我們下一集見。"})
    return {"title": f"【試跑】{topic}", "segments": segs}


def generate(brief: dict[str, Any]) -> dict[str, Any]:
    """回傳 {"title", "segments", "mock": bool}。"""
    key = config.anthropic_key()
    if not key:
        return {**_mock_script(brief), "mock": True}
    system, user = build_prompt(brief)
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": config.anthropic_model(),
                "max_tokens": min(16000, 2000 + brief["minutes"] * 700),
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=180,
        )
        r.raise_for_status()
        raw = "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
    except httpx.HTTPError as e:
        raise ScriptError(f"呼叫 Claude API 失敗：{e}") from e
    return {**parse_script(raw), "mock": False}
