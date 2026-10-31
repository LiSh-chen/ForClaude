"""Podcast Studio MVP — FastAPI 後端。

流程：建立專案(同意聲明) → 上傳/錄製聲音樣本 → 輸入主題重點 → 生成劇本
      → 使用者編修並確認 → 背景渲染(克隆聲音逐段合成 + 串接) → 試聽/下載。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import audio, config, pipeline, script_gen, store, tts

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_SAMPLE_BYTES = 25 * 1024 * 1024
MAX_SAMPLES = 10
MAX_SEGMENTS = 400

app = FastAPI(title="Podcast Studio MVP")


class NewProject(BaseModel):
    title: str = Field("", max_length=80)
    consent: bool
    consent_name: str = Field("", max_length=80)


class Brief(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    points: list[str] = Field(..., min_length=1, max_length=12)
    style: str = Field("", max_length=200)
    minutes: int = Field(5, ge=1, le=20)
    mode: Literal["solo", "dual"] = "solo"


class Segment(BaseModel):
    speaker: Literal["host", "guest"]
    emotion: str
    text: str = Field(..., max_length=1000)


class ScriptBody(BaseModel):
    title: str = Field("", max_length=120)
    segments: list[Segment] = Field(..., min_length=1, max_length=MAX_SEGMENTS)


def _get(pid: str) -> dict:
    try:
        proj = store.load(pid)
    except store.NotFound:
        raise HTTPException(404, "找不到專案")
    # 伺服器重啟會讓 running 狀態懸空，補成錯誤讓使用者可重試
    if proj["render"]["state"] == "running" and not pipeline.is_active(pid):
        proj["render"].update(state="error", error="渲染被中斷（伺服器重啟？），請重新按「生成音檔」")
        store.save(proj)
    return proj


@app.get("/api/config")
def get_config():
    provider = tts.get_provider()
    return {
        "mock_tts": provider.is_mock,
        "mock_llm": not config.anthropic_key(),
        "tts_provider": provider.name,
        "emotions": script_gen.EMOTIONS,
        "chars_per_min": script_gen.CHARS_PER_MIN,
    }


@app.post("/api/projects")
def create_project(body: NewProject):
    if not body.consent or not body.consent_name.strip():
        raise HTTPException(400, "必須勾選同意並填寫聲音本人姓名，才能建立聲音克隆專案")
    return store.create(body.title.strip(), body.consent_name.strip())


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _get(pid)


@app.post("/api/projects/{pid}/samples")
def upload_sample(pid: str, file: UploadFile = File(...)):
    proj = _get(pid)
    if len(proj["samples"]) >= MAX_SAMPLES:
        raise HTTPException(400, f"最多 {MAX_SAMPLES} 段樣本")
    d = store.project_dir(pid) / "samples"
    n = len(proj["samples"]) + 1
    while (d / f"sample{n}.mp3").exists():
        n += 1
    raw = d / f"upload{n}.bin"
    size = 0
    with raw.open("wb") as out:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_SAMPLE_BYTES:
                out.close()
                raw.unlink(missing_ok=True)
                raise HTTPException(413, "檔案太大（上限 25MB）")
            out.write(chunk)
    dst = d / f"sample{n}.mp3"
    try:
        secs = audio.normalize_sample(raw, dst)
    except audio.AudioError as e:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, f"無法解析音訊：{e}")
    finally:
        raw.unlink(missing_ok=True)
    proj["samples"].append({"file": dst.name, "seconds": round(secs, 1)})
    proj["voice_id"] = None  # 樣本變動 → 需重新克隆
    store.save(proj)
    return proj


@app.delete("/api/projects/{pid}/samples/{name}")
def delete_sample(pid: str, name: str):
    proj = _get(pid)
    if name not in [s["file"] for s in proj["samples"]]:
        raise HTTPException(404, "找不到樣本")
    (store.project_dir(pid) / "samples" / name).unlink(missing_ok=True)
    proj["samples"] = [s for s in proj["samples"] if s["file"] != name]
    proj["voice_id"] = None
    store.save(proj)
    return proj


@app.get("/api/projects/{pid}/samples/{name}")
def get_sample(pid: str, name: str):
    proj = _get(pid)
    if name not in [s["file"] for s in proj["samples"]]:
        raise HTTPException(404, "找不到樣本")
    return FileResponse(store.project_dir(pid) / "samples" / name, media_type="audio/mpeg")


@app.post("/api/projects/{pid}/script/generate")
def generate_script(pid: str, brief: Brief):
    proj = _get(pid)
    points = [p.strip() for p in brief.points if p.strip()]
    if not points:
        raise HTTPException(400, "請至少輸入一個重點")
    data = brief.model_dump() | {"points": points}
    try:
        result = script_gen.generate(data)
    except script_gen.ScriptError as e:
        raise HTTPException(502, str(e))
    proj["brief"] = data
    proj["script"] = {"title": result["title"], "segments": result["segments"], "mock": result["mock"]}
    proj["script_confirmed"] = False
    proj["final"] = None
    store.save(proj)
    return proj


@app.put("/api/projects/{pid}/script")
def save_script(pid: str, body: ScriptBody):
    proj = _get(pid)
    try:
        segs = script_gen.clean_segments([s.model_dump() for s in body.segments])
    except script_gen.ScriptError as e:
        raise HTTPException(400, str(e))
    proj["script"] = {"title": body.title, "segments": segs, "mock": (proj["script"] or {}).get("mock", False)}
    proj["script_confirmed"] = False  # 任何修改都需重新確認
    store.save(proj)
    return proj


@app.post("/api/projects/{pid}/script/confirm")
def confirm_script(pid: str):
    proj = _get(pid)
    if not proj["script"]:
        raise HTTPException(400, "尚未有劇本")
    proj["script_confirmed"] = True
    store.save(proj)
    return proj


@app.post("/api/projects/{pid}/render")
def render(pid: str):
    proj = _get(pid)
    if not proj["script"] or not proj["script_confirmed"]:
        raise HTTPException(400, "劇本需先確認才能生成音檔")
    if not proj["samples"]:
        raise HTTPException(400, "請先上傳或錄製聲音樣本")
    if not pipeline.start(pid):
        raise HTTPException(409, "正在生成中")
    proj = store.load(pid)
    proj["render"].update(state="running", done=0, total=0, error=None)
    store.save(proj)
    return proj


@app.get("/api/projects/{pid}/audio")
def get_audio(pid: str, download: bool = False):
    proj = _get(pid)
    if not proj["final"]:
        raise HTTPException(404, "尚未生成音檔")
    name = f"{proj['title']}.mp3" if download else None
    return FileResponse(store.project_dir(pid) / "final.mp3", media_type="audio/mpeg", filename=name)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    _get(pid)
    shutil.rmtree(store.project_dir(pid), ignore_errors=True)
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
