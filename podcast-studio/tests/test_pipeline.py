import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from app import audio, script_gen, tts
from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "in.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=2", str(f)], check=True)
    return f


def _project():
    r = client.post("/api/projects", json={"title": "t", "consent": True, "consent_name": "小明"})
    assert r.status_code == 200
    return r.json()["id"]


def _wait_render(pid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = client.get(f"/api/projects/{pid}").json()
        if p["render"]["state"] != "running":
            return p
        time.sleep(0.2)
    raise AssertionError("render timeout")


def test_consent_required():
    assert client.post("/api/projects", json={"title": "", "consent": False, "consent_name": "x"}).status_code == 400
    assert client.post("/api/projects", json={"title": "", "consent": True, "consent_name": " "}).status_code == 400


def test_bad_project_id_rejected():
    assert client.get("/api/projects/..%2F..%2Fetc").status_code in (404, 422)
    assert client.get("/api/projects/zzzzzzzzzzzz").status_code == 404


def test_bad_sample_rejected():
    pid = _project()
    r = client.post(f"/api/projects/{pid}/samples", files={"file": ("x.mp3", b"not audio", "audio/mpeg")})
    assert r.status_code == 400


def test_full_flow_mock(sample_file):
    pid = _project()
    r = client.post(f"/api/projects/{pid}/samples", files={"file": ("a.wav", sample_file.read_bytes(), "audio/wav")})
    assert r.status_code == 200 and r.json()["samples"][0]["seconds"] > 1.5

    # 劇本未確認前不可渲染
    assert client.post(f"/api/projects/{pid}/render").status_code == 400

    brief = {"topic": "開會", "points": ["沒目標", "人太多"], "minutes": 3, "mode": "dual"}
    p = client.post(f"/api/projects/{pid}/script/generate", json=brief).json()
    assert p["script"]["mock"] and {s["speaker"] for s in p["script"]["segments"]} == {"host", "guest"}
    assert client.post(f"/api/projects/{pid}/render").status_code == 400  # 仍未確認

    # 編修劇本 → 確認狀態被重置
    segs = p["script"]["segments"]
    segs[0]["text"] = "改過的開場白。"
    p = client.put(f"/api/projects/{pid}/script", json={"title": "我的節目", "segments": segs}).json()
    assert p["script_confirmed"] is False
    assert client.post(f"/api/projects/{pid}/script/confirm").json()["script_confirmed"] is True

    assert client.post(f"/api/projects/{pid}/render").status_code == 200
    p = _wait_render(pid)
    assert p["render"]["state"] == "done", p["render"]
    assert p["final"]["seconds"] > 3 and p["final"]["mock"]

    a = client.get(f"/api/projects/{pid}/audio")
    assert a.status_code == 200 and a.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3") or a.content[:2] == b"\xff\xfb"

    # 只改一段 → 只多合成一個新片段檔
    from app import store
    seg_dir = store.project_dir(pid) / "segments"
    before = {f.name for f in seg_dir.iterdir()}
    segs[1]["text"] = "第二段也改了。"
    client.put(f"/api/projects/{pid}/script", json={"title": "我的節目", "segments": segs})
    client.post(f"/api/projects/{pid}/script/confirm")
    client.post(f"/api/projects/{pid}/render")
    assert _wait_render(pid)["render"]["state"] == "done"
    assert len({f.name for f in seg_dir.iterdir()} - before) == 1


def test_render_without_samples_errors():
    pid = _project()
    client.post(f"/api/projects/{pid}/script/generate", json={"topic": "主題", "points": ["a"], "minutes": 1})
    client.post(f"/api/projects/{pid}/script/confirm")
    assert client.post(f"/api/projects/{pid}/render").status_code == 400


def test_parse_script_tolerates_fences_and_bad_fields():
    raw = '好的：\n```json\n{"title":"T","segments":[{"speaker":"x","emotion":"???","text":"哈囉"},{"text":""}]}\n```'
    out = script_gen.parse_script(raw)
    assert out["segments"] == [{"speaker": "host", "emotion": "neutral", "text": "哈囉"}]
    with pytest.raises(script_gen.ScriptError):
        script_gen.parse_script("沒有 json")
    with pytest.raises(script_gen.ScriptError):
        script_gen.parse_script('{"segments": []}')


def test_split_text_keeps_content_and_limit():
    text = "這是一句話。" * 60
    parts = tts.split_text(text, 100)
    assert "".join(parts) == text and all(len(p) <= 100 for p in parts)


def test_prompt_contains_requirements():
    system, user = script_gen.build_prompt({"topic": "T", "points": ["a", "b"], "minutes": 10, "mode": "dual", "style": ""})
    assert "2700" in system and "guest" in system and "- a" in user and "10 分鐘" in user


def test_assemble_adds_metadata(tmp_path, sample_file):
    out = tmp_path / "o.mp3"
    dur = audio.assemble([(sample_file, 300), (sample_file, 500)], out, "標題")
    assert 4.5 < dur < 5.5
