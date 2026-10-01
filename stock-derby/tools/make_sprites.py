"""Procedural pixel-art horse sprite library (no image model, pure code - tweak and re-run).

Each coat gets a sprite sheet: 6 gallop frames + 1 standing frame, 48x36 px each, facing right,
jockey included (silks differ per coat).  Output: ../assets/horses/<coat>.png and manifest.json.

    python stock-derby/tools/make_sprites.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 48, 36
RUN_FRAMES = 6
OUT = Path(__file__).parent.parent / "assets" / "horses"

# material ids drawn into an index image; coats colourise them
(T, BODY, SHADE, HEAD, MANE, TAIL, NEAR_UP, NEAR_LO, FAR_UP, FAR_LO, HOOF, EYE, FACE, NOSE) = range(14)
JK_SILK, JK_SILK2, JK_HELMET, JK_SKIN, JK_BOOT, OUTLINE = 20, 21, 22, 23, 24, 30


def _leg(d, hip, phase, kind, idle):
    """One leg: thigh + folding lower leg. kind: 'near'/'far'."""
    up_id, lo_id = (NEAR_UP, NEAR_LO) if kind == "near" else (FAR_UP, FAR_LO)
    a = 7 * math.sin(phase * 2 * math.pi) if idle else 40 * math.sin(phase * 2 * math.pi)
    swing = math.cos(phase * 2 * math.pi)
    bend = 0 if idle else 8 + 60 * max(0.0, swing) ** 1.2
    tu = math.radians(a)
    tl = math.radians(a - bend)
    hx, hy = hip
    kx, ky = hx + 6 * math.sin(tu), hy + 6 * math.cos(tu)
    fx, fy = kx + 6 * math.sin(tl), ky + 6 * math.cos(tl)
    d.line([(hx, hy), (round(kx), round(ky))], fill=up_id, width=3)
    d.line([(round(kx), round(ky)), (round(fx), round(fy))], fill=lo_id, width=2)
    d.rectangle([round(fx) - 1, round(fy), round(fx) + 1, round(fy) + 1], fill=HOOF)


def frame_ids(i: int, idle: bool = False) -> Image.Image:
    p = i / RUN_FRAMES
    img = Image.new("P", (W, H), T)
    d = ImageDraw.Draw(img)
    bob = 0 if idle else round(1.2 * math.sin(p * 4 * math.pi))
    oy = bob + (0 if idle else 0)

    def P(*pts):
        return [(x, y + oy) for x, y in pts]

    # far legs first, then tail, body, near legs
    if idle:
        _leg(d, (16, 21 + oy), 0.02, "far", True); _leg(d, (29, 21 + oy), 0.98, "far", True)
    else:
        _leg(d, (16, 21 + oy), p + 0.62, "far", False); _leg(d, (30, 21 + oy), p + 0.12, "far", False)
    w = 0 if idle else math.sin(p * 2 * math.pi)
    t = [(13, 14 + oy), (8, 15 + oy + round(w)), (4, 18 + oy + round(2 * w)), (2, 23 + oy + round(w))]
    d.line(t, fill=TAIL, width=3)
    d.polygon(P((12, 13), (18, 12), (29, 12), (33, 14), (34, 19), (30, 23), (18, 23), (12, 21), (10, 16)), fill=BODY)
    d.polygon(P((13, 21), (30, 21), (28, 23), (18, 23)), fill=SHADE)
    # neck + head (head dips a little when galloping)
    hd = 0 if idle else round(math.sin(p * 2 * math.pi + 1))
    d.polygon(P((28, 13), (35, 12), (41, 6 + hd), (37, 3 + hd), (31, 9)), fill=BODY)
    d.polygon(P((37, 3 + hd), (42, 4 + hd), (47, 10 + hd), (46, 13 + hd), (41, 12 + hd), (38, 8 + hd)), fill=HEAD)
    d.polygon(P((42, 4 + hd), (44, 2 + hd), (44, 5 + hd)), fill=HEAD)                       # ear
    d.rectangle([46 - 0, 10 + hd + oy, 47, 12 + hd + oy], fill=NOSE)
    d.point((42, 6 + hd + oy), fill=EYE)
    d.line([(40, 5 + hd + oy), (44, 9 + hd + oy)], fill=FACE, width=1)                     # blaze line
    # mane flowing back along the neck
    mw = 0 if idle else round(math.sin(p * 2 * math.pi + 2))
    d.line([(37, 3 + hd + oy), (33 - mw, 5 + oy), (30 - mw, 9 + oy), (28, 13 + oy)], fill=MANE, width=2)
    if idle:
        _leg(d, (16, 21 + oy), 0.0, "near", True); _leg(d, (30, 21 + oy), 0.5, "near", True)
    else:
        _leg(d, (16, 21 + oy), p + 0.5, "near", False); _leg(d, (30, 21 + oy), p, "near", False)
    # jockey, crouched forward
    d.line([(21, 12 + oy), (23, 17 + oy)], fill=JK_SKIN, width=2)
    d.line([(23, 17 + oy), (23, 19 + oy)], fill=JK_BOOT, width=2)
    d.polygon(P((19, 12), (23, 10), (29, 7), (31, 9), (26, 13), (21, 14)), fill=JK_SILK)
    d.polygon(P((24, 11), (27, 9), (28, 10), (25, 12)), fill=JK_SILK2)
    d.line([(28, 9 + oy), (34, 11 + oy)], fill=JK_SKIN, width=1)
    d.ellipse([28, 3 + oy, 33, 8 + oy], fill=JK_SKIN)
    d.pieslice([28, 2 + oy, 33, 7 + oy], 180, 360, fill=JK_HELMET)
    d.rectangle([28, 4 + oy, 31, 5 + oy], fill=JK_HELMET)
    # 1px dark outline
    px = img.load()
    out = [(x, y) for y in range(H) for x in range(W) if px[x, y] == T and any(
        0 <= x + dx < W and 0 <= y + dy < H and px[x + dx, y + dy] != T for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    for x, y in out:
        px[x, y] = OUTLINE
    return img, oy


def hx(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def shade(c, f):
    return tuple(max(0, min(255, round(v * f))) for v in c)


def h2(x, y, seed):
    n = (x * 374761393 + y * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return (n ^ (n >> 16)) / 0xFFFFFFFF


# ---- the coat library ------------------------------------------------------------------
# silks: (primary, secondary, pattern)  pattern: solid | stripe | check | sash
COATS = {
    "chestnut":  dict(zh="栗毛", base="#b5651d", mane="#7a3f12", blaze=True, silks=("#d62828", "#ffffff", "stripe")),
    "black":     dict(zh="黑駿", base="#262628", mane="#111114", socks=True, silks=("#f5c518", "#222222", "check")),
    "bay":       dict(zh="鹿毛", base="#8b4a2b", mane="#1c1412", legs="#1c1412", blaze=True, silks=("#1d4ed8", "#ffffff", "sash")),
    "grey":      dict(zh="白駒", base="#e6e8ee", mane="#b9bfcc", shade=0.82, silks=("#16a34a", "#ffffff", "stripe")),
    "dapple":    dict(zh="銀斑灰", base="#8d96a3", mane="#e4e7ec", pattern="dapple", pcolor="#b5bcc7", silks=("#7c3aed", "#fde047", "solid")),
    "palomino":  dict(zh="金鬃", base="#e0b04a", mane="#fff6dc", blaze=True, silks=("#0ea5e9", "#ffffff", "sash")),
    "cremello":  dict(zh="奶油", base="#f2e2b8", mane="#fffaf0", shade=0.88, hoof="#c8b48a", silks=("#ec4899", "#ffffff", "check")),
    "appaloosa": dict(zh="豹斑", base="#f1ece2", mane="#6b4a2b", pattern="spots", pcolor="#3b2a1d", silks=("#f97316", "#1f2937", "stripe")),
    "pinto":     dict(zh="花斑", base="#f3f0ea", mane="#3a2a1e", pattern="patch", pcolor="#8b4f2a", silks=("#14b8a6", "#ffffff", "check")),
    "roan":      dict(zh="藍斑", base="#5f6f88", mane="#2a3140", pattern="roan", pcolor="#8b9ab3", socks=True, silks=("#e11d48", "#fde68a", "sash")),
    "sorrel":    dict(zh="赤沙", base="#c8553d", mane="#f3d3b0", blaze=True, socks=True, silks=("#065f46", "#fcd34d", "solid")),
    "liver":     dict(zh="暗栗", base="#5a2e1c", mane="#2b150c", star=True, silks=("#fbbf24", "#7f1d1d", "stripe")),
    "violet":    dict(zh="紫晶", base="#8a5cc6", mane="#f0abfc", blaze=True, silks=("#f0abfc", "#4c1d95", "check")),
    "emerald":   dict(zh="翠光", base="#76b900", mane="#1a3300", legs="#2f5200", blaze=True, silks=("#111111", "#76b900", "stripe")),
    "flame":     dict(zh="火焰", base="#ef4444", mane="#fde047", legs="#7f1d1d", pattern="stripes", pcolor="#b91c1c", silks=("#fde047", "#ef4444", "sash")),
    "sky":       dict(zh="天藍", base="#60a5fa", mane="#ffffff", blaze=True, socks=True, silks=("#ffffff", "#2563eb", "check")),
    "pink":      dict(zh="櫻粉", base="#f9a8d4", mane="#be185d", blaze=True, silks=("#be185d", "#fbcfe8", "solid")),
    "steel":     dict(zh="鋼銀", base="#c3cbd6", mane="#3b82f6", legs="#8792a2", pattern="dapple", pcolor="#a2adbb", silks=("#334155", "#38bdf8", "stripe")),
    "obsidian":  dict(zh="黑金", base="#1c1c22", mane="#d4af37", hoof="#d4af37", star=True, silks=("#d4af37", "#111111", "sash")),
    "tiger":     dict(zh="虎紋", base="#d9a441", mane="#3a2410", pattern="stripes", pcolor="#5a3a12", silks=("#ea580c", "#111111", "stripe")),
}


def marks_mask(marks, oy):
    """Set of (x, y) pixels covered by a coat's logo-inspired marks (shifted with the body bob)."""
    if not marks:
        return set()
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for mk in marks:
        kind = mk[0]
        if kind == "line":
            d.line([(x, y + oy) for x, y in mk[1]], fill=255, width=mk[2])
        elif kind == "ring":
            (cx, cy, r), w = mk[1], mk[2]
            d.ellipse([cx - r, cy + oy - r, cx + r, cy + oy + r], outline=255, width=w)
        elif kind == "arc":
            (cx, cy, r), a0, a1, w = mk[1], mk[2], mk[3], mk[4]
            d.arc([cx - r, cy + oy - r, cx + r, cy + oy + r], a0, a1, fill=255, width=w)
    px = m.load()
    return {(x, y) for y in range(H) for x in range(W) if px[x, y]}


def _pick(v, k):
    return hx(v[k % len(v)] if isinstance(v, list) else v)


def colourise(ids: Image.Image, c: dict, oy: int = 0) -> Image.Image:
    base = hx(c["base"]); hoof = hx(c.get("hoof", "#2b2220")); shd = shade(base, c.get("shade", 0.78))
    legs = hx(c["legs"]) if c.get("legs") else base
    p1, p2, pat = c["silks"]; p1, p2 = hx(p1), hx(p2)
    seed = sum(map(ord, c["base"]))
    mask = marks_mask(c.get("marks"), oy)
    mc = hx(c.get("mc", "#ffffff"))
    pt, pcs = c.get("pattern"), c.get("pcolor", "#000000")
    img = Image.new("RGBA", ids.size, (0, 0, 0, 0))
    px, src = img.load(), ids.load()
    white = (246, 244, 240)
    bodylike = (BODY, HEAD, SHADE, NEAR_UP, FAR_UP)
    for y in range(ids.height):
        for x in range(ids.width):
            i = src[x, y]
            if i == T:
                continue
            col = None
            if i == OUTLINE: col = (28, 22, 26)
            elif i in (BODY, HEAD):
                col = shade(base, 1.18) if y > 0 and src[x, y - 1] in (T, OUTLINE, MANE) and i == BODY else base
            elif i == SHADE: col = shd
            elif i in (MANE, TAIL): col = _pick(c["mane"], x + y)
            elif i in (NEAR_UP, NEAR_LO): col = legs
            elif i in (FAR_UP, FAR_LO): col = shade(legs, 0.8)
            elif i == HOOF: col = hoof
            elif i == EYE: col = (15, 15, 20)
            elif i == NOSE: col = shade(base, 0.7)
            elif i == FACE: col = white if c.get("blaze") else base
            elif i == JK_SKIN: col = (240, 190, 150)
            elif i == JK_HELMET: col = p2 if pat != "solid" else shade(p1, 0.6)
            elif i == JK_BOOT: col = (40, 30, 30)
            elif i in (JK_SILK, JK_SILK2):
                alt = (pat == "stripe" and x % 3 == 0) or (pat == "check" and (x + y) % 2 == 0) \
                    or (pat == "sash" and (x - y) % 5 == 0)
                col = p2 if (alt or i == JK_SILK2) and pat != "solid" else p1
                if i == JK_SILK2 and pat == "solid": col = shade(p1, 0.8)
            if i in (NEAR_LO, FAR_LO) and c.get("socks"): col = white if i == NEAR_LO else shade(white, 0.85)
            if i == HEAD and c.get("star") and (x, y) == (42, 5): col = white
            if i in bodylike:
                r = h2(x, y, seed)
                if pt == "spots" and r < 0.13: col = _pick(pcs, int(r * 1000))
                elif pt == "dapple" and (x % 3 == 1 and y % 3 == 1): col = _pick(pcs, 0)
                elif pt == "roan" and r < 0.28: col = _pick(pcs, 0)
                elif pt == "patch" and h2(x // 5, y // 5, seed) < 0.38: col = _pick(pcs, 0)
                elif pt == "stripes" and (x + y) % 4 == 0 and i != HEAD: col = _pick(pcs, 0)
                elif pt == "hstripes" and (y - oy) % 3 == 0 and i in (BODY, SHADE): col = _pick(pcs, 0)
                elif pt == "vstripes" and x % 4 == 0 and i in (BODY, SHADE): col = _pick(pcs, 0)
                elif pt == "quad" and i in (BODY, SHADE, NEAR_UP, FAR_UP) and 10 <= x <= 34:
                    col = _pick(pcs, (1 if x >= 22 else 0) + (2 if (y - oy) >= 17 else 0))
                elif pt == "hsplit" and i in (BODY, SHADE, NEAR_UP, FAR_UP) and 10 <= x <= 34:
                    col = _pick(pcs, 0 if (y - oy) < 17 else 1)
                if (x, y) in mask and i in (BODY, SHADE, NEAR_UP, FAR_UP):
                    col = mc
            px[x, y] = (*col, 255) if col else (0, 0, 0, 0)
    return img


def sheet(coat: dict) -> Image.Image:
    sh = Image.new("RGBA", (W * (RUN_FRAMES + 1), H), (0, 0, 0, 0))
    for i in range(RUN_FRAMES):
        ids, oy = frame_ids(i)
        sh.paste(colourise(ids, coat, oy), (i * W, 0))
    ids, oy = frame_ids(0, idle=True)
    sh.paste(colourise(ids, coat, oy), (RUN_FRAMES * W, 0))
    return sh


def features(c: dict) -> str:
    f = []
    if c.get("blaze"): f.append("白面流星")
    if c.get("star"): f.append("額頭小星")
    if c.get("socks"): f.append("白襪")
    if c.get("marks"): f.append("身側品牌風格標記花紋")
    pt = {"dapple": "銀斑", "spots": "彩點", "patch": "大塊花斑", "roan": "斑駁", "stripes": "斜紋", "hstripes": "橫紋",
          "vstripes": "直紋", "quad": "四色分區", "hsplit": "上下雙色"}.get(c.get("pattern"), "")
    if pt: f.append(pt)
    if isinstance(c["mane"], list): f.append("多彩鬃毛")
    return "、".join(f) or "素色"


def main() -> None:
    from horse_specs import SPECS

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    manifest = {"frame_w": W, "frame_h": H, "run_frames": RUN_FRAMES, "brands": {}, "generic": {}}
    for tk, (company, zh, en, why, coat) in SPECS.items():
        coat = {"mane": "#333333", **coat}
        sheet(coat).save(OUT / f"{tk}.png")
        manifest["brands"][tk] = {"company": company, "zh": zh, "en": en, "inspired_by": why, "features": features(coat),
                                  "silks": dict(zip(("primary", "secondary", "pattern"), coat["silks"]))}
    for cid, c in COATS.items():
        sheet(c).save(OUT / f"generic_{cid}.png")
        manifest["generic"][cid] = {"zh": c["zh"], "features": features(c)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"{len(SPECS)} brand sheets + {len(COATS)} generic coat sheets -> {OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
