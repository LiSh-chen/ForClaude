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


class Cv:
    """Index-image canvas that draws in base-pixel coordinates scaled by k (k=1 is the original 48x36 pixel art)."""

    def __init__(self, k: int):
        self.k = k
        self.img = Image.new("P", (W * k, H * k), T)
        self.d = ImageDraw.Draw(self.img)

    def _s(self, pts):
        return [(round(x * self.k), round(y * self.k)) for x, y in pts]

    def poly(self, pts, fill):
        self.d.polygon(self._s(pts), fill=fill)

    def line(self, pts, fill, width=1):
        self.d.line(self._s(pts), fill=fill, width=max(1, round(width * self.k)))

    def _box(self, b):
        k = self.k
        return [round(b[0] * k), round(b[1] * k), round((b[2] + 1) * k) - 1, round((b[3] + 1) * k) - 1]

    def rect(self, b, fill):
        self.d.rectangle(self._box(b), fill=fill)

    def ellipse(self, b, fill):
        self.d.ellipse(self._box(b), fill=fill)

    def pie(self, b, a0, a1, fill):
        self.d.pieslice(self._box(b), a0, a1, fill=fill)


def _leg(cv, hip, phase, kind, idle):
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
    r = round if cv.k == 1 else (lambda v: v)          # pixel art snaps joints to the grid; smooth styles keep them exact
    cv.line([(hx, hy), (r(kx), r(ky))], up_id, 3)
    cv.line([(r(kx), r(ky)), (r(fx), r(fy))], lo_id, 2)
    cv.rect([r(fx) - 1, r(fy), r(fx) + 1, r(fy) + 1], HOOF)


def frame_ids(i: int, idle: bool = False, k: int = 1):
    """Material-id image of gallop frame i. k=1: the 48x36 pixel-art frame with its 1px outline; k>1: the same
    pose at k-times resolution (no outline - the smooth styles draw their own)."""
    p = i / RUN_FRAMES
    cv = Cv(k)
    bob = 0 if idle else round(1.2 * math.sin(p * 4 * math.pi))
    oy = bob

    def P(*pts):
        return [(x, y + oy) for x, y in pts]

    # far legs first, then tail, body, near legs
    if idle:
        _leg(cv, (16, 21 + oy), 0.02, "far", True); _leg(cv, (29, 21 + oy), 0.98, "far", True)
    else:
        _leg(cv, (16, 21 + oy), p + 0.62, "far", False); _leg(cv, (30, 21 + oy), p + 0.12, "far", False)
    w = 0 if idle else math.sin(p * 2 * math.pi)
    cv.line([(13, 14 + oy), (8, 15 + oy + round(w)), (4, 18 + oy + round(2 * w)), (2, 23 + oy + round(w))], TAIL, 3)
    cv.poly(P((12, 13), (18, 12), (29, 12), (33, 14), (34, 19), (30, 23), (18, 23), (12, 21), (10, 16)), BODY)
    cv.poly(P((13, 21), (30, 21), (28, 23), (18, 23)), SHADE)
    # neck + head (head dips a little when galloping)
    hd = 0 if idle else round(math.sin(p * 2 * math.pi + 1))
    cv.poly(P((28, 13), (35, 12), (41, 6 + hd), (37, 3 + hd), (31, 9)), BODY)
    cv.poly(P((37, 3 + hd), (42, 4 + hd), (47, 10 + hd), (46, 13 + hd), (41, 12 + hd), (38, 8 + hd)), HEAD)
    cv.poly(P((42, 4 + hd), (44, 2 + hd), (44, 5 + hd)), HEAD)                              # ear
    cv.rect([46, 10 + hd + oy, 47, 12 + hd + oy], NOSE)
    cv.rect([42, 6 + hd + oy, 42, 6 + hd + oy], EYE)
    cv.line([(40, 5 + hd + oy), (44, 9 + hd + oy)], FACE, 1)                               # blaze line
    # mane flowing back along the neck
    mw = 0 if idle else round(math.sin(p * 2 * math.pi + 2))
    cv.line([(37, 3 + hd + oy), (33 - mw, 5 + oy), (30 - mw, 9 + oy), (28, 13 + oy)], MANE, 2)
    if idle:
        _leg(cv, (16, 21 + oy), 0.0, "near", True); _leg(cv, (30, 21 + oy), 0.5, "near", True)
    else:
        _leg(cv, (16, 21 + oy), p + 0.5, "near", False); _leg(cv, (30, 21 + oy), p, "near", False)
    # jockey, crouched forward
    cv.line([(21, 12 + oy), (23, 17 + oy)], JK_SKIN, 2)
    cv.line([(23, 17 + oy), (23, 19 + oy)], JK_BOOT, 2)
    cv.poly(P((19, 12), (23, 10), (29, 7), (31, 9), (26, 13), (21, 14)), JK_SILK)
    cv.poly(P((24, 11), (27, 9), (28, 10), (25, 12)), JK_SILK2)
    cv.line([(28, 9 + oy), (34, 11 + oy)], JK_SKIN, 1)
    cv.ellipse([28, 3 + oy, 33, 8 + oy], JK_SKIN)
    cv.pie([28, 2 + oy, 33, 7 + oy], 180, 360, JK_HELMET)
    cv.rect([28, 4 + oy, 31, 5 + oy], JK_HELMET)
    img = cv.img
    if k == 1:                                                                             # 1px dark outline (pixel art)
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


# ---------------------------------------------------------------------------------------------------
# Smooth styles.  They re-draw the same gallop poses at K x resolution (so the animation is identical across
# styles) and differ only in how the material-id image is turned into colour:
#   flat   - flat vector: soft-edged colour blocks, low-poly facets, no outline
#   sketch - hand-drawn illustration: wobbly pencil outline, watercolour fill that is slightly mis-registered
#   neon   - cyber neon: dark body, glowing outline in the brand colour
# ---------------------------------------------------------------------------------------------------
K = 4                                   # supersampling factor (192 x 144 working canvas)
OUT_SCALE = 2.5                         # delivered frame size: 120 x 90
BODYLIKE = (BODY, HEAD, SHADE, NEAR_UP, FAR_UP)


def _np():
    import numpy as np
    return np


def _noise(shape, seed, sigma):
    """Smooth random field in [-1, 1] (box-blurred white noise) used for wobble and watercolour texture."""
    np = _np()
    rng = np.random.default_rng(seed)
    im = Image.fromarray((rng.random(shape) * 255).astype("uint8"))
    from PIL import ImageFilter
    f = np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), dtype="float32")
    f = (f - f.mean()) / (f.std() + 1e-6)
    return np.clip(f / 2.5, -1, 1)


def _warp(mask, seed, amp):
    """Displace a 2-D array by a smooth random field (hand-drawn wobble)."""
    np = _np()
    h, w = mask.shape
    dx = _noise((h, w), seed, 5) * amp
    dy = _noise((h, w), seed + 1, 5) * amp
    yy, xx = np.mgrid[0:h, 0:w]
    return mask[np.clip((yy + dy).round().astype(int), 0, h - 1), np.clip((xx + dx).round().astype(int), 0, w - 1)]


def _edges(ids, thick):
    """Boundary pixels between different materials (and against transparency), `thick` px wide."""
    np = _np()
    a = ids.copy()
    e = np.zeros(a.shape, bool)
    e[1:, :] |= a[1:, :] != a[:-1, :]
    e[:-1, :] |= a[1:, :] != a[:-1, :]
    e[:, 1:] |= a[:, 1:] != a[:, :-1]
    e[:, :-1] |= a[:, 1:] != a[:, :-1]
    if thick > 1:
        from PIL import ImageFilter
        e = np.asarray(Image.fromarray((e * 255).astype("uint8")).filter(ImageFilter.MaxFilter(2 * (thick // 2) + 1))) > 0
    return e


def _mat_colours(c):
    """Colour per material id for the smooth styles (same palette logic as the pixel version)."""
    base = hx(c["base"]); legs = hx(c["legs"]) if c.get("legs") else base
    mane = c["mane"]; mane0 = hx(mane[0] if isinstance(mane, list) else mane)
    p1, p2, pat = c["silks"]; p1, p2 = hx(p1), hx(p2)
    white = (246, 244, 240)
    return {BODY: base, HEAD: base, SHADE: shade(base, c.get("shade", 0.78)), NEAR_UP: legs, NEAR_LO: white if c.get("socks") else legs,
            FAR_UP: shade(legs, 0.82), FAR_LO: shade(white if c.get("socks") else legs, 0.82), HOOF: hx(c.get("hoof", "#2b2220")),
            MANE: mane0, TAIL: mane0, EYE: (15, 15, 20), NOSE: shade(base, 0.7), FACE: white if c.get("blaze") else base,
            JK_SKIN: (240, 190, 150), JK_HELMET: p2 if pat != "solid" else shade(p1, 0.6), JK_BOOT: (40, 30, 30),
            JK_SILK: p1, JK_SILK2: p2 if pat != "solid" else shade(p1, 0.8)}


def _paint(ids, cols, mane_list=None):
    np = _np()
    h, w = ids.shape
    rgb = np.zeros((h, w, 3), "float32")
    for i, col in cols.items():
        rgb[ids == i] = col
    if mane_list:                                                  # multi-colour mane/tail: diagonal bands
        yy, xx = np.mgrid[0:h, 0:w]
        band = ((xx + yy) // (K * 2)) % len(mane_list)
        for j, colr in enumerate(mane_list):
            m = np.isin(ids, (MANE, TAIL)) & (band == j)
            rgb[m] = hx(colr)
    return rgb


def _marks_hi(c, oy):
    np = _np()
    m = Image.new("L", (W * K, H * K), 0)
    d = ImageDraw.Draw(m)
    for mk in c.get("marks") or []:
        kind = mk[0]
        if kind == "line":
            d.line([(x * K, (y + oy) * K) for x, y in mk[1]], fill=255, width=max(1, round(mk[2] * K)))
        elif kind == "ring":
            (cx, cy, r), w_ = mk[1], mk[2]
            d.ellipse([(cx - r) * K, (cy + oy - r) * K, (cx + r) * K, (cy + oy + r) * K], outline=255, width=round(w_ * K))
        elif kind == "arc":
            (cx, cy, r), a0, a1, w_ = mk[1], mk[2], mk[3], mk[4]
            d.arc([(cx - r) * K, (cy + oy - r) * K, (cx + r) * K, (cy + oy + r) * K], a0, a1, fill=255, width=round(w_ * K))
    return np.asarray(m) > 0


def _pattern_hi(ids, rgb, c, oy):
    """Flat-colour versions of the coat patterns (blocks / stripes / dots instead of single pixels)."""
    np = _np()
    h, w = ids.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pt, pcs = c.get("pattern"), c.get("pcolor", "#000000")
    body = np.isin(ids, BODYLIKE)
    pc = lambda j: hx(pcs[j % len(pcs)] if isinstance(pcs, list) else pcs)
    by = yy / K - oy
    if pt == "quad":
        for q, (xs, ys) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            m = body & ((xx / K >= 22) == bool(xs)) & ((by >= 17) == bool(ys)) & (xx / K >= 10) & (xx / K <= 34) & (ids != HEAD)
            rgb[m] = pc(q)
    elif pt == "hsplit":
        for q in range(2):
            m = body & ((by >= 17) == bool(q)) & (xx / K >= 10) & (xx / K <= 34) & (ids != HEAD)
            rgb[m] = pc(q)
    elif pt in ("hstripes", "vstripes"):
        sel = ((by % 3) < 0.9) if pt == "hstripes" else (((xx / K) % 4) < 1.2)
        rgb[body & sel & (ids != HEAD)] = pc(0)
    elif pt == "stripes":
        rgb[body & (((xx + yy) / K) % 4 < 1.4) & (ids != HEAD)] = pc(0)
    elif pt in ("spots", "dapple", "roan", "patch"):
        n = _noise((h, w), sum(map(ord, c["base"])), 3 if pt != "patch" else 9)
        thr = {"spots": 0.55, "dapple": 0.62, "roan": 0.4, "patch": 0.25}[pt]
        rgb[body & (n > thr)] = pc(int(abs(n.sum()) * 7)) if isinstance(pcs, list) else pc(0)
    return rgb


def render_hi(ids_img, c, style, oy):
    """RGBA (working resolution) from a material-id image for one of the smooth styles."""
    np = _np()
    from PIL import ImageFilter
    raw = Image.fromarray(np.array(ids_img).astype("uint8"), "L")                                   # P-mode indices, not palette greys
    ids = np.asarray(raw.filter(ImageFilter.ModeFilter(5))).astype("int16")                          # round the polygon corners
    sil = ids != T
    cols = _mat_colours(c)
    mane = c["mane"] if isinstance(c["mane"], list) else None
    rgb = _paint(ids, cols, mane)
    rgb = _pattern_hi(ids, rgb, c, oy)
    mk = _marks_hi(c, oy) & np.isin(ids, (BODY, SHADE, NEAR_UP, FAR_UP))
    mc = np.array(hx(c.get("mc", "#ffffff")), "float32")
    h, w = ids.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w, 4), "float32")

    if style == "flat":
        # low-poly facets: alternate light/dark triangles across the body regions
        cell = K * 4
        tri = ((xx // cell + yy // cell) % 2 == 0)
        diag = ((xx % cell) > (yy % cell))
        shade_f = np.where(tri ^ diag, 1.07, 0.94)[..., None]
        facet = np.isin(ids, (BODY, SHADE, NEAR_UP, FAR_UP, HEAD)) & ~mk
        rgb = np.where(facet[..., None], np.clip(rgb * shade_f, 0, 255), rgb)
        rgb[mk] = mc
        out[..., :3] = rgb; out[..., 3] = sil * 255
        # a clean white keyline so flat shapes separate from the track
        edge = _edges(sil.astype("int16"), 1) & ~sil
        edge = np.asarray(Image.fromarray((edge * 255).astype("uint8")).filter(ImageFilter.MaxFilter(K + 1))) > 0
        out[edge & ~sil] = (255, 255, 255, 235)

    elif style == "sketch":
        paper = np.array([250, 247, 238], "float32")
        wash = 0.88 + 0.12 * (_noise((h, w), 11, 6) * 0.5 + 0.5)                                # watercolour density
        fill = rgb * wash[..., None] * 0.86 + paper * 0.14 * (1 - wash[..., None] * 0.2)
        grain = _noise((h, w), 5, 1.2)[..., None] * 9
        fill = np.clip(fill + grain, 0, 255)
        fill[mk] = mc * 0.9 + paper * 0.1
        # hand-drawn fills sit a little off the pencil lines (mis-registration)
        sh = K // 2 + 1
        def shifted(a):                                           # shift right/down without wrapping around the frame
            o = np.zeros_like(a); o[sh:, sh:] = a[:-sh, :-sh]; return o
        fill_s, sil_s = shifted(fill), shifted(sil)
        sil_loose = sil | sil_s
        out[..., :3] = np.where(sil_s[..., None], fill_s, np.where(sil[..., None], fill, 0)); out[..., 3] = sil_loose * 235
        # pencil outline: two wobbly passes + region boundaries + hatching in the shadow
        e1 = _warp(_edges(ids, 1), 21, K * 0.8)
        e2 = _warp(_edges(ids, 1), 33, K * 1.0)
        inner = _warp(_edges(np.where(np.isin(ids, (NEAR_UP, NEAR_LO, JK_SILK, JK_SILK2, JK_SKIN, JK_HELMET, MANE, TAIL)), ids, 0), 1), 41, K * 0.8)
        hatch = np.isin(ids, (SHADE, FAR_UP, FAR_LO)) & (((xx + 2 * yy) // 1) % (K * 2) < 1.4)
        pencil = np.array([70, 64, 72], "float32")
        a_line = np.clip(e1 * 0.9 + e2 * 0.4 + inner * 0.35 + hatch * 0.28, 0, 1)
        out[..., :3] = out[..., :3] * (1 - a_line[..., None]) + pencil * a_line[..., None]
        out[..., 3] = np.maximum(out[..., 3], a_line * 255)

    elif style == "neon":
        # neon colour: brightest saturated colour of the brand, pushed to full brightness
        cand = [hx(c["silks"][0]), hx(c["silks"][1]), hx(c["base"]), mc.astype(int).tolist()]
        def vividness(col):
            mx, mn = max(col), min(col)
            return (mx - mn) * 0.6 + mx * 0.4
        pr = np.array(max(cand, key=vividness), "float32")
        mxv = pr.max(); pr = np.clip(pr / (mxv + 1e-6) * 255, 0, 255)
        alt = np.array(max([hx(c["silks"][1]), hx(c["silks"][0])], key=vividness), "float32")
        alt = np.clip(alt / (alt.max() + 1e-6) * 255, 0, 255)
        dark = np.array([10, 12, 28], "float32")
        out[..., :3] = dark; out[..., 3] = sil * 215
        core = _edges(sil.astype("int16"), 3) & sil
        inner = _edges(np.where(np.isin(ids, (NEAR_UP, NEAR_LO, FAR_UP, FAR_LO, JK_SILK, JK_SILK2, JK_SKIN, JK_HELMET, MANE, TAIL, FACE)), ids, 0), 2) & sil
        jock = np.isin(ids, (JK_SILK, JK_SILK2, JK_SKIN, JK_HELMET, JK_BOOT))
        line = np.zeros((h, w, 3), "float32")
        line[core & ~jock] = pr
        line[core & jock] = alt
        line[inner & ~core] = (pr * 0.7 + alt * 0.3)
        line[mk] = np.maximum(line[mk], alt * 0.5 + 128)
        line_a = (line.sum(2) > 0).astype("float32")
        glow = Image.fromarray(line.clip(0, 255).astype("uint8"))
        g1 = np.asarray(glow.filter(ImageFilter.GaussianBlur(K * 1.6)), "float32")
        g2 = np.asarray(glow.filter(ImageFilter.GaussianBlur(K * 0.7)), "float32")
        glow_rgb = np.clip(g1 * 1.9 + g2 * 1.1, 0, 255)
        base_rgb = np.clip(dark * sil[..., None] * 0.9 + glow_rgb, 0, 255)
        hot = np.clip(line * 0.55 + 140 * line_a[..., None], 0, 255) * line_a[..., None]               # white-hot core
        rgb2 = np.where(line_a[..., None] > 0, hot, base_rgb)
        ga = np.clip(glow_rgb.max(2) * 1.6, 0, 255)
        out[..., :3] = rgb2; out[..., 3] = np.maximum(out[..., 3], ga)
    return Image.fromarray(out.clip(0, 255).astype("uint8"), "RGBA")


def sheet_smooth(coat: dict, style: str) -> Image.Image:
    fw, fh = round(W * OUT_SCALE), round(H * OUT_SCALE)
    sh = Image.new("RGBA", (fw * (RUN_FRAMES + 1), fh), (0, 0, 0, 0))
    for i in range(RUN_FRAMES + 1):
        ids, oy = frame_ids(i, idle=(i == RUN_FRAMES), k=K) if i < RUN_FRAMES else frame_ids(0, idle=True, k=K)
        hi = render_hi(ids, coat, style, oy)
        sh.paste(hi.resize((fw, fh), Image.LANCZOS), (i * fw, 0))
    return sh


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
    from horse_specs import SPECS, ART

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    manifest = {"frame_w": W, "frame_h": H, "run_frames": RUN_FRAMES, "brands": {}, "generic": {}}
    for tk, (company, zh, en, why, coat) in SPECS.items():
        coat = {"mane": "#333333", **coat}
        art = ART.get(tk, "pixel")
        (sheet(coat) if art == "pixel" else sheet_smooth(coat, art)).save(OUT / f"{tk}.png", optimize=True)
        manifest["brands"][tk] = {"art": art, "frame_w": W if art == "pixel" else round(W * OUT_SCALE), "frame_h": H if art == "pixel" else round(H * OUT_SCALE), "company": company, "zh": zh, "en": en, "inspired_by": why, "features": features(coat),
                                  "silks": dict(zip(("primary", "secondary", "pattern"), coat["silks"]))}
    for cid, c in COATS.items():
        sheet(c).save(OUT / f"generic_{cid}.png")
        for art in ("flat", "sketch", "neon"):                  # the same coat in every smooth style, picked per ticker at build time
            sheet_smooth({"mane": "#333333", **c}, art).save(OUT / f"generic_{cid}_{art}.png", optimize=True)
        manifest["generic"][cid] = {"art": "pixel", "zh": c["zh"], "features": features(c)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"{len(SPECS)} brand sheets + {len(COATS)} generic coat sheets -> {OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
