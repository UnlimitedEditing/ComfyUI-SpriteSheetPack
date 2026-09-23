"""Pure numpy/PIL/scipy core for turning diffusion-rendered "pixel art" into pixel-perfect
sprite frames and a sprite sheet. No torch here so it can be tested without ComfyUI.

Conventions: images are uint8 RGBA arrays (H, W, 4). A "native" image has one array pixel per
art pixel. The render scale k is how many rendered pixels one art pixel spans.

Scale detection and grid-offset search are adapted from ComfyUI-PixelArt-Unfaker
(https://github.com/CalaKuad1/ComfyUI-PixelArt-Unfaker, MIT).
"""

import math
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import label, sobel


# --------------------------------------------------------------------------- basics

def to_rgba(rgb, alpha=None):
    """float/uint8 RGB(A) array -> uint8 RGBA. alpha: optional float array in 0..1."""
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 4:
        out = arr.copy()
    else:
        out = np.concatenate([arr[..., :3], np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    if alpha is not None:
        out[..., 3] = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return out


def binarize_alpha(rgba, threshold=0.5):
    out = rgba.copy()
    out[..., 3] = np.where(out[..., 3] >= threshold * 255.0, 255, 0).astype(np.uint8)
    return out


def remove_border_background(rgba, tolerance=24):
    """Make the background transparent: pixels similar to the dominant corner colour that are
    connected to the image border. Interior pixels of the same colour are kept."""
    out = rgba.copy()
    h, w = out.shape[:2]
    corners = [tuple(out[0, 0]), tuple(out[0, w - 1]), tuple(out[h - 1, 0]), tuple(out[h - 1, w - 1])]
    bg = Counter(corners).most_common(1)[0][0]
    if bg[3] == 0:
        return out  # already transparent around the edges
    diff = np.abs(out[..., :3].astype(np.int32) - np.array(bg[:3], np.int32))
    similar = np.all(diff <= tolerance, axis=-1) & (out[..., 3] > 0)
    labels, _ = label(similar, structure=np.ones((3, 3), np.int32))
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border = set(int(v) for v in border if v != 0)
    if border:
        out[np.isin(labels, list(border)), 3] = 0
    return out


def gray_on_white(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def on_white_rgb(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)
    return (rgb + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- grid

def edge_profiles(rgba):
    g = gray_on_white(rgba)
    px = np.abs(np.diff(g, axis=1, prepend=g[:, :1])).sum(axis=0)
    py = np.abs(np.diff(g, axis=0, prepend=g[:1, :])).sum(axis=1)
    return px, py


def _profile_peaks(profile, rel=0.15):
    """Non-max-suppressed peaks: blur spreads one edge over 2-3 px; keep one index per edge."""
    if len(profile) < 3 or profile.max() <= 0:
        return np.array([], int)
    p = profile
    floor = rel * p.max()
    i = np.arange(1, len(p) - 1)
    keep = (p[i] >= p[i - 1]) & (p[i] > p[i + 1]) & (p[i] >= floor)
    return i[keep]


def _harmonic_score(profile, k):
    """Mean spectral power of the edge profile at harmonics h/k (h < k/2), relative to the
    profile's average power. Evaluated at exact frequencies, so no bin/jitter rounding."""
    p = profile - profile.mean()
    n = np.arange(len(p))
    base = (np.abs(np.fft.rfft(p)) ** 2)[1:].mean()
    if base <= 0:
        return 0.0
    hs = np.arange(1, (k - 1) // 2 + 1)
    power = [np.abs(np.exp(-2j * np.pi * (h / k) * n) @ p) ** 2 for h in hs]
    return float(np.mean(power) / base)


def detect_scale(rgba, k_max=64, min_score=4.0):
    """Size (in rendered px) of one art pixel. 1 if the image has no pixel grid.

    The edge profile of a k-grid is an impulse train of period k (times blur and random edge
    presence), so its power concentrates at the harmonics h/k. The true k's harmonic set contains
    the strong fundamental; a divisor of k only hits every other harmonic (and not the
    fundamental); a multiple of k puts half its harmonics on noise. So the true k maximises the
    mean harmonic power. (The Unfaker's GCD-of-spacings detector collapsed on blurred renders with
    same-colour runs, and integer peak-spacing fits were thrown off by +-1 px blur jitter.)"""
    px, py = edge_profiles(rgba)
    k_max = int(min(k_max, len(px) // 4, len(py) // 4))
    if k_max < 2:
        return 1
    # candidates start at 3: k = 2's only harmonic is Nyquist, where a true native sprite's
    # single-pixel detail also lives (native input falsely read as 2x). Set sprite_width for 2x art.
    if k_max < 3:
        return 1
    scores = {k: _harmonic_score(px, k) + _harmonic_score(py, k) for k in range(3, k_max + 1)}
    best = max(scores.values())
    if best / 2.0 < min_score:
        return 1
    # on perfectly sharp art (exact nearest upscales) the spectrum is flat, so k and all its
    # divisors tie; multiples of k sit near half the best score. Largest k near the top wins.
    return max(k for k, v in scores.items() if v >= 0.9 * best)


def detect_scale_unfaker(rgba, tile_grid_size=3, min_peak_distance=4, prominence=0.1):
    """Original Unfaker GCD-of-peak-spacings detector, kept for reference."""
    g = gray_on_white(rgba)
    h, w = g.shape
    th, tw = h // tile_grid_size, w // tile_grid_size
    if th <= 1 or tw <= 1:
        tiles = [g]
    else:
        scored = []
        for i in range(tile_grid_size):
            for j in range(tile_grid_size):
                t = g[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
                if t.size:
                    scored.append((float(np.var(t)), t))
        scored.sort(key=lambda x: x[0], reverse=True)
        tiles = [t for _, t in scored[:max(1, len(scored) // 2)]]

    def peaks(profile):
        if len(profile) < 3 or profile.max() <= 0:
            return []
        floor = profile.max() * prominence
        idx = [i for i in range(1, len(profile) - 1)
               if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] >= floor]
        kept = idx[:1]
        for i in idx[1:]:
            if i - kept[-1] >= min_peak_distance:
                kept.append(i)
        return kept

    dists = []
    for t in tiles:
        if t.shape[0] < 10 or t.shape[1] < 10:
            continue
        for axis in (1, 0):
            prof = np.abs(sobel(t, axis=axis, mode="constant")).sum(axis=1 - axis)
            p = peaks(prof)
            dists += [b - a for a, b in zip(p, p[1:]) if b - a >= 2]
    if not dists:
        return 1
    counts = Counter(dists)
    common = [d for d, _ in counts.most_common(20)]
    mode = counts.most_common(1)[0][0]
    gcd = common[0]
    for d in common[1:]:
        gcd = math.gcd(gcd, d)
        if gcd == 1:
            break
    limit = max(2, min(w, h) // 4)
    if 2 <= gcd <= limit:
        return gcd
    if 2 <= mode <= limit:
        return mode
    return 1


def grid_offset(rgba, k):
    """(dx, dy) in 0..k-1 where the art-pixel grid lines fall, from edge-energy profiles."""
    if k <= 1:
        return 0, 0
    g = gray_on_white(rgba)
    # forward differences: d[x] = |g[x] - g[x-1]| is large exactly at the first pixel of a new
    # cell (a sobel peak straddles the boundary pair and is ambiguous by one pixel)
    px = np.abs(np.diff(g, axis=1, prepend=g[:, :1])).sum(axis=0)
    py = np.abs(np.diff(g, axis=0, prepend=g[:1, :])).sum(axis=1)

    def best(profile):
        scores = [profile[o::k].sum() for o in range(k)]
        return int(np.argmax(scores))

    return best(px), best(py)


# --------------------------------------------------------------------------- fractional grids
# AI "pixel art" is not an integer upscale: its art pixels are ~6.4-7.2 px wide and drift across
# the image (measured on a real Qwen-generated skull). A fixed integer grid either misses the scale
# entirely (an integer candidate k=32 = 5 x 6.4 won by harmonic coincidence) or drifts out of phase
# within a few cells. So: estimate a fractional period, then track actual grid lines.

def _period_spectrum(profile, periods):
    p = profile - profile.mean()
    n = np.arange(len(p))
    base = (np.abs(np.fft.rfft(p)) ** 2)[1:].mean()
    if base <= 0:
        return np.zeros(len(periods))
    basis = np.exp(-2j * np.pi * np.outer(1.0 / periods, n))
    return (np.abs(basis @ p) ** 2) / base


def estimate_period(rgba, p_min=2.5, p_max=64.0, min_score=6.0):
    """Fractional size (rendered px) of one art pixel, or 1.0 if there is no pixel grid.

    The edge profile of a p-grid has spectral power at 1/p and its harmonics. The strongest peak
    can be a harmonic (p/2, p/3, ...) on sharp art where every harmonic is equally strong, so the
    answer is the largest multiple m*p_peak that still carries >= 60% of the peak power -- a true
    fundamental has ~no power at 1/(2p), so multiples beyond it are rejected."""
    px, py = edge_profiles(rgba)
    hi = min(p_max, len(px) / 4.0, len(py) / 4.0)
    if hi <= p_min:
        return 1.0
    periods = np.arange(p_min, hi, 0.02)
    s = _period_spectrum(px, periods) + _period_spectrum(py, periods)
    i = int(np.argmax(s))
    if s[i] / 2.0 < min_score:
        return 1.0
    p_peak = float(periods[i])
    best = p_peak
    for m in range(2, 7):
        q = p_peak * m
        if q >= hi:
            break
        window = (periods >= q * 0.97) & (periods <= q * 1.03)
        if window.any() and s[window].max() >= 0.6 * s[i]:
            best = float(periods[window][np.argmax(s[window])])
    return best


def track_lines(profile, p, length, search=0.35, adapt=0.3):
    """Grid-line positions (cell boundaries) along one axis for a drifting grid of period ~p.

    Anchor at the strongest edge, then walk both ways: predict the next line at +-p_local, lock
    onto the strongest edge within +-search*p of the prediction (keep the prediction when there is
    none -- same-colour runs have no edge), and let p_local follow the observed spacing."""
    prof = np.asarray(profile, np.float64)
    if prof.max() <= 0 or p < 1.5:
        return np.arange(0, length + 1, max(1.0, p)).round().astype(int)
    floor = 0.08 * prof.max()

    def lock(pred):
        lo, hi = int(np.floor(pred - search * p)), int(np.ceil(pred + search * p))
        lo, hi = max(lo, 0), min(hi, length - 1)
        if hi < lo:
            return None
        seg = prof[lo:hi + 1]
        j = int(np.argmax(seg))
        return lo + j if seg[j] >= floor else None

    anchor = int(np.argmax(prof))
    lines = [float(anchor)]
    for direction in (1, -1):
        pos, p_local = float(anchor), float(p)
        while True:
            pred = pos + direction * p_local
            if pred < 0 or pred > length:
                break
            hit = lock(pred)
            new = float(hit) if hit is not None else pred
            spacing = abs(new - pos)
            if spacing < 0.5 * p:  # locked onto the same edge again; step on
                new, spacing = pred, p_local
            p_local = min(max((1 - adapt) * p_local + adapt * spacing, 0.8 * p), 1.25 * p)
            lines.append(new)
            pos = new
    lines = np.unique(np.round(np.clip(lines, 0, length)).astype(int))
    if lines[0] > 0:
        lines = np.concatenate([[0], lines]) if lines[0] >= 0.5 * p else np.concatenate([[0], lines[1:]])
    if lines[-1] < length:
        lines = np.concatenate([lines, [length]]) if length - lines[-1] >= 0.5 * p else np.concatenate([lines[:-1], [length]])
    return lines


def snap_to_lines(rgba, xs, ys, palette, coverage=0.5):
    """One palette colour (or transparency) per tracked cell, voted on the cell interior."""
    idx = palette_indices(rgba, palette)
    n = len(palette)
    out = np.zeros((len(ys) - 1, len(xs) - 1, 4), np.uint8)
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        my = (y1 - y0) // 4
        for j in range(len(xs) - 1):
            x0, x1 = xs[j], xs[j + 1]
            cell = idx[y0:y1, x0:x1]
            if cell.size == 0 or (cell >= 0).sum() < coverage * cell.size:
                continue
            mx = (x1 - x0) // 4
            inner = idx[y0 + my:y1 - my, x0 + mx:x1 - mx]
            solid = inner[inner >= 0]
            if len(solid) == 0:
                solid = cell[cell >= 0]
            out[i, j, :3] = palette[np.bincount(solid, minlength=n).argmax()]
            out[i, j, 3] = 255
    return out


def snap_tracked(rgba, p, palette):
    """Snap a frame whose art-pixel size is ~p (fractional OK, drift OK). Returns (native, info)."""
    px, py = edge_profiles(rgba)
    xs = track_lines(px, p, rgba.shape[1])
    ys = track_lines(py, p, rgba.shape[0])
    return snap_to_lines(rgba, xs, ys, palette), {"cells": [len(xs) - 1, len(ys) - 1], "period": round(float(p), 2)}


# --------------------------------------------------------------------------- palette

def build_palette(rgba, max_colors=32):
    """(N, 3) uint8 palette from the opaque pixels. max_colors <= 0 keeps every distinct colour."""
    opaque = rgba[rgba[..., 3] > 0][:, :3]
    if len(opaque) == 0:
        return np.zeros((1, 3), np.uint8)
    uniq = np.unique(opaque, axis=0)
    if max_colors <= 0 or len(uniq) <= max_colors:
        return uniq
    strip = Image.fromarray(opaque.reshape(1, -1, 3), "RGB")
    q = strip.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = np.array(q.getpalette()[:3 * max_colors], np.uint8).reshape(-1, 3)
    used = np.unique(np.array(q).ravel())
    return pal[used]


def palette_indices(rgba, palette):
    """(H, W) int map: nearest palette entry for opaque pixels, -1 for transparent."""
    rgb = rgba[..., :3].reshape(-1, 3).astype(np.float32)
    pal = palette.astype(np.float32)
    # weighted RGB distance (cheap perceptual approximation)
    wts = np.array([0.30, 0.59, 0.11], np.float32)
    idx = np.empty(len(rgb), np.int32)
    for s in range(0, len(rgb), 65536):
        chunk = rgb[s:s + 65536]
        d = (((chunk[:, None, :] - pal[None, :, :]) ** 2) * wts).sum(-1)
        idx[s:s + 65536] = d.argmin(1)
    idx = idx.reshape(rgba.shape[:2])
    idx[rgba[..., 3] < 128] = -1  # blurred fringes below 50% alpha are background
    return idx


# --------------------------------------------------------------------------- snapping

def snap_to_grid(rgba, k, palette, offset=None, coverage=0.5):
    """Rendered frame -> native frame: one palette colour (or transparency) per k x k cell.
    offset=None detects the grid offset. Returns (native RGBA, (dx, dy))."""
    k = int(max(1, k))
    dx, dy = grid_offset(rgba, k) if offset is None else offset
    idx = palette_indices(rgba, palette)
    h, w = idx.shape
    # align the grid to 0 by padding transparent pixels before the first grid line
    pad_l, pad_t = (k - dx) % k, (k - dy) % k
    pad_r = (-(w + pad_l)) % k
    pad_b = (-(h + pad_t)) % k
    idx = np.pad(idx, ((pad_t, pad_b), (pad_l, pad_r)), constant_values=-1)
    H, W = idx.shape[0] // k, idx.shape[1] // k
    blocks = idx.reshape(H, k, W, k).transpose(0, 2, 1, 3)  # (H, W, k, k)
    # colour vote on the cell interior only: blur smears each cell's 1-px border into its
    # neighbours, and at small k those in-between colours can outvote the true one
    m = k // 4
    inner = blocks[:, :, m:k - m, m:k - m].reshape(H, W, -1)
    whole = blocks.reshape(H, W, -1)
    n = len(palette)
    out = np.zeros((H, W, 4), np.uint8)
    for i in range(H):
        for j in range(W):
            if (whole[i, j] >= 0).sum() < coverage * k * k:
                continue
            solid = inner[i, j][inner[i, j] >= 0]
            if len(solid) == 0:
                solid = whole[i, j][whole[i, j] >= 0]
            out[i, j, :3] = palette[np.bincount(solid, minlength=n).argmax()]
            out[i, j, 3] = 255
    return out, (dx, dy)


def downscale_fractional(rgba, scale, palette=None):
    """Native-ise with a non-integer scale (user-chosen sprite width): dominant colour per cell
    over rounded cell bounds. Used for the input sprite only."""
    h, w = rgba.shape[:2]
    nw, nh = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
    xs = np.round(np.linspace(0, w, nw + 1)).astype(int)
    ys = np.round(np.linspace(0, h, nh + 1)).astype(int)
    out = np.zeros((nh, nw, 4), np.uint8)
    for i in range(nh):
        for j in range(nw):
            block = rgba[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].reshape(-1, 4)
            solid = block[block[:, 3] > 0]
            if len(solid) < 0.5 * len(block) or len(block) == 0:
                continue
            cols, counts = np.unique(solid[:, :3], axis=0, return_counts=True)
            out[i, j, :3] = cols[counts.argmax()]
            out[i, j, 3] = 255
    return out


def clean_halo(rgba, lum_margin=40, passes=2):
    """Remove anti-aliasing halo pixels stuck outside the outline.

    AI pixel art blends its dark outline into the (removed) background, leaving lighter fringe
    pixels that are too far from the background colour to be keyed out. They snap to single
    opaque cells outside the outline. A halo pixel touches transparency, protrudes (<= 2 opaque
    4-neighbours; a pixel on a straight edge has 3) and is clearly lighter than the darkest pixel
    it hangs off. Legit corners and tips are outline-dark, so they stay."""
    out = rgba.copy()
    lum = out[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    h, w = out.shape[:2]
    for _ in range(passes):
        op = out[..., 3] > 0
        pad = np.pad(op, 1)
        count = (pad[:-2, 1:-1].astype(np.int32) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
        removed = 0
        for y, x in np.argwhere(op & (count >= 1) & (count <= 2)):
            nl = [lum[yy, xx] for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                  if 0 <= yy < h and 0 <= xx < w and op[yy, xx]]
            if nl and lum[y, x] > min(nl) + lum_margin:
                out[y, x, 3] = 0
                removed += 1
        if not removed:
            break
    return out


def despeckle(rgba):
    """Replace single opaque pixels whose 4 neighbours all share one other colour."""
    out = rgba.copy()
    h, w = rgba.shape[:2]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if rgba[y, x, 3] == 0:
                continue
            nb = [tuple(rgba[y - 1, x]), tuple(rgba[y + 1, x]), tuple(rgba[y, x - 1]), tuple(rgba[y, x + 1])]
            if nb[0] == nb[1] == nb[2] == nb[3] and nb[0] != tuple(rgba[y, x]):
                out[y, x] = nb[0]
    return out


# --------------------------------------------------------------------------- layout

def opaque_bbox(rgba):
    ys, xs = np.nonzero(rgba[..., 3])
    if len(xs) == 0:
        return 0, 0, rgba.shape[1], rgba.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def fit_canvas(rgba, width, height):
    """Centre the opaque content on a transparent (height, width) canvas, cropping if needed."""
    x0, y0, x1, y1 = opaque_bbox(rgba)
    content = rgba[y0:y1, x0:x1]
    ch, cw = content.shape[:2]
    if cw > width:
        c = (cw - width) // 2
        content, cw = content[:, c:c + width], width
    if ch > height:
        c = (ch - height) // 2
        content, ch = content[c:c + height], height
    out = np.zeros((height, width, 4), np.uint8)
    oy, ox = (height - ch) // 2, (width - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = content
    return out


def prepare_reference(rgba, sprite_width=0, render_scale=8, max_render_side=1024, margin=0.15,
                      bg_tolerance=24, max_colors=32, max_render_pixels=400_000, min_render_scale=4,
                      halo=True):
    """Input sprite (any resolution, real or AI 'fake' pixel art) -> (native RGBA, reference RGB on
    white at the effective render scale, effective render scale, detected source period).

    The source period may be fractional and drifting (AI pixel art), so the input is snapped on
    tracked grid lines. The native canvas gets `margin` of empty space around the sprite (so rotated
    views have room) and is padded so native * scale is a multiple of 32 -- Qwen Image 2.1 rounds
    reference sizes to multiples of 32 with a lanczos resize otherwise, smearing the pixel grid.
    The render scale is lowered (not below min_render_scale) until the reference fits
    max_render_side and max_render_pixels: 7 edits must fit the job's time budget."""
    rgba = remove_border_background(rgba, bg_tolerance) if rgba[..., 3].min() == 255 else rgba
    rgba = binarize_alpha(rgba)
    if sprite_width and sprite_width > 0:
        src_period = rgba.shape[1] / float(sprite_width)
    else:
        src_period = estimate_period(rgba)
    if src_period >= 1.5:
        native, _ = snap_tracked(rgba, src_period, build_palette(rgba, max_colors))
    else:
        native = rgba
    if halo:
        native = clean_halo(native)

    x0, y0, x1, y1 = opaque_bbox(native)
    cw, ch = x1 - x0, y1 - y0
    pad = int(math.ceil(max(cw, ch) * margin))
    nw, nh = cw + 2 * pad, ch + 2 * pad
    native = fit_canvas(native, nw, nh)

    def up32(v):
        return int(math.ceil(v / 32.0) * 32)

    k = int(max(1, render_scale))
    while k > min_render_scale:
        rw, rh = up32(nw * k), up32(nh * k)
        if max(rw, rh) <= max_render_side and rw * rh <= max_render_pixels:
            break
        k -= 1
    # pad the RENDERED reference (white) to multiples of 32 rather than the native canvas: the
    # snapper tracks grid lines, so the grid need not start at 0, and every k stays usable
    ref = on_white_rgb(upscale_nearest(native, k))
    rh, rw = up32(ref.shape[0]), up32(ref.shape[1])
    top, left = (rh - ref.shape[0]) // 2, (rw - ref.shape[1]) // 2
    padded = np.full((rh, rw, 3), 255, np.uint8)
    padded[top:top + ref.shape[0], left:left + ref.shape[1]] = ref
    return native, padded, k, src_period


def build_sheet(frames, columns=8):
    """List of equal-size native RGBA frames -> sheet RGBA."""
    h, w = frames[0].shape[:2]
    cols = max(1, min(columns, len(frames)))
    rows = int(math.ceil(len(frames) / cols))
    sheet = np.zeros((rows * h, cols * w, 4), np.uint8)
    for n, f in enumerate(frames):
        r, c = divmod(n, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return sheet


def palette_swatches(palette, swatch=16):
    n = len(palette)
    img = np.zeros((swatch, max(1, n) * swatch, 4), np.uint8)
    for i, c in enumerate(palette):
        img[:, i * swatch:(i + 1) * swatch, :3] = c
        img[:, i * swatch:(i + 1) * swatch, 3] = 255
    return img


def upscale_nearest(rgba, factor):
    return np.repeat(np.repeat(rgba, factor, axis=0), factor, axis=1)
