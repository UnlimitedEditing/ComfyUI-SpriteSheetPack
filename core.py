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


def _loop_thumb(rgba, bg_tolerance, win_h, win_w, size=40):
    """Foreground of one video frame as a small fixed-window RGB thumbnail aligned on the figure
    (x = mask centroid, y = bottom of the figure), background zero. For comparing walk poses."""
    keyed = remove_border_background(rgba, bg_tolerance)
    mask = keyed[..., 3] > 0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(size * size * 3, np.float32), (0, 0)
    cx, bottom = int(round(xs.mean())), int(ys.max()) + 1
    x0, y0 = cx - win_w // 2, bottom - win_h
    canvas = np.zeros((win_h, win_w, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(rgba.shape[1], x0 + win_w), min(rgba.shape[0], bottom)
    sub = keyed[sy0:sy1, sx0:sx1]
    region = sub[..., :3] * (sub[..., 3:4] > 0)
    canvas[sy0 - y0:sy0 - y0 + region.shape[0], sx0 - x0:sx0 - x0 + region.shape[1]] = region
    thumb = np.asarray(Image.fromarray(canvas).resize((size, size), Image.BILINEAR), np.float32) / 255.0
    return thumb.reshape(-1), (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def find_loop(frames, count=8, min_period=16, max_period=64, window=4, period=0, bg_tolerance=24,
              target_height=64, skip=3):
    """One seamless walk cycle in a video (list of RGB/RGBA uint8 frames, the character on a plain
    background) -> (frame indices, report dict).

    Walking in place repeats every full stride, but the *silhouette* repeats every step (left and
    right legs swap), so frames are compared in colour, and matched as a `window`-frame run: one
    pose occurs twice per stride (swinging forward and back), so a single-frame match finds loops
    that jump backwards at the seam. Period = the smallest local error minimum within 25% of the
    best. The loop starts at the best seam, then rotates to the widest stride so the sheet opens on
    a contact pose. `count` indices are spread evenly over the cycle."""
    rgbas = [to_rgba(f) for f in frames]
    boxes = []
    for r in rgbas:
        keyed = remove_border_background(r, bg_tolerance)
        ys, xs = np.nonzero(keyed[..., 3] > 0)
        boxes.append((int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) if len(xs) else (0, 0))
    bw = np.array([b[0] for b in boxes])
    bh = np.array([b[1] for b in boxes])
    win_h = max(8, int(np.median(bh) * 1.1))
    win_w = max(8, int(np.percentile(bw, 90) * 1.15))
    thumbs, spans = zip(*[_loop_thumb(r, bg_tolerance, win_h, win_w) for r in rgbas])
    v = np.stack(thumbs)
    t_total = len(v)
    window = max(1, min(window, t_total // 4))
    hi = min(int(max_period), t_total - window - skip - 1)
    lo = max(2, int(min_period))
    errs = {}
    if period and period > 0:
        p = int(period)
    else:
        if hi < lo:
            raise ValueError(f"video too short for a loop: {t_total} frames (min_period {lo}, "
                             f"needs more than {lo + window + skip + 1})")
        for q in range(lo, hi + 1):
            n_t = t_total - q - window + 1
            e = np.mean([np.abs(v[k:k + n_t] - v[q + k:q + k + n_t]).mean() for k in range(window)])
            errs[q] = float(e)
        best = min(errs.values())
        cands = [q for q in errs if errs[q] <= best * 1.25 + 1e-9
                 and errs[q] <= errs.get(q - 1, 9e9) and errs[q] <= errs.get(q + 1, 9e9)]
        p = min(cands)
    last = t_total - p - window
    if last < skip:
        raise ValueError(f"period {p} leaves no room for a loop in {t_total} frames")
    # best seam: the start whose window matches the same window one period later
    seam = [np.mean([np.abs(v[s + k] - v[s + p + k]).mean() for k in range(window)])
            for s in range(skip, last + 1)]
    s = skip + int(np.argmin(seam))
    # rotate to the widest stride so frame 0 is a contact pose
    widest = s + int(np.argmax([spans[s + j][0] for j in range(p)]))
    idx = [s + ((widest - s + int(round(i * p / float(count)))) % p) for i in range(int(count))]
    height = float(np.median(bh[bh > 0])) if (bh > 0).any() else float(target_height)
    scale = max(1, int(round(height / max(1, target_height))))
    pixel_scale = max(1.0, height / float(max(1, target_height)))
    report = {"frames_in": t_total, "period": p, "start": s, "widest": widest, "indices": idx,
              "seam_error": round(float(min(seam)), 4),
              "best_period_error": round(min(errs.values()), 4) if errs else None,
              "figure_px": [int(np.median(bw)), int(height)], "render_scale": scale,
              "pixel_scale": round(pixel_scale, 3)}
    return idx, report


def uniform_snap(rgba, scale, palette, alpha_threshold=0.5):
    """Video frame (RGBA, background already keyed) -> native-pixel RGBA at ONE fixed scale: the whole
    canvas is area-averaged down by `scale` (premultiplied, so edges stay clean), alpha is cut at
    `alpha_threshold`, and colours snap to the nearest palette entry. Unlike snap_tracked this never
    fits a grid to the individual frame, so every frame of a cycle has the same size and proportions
    (tracked snapping rescales each frame's axes on its own, which stretches and squashes a walk)."""
    h, w = rgba.shape[:2]
    nw, nh = max(1, int(round(w / float(scale)))), max(1, int(round(h / float(scale))))
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    pm = np.concatenate([rgba[..., :3].astype(np.float32) * a, a * 255.0], axis=-1)
    chans = [np.asarray(Image.fromarray(pm[..., c], mode="F").resize((nw, nh), Image.BOX), np.float32)
             for c in range(4)]
    alpha = chans[3] / 255.0
    rgb = np.stack(chans[:3], axis=-1) / np.maximum(alpha[..., None], 1e-6)
    out = np.zeros((nh, nw, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
    idx = palette_indices(out, palette)
    out[idx < 0, 3] = 0
    out[idx >= 0, :3] = palette[idx[idx >= 0]]
    return out


def align_on_feet(frames, pad=1):
    """Animation frames (RGBA, any sizes) -> equal-size cells on one shared pivot: ground line = the
    bottom of each figure, x = the torso (mean x of the middle rows, steadier than the feet or limbs
    while a figure walks). The cell grows to fit the largest frame, so nothing is cropped."""
    crops = []
    for f in frames:
        x0, y0, x1, y1 = opaque_bbox(f)
        crops.append(f[y0:y1, x0:x1])
    pivots = []
    for c in crops:
        h = c.shape[0]
        band = c[int(h * 0.25):max(int(h * 0.55), int(h * 0.25) + 1), :, 3]
        ys, xs = np.nonzero(band)
        pivots.append((float(xs.mean()) + 0.5 if len(xs) else c.shape[1] / 2.0, float(h)))
    left = max(p[0] for p in pivots)
    right = max(c.shape[1] - p[0] for c, p in zip(crops, pivots))
    up = max(p[1] for p in pivots)
    cw, ch = int(math.ceil(left + right)) + 2 * pad, int(math.ceil(up)) + 2 * pad
    ox, oy = int(round(pad + left)), ch - pad
    out = []
    for c, (fx, fy) in zip(crops, pivots):
        cell = np.zeros((ch, cw, 4), np.uint8)
        dx, dy = ox - int(round(fx)), oy - c.shape[0]
        cell[dy:dy + c.shape[0], dx:dx + c.shape[1]] = c[:ch - dy, :cw - dx]
        out.append(cell)
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


# --------------------------------------------------------------------------- facing

EXPECTED_FACING_8 = {1: "left", 2: "left", 3: "left", 5: "right", 6: "right", 7: "right"}


def parse_facing_report(text):
    """{"0": "left", "3": "right", ...} from a vision model's answer (fences / prose tolerated).
    Values normalised to "left" / "right" / None."""
    import json as _json
    import re as _re
    if not text:
        return {}
    obj = None
    for m in _re.finditer(r"\{", text):
        try:
            obj, _ = _json.JSONDecoder().raw_decode(text, m.start())
            break
        except ValueError:
            continue
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        try:
            idx = int(str(k).strip())
        except ValueError:
            continue
        v = str(v).lower()
        out[idx] = "left" if "left" in v and "right" not in v else "right" if "right" in v and "left" not in v else None
    return out


def frames_to_flip(report, offset, n=8):
    """Generation-order indices j (1..n-1) whose reported facing contradicts the facing expected at
    orbit position (offset + j) % n. Index 0 is the input; if the model reads the input's facing the
    opposite way to what `offset` says, its whole notion of "front" is inverted, so all answers are
    swapped before comparing. Front/back positions (no expected side) are never flipped."""
    if n != 8 or not report:
        return []
    swap = {"left": "right", "right": "left"}
    expected_input = EXPECTED_FACING_8.get(offset % n)
    got_input = report.get(0)
    invert = bool(expected_input and got_input and got_input != expected_input)
    flips = []
    for j in range(1, n):
        want = EXPECTED_FACING_8.get((offset + j) % n)
        got = report.get(j)
        if invert and got:
            got = swap[got]
        if want and got and got != want:
            flips.append(j)
    return flips


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


def save_gif(frames, path, fps=6.0, scale=6, transparent=False, loop=0):
    """Equal-size native RGBA frames -> looping animated GIF, pixel-exact.

    Pillow only (no ffmpeg/imageio: installing VideoHelperSuite's pip deps downgraded numpy on
    Graydient's image and broke ComfyUI's startup). All frames share one palette with index 0
    reserved for the background (white, or GIF-transparent), so pixel-art colours are written
    exactly -- no per-frame re-quantization, no dithering."""
    frames = [np.asarray(f) for f in frames]
    opaque = np.concatenate([f[f[..., 3] > 0][:, :3] for f in frames] or [np.zeros((0, 3), np.uint8)])
    colours = np.unique(opaque, axis=0) if len(opaque) else np.zeros((0, 3), np.uint8)
    if len(colours) > 255:  # more than a GIF palette holds: reduce to 255 shared colours
        colours = build_palette(np.dstack([opaque[None], np.full((1, len(opaque), 1), 255, np.uint8)]), 255)
    palette = np.vstack([[255, 255, 255], colours]).astype(np.uint8)
    flat = palette.ravel().tolist() + [0] * (768 - palette.size)

    images = []
    for f in frames:
        idx = palette_indices(f, colours) + 1  # transparent (-1) -> background index 0
        idx = upscale_nearest(idx.astype(np.uint8)[..., None], max(1, int(scale)))[..., 0]
        im = Image.fromarray(idx, "P")
        im.putpalette(flat)
        images.append(im)
    kwargs = dict(save_all=True, append_images=images[1:], duration=int(round(1000.0 / max(0.1, fps))),
                  loop=int(loop), disposal=2, optimize=False, background=0)
    if transparent:
        kwargs["transparency"] = 0
    images[0].save(path, format="GIF", **kwargs)
    return path


# --------------------------------------------------------------------------- sheet standardiser
# Read any sprite sheet (ours or an external one) into one canonical layout: equal-size cells,
# one row per state, every frame anchored on the same pivot. No diffusion, no model weights.

def _runs(flags, min_gap):
    """[start, end) runs of True in a 1-D bool array; runs closer than min_gap are merged."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(flags)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def _merge_specks(segs, weights, frac=0.1):
    """Fold a segment into its nearest neighbour when it holds < frac of the median pixel count
    (a stray spark or a detached weapon tip must not become a frame of its own)."""
    if len(segs) < 3:
        return segs, weights
    med = float(np.median(weights))
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, wt in enumerate(weights):
            if wt < frac * med:
                gl = segs[i][0] - segs[i - 1][1] if i > 0 else 1 << 30
                gr = segs[i + 1][0] - segs[i][1] if i < len(segs) - 1 else 1 << 30
                j = i - 1 if gl <= gr else i + 1
                lo, hi = min(i, j), max(i, j)
                segs[lo] = [segs[lo][0], segs[hi][1]]
                weights[lo] = weights[lo] + weights[hi]
                del segs[hi], weights[hi]
                changed = True
                break
    return segs, weights


def detect_sheet_cells(alpha, columns=0, rows=0, min_gap=0, min_pixels=6):
    """Find the frames of a sprite sheet from its opacity mask (H, W bool).

    columns and rows both > 0: an exact uniform grid. Otherwise rows are the horizontal bands of
    opaque pixels, and each band is cut into frames at its empty vertical gaps (wider than min_gap;
    0 = ~1% of the sheet). Returns (rows, notes): rows = [[(x0, y0, x1, y1), ...], ...]."""
    h, w = alpha.shape
    notes = []
    if columns > 0 and rows > 0:
        xs = [round(i * w / columns) for i in range(columns + 1)]
        ys = [round(j * h / rows) for j in range(rows + 1)]
        out = []
        for j in range(rows):
            cells = [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(columns)]
            while cells and alpha[cells[-1][1]:cells[-1][3], cells[-1][0]:cells[-1][2]].sum() < min_pixels:
                cells.pop()  # empty cells at the end of a row are padding, not frames
            out.append(cells)
        return out, notes
    gap = min_gap if min_gap > 0 else max(2, int(round(0.01 * max(w, h))))
    out = []
    for y0, y1 in _runs(alpha.any(axis=1), gap):
        band = alpha[y0:y1]
        segs = _runs(band.any(axis=0), gap)
        weights = [float(band[:, a:b].sum()) for a, b in segs]
        segs, weights = _merge_specks(segs, weights)
        segs = [s for s, wt in zip(segs, weights) if wt >= min_pixels]
        if not segs:
            continue
        if len(segs) == 1 and (segs[0][1] - segs[0][0]) >= 1.8 * (y1 - y0):
            # one wide blob: frames that touch. Guess square-ish cells and say so.
            n = int(round((segs[0][1] - segs[0][0]) / (y1 - y0)))
            x0, x1 = segs[0]
            segs = [[round(x0 + i * (x1 - x0) / n), round(x0 + (i + 1) * (x1 - x0) / n)] for i in range(n)]
            notes.append(f"row at y={y0}: frames touch; guessed {n} equal cells (set columns and rows to override)")
        out.append([(a, y0, b, y1) for a, b in segs])
    return out, notes


def _foot_pivot(content):
    """(x, y) of the feet in content coordinates: bottom edge, x = mean of the lowest opaque pixels."""
    ch = content.shape[0]
    band = max(1, int(round(ch * 0.12)))
    ys, xs = np.nonzero(content[ch - band:, :, 3])
    if len(xs) == 0:
        return content.shape[1] / 2.0, float(ch)
    return float(xs.mean()) + 0.5, float(ch)


def standardize_sheet(rgba, columns=0, rows=0, min_gap=0, anchor="feet", cell_width=0, cell_height=0,
                      pad=1, unscale=1, max_colors=0, labels=None, fps=8.0, bg_tolerance=24,
                      alpha_threshold=0.5, min_pixels=6):
    """Any sprite sheet (RGBA, any background) -> canonical sheet.

    Returns (sheet RGBA, frames [RGBA, ...] in grid order, meta dict). Cells are equal size, one row
    per detected row (a state), frames anchored on one pivot: anchor 'feet' = common ground line and
    foot x, 'center' = bounding-box centre. unscale N > 1 divides an upscaled sheet back to native
    pixels, 0 detects the factor. max_colors > 0 locks the sheet to one shared palette."""
    warnings = []
    rgba = binarize_alpha(rgba, alpha_threshold)
    if rgba[..., 3].min() == 255:  # no transparency at all: key out the border colour
        rgba = binarize_alpha(remove_border_background(rgba, bg_tolerance), alpha_threshold)
    layout, notes = detect_sheet_cells(rgba[..., 3] > 0, columns, rows, min_gap, min_pixels)
    warnings += notes
    if not layout:
        raise ValueError("no sprites found in the sheet (everything is transparent or background)")

    period = 1.0
    palette = build_palette(rgba, max_colors) if (max_colors > 0 or unscale != 1) else None
    if unscale == 0:
        period = estimate_period(rgba)  # whole sheet: more edges than any one frame
        if period < 1.5:
            period = 1.0
            warnings.append("unscale auto: no pixel grid found, kept the sheet as is")
    elif unscale > 1:
        period = float(unscale)

    contents = []  # per row: each frame cropped tight to its opaque bbox
    for row in layout:
        items = []
        for x0, y0, x1, y1 in row:
            region = rgba[y0:y1, x0:x1].copy()
            if period > 1.0:
                region, _ = snap_tracked(region, period, palette)
            elif max_colors > 0:
                idx = palette_indices(region, palette)
                region[idx < 0, 3] = 0
                region[idx >= 0, :3] = palette[idx[idx >= 0]]
            bx0, by0, bx1, by1 = opaque_bbox(region)
            items.append(region[by0:by1, bx0:bx1])
        contents.append(items)

    pivots = [[(c.shape[1] / 2.0, c.shape[0] / 2.0) if anchor == "center" else _foot_pivot(c) for c in items]
              for items in contents]
    flat = [(c, p) for items, pv in zip(contents, pivots) for c, p in zip(items, pv)]
    left = max(p[0] for c, p in flat)
    right = max(c.shape[1] - p[0] for c, p in flat)
    up = max(p[1] for c, p in flat)
    down = max(c.shape[0] - p[1] for c, p in flat)
    if cell_width > 0 and cell_height > 0:
        cw, ch = int(cell_width), int(cell_height)
        px = cw / 2.0
        py = float(ch - pad) if anchor != "center" else ch / 2.0
        if left + pad > px or right + pad > cw - px or up + pad > py or down + pad > ch - py:
            warnings.append(f"some frames are larger than the {cw}x{ch} cell and were cropped")
    else:
        cw = int(math.ceil(left + right)) + 2 * pad
        ch = int(math.ceil(up + down)) + 2 * pad
        px, py = pad + left, pad + up
    ox, oy = int(round(px)), int(round(py))

    cols_n = max(len(items) for items in contents)
    sheet = np.zeros((len(contents) * ch, cols_n * cw, 4), np.uint8)
    frames = []
    for r, (items, pv) in enumerate(zip(contents, pivots)):
        for c, (content, (fx, fy)) in enumerate(zip(items, pv)):
            cell = np.zeros((ch, cw, 4), np.uint8)
            dx, dy = ox - int(round(fx)), oy - int(round(fy))
            sx0, sy0 = max(0, -dx), max(0, -dy)
            tx0, ty0 = max(0, dx), max(0, dy)
            wcopy = min(content.shape[1] - sx0, cw - tx0)
            hcopy = min(content.shape[0] - sy0, ch - ty0)
            if wcopy > 0 and hcopy > 0:
                cell[ty0:ty0 + hcopy, tx0:tx0 + wcopy] = content[sy0:sy0 + hcopy, sx0:sx0 + wcopy]
            sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
            frames.append(cell)

    names = [s.strip() for s in (labels or "").split(",") if s.strip()]
    states = []
    for r, items in enumerate(contents):
        states.append({"name": names[r] if r < len(names) else f"state{r + 1}", "row": r,
                       "start": r * cols_n, "frames": len(items)})
    if names and len(names) != len(contents):
        warnings.append(f"{len(names)} labels given but {len(contents)} rows found")
    meta = {"format": "spritepack.sheet", "version": 1, "cell": [cw, ch], "columns": cols_n,
            "rows": len(contents), "pivot": [ox, oy], "anchor": anchor, "fps": fps,
            "frames": len(frames), "states": states, "sheet": [sheet.shape[1], sheet.shape[0]],
            "source_unscale": round(period, 2), "palette_size": 0 if palette is None else len(palette),
            "warnings": warnings}
    return sheet, frames, meta


def text_card(text, width=1024, margin=12):
    """Monospace text on white as an RGB uint8 image (a readable way to return JSON from hosts that
    only collect images)."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    lines = []
    for raw in text.splitlines() or [""]:
        line = raw
        while len(line) > 120:
            lines.append(line[:120])
            line = "  " + line[120:]
        lines.append(line)
    height = max(64, 2 * margin + 12 * len(lines))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((margin, margin + 12 * i), line, fill="black", font=font)
    return np.asarray(img)
