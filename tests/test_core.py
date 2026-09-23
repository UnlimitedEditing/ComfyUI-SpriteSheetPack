"""Synthetic round-trip tests: build a true native sprite, fake an AI render of it (upscale by k,
shift the grid, blur edges, add noise + partial alpha), and check the core recovers it."""

import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402

rng = np.random.default_rng(0)
PALETTE = np.array([[120, 70, 30], [200, 150, 80], [240, 230, 200], [60, 40, 90], [30, 20, 20]], np.uint8)


def make_native(w=40, h=32):
    """Random blobby sprite on transparency using a 5-colour palette."""
    out = np.zeros((h, w, 4), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    body = ((xx - w / 2) / (w * 0.35)) ** 2 + ((yy - h / 2) / (h * 0.38)) ** 2 < 1
    # real pixel art mixes runs with single-pixel detail: 2x2 runs + 30% per-pixel variation
    idx = rng.integers(0, len(PALETTE), size=(h // 2 + 1, w // 2 + 1)).repeat(2, 0).repeat(2, 1)[:h, :w]
    single = rng.random((h, w)) < 0.3
    idx[single] = rng.integers(0, len(PALETTE), size=single.sum())
    out[body, :3] = PALETTE[idx[body]]
    out[body, 3] = 255
    # 1-px dark outline, like most sprites
    edge = body & ~(np.roll(body, 1, 0) & np.roll(body, -1, 0) & np.roll(body, 1, 1) & np.roll(body, -1, 1))
    out[edge, :3] = PALETTE[4]
    return out


def fake_render(native, k, dx, dy, blur=0.8, noise=6.0):
    up = core.upscale_nearest(native, k)
    h, w = up.shape[:2]
    canvas = np.zeros((h + k, w + k, 4), np.uint8)
    canvas[dy:dy + h, dx:dx + w] = up
    rgb_white = core.on_white_rgb(canvas)
    img = Image.fromarray(rgb_white).filter(ImageFilter.GaussianBlur(blur))
    rgb = np.asarray(img).astype(np.float32) + rng.normal(0, noise, (h + k, w + k, 3))
    alpha = canvas[..., 3].astype(np.float32)
    alpha = np.asarray(Image.fromarray(alpha.astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))).astype(np.float32)
    return np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)


def match_rate(a, b, tol=0):
    """Fraction of cells where two native frames agree (after aligning opaque bboxes)."""
    fa = core.fit_canvas(a, max(a.shape[1], b.shape[1]) + 4, max(a.shape[0], b.shape[0]) + 4)
    fb = core.fit_canvas(b, fa.shape[1], fa.shape[0])
    both = (fa[..., 3] > 0) | (fb[..., 3] > 0)
    same = np.all(np.abs(fa.astype(int) - fb.astype(int)) <= tol, axis=-1) & both
    return same.sum() / max(1, both.sum())


def test_detect_scale():
    native = make_native()
    for k in (6, 8, 12):
        r = fake_render(native, k, 3, 5)
        s = core.detect_scale(r)
        assert s == k, (k, s)


def test_grid_offset_and_snap():
    native = make_native()
    for k, dx, dy in ((8, 0, 0), (8, 3, 5), (6, 5, 1), (10, 7, 2)):
        r = fake_render(native, k, dx, dy)
        off = core.grid_offset(r, k)
        assert off == (dx, dy), (k, (dx, dy), off)
        snapped, _ = core.snap_to_grid(r, k, PALETTE)
        rate = match_rate(snapped, native)
        assert rate > 0.97, (k, dx, dy, rate)


def test_prepare_reference_fake_input():
    native = make_native(44, 36)
    src = fake_render(native, 12, 4, 9)  # "AI pixel art" input at 12x, on white, opaque bg
    src[..., 3] = 255
    nat, ref, k, src_scale = core.prepare_reference(src, render_scale=8)
    assert src_scale == 12, src_scale
    assert (ref.shape[0] % 32, ref.shape[1] % 32) == (0, 0), ref.shape
    assert ref.shape[0] == nat.shape[0] * k and ref.shape[1] == nat.shape[1] * k
    # noisy input -> median-cut palette colours are close to, not equal to, the originals
    assert match_rate(nat, native, tol=16) > 0.95, match_rate(nat, native, tol=16)


def test_prepare_reference_explicit_width_and_cap():
    native = make_native(64, 64)
    src = core.upscale_nearest(native, 16)  # 1024 px wide, exact
    nat, ref, k, _ = core.prepare_reference(src, sprite_width=64, render_scale=16, max_render_side=1024)
    assert max(ref.shape[:2]) <= 1024 and k < 16, (ref.shape, k)
    assert match_rate(nat, native) > 0.99


def test_end_to_end_sheet():
    native = make_native()
    nat, ref, k, _ = core.prepare_reference(core.upscale_nearest(native, 8), render_scale=8)
    pal = core.build_palette(nat, 32)
    frames = [nat]
    for dx, dy in ((0, 0), (2, 6), (7, 3)):
        r = fake_render(nat, k, dx, dy)[: ref.shape[0], : ref.shape[1]]
        r = core.remove_border_background(r)
        f, _ = core.snap_to_grid(core.binarize_alpha(r), k, pal)
        frames.append(core.fit_canvas(f, nat.shape[1], nat.shape[0]))
        assert match_rate(frames[-1], nat) > 0.95, match_rate(frames[-1], nat)
    sheet = core.build_sheet(frames, columns=8)
    assert sheet.shape == (nat.shape[0], nat.shape[1] * 4, 4)
    colours = np.unique(sheet[sheet[..., 3] > 0][:, :3], axis=0)
    assert all(any((c == p).all() for p in pal) for c in colours)  # palette-locked


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
