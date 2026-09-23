"""Exercise the ComfyUI wrappers with torch tensors (no ComfyUI install needed)."""

import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))
spec = importlib.util.spec_from_file_location("sprite_pack", os.path.join(ROOT, "__init__.py"),
                                              submodule_search_locations=[ROOT])
pkg = importlib.util.module_from_spec(spec)
sys.modules["sprite_pack"] = pkg
spec.loader.exec_module(pkg)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_core as tc  # noqa: E402
import core  # noqa: E402


def to_t(rgba):
    return torch.from_numpy(rgba.astype(np.float32) / 255.0)[None]


def test_prepare_then_build():
    prep = pkg.NODE_CLASS_MAPPINGS["SpritePackPrepare"]()
    build = pkg.NODE_CLASS_MAPPINGS["SpritePackBuildSheet"]()
    native = tc.make_native(40, 34)
    src = core.upscale_nearest(native, 12)
    rgb, alpha = src[..., :3], src[..., 3]
    image = torch.from_numpy(rgb.astype(np.float32) / 255.0)[None]
    mask = torch.from_numpy(1.0 - alpha.astype(np.float32) / 255.0)[None]
    ref, nat, k, nw, nh, info = prep.prepare(image, 0, 8, 1024, 0.15, 32, 24, max_render_pixels=1_000_000, mask=mask)
    assert ref.shape[-1] == 3 and nat.shape[-1] == 4
    assert ref.shape[1] >= nh * k and ref.shape[2] >= nw * k and ref.shape[1] % 32 == 0 and ref.shape[2] % 32 == 0
    nat_np = (nat[0].numpy() * 255 + 0.5).astype(np.uint8)
    # fake "renders": RGBA like the Qwen 2.1 VAE, on white, shifted grids, blur + noise
    frames = {}
    for i, (dx, dy) in enumerate(((0, 0), (3, 5), (7, 2)), start=1):
        r = tc.fake_render(nat_np, k, dx, dy)[: ref.shape[1], : ref.shape[2]]
        white = core.on_white_rgb(r)
        frames[f"frame_{i}"] = to_t(np.dstack([white, np.full(white.shape[:2], 255, np.uint8)]))
    sheet, preview, fr, pal, info = build.build(nat, k, 32, 8, 4, 0.5, 24, False, **frames)
    assert sheet.shape[1:] == (nh, nw * 4, 4), sheet.shape
    assert preview.shape[1:] == (nh * 4, nw * 16, 4)
    assert fr.shape == (4, nh, nw, 4)
    for i in range(1, 4):
        got = (fr[i].numpy() * 255 + 0.5).astype(np.uint8)
        rate = tc.match_rate(got, (fr[0].numpy() * 255 + 0.5).astype(np.uint8))
        assert rate > 0.95, (i, rate)
    print(info)


def test_order_offset():
    build = pkg.NODE_CLASS_MAPPINGS["SpritePackBuildSheet"]()
    # 8 flat-colour "views"; frame j carries colour index j so the order is readable
    cols = np.array([[i * 30, 255 - i * 30, (i * 70) % 255] for i in range(8)], np.uint8)
    native = np.zeros((6, 6, 4), np.uint8); native[1:5, 1:5, :3] = cols[0]; native[1:5, 1:5, 3] = 255
    frames = {}
    for j in range(1, 8):
        f = np.full((48, 48, 4), 255, np.uint8)  # white bg, 8x render of a 6x6 native
        f[8:40, 8:40, :3] = cols[j]
        frames[f"frame_{j}"] = to_t(f)
    # palette comes from the native sprite, so give the colours used below an opaque pixel there
    for j in range(6):
        native[5, j, :3] = cols[j]
        native[5, j, 3] = 255
    nat_t = to_t(native)
    for k in (0, 1, 2, 7):
        _, _, fr, _, info = build.build(nat_t, 8, 0, 8, 1, 0.5, 24, False, False, k, **frames)
        # frame 0 (the native sprite) must land at output index k
        centre = (fr[:, 3, 3, :3].numpy() * 255 + 0.5).astype(int)
        assert (centre[k] == cols[0]).all(), (k, centre[k])
        assert (centre[(k + 1) % 8] == cols[1]).all()


def test_front_view_override():
    build = pkg.NODE_CLASS_MAPPINGS["SpritePackBuildSheet"]()
    cols = np.array([[i * 30, 255 - i * 30, (i * 70) % 255] for i in range(9)], np.uint8)
    native = np.zeros((6, 6, 4), np.uint8); native[1:5, 1:5, :3] = cols[0]; native[1:5, 1:5, 3] = 255
    for j in range(6):
        native[5, j, :3] = cols[j]; native[5, j, 3] = 255
    native[0, 0, :3] = cols[8]; native[0, 0, 3] = 255  # palette entry for the front render
    def flat(c):
        f = np.full((48, 48, 4), 255, np.uint8); f[8:40, 8:40, :3] = c
        return to_t(f)
    frames = {f"frame_{j}": flat(cols[j]) for j in range(1, 8)}
    front = flat(cols[8])
    centre = lambda fr: (fr[:, 3, 3, :3].numpy() * 255 + 0.5).astype(int)
    _, _, fr, _, _ = build.build(to_t(native), 8, 0, 8, 1, 0.5, 24, False, False, 1, front_view=front, **frames)
    assert (centre(fr)[0] == cols[8]).all()          # position 0 = dedicated front
    assert (centre(fr)[1] == cols[0]).all()          # input still at its own slot
    _, _, fr, _, _ = build.build(to_t(native), 8, 0, 8, 1, 0.5, 24, False, False, 0, front_view=front, **frames)
    assert (centre(fr)[0] == cols[0]).all()          # offset 0: the input IS the front, override ignored


if __name__ == "__main__":
    test_prepare_then_build()
    print("ok test_prepare_then_build")
    test_order_offset()
    print("ok test_order_offset")
    test_front_view_override()
    print("ok test_front_view_override")
