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
    ref, nat, k, nw, nh, info = prep.prepare(image, 0, 8, 1024, 0.15, 32, 24, mask=mask)
    assert ref.shape[-1] == 3 and nat.shape[-1] == 4
    assert ref.shape[1] == nh * k and ref.shape[2] == nw * k and ref.shape[1] % 32 == 0
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


if __name__ == "__main__":
    test_prepare_then_build()
    print("ok test_prepare_then_build")
