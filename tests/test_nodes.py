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


def test_standardize_node():
    import test_standardize as ts
    node = pkg.NODE_CLASS_MAPPINGS["SpritePackStandardizeSheet"]()
    sheet = ts.make_sheet(bg=(255, 0, 255))
    rgb = torch.from_numpy(sheet[..., :3].astype(np.float32) / 255.0)[None]
    out, prev, frames, card, info = node.standardize(rgb, 0, 0, 0, "feet", 0, 0, 1, 1, 0, "idle,attack,hit", 8.0, 4, 0.5, 24)
    import json
    meta = json.loads(info)
    assert meta["frames"] == 9 and frames.shape[0] == 9 and out.shape[-1] == 4 and card.shape[-1] == 3
    assert prev.shape[1] == out.shape[1] * 4


def test_pose_cycle_matches_reference_and_loops():
    ref = np.full((512, 384, 3), 255, np.uint8)
    ref[100:460, 140:250] = (90, 60, 40)  # a "character" block
    node = pkg.NODE_CLASS_MAPPINGS["SpritePackPoseCycle"]()
    t = torch.from_numpy(ref.astype(np.float32) / 255.0)[None]
    poses, n, info = node.draw(t, "walk", 8, "right", 24)
    assert poses.shape == (8, 512, 384, 3) and n == 8
    arr = (poses.numpy() * 255).astype(np.uint8)
    assert all(a.any() for a in arr)                      # every frame draws something
    assert len({a.tobytes() for a in arr}) == 8           # and the cycle actually moves
    rows = np.where(arr[0].any(-1).any(1))[0]
    assert rows.min() >= 90 and rows.max() <= 470         # skeleton stays in the figure's height band
    _, _, _ = node.draw(t, "run", 6, "left", 24)
    _, _, _ = node.draw(t, "idle", 4, "right", 24)


def test_build_sheet_skip_native():
    build = pkg.NODE_CLASS_MAPPINGS["SpritePackBuildSheet"]()
    native = tc.make_native(40, 34)
    frame = to_t(core.upscale_nearest(native, 8))[..., :3]
    sheet, prev, frames, pal, info = build.build(to_t(native), 8, 32, 4, 2, 0.5, 24, False,
                                                 skip_native=True, frame_1=frame, frame_2=frame)
    assert frames.shape[0] == 2


def test_skip_native_aligns_frames_on_feet_without_cropping():
    build = pkg.NODE_CLASS_MAPPINGS["SpritePackBuildSheet"]()
    native = tc.make_native(40, 34)
    big = core.upscale_nearest(tc.make_native(44, 48), 8)           # frames taller than the native cell
    sheet, prev, frames, pal, info = build.build(to_t(native), 8, 32, 2, 2, 0.5, 24, False, skip_native=True,
                                                 frame_1=to_t(big)[..., :3], frame_2=to_t(big)[..., :3])
    f = frames.numpy()
    x0, y0, x1, y1 = core.opaque_bbox(core.upscale_nearest(tc.make_native(44, 48), 1))
    assert f.shape[0] == 2 and f.shape[1] >= (y1 - y0) and f.shape[2] >= (x1 - x0)  # cell grew to the content
    assert f.shape[1] > 30                                           # bigger than the native cell (~26 rows)
    bottoms = [np.nonzero(fr[..., 3].any(1))[0].max() for fr in f]   # shared ground line
    assert len(set(bottoms)) == 1


def test_pose_scale_leaves_headroom():
    import poses
    full = poses.render_cycle("walk", 2, (256, 256), (50, 20, 200, 240), scale=1.0)[0]
    small = poses.render_cycle("walk", 2, (256, 256), (50, 20, 200, 240), scale=0.7)[0]
    top = lambda a: np.nonzero(a.any(-1).any(1))[0].min()
    assert top(small) > top(full)


def _walker_frames(total=96, period=40, size=(176, 208), drift=0.25):
    """Synthetic walk-in-place video: grey torso, two differently coloured legs swinging out of
    phase (so the SILHOUETTE repeats every half period but the colours only every full one), on
    white, drifting sideways like a real generated video."""
    import math
    from PIL import Image, ImageDraw
    out = []
    for t in range(total):
        im = Image.new("RGB", size, (255, 255, 255))
        d = ImageDraw.Draw(im)
        cx, hip_y = 80 + drift * t, 110
        d.rectangle([cx - 14, 50, cx + 14, hip_y], fill=(90, 120, 90))
        d.ellipse([cx - 12, 22, cx + 12, 50], fill=(210, 170, 130))
        for phase, color in ((0.0, (200, 40, 40)), (0.5, (40, 60, 200))):
            ang = 0.55 * math.sin(2 * math.pi * (t / period + phase))
            fx, fy = cx + 70 * math.sin(ang), hip_y + 70 * math.cos(ang)
            d.line([cx, hip_y, fx, fy], fill=color, width=9)
        out.append(np.asarray(im))
    return out


def test_find_loop_picks_full_stride_not_half():
    frames = _walker_frames(period=40)
    idx, rep = core.find_loop(frames, count=8, min_period=16, max_period=64)
    assert abs(rep["period"] - 40) <= 2, rep            # half a stride (20) would be a silhouette match
    assert len(idx) == 8 and len(set(idx)) == 8
    assert rep["seam_error"] < 0.02, rep
    # frame 0 is the widest stride
    widths = []
    for i in idx:
        k = core.remove_border_background(core.to_rgba(frames[i]), 24)
        xs = np.nonzero(k[..., 3].any(0))[0]
        widths.append(xs.max() - xs.min())
    assert widths[0] >= max(widths) - 2


def test_find_loop_period_override_and_short_video_error():
    frames = _walker_frames(total=90, period=30)
    _, rep = core.find_loop(frames, count=6, period=30)
    assert rep["period"] == 30 and len(rep["indices"]) == 6
    try:
        core.find_loop(frames[:20], count=8, min_period=16, max_period=64)
    except ValueError as e:
        assert "too short" in str(e)
    else:
        raise AssertionError("expected ValueError for a video shorter than one cycle")


def test_loop_frames_node_then_build_sheet_from_video_frame():
    frames = _walker_frames(period=36)
    t = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    picked, first, scale, info, _pscale = pkg.NODE_CLASS_MAPPINGS["SpritePackLoopFrames"]().pick(t, 6, 16, 64, 0, 40, 24)
    assert picked.shape[0] == 6 and first.shape[0] == 1 and scale >= 1
    build = pkg.NODE_CLASS_MAPPINGS["SpritePackBuildSheet"]()
    kwargs = {f"frame_{i + 1}": picked[i:i + 1] for i in range(6)}
    sheet, prev, fr, pal, binfo = build.build(first, scale, 16, 6, 2, 0.5, 24, False, skip_native=True, **kwargs)
    assert fr.shape[0] == 6
    palette = pal.numpy()[0]
    assert not (palette[..., :3] > 0.99).all(axis=-1).any()   # the white background is not in the palette


def test_uniform_scale_keeps_every_frame_the_same_size():
    frames = _walker_frames(period=36)
    t = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    picked, first, scale, info, pscale = pkg.NODE_CLASS_MAPPINGS["SpritePackLoopFrames"]().pick(t, 6, 16, 64, 0, 40, 24)
    assert pscale >= 1.0
    build = pkg.NODE_CLASS_MAPPINGS["SpritePackBuildSheet"]()
    kwargs = {f"frame_{i + 1}": picked[i:i + 1] for i in range(6)}
    *_, binfo = build.build(first, scale, 16, 6, 2, 0.5, 24, False, skip_native=True, uniform_scale=pscale, **kwargs)
    import json as _json
    cells = [d["cells"] for d in _json.loads(binfo)["details"]]
    assert len({tuple(c) for c in cells}) == 1                 # identical grid for every frame, no per-frame refit
    assert all(abs(d["period"] - pscale) < 1e-3 for d in _json.loads(binfo)["details"])
