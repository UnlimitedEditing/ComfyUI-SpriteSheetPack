"""GIF writer: frame count, timing, loop, exact colours, background handling, and the ComfyUI
node's output entry (with a stub folder_paths)."""

import importlib.util
import os
import sys
import tempfile
import types

import numpy as np
import torch
from PIL import Image, ImageSequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import core  # noqa: E402
import test_core as tc  # noqa: E402


def frames8():
    return [tc.make_native(30, 24) for _ in range(8)]


def test_save_gif_exact():
    frames = frames8()
    with tempfile.TemporaryDirectory() as d:
        path = core.save_gif(frames, os.path.join(d, "t.gif"), fps=6, scale=4)
        with Image.open(path) as im:
            got = [np.asarray(f.convert("RGBA")) for f in ImageSequence.Iterator(im)]
            info = dict(im.info)
        assert len(got) == 8
        assert got[0].shape[:2] == (24 * 4, 30 * 4)
        assert info.get("loop") == 0 and abs(info.get("duration", 0) - 167) <= 10  # GIF stores delays in 10 ms units
        for src, g in zip(frames, got):
            g = g[::4, ::4]  # back to native
            opaque = src[..., 3] > 0
            assert (g[opaque][:, :3] == src[opaque][:, :3]).all()          # colours exact
            assert (g[~opaque][:, :3] == 255).all()                        # background white


def test_save_gif_transparent():
    frames = frames8()
    with tempfile.TemporaryDirectory() as d:
        path = core.save_gif(frames, os.path.join(d, "t.gif"), fps=10, scale=2, transparent=True)
        with Image.open(path) as im:
            g = np.asarray(im.convert("RGBA"))[::2, ::2]
        assert (g[frames[0][..., 3] == 0][:, 3] == 0).all()


def test_node_entry():
    with tempfile.TemporaryDirectory() as d:
        stub = types.ModuleType("folder_paths")
        stub.get_output_directory = lambda: d
        stub.get_save_image_path = lambda prefix, out, w, h: (d, prefix, 1, "", prefix)
        sys.modules["folder_paths"] = stub
        spec = importlib.util.spec_from_file_location("sprite_pack", os.path.join(ROOT, "__init__.py"),
                                                      submodule_search_locations=[ROOT])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["sprite_pack"] = pkg
        spec.loader.exec_module(pkg)
        node = pkg.NODE_CLASS_MAPPINGS["SpritePackSaveGIF"]()
        batch = torch.from_numpy(np.stack(frames8()).astype(np.float32) / 255.0)
        res = node.save(batch, 6.0, 6, False, "sprite_turntable")
        entry = res["ui"]["gifs"][0]
        assert entry["format"] == "image/gif" and entry["filename"].endswith(".gif")
        assert os.path.isfile(res["result"][0])
        with Image.open(res["result"][0]) as im:
            assert len(list(ImageSequence.Iterator(im))) == 8


def test_save_image_gate():
    with tempfile.TemporaryDirectory() as d:
        stub = types.ModuleType("folder_paths")
        stub.get_output_directory = lambda: d
        stub.get_save_image_path = lambda prefix, out, w, h: (d, prefix, 1, "", prefix)
        sys.modules["folder_paths"] = stub
        spec = importlib.util.spec_from_file_location("sprite_pack2", os.path.join(ROOT, "__init__.py"),
                                                      submodule_search_locations=[ROOT])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["sprite_pack2"] = pkg
        spec.loader.exec_module(pkg)
        node = pkg.NODE_CLASS_MAPPINGS["SpritePackSaveImage"]()
        img = torch.from_numpy(np.stack(frames8()[:2]).astype(np.float32) / 255.0)
        on = node.save(img, "sheet", 0)
        assert len(on["ui"]["images"]) == 2 and all(os.path.isfile(os.path.join(d, e["filename"])) for e in on["ui"]["images"])
        with Image.open(os.path.join(d, on["ui"]["images"][0]["filename"])) as im:
            assert im.mode == "RGBA"
        off = node.save(img, "sheet_off", 1)
        assert off["ui"]["images"] == [] and not any(f.startswith("sheet_off") for f in os.listdir(d))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
