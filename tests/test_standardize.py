"""Sheet standardiser: synthetic sheets with ragged rows, odd backgrounds and upscaling."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402

BODY = np.array([200, 90, 60], np.uint8)
OUT = np.array([30, 20, 20], np.uint8)


def sprite(w, h, tip=0):
    """w x h blob with a dark outline; tip = a thin arm sticking out to the right."""
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = BODY
    img[:, :, 3] = 255
    img[0, :, :3] = img[-1, :, :3] = OUT
    img[:, 0, :3] = img[:, -1, :3] = OUT
    if tip:
        img = np.concatenate([img, np.zeros((h, tip, 4), np.uint8)], 1)
        img[h // 3:h // 3 + 2, w:w + tip] = (*OUT, 255)
    return img


def place(sheet, img, x, ybottom):
    h, w = img.shape[:2]
    sheet[ybottom - h:ybottom, x:x + w] = img


def make_sheet(bg=None):
    """3 rows (2, 4, 3 frames), frames of different sizes, feet at different heights."""
    sheet = np.zeros((3 * 60, 4 * 60, 4), np.uint8)
    spec = [[(20, 30, 0), (22, 32, 6)],
            [(18, 28, 0), (20, 30, 0), (24, 26, 0), (18, 34, 0)],
            [(20, 20, 0), (22, 22, 0), (30, 12, 0)]]
    for r, row in enumerate(spec):
        for c, (w, h, tip) in enumerate(row):
            place(sheet, sprite(w, h, tip), c * 60 + 6, r * 60 + 54 - (c % 2) * 3)  # feet bob between frames
    if bg is not None:
        sheet[sheet[..., 3] == 0] = (*bg, 255)
    return sheet


def feet_row(cell):
    return np.nonzero(cell[..., 3].any(axis=1))[0].max()


def test_layout_and_alignment():
    sheet, frames, meta = core.standardize_sheet(make_sheet(), labels="idle,attack,hit")
    assert meta["rows"] == 3 and meta["columns"] == 4 and meta["frames"] == 9, meta
    assert [s["frames"] for s in meta["states"]] == [2, 4, 3]
    assert [s["name"] for s in meta["states"]] == ["idle", "attack", "hit"]
    cw, ch = meta["cell"]
    assert sheet.shape == (3 * ch, 4 * cw, 4)
    assert all(f.shape == (ch, cw, 4) for f in frames)
    # one common ground line at the pivot, and no pixels lost
    assert {feet_row(f) for f in frames} == {meta["pivot"][1] - 1}
    assert int((frames[1][..., 3] > 0).sum()) == 22 * 32 + 6 * 2


def test_background_keyed_and_tip_not_split():
    _, frames, meta = core.standardize_sheet(make_sheet(bg=(255, 0, 255)))
    assert meta["frames"] == 9, meta
    assert not meta["warnings"], meta["warnings"]


def test_speck_merged():
    s = make_sheet()
    s[10, 4 * 60 - 5] = (255, 255, 255, 255)  # a lone spark in the empty right part of row 0
    _, _, meta = core.standardize_sheet(s)
    assert meta["states"][0]["frames"] == 2, meta["states"]


def test_fixed_cell_and_center():
    _, _, meta = core.standardize_sheet(make_sheet(), cell_width=48, cell_height=48, anchor="feet")
    assert meta["cell"] == [48, 48] and meta["pivot"] == [24, 47]
    _, _, m2 = core.standardize_sheet(make_sheet(), anchor="center")
    assert m2["anchor"] == "center"


def test_uniform_grid_touching():
    sheet = np.zeros((32, 96, 4), np.uint8)
    sheet[1:31, :] = (*BODY, 255)  # three cells with no gap at all
    _, _, meta = core.standardize_sheet(sheet)
    assert meta["frames"] == 3 and meta["warnings"], meta
    _, _, meta = core.standardize_sheet(sheet, columns=3, rows=1)
    assert meta["frames"] == 3 and not meta["warnings"], meta


def test_unscale_roundtrip():
    s = make_sheet()
    up = core.upscale_nearest(s, 3)
    _, rframes, _ = core.standardize_sheet(s)
    _, gframes, meta = core.standardize_sheet(up, unscale=3)
    assert meta["frames"] == 9 and meta["cell"] == list(rframes[0].shape[:2][::-1]), meta
    agree = np.mean([np.all(a == b, axis=-1).mean() for a, b in zip(rframes, gframes)])
    assert agree > 0.95, agree
    _, _, m3 = core.standardize_sheet(up, unscale=0)
    assert m3["source_unscale"] >= 2.5, m3


def test_palette_lock():
    s = make_sheet()
    s[10 + 0, 5 + 0] = (0, 0, 0, 0)
    s[12, 12, :3] = (201, 91, 61)  # near-duplicate colour inside a frame
    _, frames, meta = core.standardize_sheet(s, max_colors=2)
    cols = {tuple(px) for f in frames for px in f[f[..., 3] > 0][:, :3]}
    assert len(cols) <= 2 and meta["palette_size"] <= 2, cols
