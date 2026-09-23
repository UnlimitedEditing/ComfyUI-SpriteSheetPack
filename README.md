# ComfyUI-SpriteSheetPack

Turn diffusion-rendered "pixel art" into **pixel-perfect sprite frames and a sprite sheet**.

Image models draw pixel art as soft, slightly misaligned blocks at high resolution. These two
nodes bracket the image model in a workflow:

1. **Sprite Pack: Prepare Reference** (`SpritePackPrepare`) takes the input sprite (a real
   low-res sprite, or AI "fake" pixel art at any size), recovers its native pixel grid, adds a
   margin, and renders a clean reference at an exact integer scale (default 8 px per art pixel,
   sized to multiples of 32 so Qwen Image 2.1 does not resample it).
2. **Sprite Pack: Build Sheet** (`SpritePackBuildSheet`) takes up to 15 rendered views at that
   scale, snaps each one back to the art-pixel grid (grid-offset detection plus a per-cell colour
   vote), locks every frame to one shared palette taken from the input sprite, binarizes alpha,
   removes the background, and packs the frames into a sheet. It outputs the native sheet, a
   nearest-neighbour preview, the frame batch, and the palette.

3. **Sprite Pack: Save Turntable GIF** (`SpritePackSaveGIF`) writes the frames as a looping
   animated GIF with Pillow alone: pixel-exact colours on one shared palette, a white or transparent
   background, and a nearest-neighbour upscale. It reports the file under the `gifs` UI key, like
   VideoHelperSuite does.
4. **Sprite Pack: Save Image (switchable)** (`SpritePackSaveImage`) is SaveImage with a `disabled`
   INT input (1 = save nothing). ComfyUI always runs output nodes, so a stock SaveImage cannot be
   switched off. This one can, for example to return only the GIF on chat front-ends.

5. **Sprite Pack: Gate** (`SpritePackGate`) passes images through, or passes an empty batch when
   `disabled` is 1. Put it in front of a stock SaveImage on hosts that only collect outputs from
   stock save nodes. Graydient collected nothing from the custom save nodes above.

Everything is numpy, scipy and Pillow. It does not use ffmpeg, imageio or VideoHelperSuite, whose pip
dependencies downgraded numpy on one hosted image and stopped ComfyUI from starting. There are no compiled dependencies and no model weights.

## Outputs

| Output | Contents |
|---|---|
| `sheet` | RGBA sprite sheet at native resolution, `columns` frames per row |
| `sheet_preview` | The sheet upscaled with nearest-neighbour for viewing |
| `frames` | Batch of native RGBA frames; frame 0 is the input sprite |
| `palette` | Swatch strip of the shared palette |

## How the grid is found

AI "pixel art" is not an integer upscale: its art pixels are fractional (e.g. 6.4-7.2 px) and
their size drifts across the image. A fixed integer grid either misreads the scale or falls out of
phase within a few cells, so:

- **Period:** the peak of the edge profile's spectrum on a fine fractional grid of periods. It is
  then checked against multiples, because on sharp art every harmonic is equally strong and the
  peak can land on one of them.
- **Grid lines:** tracked, not fixed. Tracking anchors at the strongest edge, predicts the next
  line one period away, locks onto the strongest edge near the prediction, and lets the local
  period follow what it sees.
- **Halo cleanup:** light anti-aliasing pixels that protrude outside the dark outline are
  removed (`clean_halo`, on by default).

These are tested on synthetic renders with blur, noise, partial alpha, shifted grids and drifting
6/7 px cells:
`python tests/test_core.py` and `python tests/test_nodes.py`.

## Credits

The grid-offset search and the background-removal approach are adapted from
[ComfyUI-PixelArt-Unfaker](https://github.com/CalaKuad1/ComfyUI-PixelArt-Unfaker) (MIT).
