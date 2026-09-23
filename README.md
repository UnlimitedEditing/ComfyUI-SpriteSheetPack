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

Everything is numpy, scipy and Pillow. There are no compiled dependencies and no model weights.

## Outputs

| Output | Contents |
|---|---|
| `sheet` | RGBA sprite sheet at native resolution, `columns` frames per row |
| `sheet_preview` | The sheet upscaled with nearest-neighbour for viewing |
| `frames` | Batch of native RGBA frames; frame 0 is the input sprite |
| `palette` | Swatch strip of the shared palette |

## How the grid is found

- **Scale:** the edge profile of a k-pixel grid is an impulse train of period k, so its spectral
  power sits at the harmonics h/k. The detector takes the largest k whose mean harmonic power is
  near the top score. Divisors of k tie with it, and multiples of k score about half.
- **Offset:** a forward-difference edge profile, summed on each of the k possible grid phases.

Both are tested on synthetic renders with blur, noise, partial alpha and shifted grids:
`python tests/test_core.py` and `python tests/test_nodes.py`.

## Credits

The grid-offset search and the background-removal approach are adapted from
[ComfyUI-PixelArt-Unfaker](https://github.com/CalaKuad1/ComfyUI-PixelArt-Unfaker) (MIT).
