"""ComfyUI wrappers around core.py. IMAGE tensors are [B, H, W, C] float 0..1; RGBA outputs are
4-channel IMAGEs (SaveImage writes them as transparent PNGs). MASK follows the LoadImage
convention: 1 = transparent."""

import json

import numpy as np
import torch

from . import core


def _tensor_to_rgba(image, mask=None):
    arr = image[0].detach().cpu().float().numpy()
    alpha = None
    if mask is not None:
        m = mask[0].detach().cpu().float().numpy()
        if m.shape == arr.shape[:2]:
            alpha = 1.0 - m
    return core.to_rgba(arr, alpha)


def _rgba_to_tensor(rgba):
    return torch.from_numpy(rgba.astype(np.float32) / 255.0)[None]


class SpritePackPrepare:
    """Input sprite (real or AI 'fake' pixel art, any size) -> a clean reference for the image
    model at an exact integer render scale, plus the native sprite it came from."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sprite_width": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip":
                    "Native width of the input in art pixels. 0 = auto-detect the pixel grid."}),
                "render_scale": ("INT", {"default": 8, "min": 2, "max": 32, "tooltip":
                    "Rendered pixels per art pixel for the image model. Lowered automatically "
                    "to respect max_render_side."}),
                "max_render_side": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "margin": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "Empty space around the sprite, as a fraction of its larger side, so rotated "
                    "views have room."}),
                "max_colors": ("INT", {"default": 32, "min": 0, "max": 256, "tooltip":
                    "Palette size when the input has to be de-noised. 0 = keep every colour."}),
                "bg_tolerance": ("INT", {"default": 24, "min": 0, "max": 255, "tooltip":
                    "Colour tolerance for removing a solid background connected to the border."}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("reference", "native_sprite", "render_scale", "native_width", "native_height", "info")
    FUNCTION = "prepare"
    CATEGORY = "image/sprite sheet"

    def prepare(self, image, sprite_width, render_scale, max_render_side, margin, max_colors,
                bg_tolerance, mask=None):
        rgba = _tensor_to_rgba(image, mask)
        native, ref, k, src_scale = core.prepare_reference(
            rgba, sprite_width=sprite_width, render_scale=render_scale,
            max_render_side=max_render_side, margin=margin, bg_tolerance=bg_tolerance,
            max_colors=max_colors)
        info = json.dumps({"source_scale": round(float(src_scale), 3), "render_scale": k,
                           "native": [native.shape[1], native.shape[0]],
                           "reference": [ref.shape[1], ref.shape[0]]})
        print(f"[SpritePackPrepare] {info}")
        ref_t = torch.from_numpy(ref.astype(np.float32) / 255.0)[None]
        return (ref_t, _rgba_to_tensor(native), k, native.shape[1], native.shape[0], info)


class SpritePackBuildSheet:
    """Rendered views (at native * render_scale) -> pixel-perfect native frames on one shared
    palette, packed into a sprite sheet. The native sprite is frame 0 and the palette source."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"frame_{i}": ("IMAGE",) for i in range(1, 16)}
        return {
            "required": {
                "native_sprite": ("IMAGE",),
                "render_scale": ("INT", {"default": 8, "min": 1, "max": 64}),
                "max_colors": ("INT", {"default": 32, "min": 0, "max": 256, "tooltip":
                    "Shared palette size, taken from the native sprite. 0 = every colour it has."}),
                "columns": ("INT", {"default": 8, "min": 1, "max": 64}),
                "preview_scale": ("INT", {"default": 8, "min": 1, "max": 32}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "bg_tolerance": ("INT", {"default": 24, "min": 0, "max": 255}),
                "despeckle": ("BOOLEAN", {"default": False, "tooltip":
                    "Replace isolated single pixels surrounded by one other colour."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "sheet_preview", "frames", "palette", "info")
    FUNCTION = "build"
    CATEGORY = "image/sprite sheet"

    def build(self, native_sprite, render_scale, max_colors, columns, preview_scale, alpha_threshold,
              bg_tolerance, despeckle, **frames):
        native = core.binarize_alpha(_tensor_to_rgba(native_sprite), alpha_threshold)
        nh, nw = native.shape[:2]
        palette = core.build_palette(native, max_colors)
        idx = core.palette_indices(native, palette)
        front = np.zeros_like(native)
        front[idx >= 0, :3] = palette[idx[idx >= 0]]
        front[idx >= 0, 3] = 255

        out, details = [front], []
        for name in sorted(frames, key=lambda n: int(n.rsplit("_", 1)[-1])):
            img = frames[name]
            if img is None:
                continue
            rgba = core.to_rgba(img[0].detach().cpu().float().numpy())
            rgba = core.remove_border_background(rgba, bg_tolerance)
            rgba = core.binarize_alpha(rgba, alpha_threshold)
            snapped, offset = core.snap_to_grid(rgba, render_scale, palette)
            if despeckle:
                snapped = core.despeckle(snapped)
            out.append(core.fit_canvas(snapped, nw, nh))
            details.append({"frame": name, "grid_offset": list(offset)})

        sheet = core.build_sheet(out, columns)
        info = json.dumps({"frames": len(out), "cell": [nw, nh], "palette_size": len(palette),
                           "sheet": [sheet.shape[1], sheet.shape[0]], "details": details})
        print(f"[SpritePackBuildSheet] {info}")
        frames_t = torch.from_numpy(np.stack(out).astype(np.float32) / 255.0)
        return (_rgba_to_tensor(sheet), _rgba_to_tensor(core.upscale_nearest(sheet, preview_scale)),
                frames_t, _rgba_to_tensor(core.palette_swatches(palette)), info)


NODE_CLASS_MAPPINGS = {
    "SpritePackPrepare": SpritePackPrepare,
    "SpritePackBuildSheet": SpritePackBuildSheet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpritePackPrepare": "Sprite Pack: Prepare Reference",
    "SpritePackBuildSheet": "Sprite Pack: Build Sheet",
}
