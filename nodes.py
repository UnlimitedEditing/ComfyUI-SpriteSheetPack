"""ComfyUI wrappers around core.py. IMAGE tensors are [B, H, W, C] float 0..1; RGBA outputs are
4-channel IMAGEs (SaveImage writes them as transparent PNGs). MASK follows the LoadImage
convention: 1 = transparent."""

import json

import numpy as np
import torch

from . import core, poses, promptbook


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
                "max_render_pixels": ("INT", {"default": 400000, "min": 65536, "max": 4194304, "step": 1024,
                    "tooltip": "Pixel budget per rendered view; render_scale is lowered to fit it."}),
                "margin": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "Empty space around the sprite, as a fraction of its larger side, so rotated "
                    "views have room."}),
                "max_colors": ("INT", {"default": 32, "min": 0, "max": 256, "tooltip":
                    "Palette size when the input has to be de-noised. 0 = keep every colour."}),
                "bg_tolerance": ("INT", {"default": 24, "min": 0, "max": 255, "tooltip":
                    "Colour tolerance for removing a solid background connected to the border."}),
                "clean_halo": ("BOOLEAN", {"default": True, "tooltip":
                    "Remove light anti-aliasing pixels stuck outside the outline."}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("reference", "native_sprite", "render_scale", "native_width", "native_height", "info")
    FUNCTION = "prepare"
    CATEGORY = "image/sprite sheet"

    def prepare(self, image, sprite_width, render_scale, max_render_side, margin, max_colors,
                bg_tolerance, max_render_pixels=400000, clean_halo=True, mask=None):
        rgba = _tensor_to_rgba(image, mask)
        native, ref, k, src_scale = core.prepare_reference(
            rgba, sprite_width=sprite_width, render_scale=render_scale,
            max_render_side=max_render_side, margin=margin, bg_tolerance=bg_tolerance,
            max_colors=max_colors, max_render_pixels=max_render_pixels, halo=clean_halo)
        info = json.dumps({"source_period": round(float(src_scale), 2), "render_scale": k,
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
        optional["facing_report"] = ("STRING", {"forceInput": True, "tooltip":
            "Optional JSON from a vision model: which side each frame's front points to, keyed by "
            "generation order (0 = the input, 1.. = frame_1..). Frames facing the wrong way for their "
            "orbit slot are mirrored."})
        optional["skip_native"] = ("BOOLEAN", {"default": False, "tooltip":
            "Leave the native sprite out of the sheet (it still sets the palette and cell size). For "
            "animation cycles, where every frame is generated and the static reference is not one of them."})
        optional["uniform_scale"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.001, "tooltip":
            "Video frames: rendered pixels per art pixel, used for EVERY frame (area downsample onto the "
            "shared palette). 0 = fit a pixel grid to each frame (right for AI pixel art, wrong for video, "
            "where it stretches and squashes frames individually)."})
        optional["front_view"] = ("IMAGE", {"tooltip":
            "Optional dedicated straight-front render. When order_offset is not 0 (the input was not a "
            "front view), it replaces output position 0, so the sheet always starts with a true front."})
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
                "clean_halo": ("BOOLEAN", {"default": True, "tooltip":
                    "Remove light anti-aliasing pixels stuck outside the outline."}),
                "order_offset": ("INT", {"default": 0, "min": 0, "max": 63, "tooltip":
                    "Where the native sprite belongs in the output order. The frames arrive as "
                    "[sprite, next, next, ...] around the orbit; with offset k the output starts k "
                    "steps before the sprite, e.g. a 3/4-view input at orbit position 1 of 8 gets "
                    "order_offset 1 so the sheet still starts at the front view."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "sheet_preview", "frames", "palette", "info")
    FUNCTION = "build"
    CATEGORY = "image/sprite sheet"

    def build(self, native_sprite, render_scale, max_colors, columns, preview_scale, alpha_threshold,
              bg_tolerance, despeckle, clean_halo=True, order_offset=0, front_view=None,
              facing_report=None, skip_native=False, uniform_scale=0.0, **frames):
        native = _tensor_to_rgba(native_sprite)
        if native[..., 3].min() == 255:  # a plain video frame: key out the border background first
            native = core.remove_border_background(native, bg_tolerance)
        native = core.binarize_alpha(native, alpha_threshold)
        nh, nw = native.shape[:2]
        palette = core.build_palette(native, max_colors)
        idx = core.palette_indices(native, palette)
        front = np.zeros_like(native)
        front[idx >= 0, :3] = palette[idx[idx >= 0]]
        front[idx >= 0, 3] = 255

        def snap(img, fit=True):
            rgba = core.to_rgba(img[0].detach().cpu().float().numpy())
            rgba = core.remove_border_background(rgba, bg_tolerance)
            rgba = core.binarize_alpha(rgba, alpha_threshold)
            if uniform_scale and uniform_scale > 0:
                snapped = core.uniform_snap(rgba, uniform_scale, palette, alpha_threshold)
                detail = {"cells": [snapped.shape[1], snapped.shape[0]], "period": round(float(uniform_scale), 3)}
            else:
                # tracked lines, not a fixed grid: the image model's blocks drift like any AI pixel art
                snapped, detail = core.snap_tracked(rgba, render_scale, palette)
            if clean_halo:
                snapped = core.clean_halo(snapped)
            if despeckle:
                snapped = core.despeckle(snapped)
            return (core.fit_canvas(snapped, nw, nh) if fit else snapped), detail

        out, details = ([] if skip_native else [front]), []
        for name in sorted(frames, key=lambda n: int(n.rsplit("_", 1)[-1])):
            img = frames[name]
            if img is None:
                continue
            frame, detail = snap(img, fit=not skip_native)
            out.append(frame)
            details.append(dict(frame=name, **detail))

        # mirror frames whose front points the wrong way for their slot (still in generation order)
        flips = core.frames_to_flip(core.parse_facing_report(facing_report), int(order_offset), len(out))
        for j in flips:
            out[j] = out[j][:, ::-1].copy()
        k = 0 if skip_native else int(order_offset) % len(out)
        if k:  # out[j] sits at orbit position (k + j): rotate so position 0 comes first
            out = [out[(p - k) % len(out)] for p in range(len(out))]
            if front_view is not None:  # the input wasn't a front view: use the dedicated front render
                out[0], detail = snap(front_view)
                details.append(dict(frame="front_view", **detail))
        if skip_native and out:  # animation: shared ground line + torso x, cell grows to the biggest frame
            out = core.align_on_feet(out)
            nh, nw = out[0].shape[:2]
        sheet = core.build_sheet(out, columns)
        info = json.dumps({"frames": len(out), "cell": [nw, nh], "palette_size": len(palette), "order_offset": k,
                           "flipped_frames": flips,
                           "sheet": [sheet.shape[1], sheet.shape[0]], "details": details})
        print(f"[SpritePackBuildSheet] {info}")
        frames_t = torch.from_numpy(np.stack(out).astype(np.float32) / 255.0)
        return (_rgba_to_tensor(sheet), _rgba_to_tensor(core.upscale_nearest(sheet, preview_scale)),
                frames_t, _rgba_to_tensor(core.palette_swatches(palette)), info)


class SpritePackSaveGIF:
    """Native frames (e.g. Build Sheet's `frames`) -> looping turntable GIF, written with Pillow only.

    Reports the file under the UI key "gifs" with the same fields VideoHelperSuite uses, so hosts
    that collect VHS video/GIF outputs pick it up the same way."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "scale": ("INT", {"default": 6, "min": 1, "max": 32, "tooltip": "Nearest-neighbour upscale."}),
                "transparent_background": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "sprite_turntable"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gif_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "image/sprite sheet"

    def save(self, frames, fps, scale, transparent_background, filename_prefix):
        import os
        import folder_paths

        arrs = [core.to_rgba(f.detach().cpu().float().numpy()) for f in frames]
        h, w = arrs[0].shape[:2]
        out_dir = folder_paths.get_output_directory()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir, w * scale, h * scale)
        file = f"{filename}_{counter:05}_.gif"
        path = os.path.join(folder, file)
        core.save_gif(arrs, path, fps=fps, scale=scale, transparent=transparent_background)
        print(f"[SpritePackSaveGIF] {len(arrs)} frames @ {fps} fps -> {path}")
        entry = {"filename": file, "subfolder": subfolder, "type": "output", "format": "image/gif",
                 "frame_rate": fps, "fullpath": path}
        return {"ui": {"gifs": [entry]}, "result": (path,)}


class SpritePackSaveImage:
    """SaveImage with an off switch. ComfyUI always executes output nodes, so a stock SaveImage
    cannot be skipped by a switch upstream; this one saves nothing when `disabled` is 1 (an INT so
    hosts can drive it from a numeric field, e.g. "GIF only" on chat front-ends where a wide sheet
    does not display well)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "sprite"}),
                "disabled": ("INT", {"default": 0, "min": 0, "max": 1, "tooltip": "1 = save nothing."}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "image/sprite sheet"

    def save(self, images, filename_prefix, disabled):
        if int(disabled) >= 1:
            return {"ui": {"images": []}}
        import os
        import folder_paths
        from PIL import Image

        h, w = images.shape[1], images.shape[2]
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h)
        results = []
        for i, img in enumerate(images):
            arr = (img.detach().cpu().float().numpy().clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
            file = f"{filename}_{counter + i:05}_.png"
            Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB").save(
                os.path.join(folder, file), compress_level=4)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        return {"ui": {"images": results}}


class SpritePackStandardizeSheet:
    """Any sprite sheet (ours, ripped, hand-drawn) -> one canonical sheet: equal-size cells, one row
    per state, every frame on the same pivot, optionally back at native pixel size and one palette.
    No model involved. Also reports the layout as JSON for the game engine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip":
                    "Frames per row. Set columns AND rows for a sheet whose frames touch; 0 = detect from the gaps."}),
                "rows": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Number of rows (states). 0 = detect."}),
                "min_gap": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip":
                    "Empty pixels that separate two frames when detecting. 0 = about 1% of the sheet."}),
                "anchor": (["feet", "center"], {"tooltip":
                    "feet: all frames share one ground line and foot position. center: bounding-box centre."}),
                "cell_width": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip":
                    "Fixed cell size (needs both). 0 = the smallest cell that holds every frame."}),
                "cell_height": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "pad": ("INT", {"default": 1, "min": 0, "max": 64, "tooltip": "Transparent margin around the frames."}),
                "unscale": ("INT", {"default": 1, "min": 0, "max": 32, "tooltip":
                    "Divide an upscaled sheet back to native pixels: 1 = leave, N = the sheet was scaled Nx, 0 = detect."}),
                "max_colors": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip":
                    "Lock the sheet to one shared palette of this size. 0 = leave the colours alone."}),
                "labels": ("STRING", {"default": "", "tooltip":
                    "State names for the rows, top to bottom, comma separated (idle,attack,hit)."}),
                "fps": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "tooltip": "Playback rate written to the JSON."}),
                "preview_scale": ("INT", {"default": 4, "min": 1, "max": 32}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "bg_tolerance": ("INT", {"default": 24, "min": 0, "max": 255, "tooltip":
                    "Background keying tolerance when the sheet has no transparency."}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "sheet_preview", "frames", "layout_card", "info")
    FUNCTION = "standardize"
    CATEGORY = "image/sprite sheet"

    def standardize(self, image, columns, rows, min_gap, anchor, cell_width, cell_height, pad, unscale,
                    max_colors, labels, fps, preview_scale, alpha_threshold, bg_tolerance, mask=None):
        rgba = _tensor_to_rgba(image, mask)
        sheet, frames, meta = core.standardize_sheet(
            rgba, columns=columns, rows=rows, min_gap=min_gap, anchor=anchor, cell_width=cell_width,
            cell_height=cell_height, pad=pad, unscale=unscale, max_colors=max_colors, labels=labels,
            fps=fps, bg_tolerance=bg_tolerance, alpha_threshold=alpha_threshold)
        info = json.dumps(meta)
        print(f"[SpritePackStandardizeSheet] {info}")
        frames_t = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        card = core.text_card(json.dumps(meta, indent=1))
        return (_rgba_to_tensor(sheet), _rgba_to_tensor(core.upscale_nearest(sheet, preview_scale)),
                frames_t, torch.from_numpy(card.astype(np.float32) / 255.0)[None], info)


class SpritePackGate:
    """Pass images through, or an empty batch when `disabled` is 1.

    For hosts that only collect outputs from stock save nodes (Graydient collected nothing from a
    custom save node): put this in front of a stock SaveImage. An empty batch makes SaveImage save
    nothing, while the host still sees an ordinary SaveImage output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "disabled": ("INT", {"default": 0, "min": 0, "max": 1, "tooltip": "1 = output an empty batch."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "gate"
    CATEGORY = "image/sprite sheet"

    def gate(self, images, disabled):
        return (images[:0] if int(disabled) >= 1 else images,)


class SpritePackPoseCycle:
    """Draws one looping animation cycle (walk / run / idle) as OpenPose skeleton images, sized and
    placed to match the character in `reference`, for an OpenPose-trained ControlNet. Procedural: no
    pose-estimation model. Humanoid side views only; frame i is cycle phase i / frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE", {"tooltip":
                "The character on a plain background (SpritePackPrepare's reference). Sets the canvas "
                "size and where the skeleton stands."}),
            "animation": (list(poses.ANIMATIONS),),
            "frames": ("INT", {"default": 8, "min": 2, "max": 15, "tooltip": "Frames in the loop."}),
            "facing": ("STRING", {"default": "right", "tooltip": "right or left (anything starting with l = left)."}),
            "bg_threshold": ("INT", {"default": 24, "min": 1, "max": 255, "tooltip":
                "Colour distance from the border colour that counts as character when measuring its size."}),
            "scale": ("FLOAT", {"default": 0.85, "min": 0.4, "max": 1.0, "step": 0.01, "tooltip":
                "Skeleton height as a fraction of the character's height in the reference (feet stay on "
                "the same ground line). Below 1 leaves headroom, so the model does not draw the figure "
                "past the frame and crop the head."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("poses", "frames", "info")
    FUNCTION = "draw"
    CATEGORY = "image/sprite sheet"

    def draw(self, reference, animation, frames, facing, bg_threshold, scale=0.85):
        ref = (reference[0].detach().cpu().float().numpy() * 255.0).round().astype(np.uint8)
        h, w = ref.shape[:2]
        bbox = poses.figure_bbox(ref, bg_threshold)
        facing = "left" if str(facing).strip().lower().startswith("l") else "right"
        imgs = poses.render_cycle(animation, frames, (w, h), bbox, facing, scale=scale)
        info = json.dumps({"animation": animation, "frames": len(imgs), "canvas": [w, h], "figure_bbox": list(bbox)})
        print(f"[SpritePackPoseCycle] {info}")
        batch = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0)
        return (batch, len(imgs), info)


class SpritePackLoopFrames:
    """A video of a character walking in place on a plain background -> `count` frames of one
    seamless cycle, ready for SpritePackBuildSheet (skip_native). Finds the stride period by matching
    short runs of frames in colour, starts at the best seam and opens on the widest stride. Also
    suggests the render scale (rendered px per art px) for a target sprite height."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "count": ("INT", {"default": 8, "min": 2, "max": 15, "tooltip": "Frames in the sheet."}),
            "min_period": ("INT", {"default": 16, "min": 4, "max": 400, "tooltip":
                "Shortest cycle to consider, in video frames. Keep above half a step."}),
            "max_period": ("INT", {"default": 64, "min": 8, "max": 800, "tooltip":
                "Longest cycle to consider, in video frames."}),
            "period_override": ("INT", {"default": 0, "min": 0, "max": 800, "tooltip":
                "Force the cycle length in video frames. 0 = detect."}),
            "target_height": ("INT", {"default": 64, "min": 8, "max": 512, "tooltip":
                "Wanted sprite height in art pixels; sets the suggested render_scale."}),
            "bg_tolerance": ("INT", {"default": 24, "min": 0, "max": 255}),
        }, "optional": {
            "spacing": (["motion", "time"], {"default": "motion", "tooltip":
                "motion: equal steps of pose change (no hold-then-jump). time: equal steps of video time."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("frames", "first_frame", "render_scale", "info", "pixel_scale")
    FUNCTION = "pick"
    CATEGORY = "image/sprite sheet"

    def pick(self, frames, count, min_period, max_period, period_override, target_height, bg_tolerance,
             spacing="motion"):
        arr = (frames.detach().cpu().float().numpy() * 255.0).round().astype(np.uint8)
        idx, report = core.find_loop(list(arr), count=count, min_period=min_period, max_period=max_period,
                                     period=period_override, bg_tolerance=bg_tolerance,
                                     target_height=target_height, spacing=spacing)
        info = json.dumps(report)
        print(f"[SpritePackLoopFrames] {info}")
        picked = frames[idx]
        return (picked, picked[:1], report["render_scale"], info, report["pixel_scale"])


class SpritePackAnimationSheet:
    """One video with several actions in a row -> one sprite sheet, one row per action.

    Every action is cut out of the video by its segment (see core.animation_pick): cycles get the loop
    finder, one-shots get evenly spaced frames. All frames share ONE pixel scale (figure height /
    target_height), ONE palette and ONE ground line + pivot, so the actions line up in a game engine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "segments": ("STRING", {"forceInput": True, "tooltip":
                "JSON list of {label, start, end, kind, settle} in video frames (SpritePackPromptBook makes it)."}),
            "count": ("INT", {"default": 8, "min": 2, "max": 15, "tooltip": "Frames per action."}),
            "target_height": ("INT", {"default": 64, "min": 8, "max": 512, "tooltip":
                "Sprite height in art pixels; sets the one pixel scale for every frame."}),
            "max_colors": ("INT", {"default": 32, "min": 0, "max": 256, "tooltip": "Shared palette size."}),
            "min_period": ("INT", {"default": 16, "min": 4, "max": 400}),
            "max_period": ("INT", {"default": 64, "min": 8, "max": 800}),
            "bg_tolerance": ("INT", {"default": 24, "min": 0, "max": 255}),
            "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "clean_halo": ("BOOLEAN", {"default": True}),
            "despeckle": ("BOOLEAN", {"default": False}),
            "preview_scale": ("INT", {"default": 8, "min": 1, "max": 32}),
            "fps": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "tooltip": "Playback rate written to the info JSON."}),
        }, "optional": {
            "spacing": (["motion", "time"], {"default": "motion", "tooltip":
                "Cycles: motion = equal steps of pose change (no hold-then-jump), time = equal steps of video time. "
                "Optional so workflows restored before this existed keep working."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "sheet_preview", "frames", "palette", "info")
    FUNCTION = "build"
    CATEGORY = "image/sprite sheet"

    def build(self, frames, segments, count, target_height, max_colors, min_period, max_period, bg_tolerance,
              alpha_threshold, clean_halo, despeckle, preview_scale, fps, spacing="motion"):
        arr = (frames.detach().cpu().float().numpy() * 255.0).round().astype(np.uint8)
        segs = json.loads(segments) if isinstance(segments, str) else segments
        keyed, picks, scale = core.animation_pick(list(arr), segs, count=count, min_period=min_period,
                                                  max_period=max_period, bg_tolerance=bg_tolerance,
                                                  target_height=target_height, spacing=spacing)
        order = [i for p in picks for i in p["indices"]]
        native = [core.uniform_downscale(keyed[i], scale, alpha_threshold) for i in order]
        palette = core.build_palette(np.concatenate(native, axis=0), max_colors)
        native = [core.quantize_to_palette(f, palette) for f in native]
        if clean_halo:
            native = [core.clean_halo(f) for f in native]
        if despeckle:
            native = [core.despeckle(f) for f in native]
        cells = core.align_on_feet(native)
        sheet = core.build_sheet(cells, count)
        ch, cw = cells[0].shape[:2]
        states = [{"name": p["label"], "row": r, "start": r * int(count), "frames": int(count), "kind": p["kind"],
                   "report": p["report"]} for r, p in enumerate(picks)]
        info = json.dumps({"format": "spritepack.sheet", "version": 1, "cell": [cw, ch], "columns": int(count),
                           "rows": len(picks), "fps": fps, "pixel_scale": round(scale, 3),
                           "palette_size": len(palette), "states": states,
                           "sheet": [sheet.shape[1], sheet.shape[0]]})
        print(f"[SpritePackAnimationSheet] {info}")
        frames_t = torch.from_numpy(np.stack(cells).astype(np.float32) / 255.0)
        return (_rgba_to_tensor(sheet), _rgba_to_tensor(core.upscale_nearest(sheet, preview_scale)),
                frames_t, _rgba_to_tensor(core.palette_swatches(palette)), info)


class SpritePackPromptBook:
    """Short semantic tags -> one H3 prompt with a timeline, the matching frame segments and the
    snapped video length. Wording lives in prompts.json (edit it, no code change)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "actions": ("STRING", {"default": "walk", "tooltip":
                "Actions in order, joined with +: idle walk run jump attack hit death (see prompts.json)."}),
            "views": ("STRING", {"default": "side", "tooltip":
                "One view for every action, or one per action joined with +: side side-left threequarter "
                "threequarter-left front back topdown."}),
            "style": ("STRING", {"default": "pixel-white", "tooltip": "pixel-white, toon-white or raw-white."}),
            "subject": ("STRING", {"default": "<Picture 1>", "tooltip":
                "How the prompt refers to the reference image."}),
            "extra": ("STRING", {"default": "", "multiline": True, "tooltip":
                "Optional short description of the character, appended to the subject."}),
            "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
            "max_frames": ("INT", {"default": 396, "min": 22, "max": 800, "tooltip":
                "Refuse sequences longer than this many video frames."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "segments", "length", "info")
    FUNCTION = "plan"
    CATEGORY = "image/sprite sheet"

    def plan(self, actions, views, style, subject, extra, fps, max_frames):
        plan = promptbook.build_plan(actions, views, style, subject, extra, fps=fps, max_frames=max_frames)
        info = json.dumps(plan["info"])
        print(f"[SpritePackPromptBook] {info}")
        print(f"[SpritePackPromptBook] prompt: {plan['prompt']}")
        return (plan["prompt"], json.dumps(plan["segments"]), plan["length"], info)


NODE_CLASS_MAPPINGS = {
    "SpritePackPrepare": SpritePackPrepare,
    "SpritePackBuildSheet": SpritePackBuildSheet,
    "SpritePackSaveGIF": SpritePackSaveGIF,
    "SpritePackSaveImage": SpritePackSaveImage,
    "SpritePackGate": SpritePackGate,
    "SpritePackPoseCycle": SpritePackPoseCycle,
    "SpritePackLoopFrames": SpritePackLoopFrames,
    "SpritePackAnimationSheet": SpritePackAnimationSheet,
    "SpritePackPromptBook": SpritePackPromptBook,
    "SpritePackStandardizeSheet": SpritePackStandardizeSheet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpritePackPrepare": "Sprite Pack: Prepare Reference",
    "SpritePackBuildSheet": "Sprite Pack: Build Sheet",
    "SpritePackSaveGIF": "Sprite Pack: Save Turntable GIF",
    "SpritePackSaveImage": "Sprite Pack: Save Image (switchable)",
    "SpritePackGate": "Sprite Pack: Gate (pass or empty)",
    "SpritePackPoseCycle": "Sprite Pack: Pose Cycle (OpenPose skeletons)",
    "SpritePackLoopFrames": "Sprite Pack: Loop Frames (walk cycle from video)",
    "SpritePackAnimationSheet": "Sprite Pack: Animation Sheet (actions from one video)",
    "SpritePackPromptBook": "Sprite Pack: Prompt Book (tags -> prompt + timeline)",
    "SpritePackStandardizeSheet": "Sprite Pack: Standardize Sheet",
}
