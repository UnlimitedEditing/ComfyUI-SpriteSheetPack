"""Prompt book: semantic tags -> one H3 prompt with a timeline, plus the matching frame segments.

The workflow slots carry short tags (actions "walk+run+attack", views "side" or "side+front", style
"pixel-white"); prompts.json holds the wording. Because the prompt author writes the timeline, the
frame range of every action is known, and SpritePackAnimationSheet cuts each action out of the
finished video by those ranges instead of guessing where one ends and the next begins.

Pure python (no torch), so it is easy to test and to reuse outside ComfyUI."""

import json
import os
import re

BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")


def load_book(path=None):
    with open(path or BOOK_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_tags(text):
    """'walk+run, attack' -> ['walk', 'run', 'attack'] (a '+' that arrives as a space is fine too)."""
    return [t for t in re.split(r"[+,;\s]+", (text or "").strip().lower()) if t]


def fmt_time(seconds):
    """4.0 -> '00:04.000' (the MM:SS.mmm stamp H3 reads)."""
    ms = int(round(seconds * 1000))
    return f"{ms // 60000:02d}:{(ms % 60000) // 1000:02d}.{ms % 1000:03d}"


def snap_length(frames, rule):
    """Smallest valid H3 length >= frames (base + step*k)."""
    base, step = int(rule["base"]), int(rule["step"])
    frames = max(base, int(frames))
    return frames + (-(frames - base)) % step


def _lookup(table, tag, what):
    if tag not in table:
        raise ValueError(f"unknown {what} {tag!r}; valid {what}s: {', '.join(sorted(table))}")
    return table[tag]


def build_plan(actions, views, style, subject="<Picture 1>", extra="", fps=None, max_frames=None, book=None):
    """-> {"prompt", "segments" (list), "length", "info"}.

    `views` is one view for every action, or one per action (a shorter list repeats its last entry).
    The first action, and every action whose view differs from the one before, gets `turn_seconds`
    of extra settle time at its start (the camera or character is still moving there)."""
    book = book or load_book()
    fps = int(fps or book.get("fps", 24))
    act_tags, view_tags = parse_tags(actions), parse_tags(views)
    if not act_tags:
        raise ValueError("no actions given; pass e.g. 'walk' or 'walk+run+attack'")
    if not view_tags:
        view_tags = ["side"]
    style_text = _lookup(book["styles"], (style or "").strip().lower() or "pixel-white", "style")
    who = subject.strip() or "<Picture 1>"
    if (extra or "").strip():
        who = f"{who} ({extra.strip()})"

    segments, parts, cursor, prev_view = [], [], 0, None
    for n, tag in enumerate(act_tags):
        act = _lookup(book["actions"], tag, "action")
        view_tag = view_tags[min(n, len(view_tags) - 1)]
        view_text = _lookup(book["views"], view_tag, "view")
        turned = view_tag != prev_view
        frames = int(round(float(act["seconds"]) * fps))
        settle = int(round((float(act.get("settle", 0)) + (float(book.get("turn_seconds", 0)) if turned else 0)) * fps))
        camera = book["camera_first" if prev_view is None else "camera_change"].format(view=view_text) if turned else ""
        text = f"{camera} {act['text'][0].upper()}{act['text'][1:]}.".strip()
        segments.append({"label": tag, "view": view_tag, "kind": act.get("kind", "cycle"), "start": cursor,
                         "end": cursor + frames, "settle": min(settle, max(0, frames - 4))})
        parts.append((cursor, cursor + frames, text))
        cursor += frames
        prev_view = view_tag

    limit = int(max_frames or book["frame_rule"]["max"])
    length = snap_length(cursor, book["frame_rule"])
    if length > limit:
        raise ValueError(f"{len(act_tags)} actions need {length} frames at {fps} fps; the limit is {limit}. "
                         f"Use fewer actions or shorter ones (edit 'seconds' in prompts.json)")
    orig = [seg["label"] for seg in segments]       # unique labels: a repeated action is named after its view
    for i, seg in enumerate(segments):              # when the views differ, otherwise numbered
        twins = [j for j, lab in enumerate(orig) if lab == orig[i]]
        if len(twins) > 1:
            distinct_views = len({segments[j]["view"] for j in twins}) == len(twins)
            seg["label"] = f"{orig[i]}_{seg['view']}" if distinct_views else f"{orig[i]}{twins.index(i) + 1}"
    segments[-1]["end"] = length                      # the snapped-up tail belongs to the last action
    parts[-1] = (parts[-1][0], length, parts[-1][2])
    shots = []
    for n, (start, _end, text) in enumerate(parts):
        if n == 0:
            scene = book["preamble"].format(subject=who, style=style_text).strip()
            shots.append(book["shot_first"].format(n=1, text=f"{scene} {text}"))
        else:
            shots.append(book["shot_next"].format(n=n + 1, time=fmt_time(start / fps), text=text[0].lower() + text[1:]))
    prompt = (book["description_prefix"] + " ".join(shots) + "\noverall_soundscape: " + book["soundscape"]
              + "\nnon_diegetic_music: " + book["music"])
    info = {"fps": fps, "length": length, "seconds": round(length / fps, 2), "segments": segments}
    return {"prompt": prompt, "segments": segments, "length": length, "info": info}
