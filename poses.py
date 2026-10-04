"""Procedural OpenPose (COCO-18) skeleton images for animation cycles -- numpy + PIL only.

A side-view forward-kinematics rig is posed per cycle phase and drawn the way controlnet_aux draws
OpenPose bodies (coloured limb ellipses, dimmed to 60%, then coloured joint dots on black), so the
result is a valid control image for any OpenPose-trained ControlNet (e.g. InstantX Qwen-Image
ControlNet-Union, type "openpose"). No pose-estimation model, no onnx.

Units: the rig is built in "figure heights" (1.0 = sole of foot to top of head) and scaled to the
character's bounding box in the reference, so the skeleton lines up with the sprite it must redraw.
Image coordinates: x right, y down."""

import math

import numpy as np
from PIL import Image, ImageDraw

ANIMATIONS = ("walk", "run", "idle")

# OpenPose COCO-18 order
NOSE, NECK, R_SHO, R_ELB, R_WRI, L_SHO, L_ELB, L_WRI, R_HIP, R_KNE, R_ANK, L_HIP, L_KNE, L_ANK, \
    R_EYE, L_EYE, R_EAR, L_EAR = range(18)

LIMBS = [(NECK, R_SHO), (NECK, L_SHO), (R_SHO, R_ELB), (R_ELB, R_WRI), (L_SHO, L_ELB), (L_ELB, L_WRI),
         (NECK, R_HIP), (R_HIP, R_KNE), (R_KNE, R_ANK), (NECK, L_HIP), (L_HIP, L_KNE), (L_KNE, L_ANK),
         (NECK, NOSE), (NOSE, R_EYE), (R_EYE, R_EAR), (NOSE, L_EYE), (L_EYE, L_EAR)]

COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
          (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

# segment lengths, fractions of the figure height
THIGH, SHIN = 0.245, 0.240
TORSO = 0.300          # hip centre -> neck
UPPER_ARM, FOREARM = 0.165, 0.150
SHOULDER_DROP = 0.02   # neck -> shoulder, along the torso
ANKLE_ABOVE_SOLE = 0.04
NOSE_ABOVE_NECK = 0.10
SIDE_OFFSET = 0.014    # far-side limbs sit this far behind the near side so they stay distinguishable


def _rot(angle_deg):
    """Unit vector pointing straight down, rotated by angle (positive = swung forward, +x)."""
    a = math.radians(angle_deg)
    return np.array([math.sin(a), math.cos(a)])


def _chain(origin, a1, l1, a2, l2):
    """Two-segment limb: first segment angle a1, second a2 (both from straight down)."""
    mid = origin + _rot(a1) * l1
    return mid, mid + _rot(a2) * l2


def _pose(animation, phase):
    """Keypoints (18, 2) in figure-height units, ground-level ankle at y = 0 after snapping,
    facing +x. Pelvis starts at the origin and the whole body is shifted to the ground afterwards."""
    tau = 2 * math.pi
    if animation == "walk":
        swing, knee_max, knee_stance = 28.0, 60.0, 6.0
        arm_swing, elbow_base, elbow_extra, lean, hop = 24.0, 14.0, 20.0, 3.0, 0.0
    elif animation == "run":
        swing, knee_max, knee_stance = 44.0, 105.0, 12.0
        arm_swing, elbow_base, elbow_extra, lean, hop = 42.0, 85.0, 15.0, 11.0, 0.045
    else:  # idle: stand, breathe
        swing = knee_max = 0.0
        knee_stance, arm_swing, elbow_base, elbow_extra, lean, hop = 4.0, 0.0, 12.0, 0.0, 1.0, 0.0

    pts = np.zeros((18, 2))
    hip = np.array([0.0, 0.0])
    lean_vec = np.array([math.sin(math.radians(lean)), -math.cos(math.radians(lean))])
    breath = math.sin(tau * phase) if animation == "idle" else 0.0

    neck = hip + lean_vec * (TORSO + 0.004 * breath)
    pts[NECK] = neck
    # head: nose forward of the neck, ears behind, eyes between
    head_up = np.array([math.sin(math.radians(lean * 0.5)), -math.cos(math.radians(lean * 0.5))])
    pts[NOSE] = neck + head_up * NOSE_ABOVE_NECK + np.array([0.045, 0.0])
    pts[R_EYE] = pts[NOSE] + np.array([-0.020, -0.028])    # near eye
    pts[L_EYE] = pts[NOSE] + np.array([-0.042, -0.026])    # far eye
    pts[R_EAR] = pts[NOSE] + np.array([-0.070, -0.004])
    pts[L_EAR] = pts[NOSE] + np.array([-0.082, -0.002])

    shoulder = neck - lean_vec * SHOULDER_DROP
    sides = {  # leg/arm index sets and phase offsets: near side = "R", far side = "L"
        "R": dict(hip=R_HIP, knee=R_KNE, ank=R_ANK, sho=R_SHO, elb=R_ELB, wri=R_WRI, p=phase, dx=0.0),
        "L": dict(hip=L_HIP, knee=L_KNE, ank=L_ANK, sho=L_SHO, elb=L_ELB, wri=L_WRI, p=phase + 0.5, dx=-SIDE_OFFSET),
    }
    for s in sides.values():
        p = s["p"] % 1.0
        offset = np.array([s["dx"], 0.0])
        thigh_a = swing * math.sin(tau * p)
        swing_t = max(0.0, math.cos(tau * p))                       # 1 = mid-swing, knee most bent
        knee_flex = knee_stance + (knee_max - knee_stance) * swing_t ** 2
        pts[s["hip"]] = hip + offset
        knee, ank = _chain(hip + offset, thigh_a, THIGH, thigh_a - knee_flex, SHIN)
        pts[s["knee"]], pts[s["ank"]] = knee, ank

        # arms swing opposite to the same-side leg
        arm_a = -arm_swing * math.sin(tau * p)
        elbow_flex = elbow_base + elbow_extra * max(0.0, -math.sin(tau * p))  # more bend as the arm swings forward
        sh = shoulder + offset
        elb, wri = _chain(sh, arm_a, UPPER_ARM, arm_a + elbow_flex, FOREARM)
        pts[s["sho"]], pts[s["elb"]], pts[s["wri"]] = sh, elb, wri

    # plant the lowest foot on the ground; running adds a flight arc on top
    ground = max(pts[R_ANK][1], pts[L_ANK][1])
    pts[:, 1] -= ground
    if hop:  # two flight phases per cycle, one per leg
        pts[:, 1] -= hop * abs(math.sin(tau * phase))
    pts[:, 1] -= ANKLE_ABOVE_SOLE  # ankle sits above the sole; y is now "up is negative"
    pts[:, 0] -= pts[[NECK, R_HIP, L_HIP], 0].mean()  # keep the torso centred between frames
    return pts


def _draw_body(pts_px, size, stick=None):
    """controlnet_aux-style body draw: limb ellipses, dimmed to 60%, then joint dots."""
    w, h = size
    stick = stick or max(2, int(round(min(w, h) / 128)))  # 4 at 512px, like the reference renderer
    limb_layer = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(limb_layer)
    for i, (a, b) in enumerate(LIMBS):
        (x1, y1), (x2, y2) = pts_px[a], pts_px[b]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-3:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ang = math.atan2(y2 - y1, x2 - x1)
        t = np.linspace(0, 2 * math.pi, 32, endpoint=False)
        ex, ey = (length / 2) * np.cos(t), stick * np.sin(t)
        poly = [(cx + x * math.cos(ang) - y * math.sin(ang), cy + x * math.sin(ang) + y * math.cos(ang))
                for x, y in zip(ex, ey)]
        draw.polygon(poly, fill=COLORS[i])
    canvas = (np.asarray(limb_layer).astype(np.float32) * 0.6).astype(np.uint8)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    r = stick
    for i, (x, y) in enumerate(pts_px):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[i])
    return np.asarray(img)


def figure_bbox(rgb, threshold=24):
    """(x0, y0, x1, y1) of the non-background pixels in a reference on a plain background, or the
    middle 80% of the canvas when nothing (or everything) differs from the border colour."""
    arr = rgb[..., :3].astype(np.int32)
    h, w = arr.shape[:2]
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    mask = np.abs(arr - bg).sum(-1) > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0 or (xs.max() - xs.min()) >= w - 2 and (ys.max() - ys.min()) >= h - 2:
        return int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def render_cycle(animation, frames, size, bbox, facing="right", ground_y=None):
    """`frames` skeleton images (uint8 RGB, size = (w, h)) of one looping cycle, scaled to the figure
    bbox = (x0, y0, x1, y1): figure height = bbox height, feet on the bbox bottom, torso on the
    bbox centre. Frame i is phase i / frames, so frame 0 and frame N would be identical."""
    if animation not in ANIMATIONS:
        raise ValueError(f"animation must be one of {ANIMATIONS}, got {animation!r}")
    w, h = size
    x0, y0, x1, y1 = bbox
    fig_h = float(y1 - y0)
    cx = (x0 + x1) / 2.0
    ground = float(y1 if ground_y is None else ground_y)
    sign = 1.0 if facing == "right" else -1.0
    out = []
    for i in range(int(frames)):
        pts = _pose(animation, i / float(frames))
        px = np.empty_like(pts)
        px[:, 0] = cx + sign * pts[:, 0] * fig_h
        px[:, 1] = ground + pts[:, 1] * fig_h
        out.append(_draw_body(px, (w, h)))
    return out
