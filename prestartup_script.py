"""Repair a numpy/SciPy mismatch before ComfyUI imports SciPy.

Some hosted environments (seen on Graydient, 2026-09) come up with numpy 1.26 next to a SciPy that
requires numpy >= 2. ComfyUI core then dies at `import scipy.signal` (AttributeError: np.long)
before any custom node loads. ComfyUI runs every custom node's prestartup_script.py *before* that
import, in the same interpreter, so this is the one place a node pack can fix it.

It only acts when both are true, reading package metadata WITHOUT importing numpy:
  * the installed numpy is < 2, and
  * the installed SciPy declares that it needs numpy >= 2.
Then it installs numpy 2.x within SciPy's declared range. Otherwise it does nothing.
"""

import re
import subprocess
import sys


def _major_minor(version):
    m = re.match(r"(\d+)\.(\d+)", version or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def scipy_numpy_range(requires):
    """(min, max) numpy spec strings from SciPy's Requires-Dist, e.g. ('2.0.0', '2.8')."""
    for req in requires or []:
        if not re.match(r"^\s*numpy\b", req) or "extra ==" in req:
            continue
        lo = re.search(r">=\s*([\d.]+)", req)
        hi = re.search(r"<\s*([\d.]+)", req)
        return (lo.group(1) if lo else None, hi.group(1) if hi else None)
    return (None, None)


def needs_fix(numpy_version, scipy_requires):
    have = _major_minor(numpy_version)
    lo, _ = scipy_numpy_range(scipy_requires)
    need = _major_minor(lo)
    return bool(have and need and have < need)


def pin_spec(scipy_requires):
    lo, hi = scipy_numpy_range(scipy_requires)
    spec = f"numpy>={lo}"
    if hi:
        spec += f",<{hi}"
    return spec


def main():
    try:
        from importlib import metadata
        numpy_version = metadata.version("numpy")
        scipy_requires = metadata.requires("scipy")
    except Exception:
        return  # numpy or scipy not installed: nothing to repair
    if not needs_fix(numpy_version, scipy_requires):
        return
    spec = pin_spec(scipy_requires)
    print(f"[SpriteSheetPack] numpy {numpy_version} is below what SciPy needs -- installing {spec}")
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", spec],
                             capture_output=True, text=True, timeout=240)
        tail = (out.stdout or out.stderr).strip().splitlines()[-1:] or [""]
        print(f"[SpriteSheetPack] pip exit {out.returncode}: {tail[0]}")
    except Exception as e:  # never block ComfyUI's startup on the repair itself
        print(f"[SpriteSheetPack] numpy repair failed: {e}")
        return
    # ComfyUI imports a few packages (possibly torch -> numpy) before prestartup scripts run. If the
    # old numpy is already loaded, drop it so the upcoming `import scipy` loads the new one.
    stale = [m for m in sys.modules if m == "numpy" or m.startswith("numpy.")]
    if out.returncode == 0 and stale:
        for m in stale:
            del sys.modules[m]
        print(f"[SpriteSheetPack] unloaded the old numpy ({len(stale)} modules) so it re-imports as 2.x")


main()
