"""The numpy repair decision, on the exact state seen in the Graydient logs."""

import importlib.util
import os
import sys
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location("prestartup", os.path.join(ROOT, "prestartup_script.py"))
    mod = importlib.util.module_from_spec(spec)
    with mock.patch("importlib.metadata.version", side_effect=Exception("skip main")):
        spec.loader.exec_module(mod)  # main() returns early: metadata lookup "fails"
    return mod


SCIPY_NEEDS_2 = ["numpy<2.8,>=2.0.0", "pytest; extra == \"test\""]
SCIPY_OLD = ["numpy<2.3,>=1.23.5"]


def test_decision():
    m = load()
    assert m.needs_fix("1.26.4", SCIPY_NEEDS_2)          # the broken Graydient state
    assert not m.needs_fix("2.2.6", SCIPY_NEEDS_2)       # healthy
    assert not m.needs_fix("1.26.4", SCIPY_OLD)          # old scipy is happy with numpy 1.x
    assert not m.needs_fix("1.26.4", None)
    assert m.pin_spec(SCIPY_NEEDS_2) == "numpy>=2.0.0,<2.8"


def test_main_runs_pip_only_when_broken():
    m = load()
    calls = []
    fake = mock.Mock(returncode=0, stdout="Successfully installed numpy-2.2.6", stderr="")
    with mock.patch("importlib.metadata.version", return_value="1.26.4"), \
         mock.patch("importlib.metadata.requires", return_value=SCIPY_NEEDS_2), \
         mock.patch.dict(sys.modules), \
         mock.patch("subprocess.run", side_effect=lambda *a, **k: calls.append(a[0]) or fake):
        sys.modules.pop("numpy", None)
        m.main()
    assert calls and calls[0][-1] == "numpy>=2.0.0,<2.8" and "--no-deps" in calls[0]
    calls.clear()
    with mock.patch("importlib.metadata.version", return_value="2.2.6"), \
         mock.patch("importlib.metadata.requires", return_value=SCIPY_NEEDS_2), \
         mock.patch.dict(sys.modules), \
         mock.patch("subprocess.run", side_effect=lambda *a, **k: calls.append(a[0]) or fake):
        sys.modules.pop("numpy", None)
        m.main()
    assert not calls


def test_stale_numpy_is_unloaded():
    m = load()
    fake = mock.Mock(returncode=0, stdout="Successfully installed numpy-2.2.6", stderr="")
    with mock.patch("importlib.metadata.version", return_value="1.26.4"), \
         mock.patch("importlib.metadata.requires", return_value=SCIPY_NEEDS_2), \
         mock.patch.dict(sys.modules, {"numpy": object(), "numpy.core": object()}), \
         mock.patch("subprocess.run", return_value=fake):
        m.main()
        assert "numpy" not in sys.modules and "numpy.core" not in sys.modules


if __name__ == "__main__":
    test_decision()
    print("ok test_decision")
    test_main_runs_pip_only_when_broken()
    print("ok test_main_runs_pip_only_when_broken")
    test_stale_numpy_is_unloaded()
    print("ok test_stale_numpy_is_unloaded")
