import threading
import time

import pytest

from x30_modes.mode_logic import ModeCore, PathLength, ProgressWatchdog


class Recorder:
    def __init__(self):
        self.events = []

    def stop(self):
        self.events.append("stop")

    def mode(self, m):
        self.events.append(("mode", m))


def make():
    r = Recorder()
    return ModeCore(r.stop, r.mode, cancel_wait_s=2.0), r


def run_task(core, handle, out):
    """Task thread: runs until cancelled, then finishes the way mode_manager does."""
    def body():
        while not handle.cancelled:
            time.sleep(0.01)
        out.append(handle.reason)
        core.end_task(handle)
    th = threading.Thread(target=body, daemon=True)
    th.start()
    return th


def test_set_mode_validates():
    core, _ = make()
    ok, msg, prev = core.set_mode("flying")
    assert not ok and "unknown mode" in msg and core.mode == "idle"
    ok, msg, prev = core.set_mode("teleop")
    assert ok and prev == "idle" and core.mode == "teleop"


def test_switch_cancels_running_task_and_zeroes():
    core, rec = make()
    h = core.begin_task("object_nav")
    out = []
    th = run_task(core, h, out)
    time.sleep(0.05)
    ok, msg, prev = core.set_mode("teleop")
    th.join(1.0)
    assert ok and prev == "object_nav" and core.mode == "teleop"      # the dying task must not overwrite the new mode
    assert out and "teleop" in out[0]
    assert h.done and h.cancelled
    assert core.task is None
    assert rec.events.index("stop") < rec.events.index(("mode", "teleop"))


def test_new_goal_preempts_old_one_only_one_task_active():
    core, rec = make()
    h1 = core.begin_task("explore")
    out1 = []
    th1 = run_task(core, h1, out1)
    h2 = core.begin_task("goal_pose")
    th1.join(1.0)
    assert h1.done and out1 == ["preempted by a new goal"]
    assert core.task is h2 and core.mode == "goal_pose"
    core.end_task(h2)
    assert core.mode == "idle" and core.task is None


def test_end_task_returns_to_idle_and_publishes():
    core, rec = make()
    h = core.begin_task("goal_pose")
    core.end_task(h)
    assert core.mode == "idle"
    assert ("mode", "goal_pose") in rec.events and rec.events[-1] == ("mode", "idle")


def test_non_task_mode_rejected_for_begin():
    core, _ = make()
    with pytest.raises(ValueError):
        core.begin_task("teleop")


def test_unresponsive_task_reported():
    core, _ = make()
    core._cancel_wait = 0.2
    core.begin_task("explore")                         # nobody finishes it
    ok, msg, _ = core.set_mode("idle")
    assert ok and "did not stop" in msg


def test_progress_watchdog():
    w = ProgressWatchdog(window_s=10.0)
    assert not w.update(0.0, (0, 0, 0.0))
    assert not w.update(5.0, (0.05, 0, 0.0))
    assert w.update(10.5, (0.05, 0.02, 0.0))           # nothing moved for > 10 s
    w.reset(11.0, (0, 0, 0))
    assert not w.update(15.0, (0.3, 0, 0))             # moved: window restarts
    assert not w.update(24.0, (0.3, 0, 0))
    assert w.update(26.0, (0.3, 0, 0))
    w.note_progress(26.5)
    assert not w.update(30.0, (0.3, 0, 0))
    w2 = ProgressWatchdog(window_s=5.0)
    w2.update(0.0, (0, 0, 0))
    assert not w2.update(6.0, (0, 0, 1.0))             # rotating counts as progress


def test_path_length():
    p = PathLength()
    for x in (0.0, 0.5, 1.0, 1.0, 1.5):
        p.update(x, 0.0)
    assert p.total == pytest.approx(1.5)
    p.update(50.0, 0.0)                                # jump ignored
    assert p.total == pytest.approx(1.5)
