from x30_modes.mux_logic import MuxCore, ZERO


def make(mode="object_nav", **kw):
    return MuxCore(mode=mode, **kw)


def test_nav_passes_in_task_mode():
    m = make()
    m.update("nav", (0.3, 0, 0), 0.0)
    r = m.select(0.1)
    assert r.source == "nav" and r.twist == (0.3, 0, 0)


def test_teleop_beats_nav():
    m = make()
    m.update("nav", (0.3, 0, 0), 0.0)
    m.update("teleop", (0.0, 0.0, 0.5), 0.0)
    r = m.select(0.1)
    assert r.source == "teleop" and r.twist == (0.0, 0.0, 0.5) and r.teleop_active


def test_per_source_timeouts():
    m = make(timeouts={"teleop": 0.2, "nav": 0.5})
    m.update("teleop", (0.2, 0, 0), 0.0)
    m.update("nav", (0.3, 0, 0), 0.0)
    assert m.select(0.1).source == "teleop"
    r = m.select(0.3)                      # teleop expired, nav alive, but the lockout after teleop is still running
    assert r.source == "" and r.twist == ZERO
    assert m.select(0.9).source == ""      # nav timed out as well -> zero


def test_teleop_lockout_then_nav_resumes():
    m = make(teleop_lockout_s=0.5)
    m.update("teleop", (0.2, 0, 0), 0.0)
    m.select(0.05)
    m.update("teleop", (0, 0, 0), 0.1)     # joystick released
    m.update("nav", (0.3, 0, 0), 0.3)
    assert m.select(0.35).source == ""     # still locked out
    m.update("nav", (0.3, 0, 0), 0.7)
    assert m.select(0.71).source == "nav"


def test_zero_teleop_does_not_hold_priority():
    m = make(teleop_lockout_s=0.0)
    m.update("teleop", ZERO, 0.0)
    m.update("nav", (0.3, 0, 0), 0.0)
    assert m.select(0.1).source == "nav"


def test_mode_gating():
    m = make(mode="idle")
    m.update("nav", (0.3, 0, 0), 0.0)
    m.update("teleop", (0.3, 0, 0), 0.0)
    assert m.select(0.1).source == ""
    m.set_mode("teleop")
    assert m.select(0.1).source == "teleop"
    m.update("teleop", ZERO, 0.2)
    assert m.select(0.2).source == ""      # nav blocked in teleop mode
    m.set_mode("explore")
    m.update("nav", (0.3, 0, 0), 5.0)
    assert m.select(5.01).source == "nav"
    m.set_mode("bogus")
    assert m.mode == "idle"
