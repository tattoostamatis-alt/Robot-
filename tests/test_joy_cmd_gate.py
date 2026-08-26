from home_robot.joy_cmd_gate import JoyCommandGate


def test_idle_zeros_are_suppressed():
    gate = JoyCommandGate()
    assert not gate.should_forward([0.0] * 6)
    assert not gate.should_forward([0.0] * 6)


def test_motion_repeats_but_release_stops_exactly_once():
    gate = JoyCommandGate()
    assert gate.should_forward([0.2, 0, 0, 0, 0, 0])
    assert gate.should_forward([0.2, 0, 0, 0, 0, 0])
    assert gate.should_forward([0.0] * 6)
    assert not gate.should_forward([0.0] * 6)


def test_disconnect_watchdog_stops_only_active_motion():
    gate = JoyCommandGate()
    assert not gate.stop_if_active()
    gate.should_forward([0, 0, 0, 0, 0, 0.5])
    assert gate.stop_if_active()
    assert not gate.stop_if_active()
