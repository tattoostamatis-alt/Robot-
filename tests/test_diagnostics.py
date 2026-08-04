"""Unit tests for self-diagnosis (home_robot/diagnostics.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.diagnostics import CHECKS, Health, diagnose  # noqa: E402

ALIVE = {
    '/camera/camera/color/image_raw': 30.0,
    '/camera/camera/aligned_depth_to_color/image_raw': 30.0,
    '/scan': 10.0,
    '/odom': 20.0,
    '/imu/data': 100.0,
    '/mic/audio': 10.0,
}
NODES = {'camera', 'roomba_driver', 'imu_node', 'wake_word_node', 'amcl'}


def healthy(**over):
    h = Health(topic_hz=dict(ALIVE), nodes=set(NODES), load1=4.0, cores=16)
    for k, v in over.items():
        setattr(h, k, v)
    return h


def keys(findings):
    return {f.key for f in findings}


# ── a healthy robot is quiet ──────────────────────────────────────────────────

def test_healthy_robot_reports_nothing():
    assert diagnose(healthy()) == []


def test_partial_snapshot_does_not_crash():
    """A check that raises must not take the other checks down with it."""
    assert isinstance(diagnose(Health()), list)


# ── the silent failures, one at a time ────────────────────────────────────────

def test_camera_dead_while_node_alive():
    """The node runs, the topic is advertised, no frames come out — and the
    preflight still prints a tick."""
    h = healthy()
    h.topic_hz['/camera/camera/color/image_raw'] = 0.0
    assert 'camera_dead' in keys(diagnose(h))


def test_camera_not_flagged_when_the_node_is_not_running():
    h = healthy(nodes=NODES - {'camera'})
    h.topic_hz['/camera/camera/color/image_raw'] = 0.0
    assert 'camera_dead' not in keys(diagnose(h))


def test_depth_not_aligned_is_caught_separately():
    """Colour flows, depth does not: every 3D consumer silently returns []."""
    h = healthy()
    h.topic_hz['/camera/camera/aligned_depth_to_color/image_raw'] = 0.0
    found = keys(diagnose(h))
    assert 'depth_not_aligned' in found
    assert 'camera_dead' not in found        # colour is fine, so not this one


def test_base_asleep():
    h = healthy()
    h.topic_hz['/odom'] = 0.0
    assert 'base_asleep' in keys(diagnose(h))


def test_base_asleep_names_the_power_cycle():
    h = healthy()
    h.topic_hz['/odom'] = 0.0
    fix = [f.fix for f in diagnose(h) if f.key == 'base_asleep'][0]
    assert 'Power cycle' in fix or 'power cycle' in fix


def test_lidar_silent():
    h = healthy()
    h.topic_hz['/scan'] = 0.0
    assert 'lidar_silent' in keys(diagnose(h))


def test_imu_silent():
    h = healthy()
    h.topic_hz['/imu/data'] = 0.0
    assert 'imu_silent' in keys(diagnose(h))


def test_mic_dead():
    h = healthy()
    h.topic_hz['/mic/audio'] = 0.0
    assert 'mic_dead' in keys(diagnose(h))


def test_stt_stuck_from_the_log():
    """The busy watchdog's own line, which only prints when a transcription
    never came back — the actual deaf-robot case."""
    h = healthy(log_counts={'Transcription has not returned': 1})
    assert 'stt_stuck' in keys(diagnose(h))


def test_a_repeated_wake_word_is_not_a_stuck_stt():
    """‼️ 2026-08-04: this check fired on 'Already transcribing', which is what
    a second wake word inside the 5 s listening window prints. The robot was
    interrupting itself all evening, printed it 38 times, transcribed
    everything correctly — and was announced as ΚΟΥΦΟ, at CRITICAL, out loud.
    A false critical sends the owner to restart the one node that was fine."""
    h = healthy(log_counts={'Already transcribing': 38})
    assert 'stt_stuck' not in keys(diagnose(h))


def test_tts_flaky():
    h = healthy(log_counts={'TTS attempt': 5})
    assert 'tts_flaky' in keys(diagnose(h))


def test_serial_lost():
    h = healthy(log_counts={'Errno 110': 1})
    assert 'serial_lost' in keys(diagnose(h))


# ── resources ─────────────────────────────────────────────────────────────────

def test_allocated_swap_alone_is_not_a_finding():
    """‼️ Swap being USED is normal here and costs nothing. Only paging does.
    The node passes the si/so rate, not the used percentage."""
    assert 'swap_thrashing' not in keys(diagnose(healthy(swap_pct=0.0)))


def test_active_paging_is_a_finding():
    assert 'swap_thrashing' in keys(diagnose(healthy(swap_pct=5.0)))


def test_overload_relative_to_cores():
    assert 'overloaded' in keys(diagnose(healthy(load1=40.0, cores=16)))
    assert 'overloaded' not in keys(diagnose(healthy(load1=20.0, cores=16)))


# ── presentation contract ─────────────────────────────────────────────────────

def test_findings_are_sorted_most_severe_first():
    h = healthy(load1=40.0)                       # info
    h.topic_hz['/scan'] = 0.0                     # critical
    severities = [f.severity for f in diagnose(h)]
    assert severities == sorted(severities,
                                key=lambda s: {'critical': 0, 'warning': 1,
                                               'info': 2}[s])


def test_every_check_has_a_title_detail_and_fix():
    """A diagnostic without a next action is worth nothing — the operator
    already knew something was wrong."""
    for key, _pred, severity, title, detail, fix in CHECKS:
        assert key and title and detail and fix, key
        assert severity in ('critical', 'warning', 'info'), key
        assert len(fix) > 10, f'{key} has no real action'


def test_check_keys_are_unique():
    ks = [c[0] for c in CHECKS]
    assert len(ks) == len(set(ks))


def test_finding_serialises():
    h = healthy()
    h.topic_hz['/scan'] = 0.0
    d = diagnose(h)[0].to_dict()
    assert set(d) == {'key', 'severity', 'title', 'detail', 'fix'}


def test_several_failures_are_all_reported():
    h = healthy()
    h.topic_hz['/scan'] = 0.0
    h.topic_hz['/odom'] = 0.0
    h.topic_hz['/imu/data'] = 0.0
    assert {'lidar_silent', 'base_asleep', 'imu_silent'} <= keys(diagnose(h))


def test_the_node_does_not_publish_on_the_ros_diagnostics_topic():
    """‼️ '/diagnostics' is the ROS convention (diagnostic_msgs/DiagnosticArray)
    and Nav2 already publishes there. A std_msgs/String on it makes the topic
    multi-type, and `ros2 topic echo /diagnostics` then shows neither."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / 'home_robot' /
           'nodes' / 'self_diagnosis_node.py').read_text()
    assert "create_publisher(String, 'self_diagnosis'" in src
    assert "create_publisher(String, 'diagnostics'" not in src
