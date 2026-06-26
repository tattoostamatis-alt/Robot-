#!/usr/bin/env python3
"""
tracker_node.py — SORT multi-object tracker layered on top of object_detector.py.

Subscribes:  detected_objects (std_msgs/String, JSON list from object_detector.py)
Publishes:   tracked_objects  (std_msgs/String, same JSON + track_id/track_age/vel_x/vel_y)

Tracks survive temporary occlusions for up to `max_lost_frames` frames. Each class
is tracked independently (a 'cup' detection can only extend a 'cup' track), preventing
cross-class ID swaps when unrelated objects overlap in the image.
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

_NEXT_ID = [1]


def _iou_batch(bboxes_a: list, bboxes_b: list) -> np.ndarray:
    """Vectorized IoU: (N,4) × (M,4) → (N,M) matrix."""
    a = np.array(bboxes_a, dtype=float)   # N×4
    b = np.array(bboxes_b, dtype=float)   # M×4
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class KalmanTrack:
    """Single-object SORT Kalman filter.

    State: [cx, cy, s, r, vcx, vcy, vs]
      cx, cy = bounding-box centre
      s      = scale (area = w*h)
      r      = aspect ratio (w/h), held constant
      vcx, vcy, vs = corresponding velocities
    """

    def __init__(self, bbox: list, label: str, det: dict):
        self.id    = _NEXT_ID[0]; _NEXT_ID[0] += 1
        self.label = label
        self.det   = det       # latest detection dict (3D pos, clutter, …)
        self.age   = 0
        self.hits  = 1
        self.lost  = 0

        # Transition matrix (constant-velocity model)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.

        # Observation matrix (measure cx, cy, s, r)
        self.H = np.zeros((4, 7))
        np.fill_diagonal(self.H[:, :4], 1.)

        self.Q = np.diag([1., 1., 10., 10., 0.01, 0.01, 0.0001])
        self.R = np.diag([1., 1., 10., 10.])
        # High initial uncertainty on velocity
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4])

        z = self._to_z(bbox)
        self.x = np.array([*z, 0., 0., 0.])

    # ── Kalman ──────────────────────────────────────────────────────

    def predict(self) -> list:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self._to_bbox()

    def update(self, bbox: list, det: dict):
        self.lost  = 0
        self.hits += 1
        self.det   = det
        z = self._to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_z(bbox: list) -> np.ndarray:
        cx = (bbox[0] + bbox[2]) / 2.
        cy = (bbox[1] + bbox[3]) / 2.
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        return np.array([cx, cy, w * h, w / (h + 1e-6)])

    def _to_bbox(self) -> list:
        cx, cy, s, r = self.x[:4]
        w = float(np.sqrt(max(s * r, 0.)))
        h = float(abs(s) / (w + 1e-6))
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    @property
    def vel_x(self) -> float: return float(self.x[4])

    @property
    def vel_y(self) -> float: return float(self.x[5])


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.declare_parameter('iou_threshold',   0.3)
        self.declare_parameter('max_lost_frames', 3)
        self.declare_parameter('min_hits',        1)   # frames before ID is published

        self._iou_thr  = self.get_parameter('iou_threshold').value
        self._max_lost = self.get_parameter('max_lost_frames').value
        self._min_hits = self.get_parameter('min_hits').value
        self._tracks: list[KalmanTrack] = []

        self.create_subscription(String, 'detected_objects', self._cb, 10)
        self._pub = self.create_publisher(String, 'tracked_objects', 10)
        self.get_logger().info('Tracker ready (SORT, IoU-threshold=%.2f)' % self._iou_thr)

    def _cb(self, msg: String):
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not detections:
            for t in self._tracks:
                t.lost += 1
            self._tracks = [t for t in self._tracks if t.lost <= self._max_lost]
            self._pub.publish(String(data='[]'))
            return

        # ── Predict all existing tracks ──────────────────────────────
        predicted = {t.id: t.predict() for t in self._tracks}

        # ── Build IoU matrix (N tracks × M detections) ──────────────
        # Cross-class pairs are forced to 0 so the Hungarian assignment
        # never matches a 'cup' track to a 'person' detection.
        pred_bboxes = [predicted[t.id] for t in self._tracks]
        det_bboxes  = [[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections]
        iou_mat = _iou_batch(pred_bboxes, det_bboxes)

        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                if det['label'] != track.label:
                    iou_mat[ti, di] = 0.

        # ── Hungarian assignment ─────────────────────────────────────
        row_ind, col_ind = linear_sum_assignment(1. - iou_mat)

        matched_tracks = set()
        matched_dets   = set()
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self._iou_thr:
                d = detections[c]
                self._tracks[r].update([d['x1'], d['y1'], d['x2'], d['y2']], d)
                matched_tracks.add(r)
                matched_dets.add(c)

        # ── Increment lost counter for unmatched tracks ──────────────
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.lost += 1

        # ── Spawn new tracks for unmatched detections ────────────────
        for di, det in enumerate(detections):
            if di not in matched_dets:
                self._tracks.append(
                    KalmanTrack([det['x1'], det['y1'], det['x2'], det['y2']],
                                det['label'], det)
                )

        # ── Cull dead tracks ─────────────────────────────────────────
        self._tracks = [t for t in self._tracks if t.lost <= self._max_lost]

        # ── Publish confirmed tracks ─────────────────────────────────
        output = []
        for t in self._tracks:
            if t.hits >= self._min_hits:
                obj = dict(t.det)
                obj['track_id']   = t.id
                obj['track_age']  = t.age
                obj['track_hits'] = t.hits
                obj['vel_x']      = round(t.vel_x, 2)
                obj['vel_y']      = round(t.vel_y, 2)
                output.append(obj)

        self._pub.publish(String(data=json.dumps(output)))


def main():
    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
