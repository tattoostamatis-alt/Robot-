#!/usr/bin/env python3
"""Grasp geometry from a segmentation mask + depth — antipodal axis and width.

Given an instance mask, an aligned depth image (in METRES), and the camera
intrinsics, estimate where a parallel-jaw gripper should close: the grasp
centre, the grasp axis (the line the two fingers straddle) and the opening
width. object_detector.py feeds it yolo11n-seg masks; pick_place_node.py turns
the returned contacts into a wrist-roll angle and a gripper opening.

The core math (PCA / ellipse fit on the mask to recover the minor axis, plus
the depth-profile edge trace) is ported from Hello Robot's stretch4_grasping_demo
(grasp_estimators.py, Apache-2.0). See NOTICE. Adapted here to:
  * take depth already in METRES (our RealSense pipeline pre-divides by 1000),
    so the /1000 conversions from the original are removed;
  * depend on a self-contained PinholeCamera instead of Stretch's
    stretch4_gripper_modeling_and_control.gripper_camera module.

Everything is in the camera optical frame (x right, y down, z forward), matching
object_detector.py. pick_place_node.py transforms the resulting 3D contacts into
arm_base via the same tf2 path it already uses for the object centroid.
"""

import numpy as np
import cv2


# When the mask is close to circular (minor/major axis ratio above this) there
# is no meaningful grasp orientation, so we fall back to a horizontal grasp.
# 0.85 is Stretch's tuned CIRCULAR_MASK_AXIS_RATIO_THRESHOLD.
CIRCULAR_MASK_AXIS_RATIO_THRESHOLD = 0.85


class PinholeCamera:
    """Minimal pinhole projection in the optical frame (x right, y down,
    z forward), matching object_detector.py's inline math. Replaces Stretch's
    gripper_camera helper so nothing here depends on Stretch hardware."""

    def __init__(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def pixel_to_3d(self, pixel, depth_m):
        u, v = pixel
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return np.array([x, y, depth_m], dtype=np.float64)

    def pixel_from_3d(self, point_3d):
        x, y, z = point_3d
        if abs(z) < 1e-6:
            z = 1e-6
        u = x * self.fx / z + self.cx
        v = y * self.fy / z + self.cy
        return np.array([u, v], dtype=np.float64)


class GraspTarget:
    """A parallel-jaw grasp: centre, normalised axis the fingers straddle,
    the two fingertip contact points, and the opening width (metres)."""

    def __init__(self, grasp_width, grasp_axis_3d, center_3d,
                 left_contact_3d, right_contact_3d):
        self.grasp_width = grasp_width
        self.grasp_axis_3d = grasp_axis_3d
        self.center_3d = center_3d
        self.left_contact_3d = left_contact_3d
        self.right_contact_3d = right_contact_3d


class EllipsoidGraspEstimator:
    """Fit the mask to an ellipse (2D) or ellipsoid (3D PCA) and grasp across
    its minor axis. use_3d=True is more accurate for tilted objects but scans
    every mask pixel; use_3d=False (the demo default) is fast and needs only
    the centroid depth."""

    def __init__(self, cam: PinholeCamera, use_3d=False,
                 circular_ratio=CIRCULAR_MASK_AXIS_RATIO_THRESHOLD):
        self.cam = cam
        self.use_3d = use_3d
        self.circular_ratio = circular_ratio

    def estimate(self, mask, depth_image_m, object_center_3d) -> GraspTarget:
        if mask is None or object_center_3d is None:
            return None

        mask_2d = mask.squeeze()
        mask_8u = (mask_2d > 0).astype(np.uint8)

        if self.use_3d:
            return self._estimate_3d(mask_8u, depth_image_m, object_center_3d)
        return self._estimate_2d(mask_8u, object_center_3d)

    def _estimate_3d(self, mask_8u, depth_image_m, object_center_3d) -> GraspTarget:
        if depth_image_m is None:
            return None

        ys, xs = np.where(mask_8u > 0)
        if len(xs) < 10:
            return None

        depths = depth_image_m[ys, xs]
        valid = depths > 0.0
        xs, ys, depths = xs[valid], ys[valid], depths[valid]
        if len(xs) < 10:
            return None

        pts_3d = np.array([self.cam.pixel_to_3d((xs[i], ys[i]), depths[i])
                           for i in range(len(xs))])

        # PCA: the minor principal axis in the image plane is the grasp axis.
        mean_pt = np.mean(pts_3d, axis=0)
        centered = pts_3d - mean_pt
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        minor_axis_3d = eigvecs[:, 1]
        if minor_axis_3d[0] < 0:            # consistent X convention
            minor_axis_3d = -minor_axis_3d

        # Near-circular cross-section → no meaningful orientation → horizontal.
        if eigvals[0] > 0 and (np.sqrt(max(0, eigvals[1])) /
                               np.sqrt(max(0, eigvals[0]))) > self.circular_ratio:
            minor_axis_3d = np.array([1.0, 0.0, 0.0])

        projections = np.dot(centered, minor_axis_3d)
        min_proj, max_proj = np.min(projections), np.max(projections)
        left_contact = object_center_3d + minor_axis_3d * min_proj
        right_contact = object_center_3d + minor_axis_3d * max_proj
        grasp_width = max_proj - min_proj

        return GraspTarget(grasp_width, minor_axis_3d, object_center_3d,
                           left_contact, right_contact)

    def _estimate_2d(self, mask_8u, object_center_3d) -> GraspTarget:
        contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            return None

        (_, _), (MA, ma), angle = cv2.fitEllipseAMS(largest)  # MA=minor, ma=major
        if ma > 0 and (MA / ma) > self.circular_ratio:
            angle = 0.0

        # OpenCV's angle is the major axis from vertical; the minor axis is +90deg.
        minor_angle = np.deg2rad(angle) + np.pi / 2.0
        dir_x, dir_y = np.sin(minor_angle), -np.cos(minor_angle)
        half_minor = MA / 2.0

        cx, cy = self.cam.pixel_from_3d(object_center_3d)
        depth = object_center_3d[2]
        pt1_3d = self.cam.pixel_to_3d((cx + dir_x * half_minor,
                                       cy + dir_y * half_minor), depth)
        pt2_3d = self.cam.pixel_to_3d((cx - dir_x * half_minor,
                                       cy - dir_y * half_minor), depth)

        axis_3d = pt1_3d - pt2_3d
        width = np.linalg.norm(axis_3d)
        if width < 1e-4:
            return None
        axis_3d = axis_3d / width

        return GraspTarget(width, axis_3d, object_center_3d, pt2_3d, pt1_3d)


class DepthProfileGraspEstimator:
    """Find the 2D minor axis, then raycast outward from the centre along it,
    reading true per-pixel depth at each mask edge. More faithful contact
    depths than the flat-depth 2D ellipse for objects with curvature."""

    def __init__(self, cam: PinholeCamera):
        self.cam = cam
        self.base = EllipsoidGraspEstimator(cam, use_3d=False)

    def estimate(self, mask, depth_image_m, object_center_3d) -> GraspTarget:
        base = self.base.estimate(mask, depth_image_m, object_center_3d)
        if base is None or depth_image_m is None:
            return None

        mask_2d = mask.squeeze()
        h, w = mask_2d.shape
        cx, cy = self.cam.pixel_from_3d(object_center_3d)

        p1 = self.cam.pixel_from_3d(base.left_contact_3d)
        p2 = self.cam.pixel_from_3d(base.right_contact_3d)
        dir_vec = p2 - p1
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-4:
            return base
        dir_2d = dir_vec / norm

        def find_edge_3d(dx, dy, max_steps=400):
            x, y = cx, cy
            last_z = object_center_3d[2]
            last_x, last_y = cx, cy
            for _ in range(max_steps):
                x += dx
                y += dy
                ix, iy = int(round(x)), int(round(y))
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    break
                if mask_2d[iy, ix] == 0:
                    break
                z = depth_image_m[iy, ix]
                if z > 0.0:
                    last_z = z
                last_x, last_y = ix, iy
            return self.cam.pixel_to_3d((last_x, last_y), last_z)

        left_3d = find_edge_3d(-dir_2d[0], -dir_2d[1])
        right_3d = find_edge_3d(dir_2d[0], dir_2d[1])

        grasp_axis = right_3d - left_3d
        width = np.linalg.norm(grasp_axis)
        grasp_axis = grasp_axis / width if width > 1e-4 else base.grasp_axis_3d

        return GraspTarget(width, grasp_axis, object_center_3d, left_3d, right_3d)
