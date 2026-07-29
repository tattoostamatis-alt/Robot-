#!/usr/bin/env python3
"""How far away is the person the camera is looking at?

    cd ~/robot_ws \"""How far away is the person the camera is looking at?\"""How far away is the person the camera is looking at? source install/setup.bash
    python3 src/home_robot/scripts/measure_person_distance.py

Needs the camera running (bringup use_camera:=true, which localize implies).
Measured 2026-07-29: two people at 1.66 m and 1.48 m, 144 valid depth samples
each, against a tape-free eyeball check of "about a metre and a half".

Grabs one synchronized colour+depth pair, finds people with YOLO (pose model,
so ankles and knees come out as keypoints), and reads the depth at those points.

‼️ align_depth is off on this robot, so a colour pixel is NOT the same pixel in
the depth image: colour is 640x480, depth 848x480, different intrinsics. Each
colour point is therefore unprojected to a bearing with the colour intrinsics
and re-projected with the depth ones. The depth/colour baseline (~15 mm on a
D435) is ignored — at half a metre that is under one pixel.

Depth is 16UC1 in millimetres, and 0 means "no return" (too close, too shiny,
or shadowed), so zeros must be dropped before averaging rather than pulled into
the median as a fake 0 m reading.
"""
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

MODEL = '/home/dimi/robot_ws/src/home_robot/home_robot/nodes/yolo11n-pose.pt'
# COCO pose keypoint indices
KP = {'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16}
PATCH = 4          # half-size of the depth window sampled around a point, px


def _img_to_np(msg, channels, dtype):
    buf = np.frombuffer(msg.data, dtype=dtype)
    return buf.reshape(msg.height, msg.width, channels) if channels > 1 \
        else buf.reshape(msg.height, msg.width)


class Grab(Node):
    def __init__(self):
        super().__init__('measure_legs')
        self.color = self.depth = self.cinfo = self.dinfo = None
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 lambda m: setattr(self, 'color', m), 1)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw',
                                 lambda m: setattr(self, 'depth', m), 1)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 lambda m: setattr(self, 'cinfo', m), 1)
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info',
                                 lambda m: setattr(self, 'dinfo', m), 1)

    def ready(self):
        return all(x is not None for x in
                   (self.color, self.depth, self.cinfo, self.dinfo))


def color_px_to_depth_px(u, v, cinfo, dinfo):
    """Colour pixel -> depth pixel, via the shared bearing."""
    cfx, cfy, ccx, ccy = cinfo.k[0], cinfo.k[4], cinfo.k[2], cinfo.k[5]
    dfx, dfy, dcx, dcy = dinfo.k[0], dinfo.k[4], dinfo.k[2], dinfo.k[5]
    x = (u - ccx) / cfx          # tan of horizontal bearing
    y = (v - ccy) / cfy
    return x * dfx + dcx, y * dfy + dcy


def depth_at(depth_m, du, dv):
    """Median valid depth [m] in a small patch, or None if nothing returned."""
    h, w = depth_m.shape
    du, dv = int(round(du)), int(round(dv))
    if not (0 <= du < w and 0 <= dv < h):
        return None
    patch = depth_m[max(0, dv - PATCH):dv + PATCH + 1,
                    max(0, du - PATCH):du + PATCH + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size else None


def main():
    rclpy.init()
    node = Grab()
    for _ in range(300):                      # ~6 s to collect one of each
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.ready():
            break
    if not node.ready():
        missing = [n for n, v in (('color', node.color), ('depth', node.depth),
                                  ('color_info', node.cinfo),
                                  ('depth_info', node.dinfo)) if v is None]
        print(f'FAIL: no data on {", ".join(missing)}')
        return 1

    rgb = _img_to_np(node.color, 3, np.uint8)
    if node.color.encoding == 'bgr8':
        rgb = rgb[:, :, ::-1]
    depth_raw = _img_to_np(node.depth, 1, np.uint16)
    print(f'colour {node.color.width}x{node.color.height} {node.color.encoding}, '
          f'depth {node.depth.width}x{node.depth.height} {node.depth.encoding}')

    from ultralytics import YOLO
    res = YOLO(MODEL)(rgb[:, :, ::-1], verbose=False)[0]

    if res.boxes is None or len(res.boxes) == 0:
        print('No person detected in the colour image.')
        # Still useful: what is the nearest thing straight ahead?
        cx, cy = node.depth.width // 2, node.depth.height // 2
        d = depth_at(depth_raw, cx, cy)
        print(f'Depth at image centre: {d:.2f} m' if d else
              'No depth return at image centre either.')
        return 0

    for i, box in enumerate(res.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        print(f'\nPerson {i + 1}  (confidence {conf:.2f})  '
              f'bbox x {x1:.0f}-{x2:.0f}, y {y1:.0f}-{y2:.0f}')

        # Whole-body distance: median over the box, sampled on a grid.
        samples = []
        for v in np.linspace(y1, y2, 12):
            for u in np.linspace(x1, x2, 12):
                du, dv = color_px_to_depth_px(u, v, node.cinfo, node.dinfo)
                d = depth_at(depth_raw, du, dv)
                if d and 0.1 < d < 10.0:
                    samples.append(d)
        if samples:
            arr = np.array(samples)
            print(f'  distance: {np.median(arr):.2f} m   '
                  f'(nearest {arr.min():.2f}, furthest {arr.max():.2f}, '
                  f'{len(arr)} valid depth samples)')
        else:
            print('  distance: no valid depth in the box')

        if res.keypoints is None or i >= len(res.keypoints):
            continue
        kp = res.keypoints[i]
        xy = kp.xy[0].tolist()
        cf = kp.conf[0].tolist() if kp.conf is not None else [1.0] * len(xy)
        for name, idx in KP.items():
            if idx >= len(xy) or cf[idx] < 0.3:
                continue
            u, v = xy[idx]
            du, dv = color_px_to_depth_px(u, v, node.cinfo, node.dinfo)
            d = depth_at(depth_raw, du, dv)
            lateral = ''
            if d:
                # Sideways offset in the camera frame: + is image-right.
                x = (u - node.cinfo.k[2]) / node.cinfo.k[0] * d
                lateral = f', {abs(x) * 100:.0f} cm to the ' \
                          f'{"right" if x > 0 else "left"}'
            print(f'  {name:12s} {"%.2f m" % d if d else "no depth":>9s}{lateral}')

    node.destroy_node()
    rclpy.try_shutdown()
    return 0


sys.exit(main())
