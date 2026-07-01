#!/usr/bin/env python3
"""Publish the camera + lidar mast as MoveIt collision objects.

The RoArm-M3 is mounted at the robot centre; ~150mm in front of it (and just
above/below the arm base) sit the D435 camera and the SLAMTEC C1 lidar. Without
telling MoveIt about them, a dragged plan can swing the arm straight into them.

This node adds two static BOX collision objects to the planning scene so the
planner routes around the sensor mast. Coordinates are given in the arm's
planning frame `world` (= the arm mount point = robot base_link + 0.17 m up):

    robot base_link           world (arm mount)
    lidar  (0.15, 0, 0.22)  -> (0.15, 0,  0.05)
    camera (0.15, 0, 0.15)  -> (0.15, 0, -0.02)

Published as a PlanningScene diff on a timer so it survives a move_group
(re)start.
"""

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

PLANNING_FRAME = "world"

# (id, size xyz [m], position xyz [m] in PLANNING_FRAME). Sizes include a small
# safety margin over the real parts. `world` sits 0.17 m above the ground (the
# arm mount), so ground is at world z=-0.17.
#   robot_body: the Roomba disc, ground (z=-0.17) up to its ~90mm top (z=-0.08).
#     Left with a gap below the arm base (world z=0) so the arm's own base_link
#     is never reported in collision (which would block all planning).
OBSTACLES = [
    ("lidar_c1",    (0.09, 0.09, 0.07), (0.15, 0.0,  0.05)),
    ("camera_d435", (0.05, 0.13, 0.05), (0.15, 0.0, -0.02)),
    ("robot_body",  (0.36, 0.36, 0.09), (0.0,  0.0, -0.125)),
]


def _box(obj_id, size, position):
    co = CollisionObject()
    co.header.frame_id = PLANNING_FRAME
    co.id = obj_id
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(s) for s in size]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [float(p) for p in position]
    pose.orientation.w = 1.0
    co.primitives.append(prim)
    co.primitive_poses.append(pose)
    co.operation = CollisionObject.ADD
    return co


class ArmSceneObstacles(Node):
    def __init__(self):
        super().__init__('arm_scene_obstacles')
        self.pub = self.create_publisher(PlanningScene, 'planning_scene', 10)
        self._objs = [_box(*o) for o in OBSTACLES]
        self.create_timer(3.0, self._publish)
        self.get_logger().info(
            f'publishing {len(self._objs)} collision objects (lidar, camera) '
            f'in frame "{PLANNING_FRAME}"')

    def _publish(self):
        scene = PlanningScene()
        scene.is_diff = True
        for co in self._objs:
            co.header.stamp = self.get_clock().now().to_msg()
            scene.world.collision_objects.append(co)
        self.pub.publish(scene)


def main():
    rclpy.init()
    node = ArmSceneObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
