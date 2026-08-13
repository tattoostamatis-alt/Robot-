#!/usr/bin/env python3
"""RTAB-Map — 3D map of the house from the D435, alongside the live stack.

Started by the dashboard's «3D Χάρτης» tab (scripts/gui_session.sh start rtabmap),
which runs this together with rtabmap_viz on VNC display :5.

‼️ TWO RULES HERE. Both were learned the hard way, the second one live.

RULE 1 — publish_tf IS FALSE, ALWAYS.

AMCL already publishes map -> odom. RTAB-Map in its normal SLAM mode publishes
the SAME TF edge, and a TF frame with two parents is the exact failure this
project has already paid for twice — the orphaned Gazebo node publishing a
second odom -> base_link made the pose oscillate 0.49 m / 14.6 degrees on a
robot that was standing still, and it cost a long debugging session chasing a
"localization bug" that was never in the localization code. So this node
subscribes, builds its graph, and publishes NO transform. Navigation, goto_room
and AMCL are untouched while it runs, and the tab is safe to open at any time.

RULE 2 — IT RUNS IN THE `rtabmap` NAMESPACE, ALWAYS.

Caught on the live robot while building this, and it is worse than it looks:
run bare, the rtabmap node publishes into the GLOBAL namespace, and the list
includes `/map` (nav_msgs/OccupancyGrid). That is map_server's topic — the one
AMCL and both costmaps read. `ros2 topic info -v /map` grew an rtabmap
publisher next to map_server within seconds of starting it, i.e. a second
writer to the live map of the house, sending its own barely-built grid.

It does not stop at /map either: bare, the node also owns /grid_prob_map,
/octomap_grid, /goal_out, /global_path and /local_path — Nav2 names, all of
them. This is the same shape as the Gazebo tab that moved the real robot's
AMCL pose, and it would have been just as invisible: nothing errors, the live
map simply acquires a rival publisher and whoever subscribes next may get
either one.

The namespace makes all of it /rtabmap/... and the collision cannot happen.
The subscriptions are remapped to absolute topics, so they still reach the real
camera, scan and odometry from inside it. Verified after the fix: map_server is
once again the only publisher of /map.

Where the 3D map ends up, and why odom_frame_id defaults to `map`:

  RTAB-Map anchors its graph at whatever pose its odometry source reports for
  the first node. Given the plain `odom` frame, that anchor is wherever the EKF
  happened to start — so the finished 3D cloud would be offset from malou2 by
  the map->odom correction at start, and would drift with the wheels.

  Feeding it the AMCL-corrected `map` pose instead makes the graph start at
  identity with respect to malou2 and stay there: the 3D map lands in the SAME
  coordinates as the 2D map, so the rooms and the taught locations line up with
  it. That is the whole reason to build this on top of an existing good map
  rather than mapping from scratch.

  The trade: AMCL corrects in discrete jumps, and a big one (a relocalization,
  or the localization watchdog re-snapping a standing robot) looks like a
  teleport to RTAB-Map. If that turns out to hurt in practice, launch with
  anchor_frame:=odom for the smooth-but-drifting source instead.

Manual run (the tab does this for you):

    ros2 launch home_robot rtabmap.launch.py
    ros2 launch home_robot rtabmap.launch.py localization:=true   # reuse a map
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    anchor      = LaunchConfiguration('anchor_frame').perform(context)
    db_path     = LaunchConfiguration('database_path').perform(context)
    use_scan    = LaunchConfiguration('use_scan').perform(context).lower() in ('true', '1')
    localization = LaunchConfiguration('localization').perform(context).lower() in ('true', '1')
    detect_rate = LaunchConfiguration('detection_rate').perform(context)
    use_viz     = LaunchConfiguration('use_viz').perform(context).lower() in ('true', '1')

    os.makedirs(os.path.dirname(os.path.expanduser(db_path)), exist_ok=True)

    # Shared by rtabmap and rtabmap_viz: both have to agree about the frames and
    # about which topics carry the RGB-D pair, or the viewer sits empty while the
    # mapper works fine (and it is not obvious which of the two is wrong).
    common = {
        'frame_id':          'base_link',
        'odom_frame_id':     anchor,
        'map_frame_id':      'rtabmap_map',   # never published, see publish_tf
        'subscribe_depth':   True,
        'subscribe_rgb':     True,
        'subscribe_scan':    use_scan,
        'subscribe_odom_info': False,
        # The D435 timestamps its colour and depth a few ms apart and the lidar
        # runs at 10 Hz against the camera's 30 — exact-time sync would simply
        # never fire.
        'approx_sync':       True,
        'sync_queue_size':   30,
        'qos_image':         1,      # RELIABLE, matching the realsense driver
        'qos_camera_info':   1,
        'qos_scan':          1,
        # AMCL's map->base_link is not always instantly available after a jump;
        # without a wait the node drops frames and logs a wall of TF warnings.
        'wait_for_transform': 0.3,
    }

    remaps = [
        ('rgb/image',       '/camera/camera/color/image_raw'),
        # ‼️ aligned_depth_to_color, NOT depth/image_rect_raw. The raw depth is in
        # the depth sensor's own frame; RTAB-Map needs depth registered pixel-for-
        # pixel to the colour image or every point lands in the wrong place. The
        # dashboard turns align_depth.enable on for the camera while this runs —
        # the lean localize.launch.py camera starts with it off.
        ('depth/image',     '/camera/camera/aligned_depth_to_color/image_raw'),
        ('rgb/camera_info', '/camera/camera/color/camera_info'),
        ('scan',            '/scan'),
        ('odom',            '/odom'),
    ]

    # RTAB-Map's own parameters (the Rtabmap/... namespace) go on the same node.
    # These are tuned for a small flat on a machine that is already loaded: the
    # mini PC sits at load ~6 with two RViz instances before this starts.
    rtab_params = {
        # 1 Hz keyframes. The default 1 Hz is already conservative, but say it
        # explicitly — this is the single biggest CPU lever here.
        'Rtabmap/DetectionRate':      detect_rate,
        'RGBD/NeighborLinkRefining':  'true',
        'RGBD/ProximityBySpace':      'true',
        'RGBD/AngularUpdate':         '0.05',   # skip keyframes while parked
        'RGBD/LinearUpdate':          '0.05',
        'Reg/Force3DoF':              'true',   # the Roomba cannot pitch or roll
        'Reg/Strategy':               '1' if use_scan else '0',   # 1 = ICP w/ scan
        'Optimizer/Slam2D':           'true',
        'Mem/STMSize':                '30',
        # 3D occupancy from the depth cloud, decimated: full resolution is 640x480
        # points per keyframe and the export becomes unusable on this machine.
        'Grid/FromDepth':             'true',
        'Grid/3D':                    'true',
        'Grid/CellSize':              '0.05',
        'Grid/RangeMax':              '4.0',    # D435 depth is noise past ~4 m
        'Grid/DepthDecimation':       '2',
        'Grid/MaxObstacleHeight':     '2.0',
        'Grid/RayTracing':            'true',
        'Vis/MinInliers':             '15',
    }

    args = ['--delete_db_on_start'] if not localization else []

    nodes = [
        Node(
            package='rtabmap_slam', executable='rtabmap', name='rtabmap',
            namespace='rtabmap',          # ‼️ RULE 2 — see the header
            output='screen',
            parameters=[common, rtab_params, {
                'database_path': db_path,
                # ‼️ NEVER set this true. See the header.
                'publish_tf':    False,
                'Mem/IncrementalMemory': 'false' if localization else 'true',
                'Mem/InitWMWithAllNodes': 'true' if localization else 'false',
            }],
            remappings=remaps,
            arguments=args,
        ),
    ]

    if use_viz:
        nodes.append(Node(
            package='rtabmap_viz', executable='rtabmap_viz', name='rtabmap_viz',
            namespace='rtabmap',          # same namespace or it sees no map data
            output='screen',
            parameters=[common, {'database_path': db_path}],
            remappings=remaps,
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'anchor_frame', default_value='odom',
            description="Which frame RTAB-Map treats as its odometry source. "
                        "'map' (default) anchors the 3D map to the existing 2D "
                        "map so they share coordinates; 'odom' is the smooth "
                        "wheel/EKF frame, drifting but jump-free."),
        DeclareLaunchArgument(
            'database_path', default_value='~/.home_robot/rtabmap/house.db',
            description='Where the map graph is stored. Reused by '
                        'localization:=true; deleted at start otherwise.'),
        DeclareLaunchArgument(
            'localization', default_value='false',
            description='Reuse the existing database read-only instead of '
                        'wiping it and mapping from scratch.'),
        DeclareLaunchArgument(
            'use_scan', default_value='true',
            description='Fuse the 360° C1 LiDAR with the camera. The D435 sees '
                        '~87° — the scan is what keeps corridors square.'),
        DeclareLaunchArgument(
            'detection_rate', default_value='1.0',
            description='Keyframes per second. The main CPU lever; raise only '
                        'if the machine is otherwise idle.'),
        DeclareLaunchArgument(
            'use_viz', default_value='true',
            description='Also start rtabmap_viz (the Qt GUI the tab streams).'),
        OpaqueFunction(function=_setup),
    ])
