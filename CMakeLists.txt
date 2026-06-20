cmake_minimum_required(VERSION 3.8)
project(home_robot)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  home_robot/nodes/roomba_driver.py
  home_robot/nodes/object_detector.py
  home_robot/nodes/voice_control.py
  home_robot/nodes/arm_driver.py
  home_robot/nodes/wake_word_node.py
  home_robot/nodes/stt_node.py
  home_robot/nodes/llm_bridge_node.py
  home_robot/nodes/vision_node.py
  home_robot/nodes/tts_node.py
  home_robot/nodes/task_planner_node.py
  home_robot/nodes/calibrate_odom.py
  home_robot/nodes/imu_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config scripts firmware
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
