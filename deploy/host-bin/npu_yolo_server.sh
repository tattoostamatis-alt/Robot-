#!/bin/bash
# Launches the NPU YOLO inference server (~/ryzenai_venv/npu_yolo_server.py).
# Must source XRT before the venv's activate so VitisAI EP finds the NPU device.
source /opt/xilinx/xrt/setup.sh
source ~/ryzenai_venv/bin/activate
exec python3 ~/ryzenai_venv/npu_yolo_server.py
