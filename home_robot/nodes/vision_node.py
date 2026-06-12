#!/usr/bin/env python3
"""Vision node — Qwen2.5-VL (via ollama), answers questions about the latest
camera frame.

Subscribes to the RealSense color image (caches the latest frame) and to
`vision/query` (std_msgs/String, a question in Greek). Sends the cached
frame + question to a vision-language model via ollama and publishes the
answer on `vision/answer` (std_msgs/String).

Used by llm_bridge_node.py's `look` tool ("ask Max what he sees").
"""

import threading

import cv2
import ollama
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


# Wrapped in English with explicit "only what's visible" / "short" framing —
# qwen2.5vl:3b's Greek-language scene descriptions were noticeably more
# accurate and consistent with this framing than a Greek-only prompt
# (verified 2026-06-12: Greek-only prompt hallucinated unrelated objects/
# colors; this version correctly described the actual scene twice in a row).
PROMPT_TEMPLATE = (
    "Look at this image from a home robot's camera. Based only on what is "
    "visible, answer the following question in Greek, in 1-3 short "
    "sentences.\n\nQuestion: {question}"
)


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('model', 'qwen2.5vl:3b')
        self.declare_parameter('keep_alive', '5m')

        self.model = self.get_parameter('model').value
        self.keep_alive = self.get_parameter('keep_alive').value

        self.bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._busy = threading.Lock()

        self.answer_pub = self.create_publisher(String, 'vision/answer', 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._on_image, 1)
        self.create_subscription(String, 'vision/query', self._on_query, 10)

        self.get_logger().info(f'Vision node started — model={self.model}')

    def _on_image(self, msg: Image):
        with self._frame_lock:
            self._latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _on_query(self, msg: String):
        question = msg.data.strip()
        if not question:
            return
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already answering a vision query, ignoring')
            return
        threading.Thread(target=self._handle_query, args=(question,), daemon=True).start()

    def _handle_query(self, question):
        try:
            self._handle_query_inner(question)
        finally:
            self._busy.release()

    def _handle_query_inner(self, question):
        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            self.answer_pub.publish(String(
                data='Δεν βλέπω τίποτα αυτή τη στιγμή, η κάμερα δεν στέλνει εικόνα.'))
            return

        ok, jpg = cv2.imencode('.jpg', frame)
        if not ok:
            self.answer_pub.publish(String(data='Κάτι πήγε στραβά με την εικόνα της κάμερας.'))
            return

        self.get_logger().info(f'Vision query: {question}')

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': PROMPT_TEMPLATE.format(question=question),
                    'images': [jpg.tobytes()],
                }],
                keep_alive=self.keep_alive,
            )
            answer = (resp.message.content or '').strip()
        except Exception as e:
            self.get_logger().error(f'Vision model call failed: {e}')
            answer = 'Δεν μπόρεσα να δω αυτή τη στιγμή, συγγνώμη.'

        self.get_logger().info(f'Vision answer: {answer}')
        self.answer_pub.publish(String(data=answer or 'Δεν είδα κάτι ιδιαίτερο.'))


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
