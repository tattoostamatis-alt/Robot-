#!/usr/bin/env python3
"""Vision node — qwen3-vl:4b-instruct (via ollama) or Gemini (via the
google-genai API), answers questions about the latest camera frame.

Subscribes to the RealSense color image (caches the latest frame) and to
`vision/query` (std_msgs/String, a question in Greek). Sends the cached
frame + question to a vision-language model and publishes the answer on
`vision/answer` (std_msgs/String).

Used by llm_bridge_node.py's `look` tool ("ask Max what he sees").
"""

import os
import threading

import base64
import cv2
import ollama
import rclpy
import requests
from cv_bridge import CvBridge
from dotenv import load_dotenv
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

load_dotenv(os.path.expanduser('~/.env'))


# Wrapped in English with explicit "only what's visible" / "short" framing —
# Greek-language scene descriptions are noticeably more accurate and
# consistent with this framing than a Greek-only prompt (verified
# 2026-06-12 with qwen2.5vl:3b: Greek-only prompt hallucinated unrelated
# objects/colors; this version correctly described the scene twice in a
# row. Re-verified 2026-06-15 with qwen3-vl:4b-instruct: still accurate).
PROMPT_TEMPLATE = (
    "Look at this image from a home robot's camera. Based only on what is "
    "visible, answer the following question in Greek, in 1-3 short "
    "sentences.\n\nQuestion: {question}"
)


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('backend', 'lemonade')
        self.declare_parameter('model', 'qwen3-vl:4b-instruct')
        self.declare_parameter('gemini_model', 'gemini-flash-lite-latest')
        self.declare_parameter('lemonade_url', 'http://127.0.0.1:13305/api/v1')
        self.declare_parameter('lemonade_model', 'qwen3vl-it-4b-FLM')
        self.declare_parameter('keep_alive', '5m')

        self.backend = self.get_parameter('backend').value
        self.model = self.get_parameter('model').value
        self.gemini_model = self.get_parameter('gemini_model').value
        self.lemonade_url = self.get_parameter('lemonade_url').value
        self.lemonade_model = self.get_parameter('lemonade_model').value
        self.keep_alive = self.get_parameter('keep_alive').value

        self._gemini_client = None
        if self.backend == 'gemini':
            from google import genai
            self._gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

        self.bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._busy = threading.Lock()

        self.answer_pub = self.create_publisher(String, 'vision/answer', 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._on_image, 1)
        self.create_subscription(String, 'vision/query', self._on_query, 10)

        if self.backend == 'gemini':
            active_model = self.gemini_model
        elif self.backend == 'lemonade':
            active_model = self.lemonade_model
        else:
            active_model = self.model
        self.get_logger().info(f'Vision node started — backend={self.backend} model={active_model}')

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
            if self.backend == 'gemini':
                answer = self._query_gemini(jpg.tobytes(), question)
            elif self.backend == 'lemonade':
                answer = self._query_lemonade(jpg.tobytes(), question)
            else:
                answer = self._query_ollama(jpg.tobytes(), question)
        except Exception as e:
            self.get_logger().error(f'Vision model call failed: {e}')
            answer = 'Δεν μπόρεσα να δω αυτή τη στιγμή, συγγνώμη.'

        self.get_logger().info(f'Vision answer: {answer}')
        self.answer_pub.publish(String(data=answer or 'Δεν είδα κάτι ιδιαίτερο.'))

    def _query_ollama(self, jpg_bytes, question):
        resp = ollama.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': PROMPT_TEMPLATE.format(question=question),
                'images': [jpg_bytes],
            }],
            keep_alive=self.keep_alive,
        )
        return (resp.message.content or '').strip()

    def _query_lemonade(self, jpg_bytes, question):
        b64 = base64.b64encode(jpg_bytes).decode('ascii')
        payload = {
            'model': self.lemonade_model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                    {'type': 'text', 'text': PROMPT_TEMPLATE.format(question=question)},
                ],
            }],
        }
        r = requests.post(f'{self.lemonade_url}/chat/completions', json=payload, timeout=60)
        r.raise_for_status()
        return (r.json()['choices'][0]['message'].get('content') or '').strip()

    def _query_gemini(self, jpg_bytes, question):
        from google.genai import types
        resp = self._gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=[
                types.Part.from_bytes(data=jpg_bytes, mime_type='image/jpeg'),
                PROMPT_TEMPLATE.format(question=question),
            ],
        )
        return (resp.text or '').strip()


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
