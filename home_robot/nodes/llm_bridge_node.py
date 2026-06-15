#!/usr/bin/env python3
"""LLM bridge — Qwen3 (via ollama) tool calling, between speech_text and
robot actions / speech_response.

Subscribes to `speech_text` (std_msgs/String, from stt_node). Sends it to
Qwen3 along with a fixed set of tools (tidy/goto/dock/stop/patrol/
report_clutter) and a short rolling conversation history. Any tool calls
are dispatched to the relevant ROS2 topics (see _dispatch_tool); a second
LLM call, with the tool results appended, produces a short natural-language
Greek reply, published on `speech_response` (std_msgs/String) for the
planned edge-tts streaming TTS node.

Action dispatch notes (current state of the rest of the stack):
- `dock`/`goto(location='dock')` — published on `dock` (std_msgs/Bool),
  consumed by roomba_driver.py (bot.seek_dock()).
- `stop` — publishes a zero geometry_msgs/Twist on `cmd_vel`.
- `goto(location=...)` — looks up `config/locations.yaml` and publishes a
  geometry_msgs/PoseStamped on `goal_pose` (Nav2 bt_navigator's topic).
  Locations are placeholders until SLAM mapping works (see project memory).
- `tidy`/`patrol` — published as JSON on `tidy_command`/`patrol_command`,
  executed by task_planner_node.py (Nav2 navigation + clutter check via
  object_detector.py), which narrates progress on `speech_response`.
- `report_clutter` — answered from the latest `detected_objects` message
  (std_msgs/String JSON, from object_detector.py), no ROS dispatch needed.
- `look(question)` — published on `vision/query` (std_msgs/String);
  vision_node.py (qwen3-vl:4b-instruct via ollama) answers from the latest
  camera frame on `vision/answer` (std_msgs/String). Blocks (with timeout)
  for the reply since the result feeds the follow-up LLM call.
- `system_status()` — read-only host diagnostics (CPU/RAM/disk/temperature
  via psutil) plus the latest `battery/state` (sensor_msgs/BatteryState,
  from roomba_driver.py). No ROS dispatch, answered directly.
"""

import json
import math
import os
import re
import shutil
import threading

import ollama
import psutil
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String


SYSTEM_PROMPT = """Είσαι ο "Max", ο φωνητικός βοηθός ενός ρομπότ καθαρισμού σπιτιού.

Όταν ο χρήστης ζητάει μια ενέργεια (καθάρισμα, μετακίνηση, επιστροφή στη
βάση, σταμάτημα, περιπολία, αναφορά αντικειμένων) ή ρωτάει για την
κατάσταση/υγεία/μπαταρία του υπολογιστή/ρομπότ (π.χ. "πώς είσαι;",
"όλα καλά;", "πόση μπαταρία έχεις;"), κάλεσε το αντίστοιχο tool.

ΣΗΜΑΝΤΙΚΟ: Δεν έχεις δική σου όραση — δεν "θυμάσαι" και δεν φαντάζεσαι τι
υπάρχει γύρω σου. Αν ο χρήστης ρωτήσει οτιδήποτε για το τι βλέπεις, τι
υπάρχει μπροστά/γύρω σου αυτή τη στιγμή, περιγραφή του χώρου, χρώματα,
αντικείμενα κλπ, ΠΡΕΠΕΙ να καλέσεις το tool 'look' για να "δεις" μέσω της
κάμερας — ποτέ μην απαντάς μόνος σου σε τέτοιες ερωτήσεις.

Αν ο χρήστης απλώς ρωτάει κάτι άσχετο ή κάνει συζήτηση, απάντησε κανονικά
χωρίς tool.

Απάντα πάντα σύντομα, φιλικά και στα Ελληνικά, χωρίς emoji."""

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'tidy',
        'description': 'Ξεκίνα να τακτοποιείς/καθαρίζεις ένα δωμάτιο',
        'parameters': {'type': 'object', 'properties': {
            'room': {'type': 'string', 'enum': ['living_room', 'bedroom', 'kitchen', 'all'],
                     'description': 'Το δωμάτιο προς καθαρισμό'},
        }, 'required': ['room']},
    }},
    {'type': 'function', 'function': {
        'name': 'goto',
        'description': 'Πήγαινε σε μια συγκεκριμένη τοποθεσία',
        'parameters': {'type': 'object', 'properties': {
            'location': {'type': 'string', 'enum': ['living_room', 'bedroom', 'kitchen', 'dock'],
                          'description': 'Η τοποθεσία προορισμού'},
        }, 'required': ['location']},
    }},
    {'type': 'function', 'function': {
        'name': 'dock',
        'description': 'Επέστρεψε στη βάση φόρτισης',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'stop',
        'description': 'Σταμάτα αμέσως κάθε κίνηση',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'patrol',
        'description': 'Ξεκίνα περιπολία/εξερεύνηση του χώρου',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'report_clutter',
        'description': 'Πες τι αντικείμενα/ακαταστασία βλέπει αυτή τη στιγμή η κάμερα',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'look',
        'description': 'ΥΠΟΧΡΕΩΤΙΚΟ tool για κάθε ερώτηση σχετικά με το τι βλέπει/τι '
                        'υπάρχει γύρω από το ρομπότ αυτή τη στιγμή (π.χ. "τι βλέπεις;", '
                        '"τι υπάρχει μπροστά σου;", περιγραφή χώρου, χρώματα, '
                        'αντικείμενα). Κοιτάζει μέσα από την κάμερα και απαντά.',
        'parameters': {'type': 'object', 'properties': {
            'question': {'type': 'string',
                          'description': 'Η ερώτηση σχετικά με ό,τι βλέπει η κάμερα'},
        }, 'required': ['question']},
    }},
    {'type': 'function', 'function': {
        'name': 'system_status',
        'description': 'Έλεγξε την κατάσταση του υπολογιστή/ρομπότ — CPU, μνήμη, '
                        'δίσκος, θερμοκρασία, μπαταρία (π.χ. "πώς είσαι;", '
                        '"όλα καλά;", "πόση μπαταρία έχεις;")',
        'parameters': {'type': 'object', 'properties': {}},
    }},
]

_EMOJI_RE = re.compile(
    '[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF'
    '\U00002190-\U000021FF\U00002B00-\U00002BFF️]+',
    flags=re.UNICODE)


def _strip_emoji(text):
    return _EMOJI_RE.sub('', text).strip()


def _yaw_to_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def _get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return None
    for name in ('k10temp', 'coretemp', 'cpu_thermal', 'acpitz'):
        if temps.get(name):
            return round(temps[name][0].current, 1)
    return None


class LLMBridgeNode(Node):
    def __init__(self):
        super().__init__('llm_bridge_node')

        self.declare_parameter('model', 'qwen3-vl:4b-instruct')
        self.declare_parameter('keep_alive', '10m')
        self.declare_parameter('temperature', 0.1)
        self.declare_parameter('history_turns', 4)
        self.declare_parameter('vision_timeout', 60.0)

        self.model = self.get_parameter('model').value
        self.keep_alive = self.get_parameter('keep_alive').value
        self.temperature = self.get_parameter('temperature').value
        self.history_turns = self.get_parameter('history_turns').value
        self.vision_timeout = self.get_parameter('vision_timeout').value

        locations_path = os.path.join(get_package_share_directory('home_robot'),
                                        'config', 'locations.yaml')
        with open(locations_path) as f:
            self.locations = yaml.safe_load(f)

        self.response_pub = self.create_publisher(String, 'speech_response', 10)
        self.dock_pub = self.create_publisher(Bool, 'dock', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.tidy_pub = self.create_publisher(String, 'tidy_command', 10)
        self.patrol_pub = self.create_publisher(Bool, 'patrol_command', 10)
        self.vision_query_pub = self.create_publisher(String, 'vision/query', 10)

        self._history = []
        self._busy = threading.Lock()
        self._latest_objects = None
        self._vision_event = threading.Event()
        self._vision_answer = None
        self._latest_battery = None

        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)
        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)
        self.create_subscription(String, 'vision/answer', self._on_vision_answer, 10)
        self.create_subscription(BatteryState, 'battery/state', self._on_battery_state, 10)

        self.get_logger().info(f'LLM bridge started — model={self.model}')

    def _on_detected_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_vision_answer(self, msg: String):
        self._vision_answer = msg.data
        self._vision_event.set()

    def _on_battery_state(self, msg: BatteryState):
        self._latest_battery = msg

    def _on_speech_text(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already handling a request, ignoring speech_text')
            return
        threading.Thread(target=self._handle_text, args=(text,), daemon=True).start()

    def _handle_text(self, text):
        try:
            self._handle_text_inner(text)
        finally:
            self._busy.release()

    def _handle_text_inner(self, text):
        self.get_logger().info(f'Heard: {text}')

        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        for turn in self._history:
            messages.extend(turn)
        user_msg = {'role': 'user', 'content': text}
        messages.append(user_msg)

        try:
            resp = ollama.chat(model=self.model, messages=messages, tools=TOOLS,
                                options={'temperature': self.temperature}, think=False,
                                keep_alive=self.keep_alive)
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')
            return

        out = resp.message
        turn = [user_msg]

        if out.tool_calls:
            messages.append(out)
            turn.append(out)
            for tc in out.tool_calls:
                args = tc.function.arguments or {}
                self.get_logger().info(f'Tool call: {tc.function.name}({args})')
                result = self._dispatch_tool(tc.function.name, args)
                tool_msg = {'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)}
                messages.append(tool_msg)
                turn.append(tool_msg)

            try:
                resp2 = ollama.chat(model=self.model, messages=messages,
                                     options={'temperature': 0.3}, think=False,
                                     keep_alive=self.keep_alive)
                reply = (resp2.message.content or '').strip()
            except Exception as e:
                self.get_logger().error(f'LLM follow-up call failed: {e}')
                reply = ''
        else:
            reply = (out.content or '').strip()

        reply = _strip_emoji(reply)
        if not reply:
            return

        turn.append({'role': 'assistant', 'content': reply})
        self._history.append(turn)
        del self._history[:-self.history_turns]

        self.get_logger().info(f'Max: {reply}')
        self.response_pub.publish(String(data=reply))

    def _dispatch_tool(self, name, args):
        if name == 'tidy':
            room = args.get('room', 'all')
            self.tidy_pub.publish(String(data=json.dumps({'room': room})))
            return {'status': 'started', 'action': 'tidy', 'room': room}

        elif name == 'goto':
            location = args.get('location')
            if location == 'dock':
                self.dock_pub.publish(Bool(data=True))
                return {'status': 'started', 'action': 'dock'}
            loc = self.locations.get(location)
            if loc is None:
                return {'status': 'error', 'reason': f'unknown location: {location}'}
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = float(loc['x'])
            goal.pose.position.y = float(loc['y'])
            goal.pose.orientation = _yaw_to_quaternion(float(loc['yaw']))
            self.goal_pub.publish(goal)
            return {'status': 'started', 'action': 'goto', 'location': location}

        elif name == 'dock':
            self.dock_pub.publish(Bool(data=True))
            return {'status': 'started', 'action': 'dock'}

        elif name == 'stop':
            self.cmd_vel_pub.publish(Twist())
            return {'status': 'ok', 'action': 'stop'}

        elif name == 'patrol':
            self.patrol_pub.publish(Bool(data=True))
            return {'status': 'started', 'action': 'patrol'}

        elif name == 'report_clutter':
            if self._latest_objects is None:
                return {'status': 'ok', 'objects': [], 'note': 'no detections yet'}
            clutter = [o for o in self._latest_objects if o.get('clutter')]
            return {'status': 'ok', 'objects': clutter}

        elif name == 'look':
            question = args.get('question') or 'Περίγραψε τι βλέπεις.'
            self._vision_answer = None
            self._vision_event.clear()
            self.vision_query_pub.publish(String(data=question))
            if not self._vision_event.wait(timeout=self.vision_timeout):
                return {'status': 'error', 'reason': 'vision_node did not respond (timeout)'}
            return {'status': 'ok', 'description': self._vision_answer}

        elif name == 'system_status':
            mem = psutil.virtual_memory()
            disk = shutil.disk_usage('/')
            status = {
                'status': 'ok',
                'cpu_percent': psutil.cpu_percent(interval=0.5),
                'ram_percent': mem.percent,
                'ram_used_gb': round(mem.used / 1e9, 1),
                'ram_total_gb': round(mem.total / 1e9, 1),
                'disk_percent': round(disk.used / disk.total * 100, 1),
                'disk_free_gb': round(disk.free / 1e9, 1),
            }
            cpu_temp = _get_cpu_temp()
            if cpu_temp is not None:
                status['cpu_temp_c'] = cpu_temp
            if self._latest_battery is not None:
                pct = self._latest_battery.percentage
                if pct == pct:  # not NaN
                    status['battery_percent'] = round(pct * 100, 1)
                status['battery_charging'] = (
                    self._latest_battery.power_supply_status
                    == BatteryState.POWER_SUPPLY_STATUS_CHARGING)
            return status

        return {'status': 'error', 'reason': f'unknown tool: {name}'}

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = LLMBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
