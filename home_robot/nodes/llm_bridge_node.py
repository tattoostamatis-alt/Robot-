#!/usr/bin/env python3
"""LLM bridge — on-demand vision: the camera frame is included only when the
user's question is vision-related (keyword detection), keeping non-vision
commands fast (~2s) while vision queries use the full VLM path (~11s).

Qwen3-VL 4B (via Lemonade/NPU) handles vision + conversation.
The `look` tool and vision_node.py are not needed.

Subscribes to `speech_text` (std_msgs/String, from stt_node). Sends text
(+ camera frame when vision-related) to Qwen3-VL with a fixed tool set and
rolling history. Tool calls are dispatched to the relevant ROS2 topics; a
follow-up LLM call produces a Greek reply on `speech_response` for the TTS node.

Action dispatch notes:
- `dock`/`goto(location='dock')` — published on `dock` (std_msgs/Bool).
- `stop` — publishes a zero geometry_msgs/Twist on `cmd_vel`.
- `goto(location=...)` — looks up `config/locations.yaml` and sends a real
  nav2_msgs/action/NavigateToPose goal, blocking on the result.
- `tidy`/`patrol` — published as JSON on `tidy_command`/`patrol_command`.
- `report_clutter` — answered from the latest `detected_objects` message.
- `system_status()` — host diagnostics via psutil + battery/state.
- `remember` — stores a fact via rag_memory_node's memory/store topic.
"""

import base64
import cv2
import json
import math
import os
import re
import shutil
import threading
import time

import ollama
import psutil
import rclpy
import requests
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from dotenv import load_dotenv
from geometry_msgs.msg import Quaternion, Twist
from home_robot.status_query import format_status, is_status_query, wants_battery
from home_robot.stop_command import is_stop_command, strip_accents
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image
from std_msgs.msg import Bool, String

load_dotenv(os.path.expanduser('~/.env'))


# Greek keywords (accent-stripped) that signal a vision question.
# When matched, the current camera frame is included in the LLM message.
_VISION_KEYWORDS = {
    'βλεπ', 'βλεπεις', 'βλεπω', 'κοιτ', 'κοιταξ', 'κοιτα', 'δες', 'δε ',
    'εικον', 'χρωμ', 'αντικειμ', 'περιγρ', 'μπροστ', 'γυρω', 'υπαρχ',
    'τι ειν', 'τι εχ', 'τι βλ', 'χωρος', 'δωματ', 'ακαταστ', 'βρωμ',
}


def _needs_vision(text: str) -> bool:
    """Return True if the text appears to be a vision-related question."""
    return any(kw in strip_accents(text) for kw in _VISION_KEYWORDS)


SYSTEM_PROMPT = """Είσαι ο "Max", βοηθός ρομπότ καθαρισμού σπιτιού. Απάντα στα Ελληνικά, φιλικά, χωρίς emoji, με ΜΙΑ σύντομη πρόταση (το πολύ 20 λέξεις). Μην εξηγείς γιατί δεν μπορείς κάτι — πες το σύντομα.

Κάλεσε tool όταν ο χρήστης ζητάει ενέργεια (κίνηση, καθαρισμό, docking, εξερεύνηση, αναφορά αντικειμένων, πιάσιμο αντικειμένου με τον βραχίονα, να σε ακολουθήσει) ή ρωτάει κατάσταση/μπαταρία. Αν απλώς συζητά, απάντα χωρίς tool.

Αν ζητήσει να θυμηθείς κάτι ("θυμήσου...", "να θυμάσαι..."), ΠΑΝΤΑ κάλεσε 'remember' — αλλιώς χάνεται.

Αν η ερώτηση αφορά την εικόνα, η περιγραφή της κάμερας συνοδεύει ήδη το μήνυμα."""

# NOTE: keep descriptions terse. FastFlowLM (NPU) has max_prefill_len=4096 and the
# whole TOOLS schema is serialized into every prefill — verbose descriptions overflow
# it ("Max length reached!" 400). This concise set fits with room to spare.
_ROOMS = ['saloni', 'kouzina', 'diadromos', 'toualeta', 'domatio tou max', 'domatio tou mbamba']

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'tidy',
        'description': 'Τακτοποίησε/καθάρισε ένα δωμάτιο',
        'parameters': {'type': 'object', 'properties': {
            'room': {'type': 'string', 'enum': _ROOMS + ['all']},
        }, 'required': ['room']},
    }},
    {'type': 'function', 'function': {
        'name': 'goto',
        'description': 'Πήγαινε σε μια τοποθεσία',
        'parameters': {'type': 'object', 'properties': {
            'location': {'type': 'string', 'enum': _ROOMS + ['dock']},
        }, 'required': ['location']},
    }},
    {'type': 'function', 'function': {
        'name': 'dock',
        'description': 'Επέστρεψε στη βάση φόρτισης',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'stop',
        'description': 'Σταμάτα κάθε κίνηση αμέσως',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'move',
        'description': 'Μικρή χειροκίνητη κίνηση λίγων δευτ. (μπρος/πίσω/στροφή). '
                       'ΟΧΙ για μετάβαση σε δωμάτιο — γι αυτό goto.',
        'parameters': {'type': 'object', 'properties': {
            'direction': {'type': 'string', 'enum': ['forward', 'backward', 'left', 'right']},
            'duration': {'type': 'number', 'description': 'δευτερόλεπτα 0.3-3 (default 1)'},
        }, 'required': ['direction']},
    }},
    {'type': 'function', 'function': {
        'name': 'patrol',
        'description': 'Ξεκίνα περιπολία του χώρου',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'explore',
        'description': 'Αυτόνομη εξερεύνηση & χαρτογράφηση αγνώστων περιοχών',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'stop_explore',
        'description': 'Σταμάτα την εξερεύνηση',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'report_clutter',
        'description': 'Λίστα σκόρπιων αντικειμένων (clutter) από το YOLO. ΟΧΙ για "τι βλέπεις".',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'remember',
        'description': 'Θυμήσου μόνιμα ένα γεγονός. Μόνο όταν το ζητά ρητά ο χρήστης.',
        'parameters': {'type': 'object', 'properties': {
            'fact': {'type': 'string', 'description': 'το γεγονός προς απομνημόνευση'},
        }, 'required': ['fact']},
    }},
    {'type': 'function', 'function': {
        'name': 'system_status',
        'description': 'Κατάσταση ρομπότ: CPU, RAM, δίσκος, θερμοκρασία, uptime, μπαταρία',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'pick',
        'description': 'Πιάσε ένα ορατό αντικείμενο με τον βραχίονα και άφησέ το στο σημείο εναπόθεσης',
        'parameters': {'type': 'object', 'properties': {
            'object': {'type': 'string', 'description': 'αγγλικό COCO label, π.χ. cup, bottle, book'},
        }, 'required': ['object']},
    }},
    {'type': 'function', 'function': {
        'name': 'fetch',
        'description': 'Βρες, πιάσε και φέρε μου ένα αντικείμενο ("φέρε μου το X")',
        'parameters': {'type': 'object', 'properties': {
            'object': {'type': 'string', 'description': 'αγγλικό COCO label, π.χ. cup, bottle, book'},
        }, 'required': ['object']},
    }},
    {'type': 'function', 'function': {
        'name': 'follow',
        'description': 'Ακολούθησε τον χρήστη ("ακολούθησέ με")',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'stop_follow',
        'description': 'Σταμάτα να ακολουθείς τον χρήστη',
        'parameters': {'type': 'object', 'properties': {}},
    }},
]


def _build_gemini_tool():
    from google.genai import types
    decls = [types.FunctionDeclaration(name=t['function']['name'],
                                         description=t['function']['description'],
                                         parameters=t['function']['parameters'])
             for t in TOOLS]
    return types.Tool(function_declarations=decls)


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

        self.declare_parameter('backend', 'ollama')
        self.declare_parameter('model', 'qwen3-vl:4b-instruct')
        self.declare_parameter('gemini_model', 'gemini-flash-lite-latest')
        self.declare_parameter('lemonade_url', 'http://127.0.0.1:13305/api/v1')
        self.declare_parameter('lemonade_model', 'qwen3vl-it-4b-FLM')
        self.declare_parameter('keep_alive', '10m')
        self.declare_parameter('temperature', 0.1)
        self.declare_parameter('history_turns', 4)
        self.declare_parameter('nav_timeout', 60.0)
        self.declare_parameter('memory_enabled', False)
        self.declare_parameter('memory_timeout', 5.0)
        self.declare_parameter('jpeg_quality', 70)

        self.backend = self.get_parameter('backend').value
        self.model = self.get_parameter('model').value
        self.gemini_model = self.get_parameter('gemini_model').value
        self.lemonade_url = self.get_parameter('lemonade_url').value
        self.lemonade_model = self.get_parameter('lemonade_model').value
        self.keep_alive = self.get_parameter('keep_alive').value
        self.temperature = self.get_parameter('temperature').value
        self.history_turns = self.get_parameter('history_turns').value
        self.nav_timeout = self.get_parameter('nav_timeout').value
        self.memory_enabled = self.get_parameter('memory_enabled').value
        self.memory_timeout = self.get_parameter('memory_timeout').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        self._gemini_client = None
        self._gemini_tool = None
        if self.backend == 'gemini':
            from google import genai
            self._gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
            self._gemini_tool = _build_gemini_tool()

        locations_path = os.path.join(get_package_share_directory('home_robot'),
                                        'config', 'locations.yaml')
        with open(locations_path) as f:
            self.locations = yaml.safe_load(f)

        self.response_pub = self.create_publisher(String, 'speech_response', 10)
        self.dock_pub = self.create_publisher(Bool, 'dock', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tidy_pub = self.create_publisher(String, 'tidy_command', 10)
        self.patrol_pub = self.create_publisher(Bool, 'patrol_command', 10)
        self.explore_pub = self.create_publisher(Bool, 'explore_command', 10)
        self.memory_store_pub = self.create_publisher(String, 'memory/store', 10)
        self.memory_query_pub = self.create_publisher(String, 'memory/query', 10)
        self.pick_pub = self.create_publisher(String, 'pick_command', 10)
        self.follow_pub = self.create_publisher(Bool, 'follow_command', 10)
        self.mission_pub = self.create_publisher(String, 'mission/start', 10)

        self._history = []
        self._busy = threading.Lock()
        # Kept on self (not just as a _navigate local) so _emergency_stop can
        # cancel the goal from the speech callback thread.
        self._nav_goal_handle = None
        self._latest_objects = None
        self._latest_battery = None
        self._memory_event = threading.Event()
        self._memory_answer = None
        self._situation: dict = {}

        # Camera frame — encoded to JPEG once on arrival, reused per message
        self._bridge = CvBridge()
        self._frame_lock = threading.Lock()
        self._latest_frame_jpg: bytes | None = None

        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)
        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)
        self.create_subscription(BatteryState, 'battery/state', self._on_battery_state, 10)
        self.create_subscription(String, 'memory/answer', self._on_memory_answer, 10)
        self.create_subscription(String, 'situation_context', self._on_situation, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._on_image, 1)

        if self.backend == 'gemini':
            active_model = self.gemini_model
        elif self.backend == 'lemonade':
            active_model = self.lemonade_model
        else:
            active_model = self.model
        # Vision path differs per backend: `lemonade` (FastFlowLM/NPU) attaches the
        # frame to the same Qwen3-VL call, so no cloud key is involved; the ollama
        # path pre-describes the frame with Gemini Flash Lite first. Say which one
        # is actually live — a hardcoded "vision=Gemini" here sends future debugging
        # hunting for a GEMINI_API_KEY that the NPU path never reads.
        vision_path = ('Qwen3-VL on NPU (same call)' if self.backend == 'lemonade'
                       else 'Gemini Flash Lite')
        self.get_logger().info(f'LLM bridge started — backend={self.backend} model={active_model} | vision={vision_path}')

        # Warm up the NPU model at boot so the first real voice command isn't a
        # ~cold-load stall (FastFlowLM loads the 4B from disk on first use; this
        # also primes the OS page cache so later qwen<->embed-gemma swaps are fast).
        if self.backend == 'lemonade':
            threading.Thread(target=self._warmup_lemonade, daemon=True).start()

    def _warmup_lemonade(self):
        try:
            requests.post(
                f'{self.lemonade_url}/chat/completions',
                json={'model': self.lemonade_model,
                      'messages': [{'role': 'user', 'content': 'γεια'}],
                      'max_tokens': 1, 'temperature': 0.0},
                timeout=180)
            self.get_logger().info('Qwen warmed up on NPU (first command will be fast)')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'Qwen warm-up failed (non-fatal): {e}')

    # ── callbacks ──────────────────────────────────────────────────────────

    def _on_image(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self._frame_lock:
                self._latest_frame_jpg = jpg.tobytes()

    def _get_frame_jpg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_frame_jpg

    def _vision_describe(self, question: str, frame_jpg: bytes) -> str | None:
        """Fast visual description via Gemini Flash Lite (~2s)."""
        try:
            from google import genai
            from google.genai import types
            api_key = os.environ.get('GEMINI_API_KEY', '')
            if not api_key:
                return None
            client = genai.Client(api_key=api_key)
            prompt = (
                'Look at this robot camera image. Answer the following question '
                'in Greek in 1-3 short sentences, based only on what is visible.\n'
                f'Question: {question}'
            )
            resp = client.models.generate_content(
                model='gemini-flash-lite-latest',
                contents=[
                    types.Part.from_bytes(data=frame_jpg, mime_type='image/jpeg'),
                    prompt,
                ]
            )
            return (resp.text or '').strip()
        except Exception as e:
            self.get_logger().warn(f'Gemini vision failed: {e}')
            return None

    def _on_detected_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_battery_state(self, msg: BatteryState):
        self._latest_battery = msg

    def _on_memory_answer(self, msg: String):
        self._memory_answer = msg.data
        self._memory_event.set()

    def _on_situation(self, msg: String):
        try:
            self._situation = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    # ── context helpers ────────────────────────────────────────────────────

    def _situation_system_message(self) -> str | None:
        if not self._situation:
            return None
        s = self._situation
        parts = [f"Δωμάτιο: {s.get('room', '?')}"]
        if 'objects' in s:
            parts.append(f"Κοντινά αντικείμενα: {s['objects']}")
        if 'battery_pct' in s:
            parts.append(f"Μπαταρία: {s['battery_pct']}%")
        parts.append(f"CPU: {s.get('cpu_pct', '?')}%  RAM: {s.get('ram_pct', '?')}%")
        return 'Τρέχουσα κατάσταση:\n' + '\n'.join(f'- {p}' for p in parts)

    def _retrieve_memories(self, text):
        if not self.memory_enabled:
            return []
        self._memory_answer = None
        self._memory_event.clear()
        self.memory_query_pub.publish(String(data=text))
        if not self._memory_event.wait(timeout=self.memory_timeout):
            self.get_logger().warn('rag_memory_node did not respond (timeout)')
            return []
        try:
            return json.loads(self._memory_answer)
        except (json.JSONDecodeError, TypeError):
            return []

    def _memory_system_message(self, facts):
        if not facts:
            return None
        bullets = '\n'.join(f'- {f}' for f in facts)
        return f'Πράγματα που θυμάσαι από πριν:\n{bullets}'

    # ── speech entry point ─────────────────────────────────────────────────

    def _on_speech_text(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        # "Stop" is handled here, ahead of BOTH the busy lock and the LLM.
        # Ahead of the LLM because it does not reliably emit the tool call
        # (see _is_stop_command). Ahead of the lock because the lock is held
        # for the whole blocking _navigate() — up to nav_timeout (60 s) — so
        # during a goto, the exact moment the user most needs to halt the
        # robot, every utterance was being dropped with "Already handling a
        # request". Stop must never queue behind the thing it is stopping.
        if is_stop_command(text):
            self.get_logger().warn(f'STOP command (keyword gate): {text}')
            threading.Thread(target=self._emergency_stop, daemon=True).start()
            return

        # Status questions are answered from telemetry, not by the model. Same
        # failure mode as stop: it replies conversationally instead of calling
        # system_status, and having no data it makes a number up — "85%" and
        # "87%" on the same question hours apart, both while the Roomba was
        # asleep and publishing no battery topic at all. Unlike stop this still
        # takes the busy lock: it reads sensors and speaks, so it should queue
        # behind an in-flight request rather than interleave with it.
        if is_status_query(text) and self._busy.acquire(blocking=False):
            self.get_logger().info(f'Status query (keyword gate): {text}')
            threading.Thread(target=self._answer_status, args=(text,),
                             daemon=True).start()
            return

        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already handling a request, ignoring speech_text')
            return
        threading.Thread(target=self._handle_text, args=(text,), daemon=True).start()

    def _answer_status(self, text):
        """Answer a status question from the system_status tool's real data."""
        try:
            status = self._dispatch_tool('system_status', {})
            reply = format_status(status, battery_only=wants_battery(text))
            self.get_logger().info(f'Max: {reply}')
            self.response_pub.publish(String(data=reply))
        except Exception as exc:                         # noqa: BLE001
            self.get_logger().error(f'status query failed: {exc}')
            self.response_pub.publish(
                String(data='Δεν μπορώ να διαβάσω την κατάστασή μου.'))
        finally:
            self._busy.release()

    def _emergency_stop(self):
        """Halt everything: wheels, navigation, and any background behaviour."""
        # Wheels first and repeatedly — a single Twist can be lost in a DDS
        # hiccup, and Nav2's controller may still be publishing over us until
        # the goal cancel below lands.
        for _ in range(3):
            self.cmd_vel_pub.publish(Twist())
            time.sleep(0.05)

        # Cancel the in-flight NavigateToPose goal, otherwise the controller
        # simply overwrites our zero Twist on its next cycle.
        gh = self._nav_goal_handle
        if gh is not None:
            try:
                gh.cancel_goal_async()
                self.get_logger().info('Cancelled active Nav2 goal')
            except Exception as exc:                     # noqa: BLE001
                self.get_logger().warn(f'Nav2 goal cancel failed: {exc}')

        # Background behaviours run in their own nodes and ignore cmd_vel.
        self.patrol_pub.publish(Bool(data=False))
        self.explore_pub.publish(Bool(data=False))
        self.follow_pub.publish(Bool(data=False))
        self.mission_pub.publish(String(data='cancel'))

        self.cmd_vel_pub.publish(Twist())
        self.response_pub.publish(String(data='Σταμάτησα.'))

    def _handle_text(self, text):
        try:
            self._handle_text_inner(text)
        finally:
            self._busy.release()

    def _handle_text_inner(self, text):
        self.get_logger().info(f'Heard: {text}')
        if self.backend == 'gemini':
            self._handle_text_gemini(text)
        elif self.backend == 'lemonade':
            self._handle_text_lemonade(text)
        else:
            self._handle_text_ollama(text)

    # ── lemonade backend ───────────────────────────────────────────────────

    def _handle_text_lemonade(self, text):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        memory_msg = self._memory_system_message(self._retrieve_memories(text))
        if memory_msg:
            messages.append({'role': 'system', 'content': memory_msg})
        sit_msg = self._situation_system_message()
        if sit_msg:
            messages.append({'role': 'system', 'content': sit_msg})
        for turn in self._history:
            messages.extend(turn)

        # Vision: attach the camera frame directly to the user message so the
        # local Qwen3-VL sees AND answers in ONE NPU pass (no separate describe
        # call — ~8-10s vs ~12-17s two-pass). Non-vision turns send text only and
        # stay fast (~2s). The image is sent to the model but stored text-only in
        # history so later prefills stay small. Verified: image + full TOOLS fits
        # the FLM prefill (no "Max length reached" 400).
        text_user_msg = {'role': 'user', 'content': text}
        frame_jpg = self._get_frame_jpg() if _needs_vision(text) else None
        if frame_jpg is not None:
            b64 = base64.b64encode(frame_jpg).decode('ascii')
            self.get_logger().info('Vision (NPU): frame attached to query')
            send_user_msg = {'role': 'user', 'content': [
                {'type': 'text', 'text': text},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            ]}
        else:
            send_user_msg = text_user_msg
        messages.append(send_user_msg)

        def _chat(msgs, temperature, with_tools):
            # Cap the reply length: generation on the NPU runs ~18 tok/s, so a
            # runaway verbose answer (150+ tokens) takes ~9 s while a one-liner
            # takes ~3-4 s. Spoken replies should be short anyway. 120 tokens is
            # ~2-3 Greek sentences — plenty, and it bounds worst-case latency.
            payload = {'model': self.lemonade_model, 'messages': msgs,
                       'temperature': temperature, 'max_tokens': 120}
            if with_tools:
                payload['tools'] = TOOLS
            r = requests.post(f'{self.lemonade_url}/chat/completions', json=payload, timeout=120)
            r.raise_for_status()
            return r.json()['choices'][0]['message']

        try:
            out = _chat(messages, self.temperature, with_tools=True)
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')
            return

        turn = [text_user_msg]
        tool_calls = out.get('tool_calls') or []

        if tool_calls:
            messages.append(out)
            turn.append(out)
            for tc in tool_calls:
                fn = tc['function']
                args = json.loads(fn.get('arguments') or '{}')
                self.get_logger().info(f'Tool call: {fn["name"]}({args})')
                result = self._dispatch_tool(fn['name'], args)
                tool_msg = {'role': 'tool', 'tool_call_id': tc.get('id', ''),
                            'content': json.dumps(result, ensure_ascii=False)}
                messages.append(tool_msg)
                turn.append(tool_msg)

            try:
                out2 = _chat(messages, 0.3, with_tools=False)
                reply = (out2.get('content') or '').strip()
            except Exception as e:
                self.get_logger().error(f'LLM follow-up call failed: {e}')
                reply = ''
        else:
            reply = (out.get('content') or '').strip()

        reply = _strip_emoji(reply)
        if not reply:
            return

        turn.append({'role': 'assistant', 'content': reply})
        self._history.append(turn)
        del self._history[:-self.history_turns]

        self.get_logger().info(f'Max: {reply}')
        self.response_pub.publish(String(data=reply))

    # ── ollama backend ─────────────────────────────────────────────────────

    def _handle_text_ollama(self, text):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        memory_msg = self._memory_system_message(self._retrieve_memories(text))
        if memory_msg:
            messages.append({'role': 'system', 'content': memory_msg})
        sit_msg = self._situation_system_message()
        if sit_msg:
            messages.append({'role': 'system', 'content': sit_msg})
        for turn in self._history:
            messages.extend(turn)

        if _needs_vision(text):
            frame_jpg = self._get_frame_jpg()
            if frame_jpg is not None:
                vision_desc = self._vision_describe(text, frame_jpg)
                if vision_desc:
                    self.get_logger().info(f'Vision (Gemini): {vision_desc}')
                    messages.append({'role': 'system',
                                     'content': f'Η κάμερα βλέπει αυτή τη στιγμή: {vision_desc}'})

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

    # ── gemini backend ─────────────────────────────────────────────────────

    def _handle_text_gemini(self, text):
        from google.genai import types

        memory_msg = self._memory_system_message(self._retrieve_memories(text))
        sit_msg = self._situation_system_message()
        extras = '\n\n'.join(m for m in [memory_msg, sit_msg] if m)
        system_instruction = f'{SYSTEM_PROMPT}\n\n{extras}' if extras else SYSTEM_PROMPT

        config = types.GenerateContentConfig(
            system_instruction=system_instruction, tools=[self._gemini_tool],
            temperature=self.temperature)

        contents = []
        for turn in self._history:
            contents.extend(turn)

        frame_jpg = self._get_frame_jpg() if _needs_vision(text) else None
        history_user_msg = types.Content(role='user', parts=[types.Part(text=text)])
        if frame_jpg is not None:
            user_content = types.Content(role='user', parts=[
                types.Part.from_bytes(data=frame_jpg, mime_type='image/jpeg'),
                types.Part(text=text),
            ])
        else:
            user_content = history_user_msg
        contents.append(user_content)
        turn = [history_user_msg]  # text-only in history

        try:
            resp = self._gemini_client.models.generate_content(
                model=self.gemini_model, contents=contents, config=config)
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')
            return

        out = resp.candidates[0].content
        function_calls = [p.function_call for p in out.parts if p.function_call]

        if function_calls:
            contents.append(out)
            turn.append(out)
            response_parts = []
            for fc in function_calls:
                args = fc.args or {}
                self.get_logger().info(f'Tool call: {fc.name}({args})')
                result = self._dispatch_tool(fc.name, args)
                response_parts.append(types.Part.from_function_response(
                    name=fc.name, response=result))
            response_content = types.Content(role='user', parts=response_parts)
            contents.append(response_content)
            turn.append(response_content)

            try:
                resp2 = self._gemini_client.models.generate_content(
                    model=self.gemini_model, contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT, temperature=0.3))
                reply = (resp2.text or '').strip()
            except Exception as e:
                self.get_logger().error(f'LLM follow-up call failed: {e}')
                reply = ''
        else:
            reply = (out.parts[0].text or '').strip() if out.parts else ''

        reply = _strip_emoji(reply)
        if not reply:
            return

        turn.append(types.Content(role='model', parts=[types.Part(text=reply)]))
        self._history.append(turn)
        del self._history[:-self.history_turns]

        self.get_logger().info(f'Max: {reply}')
        self.response_pub.publish(String(data=reply))

    # ── navigation ─────────────────────────────────────────────────────────

    def _navigate(self, loc):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return False, 'το Nav2 δεν είναι έτοιμο'

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(loc['x'])
        goal.pose.pose.position.y = float(loc['y'])
        goal.pose.pose.orientation = _yaw_to_quaternion(float(loc['yaw']))

        accepted_ev = threading.Event()
        done_ev = threading.Event()
        goal_handle_box = [None]
        result_box = [None]

        def _on_accepted(fut):
            goal_handle_box[0] = fut.result()
            accepted_ev.set()

        def _on_result(fut):
            result_box[0] = fut.result()
            done_ev.set()

        self.nav_client.send_goal_async(goal).add_done_callback(_on_accepted)

        if not accepted_ev.wait(timeout=10.0):
            return False, 'καμία απάντηση από το Nav2'

        gh = goal_handle_box[0]
        if gh is None or not gh.accepted:
            return False, 'ο στόχος απορρίφθηκε'

        self._nav_goal_handle = gh
        try:
            gh.get_result_async().add_done_callback(_on_result)

            if not done_ev.wait(timeout=self.nav_timeout):
                return False, 'λήξη χρόνου πλοήγησης'

            result = result_box[0]
            if result is None:
                return False, 'η πλοήγηση απέτυχε'
            if result.status == GoalStatus.STATUS_CANCELED:
                # The user said "σταμάτα" mid-goto; _emergency_stop already
                # spoke, so don't let the caller report a failure on top.
                return False, None
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                return False, 'η πλοήγηση απέτυχε'
            return True, None
        finally:
            self._nav_goal_handle = None

    # ── tool dispatch ──────────────────────────────────────────────────────

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
            ok, reason = self._navigate(loc)
            if not ok:
                if reason is None:      # cancelled by the user's "σταμάτα"
                    return {'status': 'cancelled', 'action': 'goto', 'location': location}
                return {'status': 'error', 'action': 'goto', 'location': location, 'reason': reason}
            return {'status': 'ok', 'action': 'goto', 'location': location}

        elif name == 'dock':
            self.dock_pub.publish(Bool(data=True))
            return {'status': 'started', 'action': 'dock'}

        elif name == 'stop':
            self.cmd_vel_pub.publish(Twist())
            return {'status': 'ok', 'action': 'stop'}

        elif name == 'move':
            direction = args.get('direction')
            duration = max(0.3, min(float(args.get('duration', 1.0)), 3.0))

            twist = Twist()
            if direction == 'forward':
                twist.linear.x = 0.1
            elif direction == 'backward':
                twist.linear.x = -0.1
            elif direction == 'left':
                twist.angular.z = 0.5
            elif direction == 'right':
                twist.angular.z = -0.5
            else:
                return {'status': 'error', 'reason': f'unknown direction: {direction}'}

            def _drive():
                end = time.monotonic() + duration
                while time.monotonic() < end:
                    self.cmd_vel_pub.publish(twist)
                    time.sleep(0.05)
                self.cmd_vel_pub.publish(Twist())

            threading.Thread(target=_drive, daemon=True).start()
            return {'status': 'ok', 'action': 'move', 'direction': direction, 'duration': duration}

        elif name == 'patrol':
            self.patrol_pub.publish(Bool(data=True))
            return {'status': 'started', 'action': 'patrol'}

        elif name == 'explore':
            self.explore_pub.publish(Bool(data=True))
            return {'status': 'started', 'action': 'explore'}

        elif name == 'stop_explore':
            self.explore_pub.publish(Bool(data=False))
            return {'status': 'stopped', 'action': 'explore'}

        elif name == 'report_clutter':
            if self._latest_objects is None:
                return {'status': 'ok', 'objects': [], 'note': 'no detections yet'}
            clutter = [o for o in self._latest_objects if o.get('clutter')]
            return {'status': 'ok', 'objects': clutter}

        elif name == 'remember':
            fact = (args.get('fact') or '').strip()
            if not fact:
                return {'status': 'error', 'reason': 'empty fact'}
            self.memory_store_pub.publish(String(data=fact))
            return {'status': 'ok', 'action': 'remember', 'fact': fact}

        elif name == 'pick':
            obj = (args.get('object') or '').strip().lower()
            if not obj:
                return {'status': 'error', 'reason': 'no object specified'}
            self.pick_pub.publish(String(data=json.dumps({'label': obj})))
            return {'status': 'started', 'action': 'pick', 'object': obj}

        elif name == 'fetch':
            obj = (args.get('object') or '').strip().lower()
            if not obj:
                return {'status': 'error', 'reason': 'no object specified'}
            self.mission_pub.publish(String(data=f'fetch:{obj}'))
            return {'status': 'started', 'action': 'fetch', 'object': obj}

        elif name == 'follow':
            self.follow_pub.publish(Bool(data=True))
            return {'status': 'started', 'action': 'follow'}

        elif name == 'stop_follow':
            self.follow_pub.publish(Bool(data=False))
            return {'status': 'stopped', 'action': 'follow'}

        elif name == 'system_status':
            mem = psutil.virtual_memory()
            disk = shutil.disk_usage('/')
            load1, load5, load15 = os.getloadavg()
            status = {
                'status': 'ok',
                'cpu_percent': psutil.cpu_percent(interval=0.5),
                'cpu_count': psutil.cpu_count(),
                'load_avg_1m': round(load1, 2),
                'load_avg_5m': round(load5, 2),
                'load_avg_15m': round(load15, 2),
                'ram_percent': mem.percent,
                'ram_used_gb': round(mem.used / 1e9, 1),
                'ram_free_gb': round(mem.available / 1e9, 1),
                'ram_total_gb': round(mem.total / 1e9, 1),
                'disk_percent': round(disk.used / disk.total * 100, 1),
                'disk_free_gb': round(disk.free / 1e9, 1),
                'uptime_hours': round((time.time() - psutil.boot_time()) / 3600, 1),
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
