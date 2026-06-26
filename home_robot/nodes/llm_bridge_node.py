#!/usr/bin/env python3
"""LLM bridge — Qwen3 (via ollama), Qwen3.5 (via Lemonade/NPU,
OpenAI-compatible API), or Gemini (via the google-genai API) tool calling,
between speech_text and robot actions / speech_response.

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
- `goto(location=...)` — looks up `config/locations.yaml` and sends a real
  nav2_msgs/action/NavigateToPose goal (same ActionClient pattern as
  task_planner_node.py's `_navigate`), blocking on the result so the tool
  reply reflects actual success/failure instead of a blind "started".
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
import time

import ollama
import psutil
import rclpy
import requests
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from geometry_msgs.msg import Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String

load_dotenv(os.path.expanduser('~/.env'))


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

ΣΗΜΑΝΤΙΚΟ: Αν ο χρήστης σου ζητήσει ρητά να θυμηθείς κάτι (π.χ. "θυμήσου
ότι...", "να θυμάσαι ότι...", "κράτα στο μυαλό σου ότι..."), ΠΡΕΠΕΙ να
καλέσεις το tool 'remember' με το ακριβές γεγονός — μην απαντάς απλά ότι
θα το θυμηθείς χωρίς να καλέσεις το tool, αλλιώς δεν αποθηκεύεται πουθενά.

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
        'name': 'move',
        'description': 'Κάνε μια μικρή χειροκίνητη κίνηση για λίγα δευτερόλεπτα — '
                        'π.χ. "κάνε λίγο μπροστά", "πήγαινε λίγο πίσω", "γύρνα δεξιά". '
                        'ΟΧΙ για μετάβαση σε δωμάτιο (χρήση goto) ή καθαρισμό/περιπολία.',
        'parameters': {'type': 'object', 'properties': {
            'direction': {'type': 'string', 'enum': ['forward', 'backward', 'left', 'right'],
                          'description': 'Κατεύθυνση κίνησης (left/right = στροφή επί τόπου)'},
            'duration': {'type': 'number',
                         'description': 'Διάρκεια κίνησης σε δευτερόλεπτα (0.3-3, default 1)'},
        }, 'required': ['direction']},
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
        'name': 'remember',
        'description': 'Θυμήσου μόνιμα ένα γεγονός/πληροφορία για μελλοντική χρήση '
                        '(π.χ. "θυμήσου ότι το κλειδί είναι στο συρτάρι της κουζίνας"). '
                        'Χρησιμοποίησέ το ΜΟΝΟ όταν ο χρήστης ζητάει ρητά να θυμηθείς κάτι.',
        'parameters': {'type': 'object', 'properties': {
            'fact': {'type': 'string', 'description': 'Το γεγονός/πληροφορία προς απομνημόνευση'},
        }, 'required': ['fact']},
    }},
    {'type': 'function', 'function': {
        'name': 'system_status',
        'description': 'Έλεγξε την κατάσταση του υπολογιστή/ρομπότ — CPU (χρήση, '
                        'πυρήνες, φόρτος), ελεύθερη/χρησιμοποιούμενη μνήμη RAM, '
                        'δίσκος, θερμοκρασία, χρόνος λειτουργίας, μπαταρία '
                        '(π.χ. "πώς είσαι;", "όλα καλά;", "πόση μνήμη/RAM σου έχει '
                        'μείνει ελεύθερη;", "πόση μπαταρία έχεις;")',
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
        self.declare_parameter('lemonade_model', 'qwen3.5-9b-FLM')
        self.declare_parameter('keep_alive', '10m')
        self.declare_parameter('temperature', 0.1)
        self.declare_parameter('history_turns', 4)
        self.declare_parameter('vision_timeout', 60.0)
        self.declare_parameter('nav_timeout', 60.0)
        self.declare_parameter('memory_enabled', False)
        self.declare_parameter('memory_timeout', 5.0)

        self.backend = self.get_parameter('backend').value
        self.model = self.get_parameter('model').value
        self.gemini_model = self.get_parameter('gemini_model').value
        self.lemonade_url = self.get_parameter('lemonade_url').value
        self.lemonade_model = self.get_parameter('lemonade_model').value
        self.keep_alive = self.get_parameter('keep_alive').value
        self.temperature = self.get_parameter('temperature').value
        self.history_turns = self.get_parameter('history_turns').value
        self.vision_timeout = self.get_parameter('vision_timeout').value
        self.nav_timeout = self.get_parameter('nav_timeout').value
        self.memory_enabled = self.get_parameter('memory_enabled').value
        self.memory_timeout = self.get_parameter('memory_timeout').value

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
        self.vision_query_pub = self.create_publisher(String, 'vision/query', 10)
        self.memory_store_pub = self.create_publisher(String, 'memory/store', 10)
        self.memory_query_pub = self.create_publisher(String, 'memory/query', 10)

        self._history = []
        self._busy = threading.Lock()
        self._latest_objects = None
        self._vision_event = threading.Event()
        self._vision_answer = None
        self._latest_battery = None
        self._memory_event = threading.Event()
        self._memory_answer = None
        self._situation: dict = {}   # from situational_awareness_node, optional

        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)
        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)
        self.create_subscription(String, 'vision/answer', self._on_vision_answer, 10)
        self.create_subscription(BatteryState, 'battery/state', self._on_battery_state, 10)
        self.create_subscription(String, 'memory/answer', self._on_memory_answer, 10)
        self.create_subscription(String, 'situation_context', self._on_situation, 10)

        if self.backend == 'gemini':
            active_model = self.gemini_model
        elif self.backend == 'lemonade':
            active_model = self.lemonade_model
        else:
            active_model = self.model
        self.get_logger().info(f'LLM bridge started — backend={self.backend} model={active_model}')

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

    def _on_memory_answer(self, msg: String):
        self._memory_answer = msg.data
        self._memory_event.set()

    def _on_situation(self, msg: String):
        try:
            self._situation = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

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
        if self.backend == 'gemini':
            self._handle_text_gemini(text)
        elif self.backend == 'lemonade':
            self._handle_text_lemonade(text)
        else:
            self._handle_text_ollama(text)

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
        user_msg = {'role': 'user', 'content': text}
        messages.append(user_msg)

        def _chat(msgs, temperature, with_tools):
            payload = {'model': self.lemonade_model, 'messages': msgs, 'temperature': temperature}
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

        turn = [user_msg]
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
        user_content = types.Content(role='user', parts=[types.Part(text=text)])
        contents.append(user_content)
        turn = [user_content]

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

    def _navigate(self, loc):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return False, 'το Nav2 δεν είναι έτοιμο'

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(loc['x'])
        goal.pose.pose.position.y = float(loc['y'])
        goal.pose.pose.orientation = _yaw_to_quaternion(float(loc['yaw']))

        send_future = self.nav_client.send_goal_async(goal)
        goal_handle = send_future.result(timeout=10.0)
        if goal_handle is None:
            return False, 'καμία απάντηση από το Nav2'
        if not goal_handle.accepted:
            return False, 'ο στόχος απορρίφθηκε'

        result_future = goal_handle.get_result_async()
        result = result_future.result(timeout=self.nav_timeout)
        if result is None:
            return False, 'λήξη χρόνου πλοήγησης'
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            return False, 'η πλοήγηση απέτυχε'
        return True, None

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

        elif name == 'remember':
            fact = (args.get('fact') or '').strip()
            if not fact:
                return {'status': 'error', 'reason': 'empty fact'}
            self.memory_store_pub.publish(String(data=fact))
            return {'status': 'ok', 'action': 'remember', 'fact': fact}

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
