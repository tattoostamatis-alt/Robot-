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
from home_robot import api_keys
from home_robot import arm_motion
from home_robot import llm_quota
from home_robot.status_query import (ROOM_NAMES_EL, format_location,
                                     format_status, is_location_query,
                                     is_status_query, rooms_from_locations,
                                     wants_battery)
from home_robot.motion_confirm import (MOTION_TOOLS, PENDING_TTL,
                                       confirm_question, is_affirmative,
                                       is_negative)
from home_robot.simple_commands import parse_arm_command, parse_simple_motion
from home_robot.stop_command import is_stop_command, strip_accents
from home_robot.tool_router import select_tools
from home_robot.vocabulary import (greek_accusative, greek_for,
                                   needs_open_vocab, to_prompt)
from home_robot import desktop
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import BatteryState, Image, JointState
from std_msgs.msg import Bool, Empty, Float32, String

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

Για τον εαυτό σου μίλα ΠΑΝΤΑ σε α' ενικό: "χρειάζομαι", "δεν ξέρω". Ποτέ σε β' πρόσωπο. Πρόσεχε τη σειρά των λέξεων: "μου αρέσει", όχι "αρέσει μου".

ΜΗΝ εφευρίσκεις. Αν δεν ξέρεις κάτι για τον εαυτό σου, το σπίτι ή τον λόγο που κάτι πήγε στραβά, πες "δεν ξέρω" — μην μαντεύεις αιτίες.

ΟΜΩΣ: αν ένα tool γυρίσει "reason", ΑΥΤΟΣ είναι ο λόγος και τον ΞΕΡΕΙΣ. Πες τον στον χρήστη με δικά σου λόγια, σύντομα. ΠΟΤΕ "δεν ξέρω γιατί" όταν υπάρχει "reason".

Τα δωμάτια είναι ΜΟΝΟ αυτά· στα tools γράψε το greeklish, στην ομιλία πες το ελληνικό: {rooms}.

Κάλεσε tool όταν ο χρήστης ζητάει ενέργεια (κίνηση, καθαρισμό, docking, εξερεύνηση, αναφορά αντικειμένων, πιάσιμο αντικειμένου με τον βραχίονα, να σε ακολουθήσει) ή ρωτάει κατάσταση/μπαταρία. Αν απλώς συζητά, απάντα χωρίς tool.

Αν ζητήσει ΠΟΛΛΕΣ ενέργειες σε μία πρόταση, κάλεσε tool για ΚΑΘΕ μία, στη σειρά που τις είπε — ακόμη κι αν λέει "μετά", "στη συνέχεια", "έπειτα". ΠΟΤΕ μην πεις ότι έκανες κάτι για το οποίο δεν κάλεσες tool.

Αν ζητήσει να θυμηθείς κάτι ("θυμήσου...", "να θυμάσαι..."), ΠΑΝΤΑ κάλεσε 'remember' — αλλιώς χάνεται.

Αν η ερώτηση αφορά την εικόνα, η περιγραφή της κάμερας συνοδεύει ήδη το μήνυμα.

Στα "Κοντινά αντικείμενα" το `label@1.2m` σημαίνει ότι το αντικείμενο απέχει 1.2 μέτρα από σένα. Τα labels είναι αγγλικά — πες τα στα Ελληνικά (person=άτομο, chair=καρέκλα, bottle=μπουκάλι, cup=κούπα, couch=καναπές, tv=τηλεόραση). Όταν λες τι βλέπεις, ΠΡΟΣΘΕΣΕ την απόσταση αν υπάρχει: "βλέπω μια καρέκλα στο 1.2, έναν καναπέ στα 2.4 μέτρα". Αν ένα αντικείμενο ΔΕΝ είναι στη λίστα, μην εφευρίσκεις απόσταση — πες ότι δεν την ξέρεις."""

# The room list is the map's, so spell it out from the shared dict rather than
# typing it twice: a remap that renames a room must not leave the model naming
# rooms the map no longer has. (.replace, not .format — the prompt has braces.)
SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    '{rooms}', ', '.join(f'{key}={el}' for key, el in ROOM_NAMES_EL.items()))

# Nudge for the follow-up call, added only when a dispatched tool reported
# 'started' — i.e. the robot is now moving and has NOT arrived. Without it the
# model reports those as finished ("Πήγα στην κουζίνα.", "Έχω φτάσει στην βάση
# μου.") while the wheels are still turning. Blocking tools (goto returns after
# nav2 reports arrival) legitimately answer in the past tense, so they skip it.
ACTION_STARTED_NOTE = ('Η ενέργεια ΜΟΛΙΣ ξεκίνησε και συνεχίζεται — ΔΕΝ έχει '
                       'ολοκληρωθεί. Απάντησε σε ενεστώτα («Πηγαίνω…», '
                       '«Ξεκινάω…»), ποτέ σε παρελθοντικό χρόνο.')

# NOTE: keep descriptions terse. FastFlowLM (NPU) has max_prefill_len=4096 and the
# whole TOOLS schema is serialized into every prefill — verbose descriptions overflow
# it ("Max length reached!" 400). This concise set fits with room to spare.
_APPS = ['calculator', 'files', 'firefox', 'pdf_viewer', 'terminal',
         'text_editor', 'video_player', 'vscode']

# The map's rooms, from the one dict that is checked against locations.yaml.
_ROOMS = list(ROOM_NAMES_EL)

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'tidy',
        # ‼️ Said "Τακτοποίησε/καθάρισε ένα δωμάτιο", which collides head-on
        # with `sort` — "τακτοποίησε το σπίτι" went here and the robot drove
        # around reporting clutter instead of picking any of it up. This node
        # SWEEPS; it does not touch anything with the arm.
        'description': 'Σκούπισε/καθάρισε ένα δωμάτιο με τη σκούπα. ΟΧΙ πιάσιμο '
                       'αντικειμένων — γι\' αυτό είναι το sort.',
        'parameters': {'type': 'object', 'properties': {
            'room': {'type': 'string', 'enum': _ROOMS + ['all']},
        }, 'required': ['room']},
    }},
    {'type': 'function', 'function': {
        'name': 'open_app',
        'description': 'Άνοιξε εφαρμογή στην οθόνη του υπολογιστή',
        'parameters': {'type': 'object', 'properties': {
            'app': {'type': 'string', 'enum': _APPS},
        }, 'required': ['app']},
    }},
    {'type': 'function', 'function': {
        'name': 'close_app',
        'description': 'Κλείσε εφαρμογή στην οθόνη ("κλείσε όλα" → all)',
        'parameters': {'type': 'object', 'properties': {
            'app': {'type': 'string', 'enum': _APPS + ['all']},
        }, 'required': ['app']},
    }},
    {'type': 'function', 'function': {
        'name': 'list_windows',
        'description': 'Τι είναι ανοιχτό στην οθόνη του υπολογιστή',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        # ‼️ 2026-08-04: "λέω κάνε τον βραχίονα λίγο μπροστά και δεν κάνει
        # τίποτα". It was not the phrasing — there was no arm tool at all.
        # `pick`/`fetch` drive the whole pick-and-place pipeline at an object;
        # nothing could simply move the arm. The dashboard sliders and the PS5
        # stick could, and that was the whole list.
        'name': 'arm',
        'description': 'Κούνα τον βραχίονα προς μια κατεύθυνση. ΟΧΙ για πιάσιμο '
                       'αντικειμένου — γι\' αυτό είναι το pick.',
        'parameters': {'type': 'object', 'properties': {
            'direction': {'type': 'string',
                          'enum': list(arm_motion.DIRECTIONS)},
            'amount': {'type': 'string', 'enum': list(arm_motion.AMOUNTS),
                       'description': 'πόσο· default «λίγο»'},
        }, 'required': ['direction']},
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
        # Distinct from `tidy`, which only drives around reporting clutter.
        # This one actually picks things up and carries them.
        'name': 'sort',
        'description': ('ΠΙΑΣΕ σκόρπια αντικείμενα με τον βραχίονα και βάλ\' τα '
                        'στη θέση τους. Για "μάζεψε τα παιχνίδια", "τακτοποίησε '
                        'το σπίτι", "βάλ\' τα στη θέση τους".'),
        'parameters': {'type': 'object', 'properties': {
            'destination': {'type': 'string',
                            'description': 'προαιρετικό: μόνο ό,τι πάει σε αυτό '
                                           'το δωμάτιο (greeklish)'},
        }},
    }},
    {'type': 'function', 'function': {
        # The mission behind this has existed since the mission executor was
        # written and was unreachable by voice — there was no tool for it, so
        # "πήγαινε να δεις αν έκλεισα το παράθυρο" could only be triggered by
        # publishing to mission/start by hand.
        'name': 'check',
        'description': ('Πήγαινε σε ΑΛΛΟ δωμάτιο, κοίτα, γύρνα πίσω και πες τι '
                        'είδες. Για "δες αν...", "τσέκαρε αν...". ΟΧΙ για το '
                        'δωμάτιο που είναι ήδη.'),
        'parameters': {'type': 'object', 'properties': {
            'room': {'type': 'string', 'description': 'greeklish όνομα δωματίου'},
            'question': {'type': 'string',
                         'description': 'τι να ελέγξει, π.χ. "είναι κλειστό το παράθυρο;"'},
        }, 'required': ['room']},
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
        # ‼️ 2026-08-11: a user said «πήγαινε πιάστην» (go grab it) and the
        # model called pick on an object 1.83m away — pick never drives, so
        # it correctly refused (max_reach 0.45m) instead of fetching. The old
        # description didn't say pick stays put, so "πήγαινε"/"φέρε" phrasing
        # had nothing to steer it toward fetch. Now explicit both ways.
        'description': ('Πιάσε αντικείμενο που είναι ΗΔΗ σε εμβέλεια βραχίονα (~0.45m) — '
                         'ΔΕΝ οδηγεί. Αν ο χρήστης λέει "πήγαινε"/"φέρε" ή το αντικείμενο '
                         'μπορεί να είναι μακριά, χρησιμοποίησε fetch αντί γι\' αυτό.'),
        'parameters': {'type': 'object', 'properties': {
            # ‼️ NOT "COCO label" any more. That instruction made the model
            # answer «παντόφλα» with the nearest COCO-ish word it could think of
            # — 'shoe', which is not a COCO class either, so the pick was
            # unsatisfiable from the start. pick now routes anything outside the
            # 80 through the open-vocabulary detector, so the honest instruction
            # is "name the thing in English".
            'object': {'type': 'string', 'description': 'αγγλικά, π.χ. cup, book, slipper'},
        }, 'required': ['object']},
    }},
    {'type': 'function', 'function': {
        'name': 'fetch',
        'description': ('Οδήγησε ως το αντικείμενο, πιάσε το και φέρε το πίσω '
                         '("φέρε μου/πήγαινε πιάσε το X"). Χρησιμοποίησέ το ΠΑΝΤΑ '
                         'όταν χρειάζεται μετακίνηση για να το φτάσεις.'),
        'parameters': {'type': 'object', 'properties': {
            # As of commit 0c334cc (2026-08-11) fetch also routes non-COCO
            # names through the open-vocab detector, same as pick — no longer
            # COCO-only.
            'object': {'type': 'string', 'description': 'αγγλικά, π.χ. cup, book, slipper'},
        }, 'required': ['object']},
    }},
    {'type': 'function', 'function': {
        # Distinct from `goto`, which only takes taught rooms. This one goes to
        # a THING, using the map position object memory recorded when the robot
        # last saw it. Kept terse: the whole TOOLS block is the prefill cost
        # (see project_robot_llm_latency_prefill).
        'name': 'goto_object',
        'description': 'Πήγαινε δίπλα σε ένα αντικείμενο, χωρίς να το πιάσεις',
        'parameters': {'type': 'object', 'properties': {
            'object': {'type': 'string', 'description': 'αγγλικό COCO label, π.χ. chair, tv, sofa'},
        }, 'required': ['object']},
    }},
    {'type': 'function', 'function': {
        'name': 'distance_to',
        'description': 'Πόσο μακριά είναι ένα ορατό αντικείμενο/άνθρωπος (μέτρα)',
        'parameters': {'type': 'object', 'properties': {
            'object': {'type': 'string', 'description': 'COCO label, default person'},
        }},
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
    {'type': 'function', 'function': {
        # Open vocabulary lookup only — doesn't grab or drive, unlike
        # pick/fetch (both of which also take non-COCO nouns as of 0c334cc).
        'name': 'find',
        'description': ('Ψάξε με τα μάτια για ΟΠΟΙΟΔΗΠΟΤΕ αντικείμενο, ακόμη κι '
                        'άγνωστο ("βρες τα κλειδιά", "ψάξε το πορτοφόλι")'),
        'parameters': {'type': 'object', 'properties': {
            'object': {'type': 'string', 'description': 'π.χ. κλειδιά, γυαλιά, φορτιστής'},
        }, 'required': ['object']},
    }},
    {'type': 'function', 'function': {
        # The floor point comes from gesture_node, which never drives on its
        # own. Saying "πήγαινε εκεί" is the confirmation.
        'name': 'goto_pointed',
        'description': 'Πήγαινε εκεί που δείχνει ο χρήστης με το χέρι ("πήγαινε εκεί")',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        # Distinct from the RAG `remember`/recall of facts: this one is indexed
        # by TIME, so it can answer "when".
        'name': 'recall',
        'description': ('Τι έγινε σε κάποια στιγμή/μέρα ("τι έγινε σήμερα", '
                        '"πότε ήρθε ο Μαξ", "τι είπα χθες")'),
        'parameters': {'type': 'object', 'properties': {
            'when': {'type': 'string',
                     'description': 'π.χ. "σήμερα το πρωί", "χθες", "πριν από 2 ώρες"'},
        }},
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
        # 2026-08-01: the user asked for 1.5 Flash; it is GONE — ListModels no
        # longer serves any 1.5 model, and generateContent returns 404
        # NOT_FOUND. gemini-3.6-flash is the current Flash tier. A 404 naming
        # the model (rather than a 401) always means the model, not the key.
        self.declare_parameter('gemini_model', 'gemini-3.5-flash-lite')
        # Defaults match what `robot max` actually serves: FastFlowLM on :52625
        # (base path /v1, NOT Lemonade's /api/v1) running qwen3.5:4b. They used to
        # name a Lemonade endpoint that has answered model_not_found since 11.0.0
        # dropped the NPU catalogue; the launch file overrode them, so the stale
        # values only ever surfaced when a node was started by hand.
        self.declare_parameter('lemonade_url', 'http://127.0.0.1:52625/v1')
        self.declare_parameter('lemonade_model', 'qwen3.5:4b')
        self.declare_parameter('keep_alive', '10m')
        self.declare_parameter('temperature', 0.1)
        self.declare_parameter('history_turns', 4)
        self.declare_parameter('nav_timeout', 60.0)
        # Ask before driving. ‼️ Most of what reaches speech_text was never
        # addressed to the robot — see home_robot/motion_confirm.py. Off by
        # user request 2026-08-11: motion tools now execute immediately, no
        # "Να πάω;" — knowingly reopens that overheard-speech risk.
        self.declare_parameter('confirm_motion', False)
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
        self.confirm_motion = bool(self.get_parameter('confirm_motion').value)
        # (tool, args, expires_at) for a motion tool waiting on a yes/no.
        self._pending_motion = None
        self._motion_confirmed = False   # set only while running a confirmed call
        self._confirm_asked = False      # a question was spoken this turn
        self.memory_enabled = self.get_parameter('memory_enabled').value
        self.memory_timeout = self.get_parameter('memory_timeout').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        self._gemini_client = None
        self._gemini_tool = None
        if self.backend == 'gemini':
            # A missing key at startup is fatal, as it always was: the node was
            # explicitly launched to talk to Gemini and cannot.
            self._init_gemini(fatal=True)

        locations_path = os.path.join(get_package_share_directory('home_robot'),
                                        'config', 'locations.yaml')
        with open(locations_path) as f:
            self.locations = yaml.safe_load(f)

        self.response_pub = self.create_publisher(String, 'speech_response', 10)
        # How much of today's cloud allowance is left. Latched, because it
        # changes once every few minutes and a dashboard that connects in
        # between must still see a number rather than a dash.
        self.quota_pub = self.create_publisher(
            String, '/llm_quota',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # ‼️ RELATIVE 'cmd_vel', and the launch file decides where it lands.
        #
        # Two configurations, and hardcoding either one breaks the other:
        #   * with use_obstacle_safety:=true, obstacle_safety_node sits between
        #     /cmd_vel and the wheels and zeroes forward motion when the camera
        #     sees something ahead. Its docstring names THIS node as the reason
        #     it exists. Publishing straight to cmd_vel_safe would walk past it.
        #   * localize.launch.py (what `robot max` runs) leaves it off, because
        #     without object_detector its fail-safe blocks all forward motion.
        #     Nothing then relays /cmd_vel, which has ZERO subscribers —
        #     measured live 2026-08-03 — so every spoken "προχώρα μπροστά"
        #     computed a correct twist and published it into nothing.
        # So bringup remaps this to cmd_vel_safe exactly when the safety net is
        # not running. See the `voice_cmd_vel` remapping there.
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tidy_pub = self.create_publisher(String, 'tidy_command', 10)
        # The arm, spoken. Positions are ABSOLUTE on the wire, so a relative
        # "λίγο μπροστά" needs the current pose — hence the joint_states
        # subscription. arm_driver clamps to its own limits; nothing here
        # second-guesses those numbers.
        self.arm_cmd_pub = self.create_publisher(JointState, '/arm/joint_cmd', 10)
        self.gripper_pub = self.create_publisher(Float32, '/arm/gripper_cmd', 10)
        self._arm_pose = {}
        self.create_subscription(JointState, '/arm/joint_states',
                                 self._on_arm_state, 10)
        self.patrol_pub = self.create_publisher(Bool, 'patrol_command', 10)
        self.explore_pub = self.create_publisher(Bool, 'explore_command', 10)
        self.memory_store_pub = self.create_publisher(String, 'memory/store', 10)
        self.memory_query_pub = self.create_publisher(String, 'memory/query', 10)
        # Gestures and the time-indexed timeline, both request/response like
        # memory/query above.
        self.gesture_go_pub = self.create_publisher(Empty, '/gesture_go', 10)
        self.episodic_query_pub = self.create_publisher(
            String, '/episodic/query', 10)
        self.vocab_pub = self.create_publisher(String, '/vocab/set', 10)
        self.pick_pub = self.create_publisher(String, 'pick_command', 10)
        self.follow_pub = self.create_publisher(Bool, 'follow_command', 10)
        self.mission_pub = self.create_publisher(String, 'mission/start', 10)

        self._history = []
        # Registered only now that _history exists: the callback clears it on a
        # backend switch, and rclpy fires this on the very next parameter set.
        #
        # Lets the dashboard flip Gemini <-> the local NPU model without
        # restarting the stack. A restart would drop every subscription and
        # (via `robot max` defaults) risk bringing the backend back to
        # something nobody asked for.
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._busy = threading.Lock()
        # Kept on self (not just as a _navigate local) so _emergency_stop can
        # cancel the goal from the speech callback thread.
        self._nav_goal_handle = None
        self._latest_objects = None
        self._objects_stamp = None      # when they arrived; see _distance_to
        self._latest_battery = None
        self._memory_event = threading.Event()
        self._memory_answer = None
        self._situation: dict = {}
        # Last floor point someone pointed at, and the episodic recall channel.
        self._gesture_point = None
        self._episodic_event = threading.Event()
        self._episodic_answer = None
        # Open-vocabulary search. `_vocab_baseline` is the inference count at
        # request time; see _cb_vocab_state.
        self._vocab_lock = threading.Lock()
        self._vocab_event = threading.Event()
        self._vocab_hits = None
        self._vocab_inferences = 0
        self._vocab_baseline = None
        self.declare_parameter('find_timeout', 8.0)
        self.find_timeout = self.get_parameter('find_timeout').value

        # `pick` used to be fire-and-forget, so the model was free to answer
        # «Απλώνομαι για να πιάσω την παντόφλα» while pick_place_node had
        # already given up 9 ms earlier — on 2026-08-06 the user heard the
        # refusal AND the confident lie, the lie last. We now wait just long
        # enough to catch the fast rejections (no matching detection, TF
        # failure, busy), which pick_place_node emits immediately. A pick that
        # is genuinely under way takes ~10 s of arm moves and reports later; we
        # do NOT wait for that, we just stop claiming success we cannot see.
        self._pick_lock = threading.Lock()
        self._pick_event = threading.Event()
        self._pick_result = None
        self.declare_parameter('pick_reject_window', 2.5)
        self.pick_reject_window = self.get_parameter('pick_reject_window').value

        # Closing the last metre to something the arm can see but not reach.
        # ‼️ _approach_cancel must be honoured by the drive loop: that loop
        # publishes Twist at 20 Hz, so without it a spoken «σταμάτα» would have
        # _emergency_stop's zero Twist overwritten on the very next cycle and
        # the robot would keep rolling at the user.
        self._approach_cancel = threading.Event()
        self.declare_parameter('pick_approach_max', 1.5)      # m — further than this is a `fetch`, not a nudge
        self.declare_parameter('pick_approach_margin', 0.7)   # fraction of reach to stop short at
        self.declare_parameter('pick_approach_timeout', 15.0) # s — hard cap on one open-loop drive
        self.declare_parameter('pick_approach_step', 0.25)    # m — max per nudge, re-measure between
        self.declare_parameter('pick_approach_tries', 8)      # nudge/re-measure cycles before giving up
        # ‼️ open_vocab_detector runs every 3rd frame and pick_place_node acts
        # on the last message it received, so a retry fired too soon is
        # answered from detections taken before the robot moved — which is how
        # a successful approach reported «δεν βλέπω τέτοιο αντικείμενο».
        self.declare_parameter('pick_approach_settle', 2.5)   # s — wait for fresh detections after moving
        self.pick_approach_max = self.get_parameter('pick_approach_max').value
        self.pick_approach_margin = self.get_parameter('pick_approach_margin').value
        self.pick_approach_timeout = self.get_parameter('pick_approach_timeout').value
        self.pick_approach_step = self.get_parameter('pick_approach_step').value
        self.pick_approach_tries = self.get_parameter('pick_approach_tries').value
        self.pick_approach_settle = self.get_parameter('pick_approach_settle').value

        # Camera frame — encoded to JPEG once on arrival, reused per message
        self._bridge = CvBridge()
        self._frame_lock = threading.Lock()
        self._latest_frame_jpg: bytes | None = None

        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)
        # Real position, for the location gate. room_markers_node derives this
        # from /amcl_pose, so an empty string genuinely means "not localized"
        # — which is the honest answer, not a room to guess at.
        #
        # TRANSIENT_LOCAL to match the latched publisher. /current_room follows
        # /amcl_pose, which a stationary robot does not publish at all, so with
        # the default volatile QoS this node learns the room only if it happens
        # to be running while the robot drives. Restarted mid-session on
        # 2026-07-31 next to a parked robot, it then claimed "ο εντοπισμός δεν
        # είναι ενεργός" indefinitely — pointing the user at a subsystem that
        # was working fine.
        self._current_room = ''
        self.create_subscription(
            String, '/current_room', self._on_current_room,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)
        self.create_subscription(BatteryState, 'battery/state', self._on_battery_state, 10)
        self.create_subscription(String, 'memory/answer', self._on_memory_answer, 10)
        self.create_subscription(String, '/gesture_status',
                                 self._cb_gesture_status, 10)
        self.create_subscription(String, '/episodic/answer',
                                 self._cb_episodic_answer, 10)
        self.create_subscription(String, '/vocab/state',
                                 self._cb_vocab_state, 10)
        self.create_subscription(String, 'pick_result', self._cb_pick_result, 10)
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

        # Latch today's allowance right away. The stack restarts several times
        # a day and the count does not reset with it, so the first thing the
        # dashboard should see is where the day actually stands.
        if self.backend == 'gemini':
            view = self._publish_quota()
            self.get_logger().info(
                f'Gemini quota: {view["used"]}/{view["limit"]} used today '
                f'({view["source"]} limit), resets in {view["resets_in"] // 3600}h')

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
            # Through the helper, not os.environ: a node started without the
            # dotenv path would silently return None here and the robot would
            # answer vision questions with nothing, blaming the camera.
            api_key = api_keys.gemini_key()
            if not api_key:
                return None
            client = genai.Client(api_key=api_key)
            prompt = (
                'Look at this robot camera image. Answer the following question '
                'in Greek in 1-3 short sentences, based only on what is visible.\n'
                f'Question: {question}'
            )
            resp = client.models.generate_content(
                # Was pinned to 'gemini-flash-lite-latest', which ListModels no
                # longer serves at all — this path 404'd while the main one
                # worked. Follow the node's own parameter instead.
                model=self.gemini_model,
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
            self._objects_stamp = time.time()
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
        # Date and time go in unconditionally — asked for 2026-07-30, after "τι
        # ώρα είναι;" was answered with "Δεν ξέρω την ώρα". A tool would cost a
        # second NPU round trip (~6s) plus its schema in every prefill; two
        # lines of context cost ~20 tokens and are always right.
        now = time.localtime()
        days = ['Δευτέρα', 'Τρίτη', 'Τετάρτη', 'Πέμπτη', 'Παρασκευή',
                'Σάββατο', 'Κυριακή']
        months = ['Ιανουαρίου', 'Φεβρουαρίου', 'Μαρτίου', 'Απριλίου', 'Μαΐου',
                  'Ιουνίου', 'Ιουλίου', 'Αυγούστου', 'Σεπτεμβρίου',
                  'Οκτωβρίου', 'Νοεμβρίου', 'Δεκεμβρίου']
        clock = [f'Ώρα: {now.tm_hour:02d}:{now.tm_min:02d}',
                 f'Ημερομηνία: {days[now.tm_wday]} {now.tm_mday} '
                 f'{months[now.tm_mon - 1]} {now.tm_year}']
        if not self._situation:
            return 'Τρέχουσα κατάσταση:\n' + '\n'.join(f'- {p}' for p in clock)
        s = self._situation
        parts = clock + [f"Δωμάτιο: {s.get('room', '?')}"]
        if 'objects' in s:
            parts.append(f"Κοντινά αντικείμενα: {s['objects']}")
        if 'battery_pct' in s:
            parts.append(f"Μπαταρία: {s['battery_pct']}%")
        parts.append(f"CPU: {s.get('cpu_pct', '?')}%  RAM: {s.get('ram_pct', '?')}%")
        return 'Τρέχουσα κατάσταση:\n' + '\n'.join(f'- {p}' for p in parts)

    def _find_object(self, name):
        """Look for an arbitrary object via the open-vocabulary detector.

        Distinct from `pick`/`fetch`, which go through object_detector and are
        therefore limited to the COCO 80. This one takes any noun — but it only
        LOOKS; it does not drive or grasp.
        """
        prompt = to_prompt(name)
        if prompt is None:
            return {'status': 'error', 'action': 'find', 'object': name,
                    'reason': f'δεν ξέρω τι είναι «{name}»'}

        # ‼️ Wait for an inference that happened AFTER this request. The
        # detector publishes its state on set too, with hits == [], and
        # settling on that would answer "δεν το βλέπω" before it had looked.
        with self._vocab_lock:
            self._vocab_baseline = self._vocab_inferences
            self._vocab_hits = None
        self._vocab_event.clear()
        self.vocab_pub.publish(String(data=json.dumps([prompt])))

        # The detector has to encode the prompt and then wait for a camera
        # frame, so give it noticeably longer than the RAG round trip.
        if not self._vocab_event.wait(timeout=self.find_timeout):
            return {'status': 'error', 'action': 'find', 'object': name,
                    'reason': 'open_vocab_detector did not answer — is '
                              'use_open_vocab:=true?'}

        hits = self._vocab_hits or []
        if not hits:
            return {'status': 'ok', 'action': 'find', 'object': name,
                    'found': False, 'greek': greek_for(prompt)}
        best = max(hits, key=lambda h: h.get('conf', 0))
        return {'status': 'ok', 'action': 'find', 'object': name,
                'found': True, 'greek': greek_for(prompt),
                'confidence': best.get('conf'),
                'distance_m': best.get('z')}

    def _cb_pick_result(self, msg):
        try:
            result = json.loads(msg.data) or {}
        except (json.JSONDecodeError, TypeError):
            return
        with self._pick_lock:
            self._pick_result = result
        self._pick_event.set()

    def _prepare_open_vocab_target(self, prompt):
        """Point the open-vocabulary detector at `prompt` and confirm it sees it.

        pick_place_node can only grasp what is in a detections message. For the
        80 COCO classes object_detector supplies that continuously; for anything
        else — a slipper, a charger — nothing is published until the
        open-vocabulary detector is told what to hunt for. So `pick` has to ask
        for it and wait for a hit BEFORE commanding the arm, otherwise the node
        looks for a label nobody is producing and reports «δεν βλέπω κάτι να
        σηκώσω» however plainly the object is sitting there.

        Returns None on success, or an error dict ready to hand back to the LLM.
        """
        with self._vocab_lock:
            self._vocab_baseline = self._vocab_inferences
            self._vocab_hits = None
        self._vocab_event.clear()
        self.vocab_pub.publish(String(data=json.dumps([prompt])))

        # ‼️ Keep looking until the deadline instead of settling for the first
        # state message. `find` can settle, because "δεν το βλέπω" is a valid
        # answer to a question. `pick` is an ACTION, and one empty frame is not
        # a reason to refuse it — measured 2026-08-06: this returned hits=0
        # after 128 ms while the slipper was being detected continuously at
        # conf 0.25-0.33. Two ways that happens, both real:
        #   * open_vocab_detector publishes its state on set_classes too, with
        #     hits cleared. When the detector was ALREADY running (an earlier
        #     find, or a previous pick), its inference count is past our
        #     baseline the moment we look, so that empty publish satisfies the
        #     "an inference ran after my request" test without a frame having
        #     been examined for us.
        #   * open-vocab confidences hover near the threshold, so any single
        #     frame can legitimately come back empty.
        # Re-arming the baseline on each pass makes every subsequent message a
        # fresh inference, so this waits for frames rather than for messages.
        deadline = time.monotonic() + self.find_timeout
        answered = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not self._vocab_event.wait(timeout=remaining):
                break
            answered = True
            with self._vocab_lock:
                hits = self._vocab_hits or []
                self._vocab_baseline = self._vocab_inferences
                self._vocab_hits = None
            self._vocab_event.clear()
            if hits:
                return None

        self.get_logger().info(
            f'open-vocab prepare({prompt}): answered={answered}, no hits '
            f'within {self.find_timeout:.0f}s')
        if not answered:
            return {'status': 'error', 'action': 'pick',
                    'reason': 'ο ανιχνευτής ανοιχτού λεξιλογίου δεν απάντησε — '
                              'τρέχει με use_open_vocab:=true;'}
        return {'status': 'ok', 'action': 'pick', 'found': False,
                'reason': f'δεν βλέπω {greek_for(prompt)} μπροστά μου'}

    def _pick_object(self, name):
        """`pick`, routed through whichever detector can actually see `name`."""
        prompt = to_prompt(name)
        if prompt is None:
            return {'status': 'error', 'action': 'pick', 'object': name,
                    'reason': f'δεν ξέρω τι είναι «{name}»'}

        if needs_open_vocab(prompt):
            failure = self._prepare_open_vocab_target(prompt)
            if failure is not None:
                failure['object'] = name
                return failure
            label = prompt
        else:
            # object_detector emits COCO labels with underscores ('cell_phone'),
            # while to_prompt speaks them with spaces.
            label = prompt.replace(' ', '_')

        # ‼️ Nobody listening means nothing will ever happen, and the silence is
        # indistinguishable from a grasp in progress — so the wait below would
        # time out and we would report 'started'. Caught live on 2026-08-06:
        # pick_place_node was down, the robot cheerfully said «Απλώνω τον
        # βραχίονα για να πιάσω την παντόφλα», and the arm never moved. This is
        # the same class of bug as tidy_command/patrol_command answering
        # "Ξεκινάω περιπολία" into a topic with no subscriber.
        if self.pick_pub.get_subscription_count() == 0:
            return {'status': 'error', 'action': 'pick', 'object': name,
                    'reason': 'ο κόμβος του βραχίονα δεν τρέχει (pick_place_node)'}

        # ‼️ Drive up to it rather than refusing. An object on the floor in
        # front of the robot is typically ~1 m away and the arm reaches 0.45 m,
        # so "πιάσε την παντόφλα" answered «είναι πολύ μακριά» essentially
        # always — technically true and useless: the user pointed at a thing
        # one step away and expected the robot to take that step. `fetch` can
        # already approach, but it resolves the target through object memory
        # and the map, which is the wrong machinery for something that is
        # visible right now. So pick closes the gap itself and retries once.
        # ‼️ Closed loop, not one corrective step. Measured on hardware: asked
        # for 0.74 m, the robot covered ~0.30 m — soft_start_accel ramps the
        # wheels slowly and obstacle_safety trims cmd_vel, so an open-loop
        # timed drive under-shoots badly and unpredictably. Re-measuring from
        # the camera after each nudge converges anyway; guessing a fudge factor
        # for the drive would not.
        for attempt in range(self.pick_approach_tries):
            result = self._send_pick(label)
            if result.get('status') in ('ok', 'started'):
                break
            last = attempt == self.pick_approach_tries - 1
            if result.get('status') != 'out_of_reach' or last:
                reason = (result.get('detail') or result.get('status')
                          or 'ο βραχίονας δεν μπόρεσε να το πιάσει')
                self.get_logger().info(f'pick({label}) refused: {reason}')
                return {'status': 'error', 'action': 'pick', 'object': name,
                        'reason': reason}

            gap = self._approach_gap(result)
            if gap is None:
                return {'status': 'error', 'action': 'pick', 'object': name,
                        'reason': result.get('detail') or 'είναι πολύ μακριά'}
            if not attempt:            # announce the approach once, not per nudge
                self.response_pub.publish(
                    String(data=f'Πλησιάζω {greek_accusative(prompt)}.'))
            if not self._drive_forward(gap):
                return {'status': 'error', 'action': 'pick', 'object': name,
                        'reason': 'με σταμάτησες ενώ πλησίαζα'}

        return {'status': 'started', 'action': 'pick', 'object': name,
                'greek': greek_accusative(prompt)}

    def _send_pick(self, label):
        """One pick_command round trip. {} when nothing answered in time."""
        with self._pick_lock:
            self._pick_result = None
        self._pick_event.clear()
        self.pick_pub.publish(String(data=json.dumps({'label': label})))

        # Only a rejection arrives this fast; a real grasp is still moving, so
        # a timeout here means "under way", not "failed".
        if not self._pick_event.wait(timeout=self.pick_reject_window):
            return {'status': 'started'}
        with self._pick_lock:
            return self._pick_result or {}

    def _approach_gap(self, result):
        """How far to drive so the target lands inside the arm's envelope.

        None when the object is further than one approach should cover — at
        that range it is a `fetch` (navigate, avoid obstacles, re-look), not a
        nudge, and driving blindly toward something metres away is exactly the
        kind of open-loop motion this codebase keeps regretting.
        """
        distance, reach = result.get('distance'), result.get('max_reach')
        if not isinstance(distance, (int, float)) or distance > self.pick_approach_max:
            return None
        reach = reach if isinstance(reach, (int, float)) else 0.45
        # Stop short of the reach limit: the drive is open-loop (no odometry
        # feedback here) and overshooting puts the robot on top of the object.
        want = max(0.0, distance - reach * self.pick_approach_margin)
        # ‼️ And never in one go. The camera looks forward, not down, so an
        # object on the floor leaves the bottom of the frame as the robot gets
        # close — overshoot is not "slightly too far", it is losing sight of
        # the target entirely. Live 2026-08-06: one 0.45 m nudge drove past the
        # slipper, and the next look saw only floor, which read as «δεν βλέπω
        # τέτοιο αντικείμενο» right after a working approach. Small steps with
        # a fresh measurement between them cost seconds and cannot overshoot.
        return min(want, self.pick_approach_step)

    def _drive_forward(self, metres):
        """Blocking, open-loop nudge forward at the teleop speed.

        Goes out on cmd_vel, which bringup remaps to cmd_vel_safe — so
        obstacle_safety and collision_monitor can still stop it. Blocking on
        purpose: the retry pick must not fire while the robot is still rolling.
        """
        self._approach_cancel.clear()
        speed = 0.1                                   # m/s, matches `move`
        duration = min(metres / speed, self.pick_approach_timeout)
        self.get_logger().info(f'pick approach: {metres:.2f} m ({duration:.1f} s)')
        twist = Twist()
        twist.linear.x = speed
        end = time.monotonic() + duration
        while time.monotonic() < end and not self._approach_cancel.is_set():
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        self.cmd_vel_pub.publish(Twist())
        if self._approach_cancel.is_set():
            return False
        time.sleep(self.pick_approach_settle)
        return True

    def _cb_vocab_state(self, msg):
        try:
            state = json.loads(msg.data) or {}
        except (json.JSONDecodeError, TypeError):
            return
        count = state.get('inferences', 0)
        with self._vocab_lock:
            self._vocab_inferences = count
            baseline = self._vocab_baseline
            # A miss is a legitimate answer, so settle on the first inference
            # that ran after the request rather than on a non-empty hit list —
            # otherwise "δεν το βλέπω" would always time out instead.
            if baseline is not None and count > baseline:
                self._vocab_hits = state.get('hits') or []
                self._vocab_baseline = None
                self._vocab_event.set()

    def _goto_pointed(self):
        """Act on the last confirmed pointing gesture.

        gesture_node deliberately never drives itself, so this tool IS the
        confirmation step — the user pointed, then said "πήγαινε εκεί".
        """
        if self._gesture_point is None:
            return {'status': 'error', 'action': 'goto_pointed',
                    'reason': 'no gesture seen — δείξε με το χέρι στο πάτωμα '
                              'και κράτα το για λίγο'}
        self.gesture_go_pub.publish(Empty())
        x, y = self._gesture_point
        return {'status': 'started', 'action': 'goto_pointed',
                'x': round(x, 2), 'y': round(y, 2)}

    def _recall(self, when):
        """Ask episodic_memory_node what happened in a period."""
        self._episodic_answer = None
        self._episodic_event.clear()
        self.episodic_query_pub.publish(String(data=when))
        if not self._episodic_event.wait(timeout=self.memory_timeout):
            self.get_logger().warn('episodic_memory_node did not respond (timeout)')
            return {'status': 'error', 'action': 'recall',
                    'reason': 'episodic memory unavailable'}
        return {'status': 'ok', 'action': 'recall', 'when': when,
                'answer': self._episodic_answer}

    def _cb_gesture_status(self, msg):
        try:
            last = (json.loads(msg.data) or {}).get('last')
        except (json.JSONDecodeError, TypeError):
            return
        self._gesture_point = (last['x'], last['y']) if last else None

    def _cb_episodic_answer(self, msg):
        self._episodic_answer = msg.data
        self._episodic_event.set()

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
            self._pending_motion = None      # a stop also drops the question
            threading.Thread(target=self._emergency_stop, daemon=True).start()
            return

        # Answering "να πάω στο σαλόνι;". Ahead of the busy lock and the model,
        # like stop, and for the same reason: the answer must not queue behind
        # anything, and a yes/no is not a question worth a model call.
        if self._pending_motion is not None:
            if self._answer_pending_motion(text):
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

        # "Where are you" is the same trap. Measured 2026-07-27 with no AMCL
        # running: it answered "Είμαι στη βάση φόρτισης" with no tool call at
        # all — a room invented as confidently as a real one. Answered from
        # /current_room, which room_markers_node derives from /amcl_pose.
        # No lock needed: this reads one cached string and speaks.
        if is_location_query(text):
            self.get_logger().info(f'Location query (keyword gate): {text}')
            reply = format_location(self._current_room)
            self.get_logger().info(f'Max: {reply}')
            self.response_pub.publish(String(data=reply))
            return

        # Bare goto/move/turn/arm, ahead of the LLM for latency (see
        # simple_commands.py). Unlike these same tools called by the model,
        # a match here skips motion_confirm too — the phrasing is narrow and
        # verb-anchored enough not to be the overheard-conversation shape
        # confirm_motion exists for. Takes the busy lock like a status query:
        # a real action, so it must not interleave with one already running.
        simple = (parse_arm_command(text)
                  or parse_simple_motion(text, rooms_from_locations(self.locations)))
        if simple is not None and self._busy.acquire(blocking=False):
            name, args = simple
            self.get_logger().info(f'Simple motion command (keyword gate): {text} -> {name}{args}')

            def _run():
                try:
                    self._motion_confirmed = True
                    result = self._dispatch_tool(name, args)
                finally:
                    self._motion_confirmed = False
                    self._busy.release()
                self.get_logger().info(f'{name} -> {result}')
                self._speak_dispatch_result(name, result)

            threading.Thread(target=_run, daemon=True).start()
            return

        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already handling a request, ignoring speech_text')
            return
        threading.Thread(target=self._handle_text, args=(text,), daemon=True).start()

    def _publish_reply(self, reply):
        """The model's final word — unless a confirmation question replaced it.

        ‼️ When a motion tool asks "Να πάω στο σαλόνι;" the question has already
        been spoken by _dispatch_tool. Whatever the model composes afterwards is
        a second sentence on top of it, and the first live test showed what that
        sentence is: «Πηγαίνω στο σαλόνι» — announcing the trip it was just
        stopped from taking. Two voices, one of them wrong. The question wins.
        """
        if self._confirm_asked:
            self._confirm_asked = False
            if reply:
                self.get_logger().info(f'[suppressed after confirmation] {reply}')
            return
        self.get_logger().info(f'Max: {reply}')
        self.response_pub.publish(String(data=reply))

    def _answer_pending_motion(self, text) -> bool:
        """Handle a yes/no to a motion question. True if it was consumed here.

        Three outcomes, and the third is the important one: anything that is
        neither yes nor no means the room moved on to something else, so the
        pending action is DROPPED rather than left armed waiting for a "ναι"
        that belongs to a different sentence.
        """
        pending = self._pending_motion
        if pending is None:
            return False
        tool, args, expires = pending
        if time.monotonic() > expires:
            self._pending_motion = None
            self.get_logger().info(f'Motion confirmation for {tool} expired')
            return False

        if is_affirmative(text):
            self._pending_motion = None
            self.get_logger().info(f'Motion confirmed by voice: {tool} {args}')

            def _run():
                try:
                    self._motion_confirmed = True
                    result = self._dispatch_tool(tool, args)
                finally:
                    self._motion_confirmed = False
                self.get_logger().info(f'{tool} → {result}')
                self._speak_dispatch_result(tool, result)

            threading.Thread(target=_run, daemon=True).start()
            return True

        if is_negative(text):
            self._pending_motion = None
            self.get_logger().info(f'Motion declined by voice: {tool}')
            self.response_pub.publish(String(data='Εντάξει, δεν πάω.'))
            return True

        self._pending_motion = None
        self.get_logger().info(
            f'No yes/no for {tool} — dropping it and treating this as new speech')
        return False

    def _speak_dispatch_result(self, tool, result):
        """Voice the outcome of a motion tool confirmed by "ναι".

        ‼️ _run() (above) calls _dispatch_tool() directly, bypassing the model
        entirely — same reliability reason the confirmation itself is a
        keyword gate, not a model call. That means nothing else turns this
        result dict into words: found live 2026-08-11 as total silence after
        "Να ψάξω μαύρη παντόφλα;" → "ναι" — the search ran (8s, no hits) and
        nobody ever heard the answer.
        """
        status = result.get('status')
        if tool == 'find':
            if status == 'error':
                reply = f"Δεν μπόρεσα να ψάξω: {result.get('reason', 'άγνωστο πρόβλημα')}"
            else:
                greek = result.get('greek')
                acc = greek_accusative(greek) if greek else 'το αντικείμενο'
                if result.get('found'):
                    reply = f"Βρήκα {acc}."
                else:
                    reply = f"Δεν βλέπω {acc}."
        elif status == 'error':
            reply = f"Δεν μπόρεσα: {result.get('reason', 'άγνωστο πρόβλημα')}"
        elif status == 'cancelled':
            reply = 'Ακυρώθηκε.'
        elif status == 'started':
            reply = 'Ξεκίνησα.'
        else:
            reply = 'Έγινε.'
        self._publish_reply(reply)

    def _on_arm_state(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._arm_pose[name] = float(pos)

    def _move_arm(self, direction, amount):
        """"Κάνε τον βραχίονα λίγο μπροστά" — relative, from where it is now.

        ‼️ Requires a live pose. The wire format is absolute positions, so with
        no /arm/joint_states the only options are to guess a starting point or
        to refuse; guessing would fling the arm from wherever it happens to be
        to wherever the guess said. It refuses, and says why.
        """
        deltas = arm_motion.deltas_for(direction, amount)
        if deltas is None:
            return {'status': 'error', 'action': 'arm',
                    'reason': f'unknown direction: {direction}',
                    'known': list(arm_motion.DIRECTIONS)}
        if not self._arm_pose:
            return {'status': 'error', 'action': 'arm',
                    'reason': 'δεν παίρνω θέση από τον βραχίονα — τρέχει ο arm_driver;'}

        js = JointState()
        moved = {}
        for joint, delta in deltas.items():
            cur = self._arm_pose.get(joint)
            if cur is None:
                continue
            target = cur + delta
            js.name.append(joint)
            js.position.append(target)
            moved[joint] = round(target, 3)
        if not js.name:
            return {'status': 'error', 'action': 'arm',
                    'reason': f'no joint state for {list(deltas)}'}

        js.header.stamp = self.get_clock().now().to_msg()
        self.arm_cmd_pub.publish(js)
        return {'status': 'ok', 'action': 'arm',
                'moved': arm_motion.describe(direction, amount), 'targets': moved}

    def _on_current_room(self, msg: String):
        self._current_room = msg.data.strip()

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
        # Before any zero Twist: a pick approach drive publishes at 20 Hz and
        # would simply overwrite it.
        self._approach_cancel.set()

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
        # Cleared per turn, not only in _publish_reply: a backend that produces
        # an empty reply returns early and never reaches it, and a flag left set
        # would swallow the NEXT answer instead of this one.
        self._confirm_asked = False
        try:
            self._handle_text_inner(text)
        finally:
            self._busy.release()

    def _init_gemini(self, fatal: bool = False) -> bool:
        """Build the Gemini client and tool schema. False if there is no key."""
        key = api_keys.gemini_key()
        if not key:
            # Loud and actionable. os.environ['GEMINI_API_KEY'] used to raise a
            # bare KeyError here, which named the variable but not what to do
            # about it.
            self.get_logger().error(api_keys.MISSING_MSG)
            if fatal:
                raise RuntimeError(api_keys.MISSING_MSG)
            return False
        from google import genai
        self.get_logger().info(f'Gemini key from {api_keys.describe()}')
        self._gemini_client = genai.Client(api_key=key)
        self._gemini_tool = _build_gemini_tool()
        return True

    def _on_set_parameters(self, params):
        """Allow `backend` to be switched at runtime.

        Only the backend is handled here; everything else is accepted and
        stored so a `ros2 param set` does not silently do nothing.

        ‼️ Switching TO lemonade does not start FastFlowLM — `robot max`
        deliberately leaves it stopped when the backend is gemini, to keep its
        4.7 GB free. The dashboard starts the server first and only then sets
        this parameter; from the CLI you must start it yourself.
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name != 'backend':
                continue
            new = p.value
            if new not in ('gemini', 'lemonade', 'ollama'):
                return SetParametersResult(
                    successful=False,
                    reason=f"unknown backend '{new}' "
                           "(gemini | lemonade | ollama)")
            if new == 'gemini' and self._gemini_client is None:
                if not self._init_gemini():
                    return SetParametersResult(
                        successful=False, reason='no Gemini API key')
            # The history is Gemini `types.Content` on one backend and OpenAI
            # dicts on the other. Carrying it across would hand the new backend
            # objects it cannot parse, so drop it: the cost is that the robot
            # forgets the last few turns, which beats a backend that throws on
            # the first thing you say to it.
            self._history.clear()
            self.backend = new
            self.get_logger().info(f'LLM backend switched to {new}')
        return SetParametersResult(successful=True)

    def _handle_text_inner(self, text):
        self.get_logger().info(f'Heard: {text}')
        if self.backend == 'gemini':
            self._handle_text_gemini(text)
        elif self.backend == 'lemonade':
            self._handle_text_lemonade(text)
        else:
            self._handle_text_ollama(text)

    def _remember_turn(self, turn, acted):
        """Commit a turn to the rolling history — unless it fired a tool.

        Measured on qwen3.5:4b: leaving an action turn (tool_call + tool result
        + "Πήγα στην κουζίνα.") in the history makes a REPEAT of the same
        command answer from that history instead of calling the tool again —
        1/3 on explore/follow/dock/goto, and the reply claims the action
        happened while the robot stands still. The tell is latency collapsing
        (8.6s → 5.4s → 2.2s): it is recalling, not working.

        Commands are idempotent and need no continuity; chat does. So chat
        turns are remembered and action turns are not. The cost is that "τι σου
        είπα πριν;" cannot recall a past command — worth it to never claim an
        action that did not happen.
        """
        if acted:
            return
        self._history.append(turn)
        del self._history[:-self.history_turns]

    # ── lemonade backend ───────────────────────────────────────────────────

    def _handle_text_lemonade(self, text):
        # ‼️ FastFlowLM's chat template raises "System message must be at the
        # beginning" for a system message that follows any non-system message —
        # and returns it as {"error": ...} with HTTP 200, so it surfaces as a
        # KeyError, not a 4xx. Leading system messages are fine (measured), but
        # merging them into one keeps that invariant trivially true; the
        # follow-up call below depends on it.
        parts = [SYSTEM_PROMPT]
        parts += [m for m in (self._memory_system_message(self._retrieve_memories(text)),
                              self._situation_system_message()) if m]
        system_msg = {'role': 'system', 'content': '\n\n'.join(parts)}
        messages = [system_msg]
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

        def _chat(msgs, temperature, tools=None):
            # Cap the reply length. Note: generation is NOT what makes answers
            # slow — measured 2026-07-31, a reply takes ~0.4 s to write while
            # the prefill takes 5.3 s, because FastFlowLM re-reads the whole
            # prompt every call (no prefix caching). The cap stays as a bound on
            # rambling, but latency is won in the prefill — see the `tools`
            # argument and tool_router.py.
            payload = {'model': self.lemonade_model, 'messages': msgs,
                       'temperature': temperature, 'max_tokens': 120}
            if tools:
                payload['tools'] = tools
            r = requests.post(f'{self.lemonade_url}/chat/completions', json=payload, timeout=120)
            r.raise_for_status()
            body = r.json()
            # FLM reports template/prefill errors in a 200 body. Without this the
            # caller dies on a KeyError that says nothing about the real cause.
            if 'choices' not in body:
                raise RuntimeError(f'FLM error: {body.get("error", body)}')
            return body['choices'][0]['message']

        # Send only the tools this utterance could plausibly need. The full
        # schema is 1847 of 1941 prompt tokens, so "τι ώρα είναι;" was paying
        # ~3 s to re-read how to pick up a cup. Unrecognised-but-action-ish
        # phrasings still get the full set; see tool_router.py.
        tools = select_tools(text, TOOLS)
        if len(tools) != len(TOOLS):
            self.get_logger().debug(
                f'Tool routing: {len(tools)}/{len(TOOLS)} tools '
                f'({[t["function"]["name"] for t in tools]})')

        try:
            out = _chat(messages, self.temperature, tools=tools)
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')
            return

        turn = [text_user_msg]
        tool_calls = out.get('tool_calls') or []

        if tool_calls:
            messages.append(out)
            turn.append(out)
            started = False
            for tc in tool_calls:
                fn = tc['function']
                args = json.loads(fn.get('arguments') or '{}')
                self.get_logger().info(f'Tool call: {fn["name"]}({args})')
                result = self._dispatch_tool(fn['name'], args)
                started |= result.get('status') == 'started'
                tool_msg = {'role': 'tool', 'tool_call_id': tc.get('id', ''),
                            'content': json.dumps(result, ensure_ascii=False)}
                messages.append(tool_msg)
                turn.append(tool_msg)
            if started:
                # Same template rule: fold the note into the leading system
                # message rather than appending a second one.
                messages[0] = {'role': 'system',
                               'content': f'{system_msg["content"]}\n\n{ACTION_STARTED_NOTE}'}

            try:
                out2 = _chat(messages, 0.3)
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
        self._remember_turn(turn, bool(tool_calls))

        self._publish_reply(reply)

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
            started = False
            for tc in out.tool_calls:
                args = tc.function.arguments or {}
                self.get_logger().info(f'Tool call: {tc.function.name}({args})')
                result = self._dispatch_tool(tc.function.name, args)
                started |= result.get('status') == 'started'
                tool_msg = {'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)}
                messages.append(tool_msg)
                turn.append(tool_msg)
            if started:
                messages.append({'role': 'system', 'content': ACTION_STARTED_NOTE})

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
        self._remember_turn(turn, bool(out.tool_calls))

        self._publish_reply(reply)

    # ── gemini backend ─────────────────────────────────────────────────────

    def _publish_quota(self, view=None):
        """Put the current allowance on /llm_quota for the dashboard."""
        view = view if view is not None else llm_quota.snapshot()
        try:
            self.quota_pub.publish(String(data=json.dumps(view)))
        except Exception as e:            # never take the LLM down with it
            self.get_logger().debug(f'quota publish failed: {e}')
        return view

    def _gemini_generate(self, **kwargs):
        """The single door every Gemini request goes through, so it is counted.

        ‼️ A rejected request is NOT counted — Google does not charge a 429
        against the daily quota, and counting it here would make the number
        drift further from the truth every time we ran out. The rejection is
        worth more than a count anyway: it is the one place the real limit is
        stated, so it retunes the counter instead of incrementing it.
        """
        try:
            resp = self._gemini_client.models.generate_content(**kwargs)
        except Exception as e:
            view = llm_quota.note_rate_limit(e)
            if view is not None:
                self._publish_quota(view)
                self.get_logger().error(
                    f'Gemini refused: rate limit / quota. '
                    f'{view["used"]}/{view["limit"]} today, resets in '
                    f'{view["resets_in"] // 60} min')
            raise
        self._publish_quota(llm_quota.record())
        return resp

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
            resp = self._gemini_generate(
                model=self.gemini_model, contents=contents, config=config)
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')
            # ‼️ Out of quota is the one failure the person can act on, and
            # silence is the worst way to report it — it looks exactly like
            # the STT being deaf. Say it, once, in Greek.
            if llm_quota.is_rate_limit(e):
                self._publish_reply(llm_quota.describe())
            return

        out = resp.candidates[0].content
        function_calls = [p.function_call for p in out.parts if p.function_call]

        if function_calls:
            contents.append(out)
            turn.append(out)
            response_parts = []
            started = False
            for fc in function_calls:
                args = fc.args or {}
                self.get_logger().info(f'Tool call: {fc.name}({args})')
                result = self._dispatch_tool(fc.name, args)
                started |= result.get('status') == 'started'
                response_parts.append(types.Part.from_function_response(
                    name=fc.name, response=result))
            response_content = types.Content(role='user', parts=response_parts)
            contents.append(response_content)
            turn.append(response_content)

            try:
                followup_system = (f'{SYSTEM_PROMPT}\n\n{ACTION_STARTED_NOTE}'
                                   if started else SYSTEM_PROMPT)
                resp2 = self._gemini_generate(
                    model=self.gemini_model, contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=followup_system, temperature=0.3))
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
        self._remember_turn(turn, bool(function_calls))

        self._publish_reply(reply)

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

    def _start_docking(self):
        """Run the whole dock sequence, not just the last 60 cm of it.

        This used to publish `dock` directly, which starts the driver's IR
        homing wherever the robot happens to be standing. From another room
        there is no beam to find, so it swept for 20 s and gave up — while the
        reply had already said it was on its way. Docking is a mission
        (navigate to the handover point, relocalize on the tag, then IR home),
        so it goes to the mission executor like the other multi-step jobs.
        """
        self.mission_pub.publish(String(data='dock'))
        return {'status': 'started', 'action': 'dock'}

    def _dispatch_tool(self, name, args):
        # ‼️ Ask before moving. The model is not wrong to read "πάμε στο σαλόνι"
        # as a goto — it just was not said to the robot. See motion_confirm.py
        # for the timeline entry that started this. Confirmed calls come back
        # through here with _motion_confirmed set and fall straight through.
        if (self.confirm_motion and name in MOTION_TOOLS
                and not self._motion_confirmed):
            # ‼️ The question is SPOKEN HERE, not left to the model. First live
            # test, Gemini, with the tool result carrying an explicit "say
            # exactly «Να πάω στο σαλόνι;» and nothing else": it answered
            # «Πηγαίνω στο σαλόνι και μετά στον διάδρομο» — a confident lie
            # about an action that had just been refused. Exactly the failure
            # stop_command.py exists for. The gate is only worth anything if
            # what the person hears is also not the model's choice.
            if self._pending_motion is not None:
                # A second motion call in the same turn ("σαλόνι and then
                # διάδρομο"). One question at a time; the rest are dropped
                # rather than queued, because a single "ναι" cannot mean yes
                # to a journey the person never heard described.
                return {'status': 'ignored', 'action': name,
                        'reason': 'already waiting on a confirmation'}
            question = confirm_question(name, args)
            self._pending_motion = (name, args, time.monotonic() + PENDING_TTL)
            self._confirm_asked = True
            self.get_logger().info(f'Motion tool {name} needs confirmation: {question}')
            self.get_logger().info(f'Max: {question}')
            self.response_pub.publish(String(data=question))
            return {'status': 'needs_confirmation', 'action': name,
                    'question': question,
                    'instruction': 'The question has already been asked out '
                                   'loud. Reply with an empty string.'}

        if name == 'tidy':
            room = args.get('room', 'all')
            self.tidy_pub.publish(String(data=json.dumps({'room': room})))
            return {'status': 'started', 'action': 'tidy', 'room': room}

        elif name == 'goto':
            location = args.get('location')
            if location == 'dock':
                return self._start_docking()
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
            return self._start_docking()

        elif name == 'find':
            return self._find_object(args.get('object') or '')

        elif name == 'goto_pointed':
            return self._goto_pointed()

        elif name == 'recall':
            return self._recall(args.get('when') or '')

        elif name == 'open_app':
            return desktop.open_app(args.get('app'))

        elif name == 'close_app':
            return desktop.close_app(args.get('app'))

        elif name == 'list_windows':
            return desktop.list_windows()

        elif name == 'distance_to':
            return self._distance_to(args.get('object') or 'person')

        elif name == 'arm':
            return self._move_arm(args.get('direction'), args.get('amount'))

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
            return self._pick_object(obj)

        elif name == 'fetch':
            obj = (args.get('object') or '').strip().lower()
            if not obj:
                return {'status': 'error', 'reason': 'no object specified'}
            self.mission_pub.publish(String(data=f'fetch:{obj}'))
            return {'status': 'started', 'action': 'fetch', 'object': obj}

        elif name == 'sort':
            group = (args.get('destination') or '').strip().lower()
            if group and group not in self.locations:
                return {'status': 'error', 'reason': f'unknown room: {group}'}
            self.mission_pub.publish(
                String(data=f'sort:{group}' if group else 'sort'))
            return {'status': 'started', 'action': 'sort',
                    'destination': group or 'all'}

        elif name == 'check':
            room = (args.get('room') or '').strip().lower()
            if room not in self.locations:
                return {'status': 'error',
                        'reason': f'unknown room: {room or "(none)"}'}
            question = (args.get('question') or '').strip() or 'τι βλέπεις;'
            # ':' separates the mission's own fields, so a question containing
            # one would be parsed as a third field and silently truncated.
            question = question.replace(':', ' ')
            self.mission_pub.publish(String(data=f'check:{room}:{question}'))
            return {'status': 'started', 'action': 'check',
                    'room': room, 'question': question}

        elif name == 'goto_object':
            obj = (args.get('object') or '').strip().lower()
            if not obj:
                return {'status': 'error', 'reason': 'no object specified'}
            self.mission_pub.publish(String(data=f'goto_object:{obj}'))
            return {'status': 'started', 'action': 'goto_object', 'object': obj}

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

    # Detections carry no timestamp of their own, so note when they landed.
    # object_detector runs at ~2 Hz; anything older than this means it stopped
    # publishing (node down, camera segfaulted — that happens on ~2.5% of
    # boots). Answering "1.5 μέτρα" off a frozen frame would invent a person
    # who has already left the room, which is worse than saying "δεν βλέπω".
    DETECTION_MAX_AGE = 4.0

    def _distance_to(self, label: str) -> dict:
        """How far away is the nearest `label`, from the depth camera."""
        label = (label or '').strip().lower()
        if not self._latest_objects or self._objects_stamp is None:
            return {'status': 'error', 'object': label,
                    'reason': 'δεν φτάνουν ανιχνεύσεις — τρέχει ο object_detector;'}

        age = time.time() - self._objects_stamp
        if age > self.DETECTION_MAX_AGE:
            return {'status': 'error', 'object': label,
                    'reason': f'παλιές ανιχνεύσεις ({age:.0f}s) — η κάμερα σταμάτησε'}

        def distance_of(o):
            # box_distance is a median over the whole box and survives a single
            # bad depth pixel; z is the centre pixel and is the fallback.
            d = o.get('box_distance')
            return d if d is not None else o.get('z')

        matches = [o for o in self._latest_objects
                   if str(o.get('label', '')).lower() == label
                   and distance_of(o) is not None]
        if not matches:
            return {'status': 'ok', 'object': label, 'seen': False}

        best = min(matches, key=distance_of)
        dist = float(distance_of(best))
        # Camera optical frame: +x right, +z forward. Positive bearing = left,
        # matching how the answer gets spoken ("στα αριστερά σου").
        bearing = math.degrees(math.atan2(-float(best.get('x') or 0.0),
                                          max(float(best.get('z') or dist), 1e-3)))
        side = 'μπροστά' if abs(bearing) < 12 else (
            'αριστερά' if bearing > 0 else 'δεξιά')
        return {'status': 'ok', 'object': label, 'seen': True,
                'distance_m': round(dist, 2), 'bearing_deg': round(bearing),
                'side': side, 'count': len(matches)}

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
