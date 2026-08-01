"""Where cloud API keys come from, for nodes that need one.

The ROS nodes start from a tty, from `robot max`, or by hand with `ros2 run` —
three different environments, only one of which reliably inherits an exported
variable. Reading `os.environ` alone meant a node started the "wrong" way died
with a bare `KeyError: 'GEMINI_API_KEY'`, which says nothing about what to do
about it.

So the key is resolved from the environment first (so an export still wins, and
CI can inject one), then from a file under ~/.home_robot — the same place the
dashboard token lives, for the same reason: it survives reboots and is not in
git.

‼️ Never log or publish the value. `describe()` exists so a node can say where
the key came from without saying what it is.
"""
import os

GEMINI_KEY_FILE = os.path.expanduser('~/.home_robot/gemini_api_key')
GEMINI_ENV = 'GEMINI_API_KEY'


def _read_file(path):
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def gemini_key():
    """The Gemini API key, or None. Environment wins over the file."""
    return os.environ.get(GEMINI_ENV) or _read_file(GEMINI_KEY_FILE)


def describe():
    """Where the key came from, for logging. Never includes the key itself."""
    if os.environ.get(GEMINI_ENV):
        return f'${GEMINI_ENV}'
    if _read_file(GEMINI_KEY_FILE):
        return GEMINI_KEY_FILE
    return 'nowhere'


MISSING_MSG = (
    f'No Gemini API key. Put it in {GEMINI_KEY_FILE} '
    f'(chmod 600) or export {GEMINI_ENV}. Get one at '
    'https://aistudio.google.com/apikey'
)
