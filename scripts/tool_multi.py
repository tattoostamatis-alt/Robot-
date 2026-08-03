#!/usr/bin/env python3
"""Multi-command tool-calling bench for the Max voice pipeline.

The third scenario the other two benches miss. `tool_ab.py` sends ONE command
per utterance; `tool_repeat.py` sends the SAME command several times. Neither
covers what was seen live on 2026-07-30:

    Heard: «Πήγαινε στη βάση, ακολούθησέ με, τι ώρα είναι.»
    Max:   «Πήγαινα στη βάση, ακολουθώ τον χρήστη και η ώρα είναι 22:30.»
    Tool calls: NONE. The robot never moved.

The model narrates every requested action as if it had happened, without
calling a single tool. That failure only appears when one utterance carries
SEVERAL DIFFERENT commands, so it needs its own bench.

Scoring is deliberately asymmetric, because the two ways of being wrong are not
equally bad:

  * a MISSING tool is the bug — the robot silently does nothing while claiming
    it did. Every expected tool must fire.
  * an EXTRA tool is noise, reported but not failed: the dispatcher runs what it
    is given, and an extra `system_status` costs a query, not a lie.

SYSTEM_PROMPT and TOOLS are lifted out of llm_bridge_node.py with ast, exactly
as tool_ab.py does it, so the bench can never drift from the node. The node
imports rclpy, so only the top-level assignments we need are exec'd.

Both backends are supported. Pick the one you actually run:

    ./tool_multi.py                      # gemini (the default backend)
    ./tool_multi.py --backend flm        # FastFlowLM on :52625, strictly serial
    ./tool_multi.py --repeat 5           # tool calls are probabilistic
"""
import argparse
import collections
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tool_ab import extract  # noqa: E402  — one extractor, shared by all benches

WANT = {'SYSTEM_PROMPT', '_ROOMS', '_APPS', 'TOOLS'}
FLM_URL = 'http://localhost:52625/v1/chat/completions'


def load_contract():
    """SYSTEM_PROMPT/TOOLS as the node sends them."""
    ns = extract(WANT)
    return ns['SYSTEM_PROMPT'], ns['TOOLS']


# (utterance, Counter of tools that MUST fire)
#
# Counts, not a set: "προχώρα μπροστά και μετά στρίψε δεξιά" is TWO `move`
# calls, and scoring by name alone scored a single one as a pass -- hiding
# exactly the dropped-second-command this bench exists to catch.
#
# Not everything conversational is a tool, and expecting one where the node
# doesn't use one produces fake failures:
#   * the time/date ride in the system context (_situation_system_message), so
#     "τι ώρα είναι" must NOT call system_status;
#   * "τι βλέπεις" is answered from the camera description already attached to
#     the message.
# Both appear below as commands paired with a question, because that mix is
# what was actually spoken on 2026-07-30.
#
# The shapes vary on purpose so a fix cannot be overfitted to one sentence:
# «και»-joined vs «μετά»-joined vs comma-joined, two commands vs three.
CASES = [
    ('Πήγαινε στη βάση, ακολούθησέ με, τι ώρα είναι.',
     collections.Counter({'dock': 1, 'follow': 1})),
    ('Πήγαινε στην κουζίνα και μετά καθάρισε το σαλόνι.',
     collections.Counter({'goto': 1, 'tidy': 1})),
    ('Άνοιξε το firefox και πες μου την κατάσταση του συστήματος.',
     collections.Counter({'open_app': 1, 'system_status': 1})),
    ('Σταμάτα να με ακολουθείς και πήγαινε στο σαλόνι.',
     collections.Counter({'stop_follow': 1, 'goto': 1})),
    ('Καθάρισε την κουζίνα, μετά πήγαινε στη βάση.',
     collections.Counter({'tidy': 1, 'dock': 1})),
    ('Θυμήσου ότι τα κλειδιά είναι στο τραπέζι και πήγαινε στο σαλόνι.',
     collections.Counter({'remember': 1, 'goto': 1})),
    ('Προχώρα μπροστά ένα μέτρο και μετά στρίψε δεξιά.',
     collections.Counter({'move': 2})),
    ('Πες μου τι βλέπεις και εξερεύνησε το σπίτι.',
     collections.Counter({'explore': 1})),
]


# The free tier allows 15 generate_content calls per minute per model. A full
# --repeat 3 run is 24 calls, so an unthrottled bench dies half way through with
# a 429 and reports nothing. Pace ourselves just under the limit and still
# honour the server's retryDelay if we misjudge it.
_MIN_INTERVAL_S = 4.2
_last_call = [0.0]


def call_gemini(system_prompt, tools, utterance, temperature):
    """One turn against Gemini, same contract as _handle_text_gemini()."""
    from google import genai
    from google.genai import types

    sys.path.insert(0, str(pathlib.Path.home() / 'robot_ws/src/home_robot'))
    from home_robot import api_keys

    key = api_keys.gemini_key()
    if not key:
        sys.exit('no Gemini key — see home_robot/api_keys.py')

    decls = [types.FunctionDeclaration(name=t['function']['name'],
                                       description=t['function']['description'],
                                       parameters=t['function']['parameters'])
             for t in tools]
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(function_declarations=decls)],
        temperature=temperature)
    contents = [types.Content(role='user', parts=[types.Part(text=utterance)])]

    for attempt in range(4):
        wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.models.generate_content(
                model='gemini-3.5-flash-lite', contents=contents, config=config)
            break
        except Exception as e:
            if '429' not in str(e) or attempt == 3:
                raise
            # "Please retry in 37.2s" -- believe the server over our own pacing.
            delay = 30.0
            match = re.search(r'retry in ([\d.]+)s', str(e))
            if match:
                delay = float(match.group(1)) + 1
            print(f'      … 429, αναμονή {delay:.0f}s')
            time.sleep(delay)
        finally:
            _last_call[0] = time.monotonic()

    out = resp.candidates[0].content
    called = [p.function_call.name for p in (out.parts or []) if p.function_call]
    text = ''.join(p.text or '' for p in (out.parts or []) if p.text).strip()
    return called, text


def call_flm(system_prompt, tools, utterance, temperature):
    """One turn against FastFlowLM. ONE request in flight at a time."""
    import requests

    body = {
        'model': 'local', 'temperature': temperature, 'max_tokens': 120,
        'tools': tools,
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': utterance}],
    }
    r = requests.post(FLM_URL, json=body, timeout=180)
    r.raise_for_status()
    msg = r.json()['choices'][0]['message']
    called = [tc['function']['name'] for tc in (msg.get('tool_calls') or [])]
    return called, (msg.get('content') or '').strip()


def _fmt(counter):
    """`goto, move×2` — the multiplier only shows when it matters."""
    return ', '.join(f'{name}×{n}' if n > 1 else name
                     for name, n in sorted(counter.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('gemini', 'flm'), default='gemini')
    ap.add_argument('--repeat', type=int, default=1,
                    help='runs per case; tool calling is probabilistic')
    ap.add_argument('--temperature', type=float, default=0.1,
                    help="matches the node's default")
    ap.add_argument('-v', '--verbose', action='store_true',
                    help="print the model's reply — that is where the lie shows")
    args = ap.parse_args()

    system_prompt, tools = load_contract()
    call = call_gemini if args.backend == 'gemini' else call_flm

    print(f'backend={args.backend}  repeat={args.repeat}  '
          f'temp={args.temperature}  {len(CASES)} cases\n')

    total = passed = 0
    silent = 0                       # the actual bug: narrated, called nothing
    missing_by_tool = collections.Counter()
    latencies = []

    for utterance, expected in CASES:
        for run in range(args.repeat):
            total += 1
            t0 = time.monotonic()
            try:
                called, text = call(system_prompt, tools, utterance, args.temperature)
            except Exception as e:
                print(f'✗ ERROR {utterance!r}\n    {type(e).__name__}: {e}')
                continue
            latencies.append(time.monotonic() - t0)

            got = collections.Counter(called)
            missed = expected - got          # Counter subtraction: keeps counts
            extra = got - expected

            if not missed:
                passed += 1
                mark, note = '✓', ''
            else:
                mark = '✗'
                note = f'  ΛΕΙΠΕΙ: {_fmt(missed)}'
                missing_by_tool.update(missed)
                if not got:
                    silent += 1
                    note += '  ← ΚΑΝΕΝΑ tool (το bug)'

            tag = f'[{run + 1}]' if args.repeat > 1 else '   '
            print(f'{mark} {tag} {utterance}')
            print(f'      κλήθηκαν: {_fmt(got) or "—"}{note}')
            if extra:
                print(f'      επιπλέον (δεν μετράει): {_fmt(extra)}')
            if args.verbose and text:
                print(f'      είπε: {text}')

    print(f'\n{"=" * 62}')
    print(f'ΠΕΡΑΣΑΝ  {passed}/{total}')
    print(f'ΣΙΩΠΗΛΑ  {silent}/{total}  (αφηγήθηκε χωρίς να καλέσει ΚΑΝΕΝΑ tool)')
    if missing_by_tool:
        worst = ', '.join(f'{t}×{c}' for t, c in missing_by_tool.most_common())
        print(f'ΧΑΜΕΝΑ   {worst}')
    if latencies:
        print(f'ΧΡΟΝΟΣ   μέσος {sum(latencies) / len(latencies):.2f}s  '
              f'max {max(latencies):.2f}s')

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
