#!/usr/bin/env python3
"""Repeat bench — does the SAME command still fire its tool the 2nd and 3rd time?

tool_ab.py sends every utterance as a fresh single-turn request, so it can only
ever measure first-time accuracy. This bench replays a command inside ONE
conversation, rebuilding the history exactly the way llm_bridge_node does
(user / assistant tool_call / tool result / assistant reply, history_turns,
temp 0.1 with tools then 0.3 without for the follow-up), including the node's
_remember_turn policy of not keeping action turns.

That policy is the thing under test: with action turns left in the history,
qwen3.5:4b answers repeats from memory ("Πήγα στην κουζίνα.") without calling
the tool at all — the robot says it moved and stands still. Run this alongside
tool_ab.py whenever the model or the history handling changes.

  --keep-action-turns  restores the old (broken) behaviour, to show the bug.

FLM rule: exactly ONE request in flight at a time. This is strictly serial.
"""
import argparse, json, pathlib, statistics, sys, time
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tool_ab import extract  # noqa: E402

# Dispatch results the node returns, so the follow-up call sees a real payload.
# 'started' = the robot is now moving and has NOT arrived; the node appends
# ACTION_STARTED_NOTE for those. Blocking tools (goto waits for nav2) return 'ok'.
STARTED = {'tidy', 'dock', 'patrol', 'explore', 'pick', 'fetch', 'follow'}

# (utterance, expected tool)
CASES = [('ξεκίνα εξερεύνηση', 'explore'),
         ('ακολούθησέ με', 'follow'),
         ('γύρνα στη βάση σου', 'dock'),
         ('πήγαινε στην κουζίνα', 'goto'),
         ('καθάρισε το σαλόνι', 'tidy')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--url', default='http://127.0.0.1:52625/v1')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--history-turns', type=int, default=4)
    ap.add_argument('--keep-action-turns', action='store_true',
                    help='reproduce the old bug: remember action turns too')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ns = extract({'SYSTEM_PROMPT', '_ROOMS', 'TOOLS', 'ACTION_STARTED_NOTE'})
    system, tools, note = (ns['SYSTEM_PROMPT'], ns['TOOLS'],
                           ns['ACTION_STARTED_NOTE'])

    def call(messages, temperature, with_tools):
        payload = {'model': args.model, 'messages': messages,
                   'temperature': temperature, 'max_tokens': 120}
        if with_tools:
            payload['tools'] = tools
        t0 = time.monotonic()
        r = requests.post(f'{args.url}/chat/completions', json=payload, timeout=180)
        r.raise_for_status()
        body = r.json()
        if 'choices' not in body:      # FLM reports errors in a 200 body
            sys.exit(f'FLM error: {body.get("error", body)}')
        return body['choices'][0]['message'], time.monotonic() - t0

    def turn(history, text):
        """One llm_bridge turn. Returns (tool_fired_or_None, reply, seconds)."""
        messages = [{'role': 'system', 'content': system}]
        for t in history:
            messages.extend(t)
        user_msg = {'role': 'user', 'content': text}
        messages.append(user_msg)

        out, dt = call(messages, 0.1, True)
        rec, fired = [user_msg], None
        calls = out.get('tool_calls') or []
        if calls:
            fired = calls[0]['function']['name']
            messages.append(out)
            rec.append(out)
            started = False
            for tc in calls:
                name = tc['function']['name']
                result = {'status': 'started' if name in STARTED else 'ok',
                          'action': name}
                started |= result['status'] == 'started'
                tool_msg = {'role': 'tool', 'tool_call_id': tc.get('id', ''),
                            'content': json.dumps(result, ensure_ascii=False)}
                messages.append(tool_msg)
                rec.append(tool_msg)
            if started:      # folded in, not appended: system must stay first
                messages[0] = {'role': 'system',
                               'content': f'{system}\n\n{note}'}
            out2, dt2 = call(messages, 0.3, False)
            dt += dt2
            reply = (out2.get('content') or '').strip()
        else:
            reply = (out.get('content') or '').strip()

        rec.append({'role': 'assistant', 'content': reply})
        if not calls or args.keep_action_turns:      # node's _remember_turn
            history.append(rec)
            del history[:-args.history_turns]
        return fired, reply, dt

    mode = 'KEEP action turns (old, broken)' if args.keep_action_turns \
        else 'drop action turns (current node)'
    print(f'model={args.model}  repeat={args.repeat}  history={mode}', flush=True)

    rows, lat = [], []
    for utt, want in CASES:
        history, turns, hits = [], [], 0
        for i in range(args.repeat):
            fired, reply, dt = turn(history, utt)
            ok = fired == want
            hits += ok
            lat.append(dt)
            turns.append({'n': i + 1, 'fired': fired, 'ok': ok, 'reply': reply,
                          's': round(dt, 1)})
            print(f'  {"✔" if ok else "✗"} #{i+1} {want:<8} fired={fired!s:<14} '
                  f'{dt:4.1f}s  «{reply[:70]}»', flush=True)
        rate = hits / args.repeat
        rows.append({'utt': utt, 'want': want, 'rate': rate, 'turns': turns})
        print(f'    → {rate:.0%}\n', flush=True)

    overall = statistics.mean(r['rate'] for r in rows)
    print(f'  REPEAT accuracy : {overall:.1%}  ({len(rows)} commands '
          f'x {args.repeat})')
    print(f'  latency med/p90 : {statistics.median(lat):.1f}s / '
          f'{sorted(lat)[int(len(lat) * 0.9)]:.1f}s')

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {'model': args.model, 'keep_action_turns': args.keep_action_turns,
             'overall': overall, 'rows': rows}, ensure_ascii=False, indent=2))
        print(f'  → {args.out}')
    return 0 if overall == 1 else 1


if __name__ == '__main__':
    sys.exit(main())
