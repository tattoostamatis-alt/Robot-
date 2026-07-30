#!/usr/bin/env python3
"""A/B tool-calling bench for the Max voice pipeline.

Sends the SAME prompts llm_bridge_node sends -- same SYSTEM_PROMPT, same TOOLS
schema, same temperature (0.1) and max_tokens (120) -- to whatever model is
currently served on :52625, and scores whether the right tool fired.

SYSTEM_PROMPT and TOOLS are lifted straight out of llm_bridge_node.py via ast so
this bench can never drift from the node. The node imports rclpy, so we exec
only the three top-level assignments we need.

FLM rule: exactly ONE request in flight at a time. This is strictly serial.
"""
import argparse, ast, json, pathlib, statistics, sys, time
import requests

NODE = pathlib.Path.home() / 'robot_ws/src/home_robot/home_robot/nodes/llm_bridge_node.py'
WANT = {'SYSTEM_PROMPT', '_ROOMS', 'TOOLS'}


def load_contract():
    """Pull SYSTEM_PROMPT/_ROOMS/TOOLS out of the node without importing ROS."""
    tree = ast.parse(NODE.read_text(encoding='utf-8'))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in WANT:
                exec(compile(ast.Module([node], []), '<contract>', 'exec'), ns)
    missing = WANT - ns.keys()
    if missing:
        sys.exit(f'could not extract {missing} from {NODE}')
    return ns['SYSTEM_PROMPT'], ns['TOOLS']


# (utterance, expected_tool_or_None, expected_args_subset)
# None means "must NOT call a tool" -- chat, not action.
CASES = [
    ('πήγαινε στην κουζίνα',                      'goto',           {'location': 'kouzina'}),
    ('πήγαινε στο σαλόνι',                        'goto',           {'location': 'saloni'}),
    ('πήγαινε στο δωμάτιο του Μαξ',               'goto',           {'location': 'domatio tou max'}),
    ('καθάρισε το σαλόνι',                        'tidy',           {'room': 'saloni'}),
    ('σκούπισε όλο το σπίτι',                     'tidy',           {'room': 'all'}),
    ('πήγαινε στη βάση φόρτισης',                 'dock',           {}),
    ('σταμάτα',                                   'stop',           {}),
    ('σταμάτα αμέσως',                            'stop',           {}),
    ('προχώρα λίγο μπροστά',                      'move',           {'direction': 'forward'}),
    ('κάνε πίσω',                                 'move',           {'direction': 'backward'}),
    ('ξεκίνα εξερεύνηση',                         'explore',        {}),
    ('σταμάτα την εξερεύνηση',                    'stop_explore',   {}),
    ('ξεκίνα περιπολία',                          'patrol',         {}),
    ('τι σκόρπια αντικείμενα υπάρχουν;',          'report_clutter', {}),
    ('πόση μπαταρία έχεις;',                      'system_status',  {}),
    ('πες μου την κατάσταση του συστήματος',      'system_status',  {}),
    ('θυμήσου ότι τα κλειδιά είναι στο συρτάρι',  'remember',       {}),
    ('φέρε μου το μπουκάλι',                      'fetch',          {'object': 'bottle'}),
    ('πιάσε το κύπελλο',                          'pick',           {'object': 'cup'}),
    ('ακολούθησέ με',                             'follow',         {}),
    ('μη με ακολουθείς άλλο',                     'stop_follow',    {}),
    # negatives: chatting, must NOT fire a tool
    ('γεια σου Μαξ, τι κάνεις;',                  None,             {}),
    ('ποιος σε έφτιαξε;',                         None,             {}),
    ('τι καιρό κάνει σήμερα;',                    None,             {}),
]


def chat(url, model, system, tools, text, temperature, max_tokens):
    payload = {'model': model,
               'messages': [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': text}],
               'temperature': temperature, 'max_tokens': max_tokens,
               'tools': tools}
    t0 = time.monotonic()
    r = requests.post(f'{url}/chat/completions', json=payload, timeout=180)
    dt = time.monotonic() - t0
    r.raise_for_status()
    return r.json()['choices'][0]['message'], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--url', default='http://127.0.0.1:52625/v1')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--max-tokens', type=int, default=120)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    system, tools = load_contract()
    print(f'model={args.model}  cases={len(CASES)}  repeat={args.repeat}  '
          f'temp={args.temperature}  max_tokens={args.max_tokens}', flush=True)

    rows, lat = [], []
    for utt, want_tool, want_args in CASES:
        hits = 0
        detail = []
        for _ in range(args.repeat):
            try:
                msg, dt = chat(args.url, args.model, system, tools, utt,
                               args.temperature, args.max_tokens)
            except Exception as e:
                detail.append(f'ERR:{e}')
                continue
            lat.append(dt)
            calls = msg.get('tool_calls') or []
            if want_tool is None:
                ok = not calls
                detail.append('-' if ok else f'!{calls[0]["function"]["name"]}')
            elif not calls:
                ok = False
                detail.append('none')
            else:
                fn = calls[0]['function']
                got = fn['name']
                try:
                    got_args = json.loads(fn.get('arguments') or '{}')
                except Exception:
                    got_args = {}
                name_ok = got == want_tool
                args_ok = all(str(got_args.get(k, '')).strip().lower()
                              == str(v).strip().lower() for k, v in want_args.items())
                ok = name_ok and args_ok
                detail.append(got if name_ok and not args_ok
                              else (got if not name_ok else 'ok'))
                if name_ok and not args_ok:
                    detail[-1] = f'{got}{got_args}'
            hits += ok
        rate = hits / args.repeat
        rows.append((utt, want_tool or 'CHAT', rate, detail))
        flag = '✔' if rate == 1 else ('~' if rate > 0 else '✗')
        print(f'  {flag} {rate:4.0%}  {want_tool or "CHAT":<14} {utt[:40]:<42} {detail}',
              flush=True)

    actions = [r for r in rows if r[1] != 'CHAT']
    chats = [r for r in rows if r[1] == 'CHAT']
    a_score = statistics.mean(r[2] for r in actions) if actions else 0
    c_score = statistics.mean(r[2] for r in chats) if chats else 0
    overall = statistics.mean(r[2] for r in rows)
    print(f'\n  ACTION accuracy : {a_score:.1%}  ({len(actions)} cases)')
    print(f'  CHAT  restraint : {c_score:.1%}  ({len(chats)} cases)')
    print(f'  OVERALL         : {overall:.1%}')
    if lat:
        print(f'  latency med/p90 : {statistics.median(lat):.1f}s / '
              f'{sorted(lat)[int(len(lat) * 0.9)]:.1f}s')

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {'model': args.model, 'action': a_score, 'chat': c_score,
             'overall': overall,
             'median_latency': statistics.median(lat) if lat else None,
             'rows': [{'utt': u, 'want': w, 'rate': r, 'detail': d}
                      for u, w, r, d in rows]}, ensure_ascii=False, indent=2))
        print(f'  → {args.out}')


if __name__ == '__main__':
    main()
