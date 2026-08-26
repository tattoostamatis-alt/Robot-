#!/usr/bin/env python3
"""Interactive chat with the local Lemonade server, with proper UTF-8/Greek
input support and read-only system-stats tool access."""
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
import json
import urllib.request

import edge_tts
import numpy as np
import psutil
import requests
import sounddevice as sd
import chromadb
from chromadb.api.types import EmbeddingFunction

HOST = "http://127.0.0.1:13305"
MEMORY_DB_PATH = os.path.expanduser("~/.robot_memory")
EMBEDDING_MODEL = "embed-gemma-300m-FLM"
MEMORY_MAX_DISTANCE = 1.0
MEMORY_TOP_K = 3
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")

TTS_VOICE = "el-GR-AthinaNeural"
TTS_SAMPLE_RATE = 24000

SYSTEM_PROMPT = (
    "Είσαι ένας βοηθός που τρέχει τοπικά. Αν ο χρήστης ρωτήσει για την "
    "κατάσταση του υπολογιστή (CPU, μνήμη/RAM, δίσκος, θερμοκρασία, uptime), "
    "ΠΡΕΠΕΙ να καλέσεις το tool 'system_status' — μην απαντάς ποτέ μόνος σου "
    "σε τέτοιες ερωτήσεις χωρίς να το καλέσεις πρώτα.\n\n"
    "Αν ο χρήστης σου ζητήσει ρητά να θυμηθείς κάτι (π.χ. \"θυμήσου ότι...\", "
    "\"να θυμάσαι ότι...\"), ΠΡΕΠΕΙ να καλέσεις το tool 'remember' με το "
    "ακριβές γεγονός — μην απαντάς απλά ότι θα το θυμηθείς χωρίς να καλέσεις "
    "το tool, αλλιώς δεν αποθηκεύεται πουθενά.")

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'system_status',
        'description': 'Έλεγξε την κατάσταση του υπολογιστή — CPU, μνήμη, δίσκος, '
                        'θερμοκρασία, uptime.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'remember',
        'description': 'Θυμήσου μόνιμα ένα γεγονός/πληροφορία για μελλοντική χρήση. '
                        'Χρησιμοποίησέ το ΜΟΝΟ όταν ο χρήστης ζητάει ρητά να θυμηθείς κάτι.',
        'parameters': {'type': 'object', 'properties': {
            'fact': {'type': 'string', 'description': 'Το γεγονός προς απομνημόνευση'},
        }, 'required': ['fact']},
    }},
]


class LemonadeEmbeddingFunction(EmbeddingFunction):
    def __init__(self, url, model):
        self.url = url
        self.model = model

    def name(self):
        return f'lemonade-{self.model}'

    def __call__(self, input):
        r = requests.post(f'{self.url}/embeddings',
                           json={'model': self.model, 'input': list(input)}, timeout=30)
        r.raise_for_status()
        data = sorted(r.json()['data'], key=lambda e: e['index'])
        return [e['embedding'] for e in data]


_memory_collection = None


def get_memory_collection():
    global _memory_collection
    if _memory_collection is None:
        ef = LemonadeEmbeddingFunction(f'{HOST}/api/v1', EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
        _memory_collection = client.get_or_create_collection('robot_memory', embedding_function=ef)
    return _memory_collection


def remember_fact(fact):
    import uuid
    col = get_memory_collection()
    col.add(documents=[fact], ids=[str(uuid.uuid4())], metadatas=[{'stored_at': time.time()}])
    return {'status': 'ok', 'fact': fact}


def retrieve_memories(text):
    col = get_memory_collection()
    if col.count() == 0:
        return []
    try:
        res = col.query(query_texts=[text], n_results=min(MEMORY_TOP_K, col.count()))
    except Exception:
        return []
    return [doc for doc, dist in zip(res['documents'][0], res['distances'][0])
            if dist <= MEMORY_MAX_DISTANCE]


def sanitize(text):
    """Strip lone UTF-16 surrogates the backend occasionally emits when it
    splits a multi-byte Greek character across two streaming chunks."""
    return _LONE_SURROGATE.sub("", text)


def _get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return None
    for name in ('k10temp', 'coretemp', 'cpu_thermal', 'acpitz'):
        if temps.get(name):
            return round(temps[name][0].current, 1)
    return None


def get_system_status():
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage('/')
    load1, load5, load15 = psutil.getloadavg()
    status = {
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
    return status


def dispatch_tool(name, args):
    if name == 'system_status':
        return get_system_status()
    if name == 'remember':
        fact = (args.get('fact') or '').strip()
        if not fact:
            return {'status': 'error', 'reason': 'empty fact'}
        return remember_fact(fact)
    return {'status': 'error', 'reason': f'unknown tool: {name}'}


def chat_once(model, messages, tools=None):
    body = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{HOST}/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]


def stream_chat(model, messages):
    body = json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{HOST}/api/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    reply = ""
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            delta = sanitize(chunk["choices"][0]["delta"].get("content") or "")
            print(delta, end="", flush=True)
            reply += delta
    print()
    return reply


async def _synthesize(text):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def _decode_mp3(mp3_bytes):
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ar", str(TTS_SAMPLE_RATE), "-ac", "1", "pipe:1"],
        input=mp3_bytes, capture_output=True, check=True)
    return proc.stdout


def speak(text):
    text = text.strip()
    if not text:
        return
    try:
        mp3 = asyncio.run(_synthesize(text))
        pcm = _decode_mp3(mp3)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=TTS_SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"[tts σφάλμα] {e}")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-9b-FLM"
    print(f"--- {model} (γράψε /exit για έξοδο, /model <name> για αλλαγή μοντέλου) ---")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            text = sanitize(input("\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text.startswith("/model "):
            model = text.split(" ", 1)[1].strip()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"--- εναλλαγή σε {model} ---")
            continue
        facts = retrieve_memories(text)
        if facts:
            bullets = '\n'.join(f'- {f}' for f in facts)
            messages.append({"role": "system",
                              "content": f"Πράγματα που θυμάσαι από πριν:\n{bullets}"})
        messages.append({"role": "user", "content": text})
        try:
            msg = chat_once(model, messages, tools=TOOLS)
        except urllib.error.HTTPError as e:
            print(f"[σφάλμα HTTP {e.code}] {e.read().decode('utf-8', errors='replace')}")
            messages.pop()
            continue

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            reply = sanitize(msg.get("content") or "")
            print(reply)
            speak(reply)
            messages.append({"role": "assistant", "content": reply})
            continue

        messages.append(msg)
        for tc in tool_calls:
            fn = tc["function"]
            args = json.loads(fn.get("arguments") or "{}")
            result = dispatch_tool(fn["name"], args)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                              "content": json.dumps(result, ensure_ascii=False)})
        try:
            reply = stream_chat(model, messages)
        except urllib.error.HTTPError as e:
            print(f"[σφάλμα HTTP {e.code}] {e.read().decode('utf-8', errors='replace')}")
            continue
        speak(reply)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
