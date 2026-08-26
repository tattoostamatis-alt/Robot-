#!/usr/bin/env bash
# flm_keepalive — keep the FastFlowLM NPU server alive, safely.
#
# `flm serve` exits on its own two ways seen on this rig:
#   • silently after a successful request (clean, no traceback, RAM free), and
#   • "double free or corruption" when it gets CONCURRENT requests (its NPU
#     queue reports "5/10" then aborts). So the golden rule is: exactly ONE
#     client hits :52625 at a time. This wrapper therefore does NOT send its own
#     warm-up probes (an earlier version fired `( warm ) & ` every loop and,
#     combined with fast crash-restarts, spawned dozens of curl'ing subshells
#     that DoS'd the server into constant double-frees). Keep it dumb: run the
#     server, and if it dies, back off and rerun it. Nothing else talks here
#     except llm_bridge, one request at a time.
#
# Usage:  setsid nohup ~/bin/flm_keepalive.sh >/tmp/flm_serve.log 2>&1 &
# Stop:   touch /tmp/flm_keepalive.stop
set -u
# qwen3.5:4b replaced qwen3vl-it:4b on 2026-07-30 after an A/B on the real voice
# tool set (24 Greek utterances x3, same SYSTEM_PROMPT and TOOLS): 100 % vs
# 90.5 % tool accuracy. The VL model never called `stop` for a bare "σταμάτα"
# nor `stop_follow` for "μη με ακολουθείς άλλο" — it answered "Σταμάτησα." while
# the robot kept moving. qwen3.5 gets both 3/3. It is multimodal too (verified:
# reads text off a frame), so the vision path is unaffected. RSS is the same
# (6.14 vs 6.19 GB); median latency 5.9 s vs 4.8 s is the price.
MODEL="${1:-qwen3.5:4b}"
# Context length. Left at flm's default (-1) the server sizes its NPU KV-cache
# buffers from the model's max_position_embeddings — 40960 tokens for
# a 4B Qwen — and reserves 8.4 GB of /dev/accel/accel0 DMA buffers (36 per
# layer, measured 2026-07-26). On a 13 GB box that is most of RAM: with the ROS
# stack up, `free` showed 744 MB available, 3.8/4.0 GB swap used and ~10 MB/s
# of continuous swap-in, which pushed faster-whisper from ~6 s to 42-59 s per
# utterance and made the voice pipeline look dead.
#
# We never come close to 40 k: the real prompt is ~2050 tokens (system + the
# 15-tool schema) and ~2974 with a 720p camera frame attached — the server
# itself reported kv_token_occupancy 0.7 % at the default. 8192 keeps a 63 %
# margin over the worst case (image + full TOOLS + 4 history turns) while
# cutting the reservation to 4.9 GB. Measured: RSS 8.53 GB → 5.87 GB, with
# tool-calling and vision both re-verified unchanged.
CTX_LEN="${2:-8192}"
STOP=/tmp/flm_keepalive.stop
rm -f "$STOP"
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true
backoff=2
while true; do
    [ -f "$STOP" ] && { echo "[keepalive] stop file present, exiting"; break; }
    echo "[keepalive] $(date '+%H:%M:%S') starting flm serve $MODEL (ctx-len $CTX_LEN)"
    start=$(date +%s)
    flm serve "$MODEL" -c "$CTX_LEN"
    code=$?
    [ -f "$STOP" ] && { echo "[keepalive] stopped"; break; }
    ran=$(( $(date +%s) - start ))
    # If it survived a while, reset backoff; if it died fast, ramp it up so a
    # persistent crash loop stays gentle (2 → 5 → 10 → 20 → 30 s cap).
    if [ "$ran" -ge 30 ]; then backoff=2; else backoff=$(( backoff < 30 ? backoff*2 : 30 )); fi
    echo "[keepalive] $(date '+%H:%M:%S') flm exited (code $code) after ${ran}s — restarting in ${backoff}s"
    sleep "$backoff"
done
