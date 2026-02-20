#!/usr/bin/env bash
set -euo pipefail
cd /home/openclawd/.openclaw/workspace/simmer_bot
set -a
source /home/openclawd/.openclaw/workspace/simmer_bot/runtime/.env
set +a
exec python3 /home/openclawd/.openclaw/workspace/simmer_bot/simmer_minute_loop.py
