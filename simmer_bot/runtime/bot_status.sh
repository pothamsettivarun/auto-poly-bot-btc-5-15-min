#!/usr/bin/env bash
set -euo pipefail

echo "== service =="
systemctl --user --no-pager --full status simmer-bot.service | sed -n '1,20p'

echo
echo "== latest log lines =="
tail -n 20 /home/openclawd/.openclaw/workspace/memory/simmer-loop.log || true

echo
echo "== latest events =="
tail -n 20 /home/openclawd/.openclaw/workspace/memory/simmer-events.jsonl || true

echo
echo "== healthcheck =="
/home/openclawd/.openclaw/workspace/simmer_bot/runtime/bot_healthcheck.sh || true
