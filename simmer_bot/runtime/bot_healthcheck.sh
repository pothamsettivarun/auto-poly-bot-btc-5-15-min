#!/usr/bin/env bash
set -euo pipefail

STATE="/home/openclawd/.openclaw/workspace/memory/simmer-loop-state.json"
METRICS="/home/openclawd/.openclaw/workspace/memory/simmer-metrics.jsonl"
EVENTS="/home/openclawd/.openclaw/workspace/memory/simmer-events.jsonl"

NOW=$(date -u +%s)
STATUS="ok"
DETAILS=()

# 1) systemd status
if ! systemctl --user is-active --quiet simmer-bot.service; then
  STATUS="critical"
  DETAILS+=("service_not_active")
fi

# 2) stall detection via last_cycle_ts
if [[ -f "$STATE" ]]; then
  LAST_CYCLE=$(python3 - << 'PY'
import json,datetime,sys
p='/home/openclawd/.openclaw/workspace/memory/simmer-loop-state.json'
try:
    st=json.load(open(p))
    ts=st.get('last_cycle_ts')
    if not ts:
        print('none'); sys.exit(0)
    dt=datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
    print(int(dt.timestamp()))
except Exception:
    print('none')
PY
)
  if [[ "$LAST_CYCLE" == "none" ]]; then
    DETAILS+=("no_last_cycle_ts")
  else
    AGE=$((NOW - LAST_CYCLE))
    if (( AGE > 180 )); then
      [[ "$STATUS" == "ok" ]] && STATUS="warn"
      DETAILS+=("cycle_stale_${AGE}s")
    fi
  fi
else
  [[ "$STATUS" == "ok" ]] && STATUS="warn"
  DETAILS+=("state_missing")
fi

# 3) recent error rate
if [[ -f "$EVENTS" ]]; then
  ERR_5M=$(python3 - << 'PY'
import json,time
p='/home/openclawd/.openclaw/workspace/memory/simmer-events.jsonl'
now=time.time(); c=0
for line in open(p,'r',errors='ignore'):
    try:
        o=json.loads(line)
    except: continue
    if o.get('type')!='error':
        continue
    ts=o.get('ts')
    if not ts: continue
    try:
        t=time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S'))
    except: continue
    if now - t <= 300:
        c+=1
print(c)
PY
)
  if (( ERR_5M >= 5 )); then
    [[ "$STATUS" == "ok" ]] && STATUS="warn"
    DETAILS+=("high_error_rate_5m=${ERR_5M}")
  fi
fi

echo "simmer_bot_health status=${STATUS} details=${DETAILS[*]:-none}"
