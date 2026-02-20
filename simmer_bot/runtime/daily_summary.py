#!/usr/bin/env python3
import json
from datetime import datetime, timezone

EVENTS='/home/openclawd/.openclaw/workspace/memory/simmer-events.jsonl'
STATE='/home/openclawd/.openclaw/workspace/memory/simmer-loop-state.json'

def today_utc():
    return datetime.now(timezone.utc).date().isoformat()

def parse_ts(ts):
    try:
        return datetime.fromisoformat(ts.replace('Z','+00:00'))
    except Exception:
        return None

today=today_utc()
orders=0
fills=0
partials=0
fails=0
no_trades=0

try:
    with open(EVENTS,'r',errors='ignore') as f:
        for line in f:
            try:
                o=json.loads(line)
            except Exception:
                continue
            ts=parse_ts(o.get('ts',''))
            if not ts or ts.date().isoformat()!=today:
                continue
            if o.get('type')=='order':
                orders+=1
                st=o.get('status')
                if st=='success': fills+=1
                elif st=='failed': fails+=1
                elif st=='partial': partials+=1
            elif o.get('type')=='decision' and o.get('action')=='no-trade':
                no_trades+=1
except FileNotFoundError:
    pass

trades_today='unknown'
try:
    st=json.load(open(STATE))
    trades_today=st.get('trades_today','unknown')
except Exception:
    pass

print(f"daily_summary utc_date={today} orders={orders} fills={fills} partials={partials} fails={fails} no_trades={no_trades} trades_today_counter={trades_today}")
