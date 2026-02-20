import os
import re
import time
import json
import math
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from core.constraints import ConstraintModel
from core.ip_oracle import IPOOracle
from core.initfw import init_fw
from core.barrier_fw import run_barrier_fw
from core.stopping import should_execute
from core.execution import ExecutionAdapter, IdempotencyStore, reconcile_fill
from risk.limits import RiskConfig, risk_check

SIM_START_BAL = 10000.0
MAX_TRADE = 100.0
MAX_TRADES_PER_DAY = 1000

# Hard safety sizing controls
SAFE_MAX_ORDER_NOTIONAL = float(os.getenv("SAFE_MAX_ORDER_NOTIONAL", "20"))
EXTREME_PRICE_BLOCK_LOW = float(os.getenv("EXTREME_PRICE_BLOCK_LOW", "0.08"))
EXTREME_PRICE_BLOCK_HIGH = float(os.getenv("EXTREME_PRICE_BLOCK_HIGH", "0.98"))
MARKET_COOLDOWN_SEC = 125
# FAIL_STREAK_WINDOW_SEC = 180
# FAIL_COOLDOWN_SEC = 240
# CADENCE_MARKET_COOLDOWN_SEC = 120
# CADENCE_GUARANTEE_FLOOR = -0.02

# Phase-D stop-and-trade defaults
ALPHA_STOP = 0.9
NET_THRESHOLD = 0.02
FEE_EST = 0.003
SLIPPAGE_EST = 0.006
RISK_BUFFER_EST = 0.004

# Phase-E risk limits
RISK_CFG = RiskConfig(
    max_position_per_market=400.0,
    max_gross_exposure=1500.0,
    max_daily_drawdown=-500.0,
)

API = "https://api.simmer.markets/api/sdk"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=30"
ASSET_MARKETS = [
    {"name": "BTC", "query": "Bitcoin Up or Down", "symbol": "BTCUSDT"},
    {"name": "ETH", "query": "Ethereum Up or Down", "symbol": "ETHUSDT"},
    {"name": "SOL", "query": "Solana Up or Down", "symbol": "SOLUSDT"},
]

# Selection behavior toggles (easy rollback controls)
ENABLE_ASSET_DIVERSIFICATION = os.getenv("ENABLE_ASSET_DIVERSIFICATION", "1") == "1"
LOOKAHEAD_5M_SEC = int(os.getenv("LOOKAHEAD_5M_SEC", "480"))
LOOKAHEAD_15M_SEC = int(os.getenv("LOOKAHEAD_15M_SEC", "1200"))
CADENCE_OVERRIDE_DAY_ONLY = os.getenv("CADENCE_OVERRIDE_DAY_ONLY", "1") == "1"
CADENCE_DAY_START_UTC_HOUR = int(os.getenv("CADENCE_DAY_START_UTC_HOUR", "8"))
CADENCE_DAY_END_UTC_HOUR = int(os.getenv("CADENCE_DAY_END_UTC_HOUR", "23"))

# A(t)-style spread/edge gate (hard pre-trade filter)
ENABLE_SPREAD_GATE = os.getenv("ENABLE_SPREAD_GATE", "1") == "1"
SPREAD_EPSILON = float(os.getenv("SPREAD_EPSILON", "0.03"))

# Adaptive spread-epsilon controls (self-tighten when execution quality degrades)
ADAPTIVE_SPREAD_EPSILON = os.getenv("ADAPTIVE_SPREAD_EPSILON", "1") == "1"
SPREAD_EPSILON_MIN = float(os.getenv("SPREAD_EPSILON_MIN", "0.02"))
SPREAD_EPSILON_MAX = float(os.getenv("SPREAD_EPSILON_MAX", "0.06"))
SPREAD_EPSILON_STEP_UP = float(os.getenv("SPREAD_EPSILON_STEP_UP", "0.005"))
SPREAD_EPSILON_STEP_DOWN = float(os.getenv("SPREAD_EPSILON_STEP_DOWN", "0.002"))
SPREAD_EPSILON_RECALC_SEC = int(os.getenv("SPREAD_EPSILON_RECALC_SEC", "600"))
CHAINLINK_GUARD_ENABLED = os.getenv("CHAINLINK_GUARD_ENABLED", "0") == "1"
CHAINLINK_MAX_DEVIATION_BPS = float(os.getenv("CHAINLINK_MAX_DEVIATION_BPS", "40"))
CHAINLINK_RPC_URL = os.getenv("CHAINLINK_RPC_URL", "https://cloudflare-eth.com")

CHAINLINK_FEEDS = {
    "BTCUSDT": "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c",  # BTC / USD
    "ETHUSDT": "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419",  # ETH / USD
    "SOLUSDT": "0x4ffC43a60e009B551865A93d232E33Fce9f01507",  # SOL / USD
}

LOG = "/home/openclawd/.openclaw/workspace/memory/simmer-loop.log"
STATE_PATH = "/home/openclawd/.openclaw/workspace/memory/simmer-loop-state.json"
EVENTS_PATH = "/home/openclawd/.openclaw/workspace/memory/simmer-events.jsonl"
METRICS_PATH = "/home/openclawd/.openclaw/workspace/memory/simmer-metrics.jsonl"


def utc_now():
    return datetime.now(timezone.utc)


def iso_now():
    return utc_now().isoformat()


def http_json(url, method="GET", headers=None, body=None, timeout=12):
    req = Request(url, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    with urlopen(req, data=data, timeout=timeout) as r:
        return json.loads(r.read().decode())


def read_key():
    # Priority: explicit env var -> ~/.openclaw/.env -> ~/.config/simmer/credentials.json
    v = os.getenv("SIMMER_API_KEY")
    if v:
        return v.strip()

    env = "/home/openclawd/.openclaw/.env"
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("SIMMER_API_KEY="):
                return line.strip().split("=", 1)[1]

    creds = "/home/openclawd/.config/simmer/credentials.json"
    if os.path.exists(creds):
        try:
            with open(creds, "r") as f:
                data = json.load(f)
            k = (data or {}).get("api_key")
            if k:
                return str(k).strip()
        except Exception:
            pass

    return None


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def duration_minutes_from_question(q):
    m = re.search(r"(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM) ET", q)
    if not m:
        return None
    h1, m1, ap1, h2, m2, ap2 = m.groups()
    h1 = int(h1) % 12 + (12 if ap1 == "PM" else 0)
    h2 = int(h2) % 12 + (12 if ap2 == "PM" else 0)
    t1 = h1 * 60 + int(m1)
    t2 = h2 * 60 + int(m2)
    if t2 < t1:
        t2 += 24 * 60
    return t2 - t1


def log(msg):
    ts = iso_now()
    with open(LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def append_jsonl(path, obj):
    row = dict(obj)
    row.setdefault("ts", iso_now())
    with open(path, "a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def ensure_state_schema(st):
    st = st or {}
    st.setdefault("schema_version", 2)
    st.setdefault("last_trade_ts_by_market", {})
    st.setdefault("last_trade_ts_by_asset", {})
    # st.setdefault("last_cadence_ts_by_market", {})
    # st.setdefault("market_failures", {})
    st.setdefault("last_cycle_ts", None)
    st.setdefault("spread_epsilon_live", SPREAD_EPSILON)
    st.setdefault("last_spread_recalc_ts", 0.0)
    st.setdefault("last_success_trade_ts", None)
    st.setdefault("daily_start_equity", SIM_START_BAL)
    st.setdefault("trades_today", 0)
    st.setdefault("date", utc_now().date().isoformat())
    st.setdefault("last_secs_left", None)
    st.setdefault("shadow_position_by_market", {})  # conservative internal inventory tracking from fills
    st.setdefault(
        "stats",
        {
            "cycles": 0,
            "orders_success": 0,
            "orders_failed": 0,
            "duplicate_blocks": 0,
            "risk_blocks": 0,
            "partial_fills": 0,
        },
    )
    return st


def load_state():
    if not os.path.exists(STATE_PATH):
        return ensure_state_schema({})
    try:
        with open(STATE_PATH, "r") as f:
            st = json.load(f)
    except Exception:
        st = {}
    return ensure_state_schema(st)


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def reset_day_if_needed(state, equity):
    today = utc_now().date().isoformat()
    if state.get("date") != today:
        state["date"] = today
        state["trades_today"] = 0
        state["daily_start_equity"] = equity
        state["last_trade_ts_by_market"] = {}


def get_snapshot(key):
    h = {"Authorization": f"Bearer {key}"}

    t0 = time.time()
    me = http_json(f"{API}/agents/me", headers=h)
    t_me_ms = int((time.time() - t0) * 1000)

    t1 = time.time()
    pos = http_json(f"{API}/positions", headers=h)
    t_pos_ms = int((time.time() - t1) * 1000)

    balance = float(me.get("balance", 0.0) or 0.0)
    positions = pos.get("positions", []) or []
    positions_value = float(pos.get("total_value", 0.0) or 0.0)
    equity = balance + positions_value

    return {
        "headers": h,
        "me": me,
        "pos": pos,
        "balance": balance,
        "positions": positions,
        "positions_value": positions_value,
        "equity": equity,
        "t_me_ms": t_me_ms,
        "t_pos_ms": t_pos_ms,
    }


def detect_asset(question: str):
    ql = (question or "").lower()
    if "bitcoin up or down" in ql:
        return "BTC", "BTCUSDT"
    if "ethereum up or down" in ql:
        return "ETH", "ETHUSDT"
    if "solana up or down" in ql:
        return "SOL", "SOLUSDT"
    return None, None


def select_candidate_market(markets, state, snapshot):
    now = utc_now()
    now_ts = time.time()
    cand = []
    ltt = state.get("last_trade_ts_by_market", {})
    lta = state.get("last_trade_ts_by_asset", {})
    total_considered = 0
    cooldown_blocked = 0
    # failure_cooldown_blocked = 0

    for m in markets:
        q = m.get("question", "")
        asset, _sym = detect_asset(q)
        if not asset:
            continue
        res = m.get("resolves_at")
        if not res:
            continue
        try:
            dt = parse_iso(res)
        except Exception:
            continue

        secs_left = (dt - now).total_seconds()
        if secs_left <= 0:
            continue

        dur = duration_minutes_from_question(q)
        if dur not in (5, 15, None):
            continue

        # Configurable lookahead windows.
        max_window_sec = LOOKAHEAD_5M_SEC if dur == 5 else LOOKAHEAD_15M_SEC
        if secs_left > max_window_sec:
            continue

        total_considered += 1
        mid = m.get("id")
        last_ts = ltt.get(mid)
        if last_ts is not None and (now_ts - float(last_ts)) < MARKET_COOLDOWN_SEC:
            cooldown_blocked += 1
            continue

        # refinement cooldown disabled (reverted to pre-refinement flow)

        priority = 0 if dur == 5 else 1
        if ENABLE_ASSET_DIVERSIFICATION:
            # Diversify by least-recently-traded asset.
            asset_last = float(lta.get(asset, 0.0) or 0.0)
            asset_idle = now_ts - asset_last if asset_last > 0 else 1e9
            cand.append((priority, -asset_idle, secs_left, asset, m))
        else:
            # Rollback behavior: strict soonest-resolving within priority.
            cand.append((priority, secs_left, asset, m))

    if ENABLE_ASSET_DIVERSIFICATION:
        cand.sort(key=lambda x: (x[0], x[1], x[2]))
    else:
        cand.sort(key=lambda x: (x[0], x[1]))
    stats = {
        "total_considered": total_considered,
        "cooldown_blocked": cooldown_blocked,
        # "failure_cooldown_blocked": failure_cooldown_blocked,
        "eligible_count": len(cand),
    }
    if not cand:
        return None, stats
    selected = cand[0][-1]
    return selected, stats


def _eth_rpc(method, params, timeout=6):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return http_json(CHAINLINK_RPC_URL, method="POST", body=body, timeout=timeout)


def _eth_call(to_addr, data, timeout=6):
    r = _eth_rpc("eth_call", [{"to": to_addr, "data": data}, "latest"], timeout=timeout)
    res = (r or {}).get("result")
    if not isinstance(res, str) or not res.startswith("0x"):
        raise RuntimeError("bad_eth_call_result")
    return res


def _decode_int256(hex_word):
    v = int(hex_word, 16)
    if v >= 1 << 255:
        v -= 1 << 256
    return v


def chainlink_spot_usd(symbol):
    feed = CHAINLINK_FEEDS.get(symbol)
    if not feed:
        return None

    # decimals()
    dec_hex = _eth_call(feed, "0x313ce567")
    decimals = int(dec_hex, 16)

    # latestRoundData() -> 5 words, answer is 2nd word
    rd_hex = _eth_call(feed, "0xfeaf968c")
    payload = rd_hex[2:]
    if len(payload) < 64 * 5:
        raise RuntimeError("short_latestRoundData")
    answer_word = payload[64:128]
    answer = _decode_int256(answer_word)
    px = float(answer) / (10 ** decimals)
    if px <= 0:
        raise RuntimeError("non_positive_chainlink_price")
    return px


def chainlink_deviation_bps(symbol, binance_spot):
    cl = chainlink_spot_usd(symbol)
    if cl is None or binance_spot is None:
        return None, cl
    dev = abs(float(binance_spot) - float(cl)) / float(cl) * 10000.0
    return dev, cl


def build_signal(symbol):
    d = http_json(BINANCE_KLINES.format(symbol=symbol))
    closes = [float(x[4]) for x in d]
    r1 = (closes[-1] / closes[-2] - 1) * 100
    r3 = (closes[-1] / closes[-4] - 1) * 100
    r5 = (closes[-1] / closes[-6] - 1) * 100
    rets = [(closes[i] / closes[i - 1] - 1) * 100 for i in range(1, len(closes))]
    mean15 = sum(rets[-15:]) / 15
    vol = (sum((x - mean15) ** 2 for x in rets[-15:]) / 15) ** 0.5

    score = 0
    if r1 > 0:
        score += 1
    if r3 > 0:
        score += 2
    if r5 > 0:
        score += 2
    if r1 < 0:
        score -= 1
    if r3 < 0:
        score -= 2
    if r5 < 0:
        score -= 2

    strength = abs(r3) + abs(r5)
    if vol > 0.18 or strength < 0.11:
        return {
            "side": None,
            "confidence": 0.0,
            "r1": r1,
            "r3": r3,
            "r5": r5,
            "vol": vol,
            "score": score,
            "spot": closes[-1],
            "reason": "weak_or_choppy",
        }

    side = None
    if score >= 3:
        side = "yes"
    elif score <= -3:
        side = "no"

    conf = min(1.0, max(0.0, (strength / 0.30) * (1.0 - min(vol, 0.20) / 0.20)))

    return {
        "side": side,
        "confidence": conf if side else 0.0,
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "vol": vol,
        "score": score,
        "spot": closes[-1],
        "reason": "ok" if side else "no_clear_edge",
    }


def size_order(signal, secs_left, snapshot):
    conf = float(signal.get("confidence", 0.0))
    amount = 40.0 + 80.0 * conf
    if 0 < secs_left <= 30:
        amount *= 1.20
    amount = min(amount, 160.0)

    max_afford = max(0.0, snapshot["balance"] - 5.0)
    amount = min(amount, max_afford)

    if secs_left > 30:
        amount = min(amount, MAX_TRADE)

    # Global hard cap to prevent sudden oversized orders.
    amount = min(amount, SAFE_MAX_ORDER_NOTIONAL)

    return round(max(0.0, amount), 2)


def should_execute_projection_proxy(signal, warnings, price):
    if signal.get("side") not in ("yes", "no"):
        return False, f"signal={signal.get('reason')}"
    if float(signal.get("confidence", 0.0)) < 0.42:
        return False, "low_confidence"
    if price < 0.08 or price > 0.92:
        return False, "extreme_price"
    _ = warnings
    return True, "ok"


def cadence_override_allowed_now():
    if not CADENCE_OVERRIDE_DAY_ONLY:
        return True
    h = utc_now().hour
    start = int(CADENCE_DAY_START_UTC_HOUR)
    end = int(CADENCE_DAY_END_UTC_HOUR)
    # active on [start, end)
    if start <= end:
        return start <= h < end
    # overnight wrap support
    return h >= start or h < end


def can_cadence_override(state, signal, price):
    # Reverted to permissive cadence behavior for higher opportunity flow.
    _ = state
    _ = signal
    _ = price
    return True, "ok"


def current_spread_epsilon(state):
    if not ADAPTIVE_SPREAD_EPSILON:
        return SPREAD_EPSILON

    now_ts = time.time()
    last_calc = float(state.get("last_spread_recalc_ts", 0.0) or 0.0)
    eps = float(state.get("spread_epsilon_live", SPREAD_EPSILON) or SPREAD_EPSILON)

    if (now_ts - last_calc) < SPREAD_EPSILON_RECALC_SEC:
        return max(SPREAD_EPSILON_MIN, min(SPREAD_EPSILON_MAX, eps))

    stats = state.get("stats", {})
    succ = float(stats.get("orders_success", 0) or 0)
    fail = float(stats.get("orders_failed", 0) or 0)
    partial = float(stats.get("partial_fills", 0) or 0)

    attempts = succ + fail
    reject_rate = (fail / attempts) if attempts > 0 else 0.0
    partial_rate = (partial / succ) if succ > 0 else 0.0

    # Degrade -> require larger edge. Clean -> allow slightly tighter edge.
    if reject_rate > 0.08 or partial_rate > 0.35:
        eps += SPREAD_EPSILON_STEP_UP
    elif reject_rate < 0.04 and partial_rate < 0.20:
        eps -= SPREAD_EPSILON_STEP_DOWN

    eps = max(SPREAD_EPSILON_MIN, min(SPREAD_EPSILON_MAX, eps))
    state["spread_epsilon_live"] = eps
    state["last_spread_recalc_ts"] = now_ts
    return eps


def projection_stop_decision(signal, price, risk_ok=True):
    conf = float(signal.get("confidence", 0.0))
    side = signal.get("side")
    edge = min(0.20, 0.02 + 0.25 * conf)

    p_model = float(price)
    if side == "yes":
        p_model = min(0.99, p_model + edge)
    elif side == "no":
        p_model = max(0.01, p_model - edge)

    target_q = {0: p_model, 1: 1.0 - p_model}

    constraints = ConstraintModel(universe={0, 1}, groups_exactly_one=[{0, 1}])
    oracle = IPOOracle(constraints, timeout_ms=200, max_retries=1)

    sigma_hat, Z0, u, certs, init_metrics = init_fw({}, constraints, oracle)
    best, hist = run_barrier_fw(
        target_q=target_q,
        sigma_hat=sigma_hat,
        Z0=Z0,
        u=u,
        oracle=oracle,
        alpha=ALPHA_STOP,
        max_iters=50,
        epsilon0=0.15,
        epsilon_floor=1e-4,
        prob_floor=1e-9,
    )

    D_t = float(best.get("divergence", 0.0))
    g_t = float(best.get("gap", 0.0))

    dec = should_execute(
        D_t=D_t,
        g_t=g_t,
        fees_est=FEE_EST,
        slippage_est=SLIPPAGE_EST,
        risk_buffer=RISK_BUFFER_EST,
        risk_ok=risk_ok,
        alpha=ALPHA_STOP,
        min_net_threshold=NET_THRESHOLD,
    )

    mu_yes = float((best.get("mu") or {}).get(0, 0.5))
    suggested_side = "yes" if mu_yes >= price else "no"

    spread_abs = abs(mu_yes - float(price))

    return {
        "execute": bool(dec.execute),
        "reason": dec.reason,
        "side": suggested_side,
        "D": D_t,
        "g": g_t,
        "guarantee": float(dec.guarantee),
        "alpha_ok": bool(dec.alpha_ok),
        "guarantee_ok": bool(dec.guarantee_ok),
        "iters": len(hist),
        "epsilon": float(best.get("epsilon", 0.0)),
        "init_runtime_ms": int(init_metrics.get("runtime_ms", 0)),
        "init_z0_size": int(init_metrics.get("z0_size", 0)),
        "mu_yes": mu_yes,
        "spread_abs": spread_abs,
    }


def build_execution_adapter():
    def submit_fn(headers, body, timeout_sec):
        return http_json(f"{API}/trade", method="POST", headers=headers, body=body, timeout=timeout_sec)

    return ExecutionAdapter(submit_fn=submit_fn, store=IdempotencyStore())


def run_cycle(state, exec_adapter):
    cycle_start = time.time()
    key = read_key()
    if not key:
        log("no SIMMER_API_KEY; skip")
        append_jsonl(EVENTS_PATH, {"type": "skip", "reason": "no_api_key"})
        return state

    snap = get_snapshot(key)
    reset_day_if_needed(state, snap["equity"])

    # A/E baseline non-order risk and ops gates
    if int(state.get("trades_today", 0)) >= MAX_TRADES_PER_DAY:
        state["stats"]["risk_blocks"] = int(state["stats"].get("risk_blocks", 0)) + 1
        log(f"skip risk_gate trades/day cap hit {state.get('trades_today')}")
        append_jsonl(EVENTS_PATH, {"type": "skip", "reason": "trades_day_cap"})
        return state

    if (snap["equity"] - float(state.get("daily_start_equity", SIM_START_BAL))) <= RISK_CFG.max_daily_drawdown:
        state["stats"]["risk_blocks"] = int(state["stats"].get("risk_blocks", 0)) + 1
        log("skip risk_gate kill-switch drawdown")
        append_jsonl(EVENTS_PATH, {"type": "skip", "reason": "kill_switch_drawdown"})
        raise SystemExit(0)

    t_mk0 = time.time()
    mk = []
    seen = set()
    for am in ASSET_MARKETS:
        q = am["query"].replace(" ", "%20")
        rows = http_json(
            f"{API}/markets?q={q}&status=active&limit=100",
            headers=snap["headers"],
        ).get("markets", [])
        for r in rows:
            rid = r.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                mk.append(r)
    t_markets_ms = int((time.time() - t_mk0) * 1000)

    m, sel_stats = select_candidate_market(mk, state, snap)
    dup_rate = (sel_stats["cooldown_blocked"] / sel_stats["total_considered"]) if sel_stats["total_considered"] else 0.0
    state["stats"]["duplicate_blocks"] = int(state["stats"].get("duplicate_blocks", 0)) + int(sel_stats["cooldown_blocked"])

    if not m:
        state["last_secs_left"] = None
        log("no eligible 5m/15m BTC/ETH/SOL market in active lookahead window")
        append_jsonl(EVENTS_PATH, {"type": "skip", "reason": "no_market", "selection": sel_stats})
        append_jsonl(
            METRICS_PATH,
            {
                "type": "cycle",
                "cycle_time_ms": int((time.time() - cycle_start) * 1000),
                "markets_ms": t_markets_ms,
                "me_ms": snap["t_me_ms"],
                "positions_ms": snap["t_pos_ms"],
                "decision": "no_market",
                "trades_today": state.get("trades_today", 0),
                "duplicate_attempt_rate": round(dup_rate, 4),
                "selection": sel_stats,
            },
        )
        return state

    m_id = m.get("id")
    qtxt = m.get("question", "")
    price = float(m.get("current_probability", 0.5) or 0.5)
    resolves_at = parse_iso(m.get("resolves_at"))
    secs_left = (resolves_at - utc_now()).total_seconds()
    state["last_secs_left"] = secs_left

    # hard tail blocks to prevent thin-book and terminal-price drift.
    if price <= EXTREME_PRICE_BLOCK_LOW:
        reason = f"hard_low_price_block:{price:.3f}"
        log(f"no-trade {m_id} reason={reason}")
        append_jsonl(EVENTS_PATH, {"type": "decision", "action": "no-trade", "market_id": m_id, "reason": reason, "price": price, "secs_left": secs_left})
        return state
    if price >= EXTREME_PRICE_BLOCK_HIGH:
        reason = f"hard_high_price_block:{price:.3f}"
        log(f"no-trade {m_id} reason={reason}")
        append_jsonl(EVENTS_PATH, {"type": "decision", "action": "no-trade", "market_id": m_id, "reason": reason, "price": price, "secs_left": secs_left})
        return state

    t_ctx0 = time.time()
    ctx = http_json(f"{API}/context/{m_id}", headers=snap["headers"])
    t_context_ms = int((time.time() - t_ctx0) * 1000)
    warnings = ctx.get("warnings") or []

    asset_name, signal_symbol = detect_asset(qtxt)
    if not signal_symbol:
        log(f"no-trade {m_id} reason=unsupported_asset question='{qtxt[:80]}'")
        return state

    signal = build_signal(signal_symbol)

    if CHAINLINK_GUARD_ENABLED:
        try:
            dev_bps, cl_spot = chainlink_deviation_bps(signal_symbol, signal.get("spot"))
            if dev_bps is not None and dev_bps > CHAINLINK_MAX_DEVIATION_BPS:
                reason = f"chainlink_guard:{dev_bps:.1f}bps>{CHAINLINK_MAX_DEVIATION_BPS:.1f}"
                log(
                    f"no-trade {m_id} reason={reason} symbol={signal_symbol} "
                    f"binance_spot={signal.get('spot')} chainlink_spot={cl_spot}"
                )
                append_jsonl(
                    EVENTS_PATH,
                    {
                        "type": "decision",
                        "action": "no-trade",
                        "market_id": m_id,
                        "reason": reason,
                        "symbol": signal_symbol,
                        "binance_spot": signal.get("spot"),
                        "chainlink_spot": cl_spot,
                        "deviation_bps": dev_bps,
                    },
                )
                return state
        except Exception as e:
            # Fail-open by default so a transient RPC issue doesn't halt trading.
            log(f"chainlink_guard_warn symbol={signal_symbol} err={type(e).__name__}:{e}")

    # Cadence goal: at least one trade attempt per minute (tiny size override if needed)
    now_ts = time.time()
    last_success = state.get("last_success_trade_ts")
    cadence_due = ((last_success is None) or ((now_ts - float(last_success)) >= 60.0)) and cadence_override_allowed_now()

    ok_proxy, reason_proxy = should_execute_projection_proxy(signal, warnings, price)
    if not ok_proxy and not cadence_due:
        log(f"no-trade {m_id} reason={reason_proxy} sig={signal}")
        append_jsonl(
            EVENTS_PATH,
            {
                "type": "decision",
                "action": "no-trade",
                "market_id": m_id,
                "reason": reason_proxy,
                "signal": signal,
                "price": price,
                "secs_left": secs_left,
            },
        )
        append_jsonl(
            METRICS_PATH,
            {
                "type": "cycle",
                "cycle_time_ms": int((time.time() - cycle_start) * 1000),
                "markets_ms": t_markets_ms,
                "context_ms": t_context_ms,
                "me_ms": snap["t_me_ms"],
                "positions_ms": snap["t_pos_ms"],
                "decision": "no-trade",
                "reject_reason": reason_proxy,
                "duplicate_attempt_rate": round(dup_rate, 4),
                "selection": sel_stats,
            },
        )
        return state

    cadence_ok, cadence_reason = can_cadence_override(state, signal, price)

    if cadence_due and signal.get("side") not in ("yes", "no") and cadence_ok:
        # conservative fallback side only when momentum and volatility are supportive
        mom = float(signal.get("r1", 0.0)) + float(signal.get("r3", 0.0)) + float(signal.get("r5", 0.0))
        vol = float(signal.get("vol", 999.0))
        if abs(mom) >= 0.12 and vol <= 0.14:
            signal["side"] = "yes" if mom >= 0 else "no"
            signal["confidence"] = max(float(signal.get("confidence", 0.0)), 0.33)

    proj = projection_stop_decision(signal=signal, price=price, risk_ok=True)

    # Hard spread gate: require model-vs-market edge magnitude to clear epsilon.
    spread_epsilon_live = current_spread_epsilon(state)
    if ENABLE_SPREAD_GATE and float(proj.get("spread_abs", 0.0)) < spread_epsilon_live:
        reason = f"spread_below_epsilon:{proj.get('spread_abs', 0.0):.4f}<{spread_epsilon_live:.4f}"
        log(
            f"no-trade {m_id} reason={reason} side={proj.get('side')} "
            f"mu_yes={proj.get('mu_yes', 0.0):.4f} price={price:.4f} eps={spread_epsilon_live:.4f}"
        )
        append_jsonl(
            EVENTS_PATH,
            {
                "type": "decision",
                "action": "no-trade",
                "market_id": m_id,
                "reason": reason,
                "signal": signal,
                "projection": proj,
                "price": price,
                "secs_left": secs_left,
            },
        )
        append_jsonl(
            METRICS_PATH,
            {
                "type": "cycle",
                "cycle_time_ms": int((time.time() - cycle_start) * 1000),
                "markets_ms": t_markets_ms,
                "context_ms": t_context_ms,
                "me_ms": snap["t_me_ms"],
                "positions_ms": snap["t_pos_ms"],
                "decision": "no-trade",
                "reject_reason": reason,
                "D": proj["D"],
                "g": proj["g"],
                "guarantee": proj["guarantee"],
                "spread_abs": proj.get("spread_abs", 0.0),
                "spread_epsilon": spread_epsilon_live,
                "iters": proj["iters"],
                "epsilon": proj["epsilon"],
                "duplicate_attempt_rate": round(dup_rate, 4),
                "selection": sel_stats,
            },
        )
        return state

    cadence_override = False
    if not proj["execute"]:
        spread_ok_for_cadence = float(proj.get("spread_abs", 0.0)) >= max(spread_epsilon_live, 0.035)
        conf_ok_for_cadence = float(signal.get("confidence", 0.0)) >= 0.50
        vol_ok_for_cadence = float(signal.get("vol", 999.0)) <= 0.14
        if cadence_due and cadence_ok and spread_ok_for_cadence and conf_ok_for_cadence and vol_ok_for_cadence:
            cadence_override = True
            proj["reason"] = f"cadence_override:{proj['reason']}"
        else:
            reason = f"stop_rule:{proj['reason']}"
            log(f"no-trade {m_id} reason={reason} D={proj['D']:.4f} g={proj['g']:.4f} G={proj['guarantee']:.4f}")
            append_jsonl(
                EVENTS_PATH,
                {
                    "type": "decision",
                    "action": "no-trade",
                    "market_id": m_id,
                    "reason": reason,
                    "signal": signal,
                    "projection": proj,
                    "price": price,
                    "secs_left": secs_left,
                },
            )
            append_jsonl(
                METRICS_PATH,
                {
                    "type": "cycle",
                    "cycle_time_ms": int((time.time() - cycle_start) * 1000),
                    "markets_ms": t_markets_ms,
                    "context_ms": t_context_ms,
                    "me_ms": snap["t_me_ms"],
                    "positions_ms": snap["t_pos_ms"],
                    "decision": "no-trade",
                    "reject_reason": reason,
                    "D": proj["D"],
                    "g": proj["g"],
                    "guarantee": proj["guarantee"],
                    "iters": proj["iters"],
                    "epsilon": proj["epsilon"],
                    "duplicate_attempt_rate": round(dup_rate, 4),
                    "selection": sel_stats,
                },
            )
            return state

    # Use projection side recommendation
    signal["side"] = proj.get("side") or signal.get("side")
    amount = size_order(signal, secs_left, snap)
    if cadence_override:
        amount = min(amount, 10.0)
    if amount <= 0:
        log(f"no-trade {m_id} reason=amount_zero")
        return state

    # Refresh snapshot immediately before order-time risk check to reduce stale-state risk.
    try:
        snap_live = get_snapshot(key)
    except Exception:
        snap_live = snap

    # Phase-E strict risk check at order time
    risk_ok, risk_reason = risk_check(
        config=RISK_CFG,
        positions=snap_live["positions"],
        market_id=m_id,
        current_equity=snap_live["equity"],
        day_start_equity=float(state.get("daily_start_equity", SIM_START_BAL)),
        new_order_notional=amount,
    )
    if not risk_ok:
        state["stats"]["risk_blocks"] = int(state["stats"].get("risk_blocks", 0)) + 1
        log(f"no-trade {m_id} risk_block {risk_reason}")
        append_jsonl(
            EVENTS_PATH,
            {
                "type": "decision",
                "action": "no-trade",
                "market_id": m_id,
                "reason": f"risk:{risk_reason}",
                "projection": proj,
                "amount": amount,
            },
        )
        return state

    reasoning = (
        f"1m momentum near 5m close: r1={signal['r1']:.3f}%, r3={signal['r3']:.3f}%, "
        f"r5={signal['r5']:.3f}%, vol={signal['vol']:.3f}%, conf={signal['confidence']:.2f}. "
        f"Stop-rule D={proj['D']:.4f}, g={proj['g']:.4f}, netG={proj['guarantee']:.4f}. "
        f"Sizing {amount:.2f} $SIM."
    )

    # Phase-E idempotent execution
    idem_bucket = int(time.time() // 30)
    idem_key = f"{state.get('date')}:{m_id}:{signal['side']}:{idem_bucket}"
    body = {
        "market_id": m_id,
        "side": signal["side"],
        "amount": amount,
        "venue": "polymarket",
        "source": "sdk:openclawd-1m-loop",
        "reasoning": reasoning,
    }

    t_trade0 = time.time()
    exr = exec_adapter.submit_idempotent(idem_key=idem_key, headers=snap["headers"], body=body, timeout_sec=10)
    t_trade_ms = int((time.time() - t_trade0) * 1000)

    if exr.status in ("rejected", "timeout", "error"):
        state["stats"]["orders_failed"] = int(state["stats"].get("orders_failed", 0)) + 1

        log(f"trade-failed market={m_id} side={signal['side']} amt={amount} status={exr.status}")
        append_jsonl(
            EVENTS_PATH,
            {
                "type": "order",
                "status": "failed",
                "market_id": m_id,
                "side": signal["side"],
                "amount": amount,
                "projection": proj,
                "cadence_override": cadence_override,
                "execution": {
                    "status": exr.status,
                    "client_order_id": exr.client_order_id,
                    "provider_order_id": exr.provider_order_id,
                },
                "response": exr.raw,
            },
        )
        append_jsonl(
            METRICS_PATH,
            {
                "type": "cycle",
                "cycle_time_ms": int((time.time() - cycle_start) * 1000),
                "markets_ms": t_markets_ms,
                "context_ms": t_context_ms,
                "trade_ms": t_trade_ms,
                "decision": "trade_failed",
                "exec_status": exr.status,
                "cadence_override": cadence_override,
                "D": proj["D"],
                "g": proj["g"],
                "guarantee": proj["guarantee"],
                "iters": proj["iters"],
                "epsilon": proj["epsilon"],
                "duplicate_attempt_rate": round(dup_rate, 4),
                "selection": sel_stats,
            },
        )
        return state

    # success / partial path
    if exr.status == "partial":
        state["stats"]["partial_fills"] = int(state["stats"].get("partial_fills", 0)) + 1

    if exr.status in ("filled", "partial"):
        # conservative internal reconciliation for inventory sanity
        shadow = state.get("shadow_position_by_market", {})
        prev_pos = float(shadow.get(m_id, 0.0))
        new_pos = reconcile_fill(prev_pos, signal["side"], exr)
        shadow[m_id] = new_pos
        state["shadow_position_by_market"] = shadow

        now_fill_ts = time.time()
        state["last_trade_ts_by_market"][m_id] = now_fill_ts
        # cadence override per-market timestamp disabled (reverted)
        if asset_name:
            state["last_trade_ts_by_asset"][asset_name] = now_fill_ts
        state["last_success_trade_ts"] = now_fill_ts
        state["trades_today"] = int(state.get("trades_today", 0)) + 1
        state["stats"]["orders_success"] = int(state["stats"].get("orders_success", 0)) + 1
    # market failure cooldown state disabled (reverted)

    log(
        f"TRADE side={signal['side']} amt={amount} market={m_id} p={price:.3f} status={exr.status} "
        f"filled={exr.filled_amount:.2f} rem={exr.remaining_amount:.2f} q='{qtxt[:80]}'"
    )

    append_jsonl(
        EVENTS_PATH,
        {
            "type": "order",
            "status": "success" if exr.status in ("filled", "partial") else exr.status,
            "market_id": m_id,
            "side": signal["side"],
            "amount": amount,
            "price": price,
            "secs_left": secs_left,
            "signal": signal,
            "projection": proj,
            "cadence_override": cadence_override,
            "execution": {
                "status": exr.status,
                "client_order_id": exr.client_order_id,
                "provider_order_id": exr.provider_order_id,
                "requested_amount": exr.requested_amount,
                "filled_amount": exr.filled_amount,
                "remaining_amount": exr.remaining_amount,
            },
            "response": exr.raw,
        },
    )

    append_jsonl(
        METRICS_PATH,
        {
            "type": "cycle",
            "cycle_time_ms": int((time.time() - cycle_start) * 1000),
            "markets_ms": t_markets_ms,
            "context_ms": t_context_ms,
            "trade_ms": t_trade_ms,
            "me_ms": snap["t_me_ms"],
            "positions_ms": snap["t_pos_ms"],
            "decision": "trade_success",
            "exec_status": exr.status,
            "cadence_override": cadence_override,
            "trades_today": state.get("trades_today", 0),
            "D": proj["D"],
            "g": proj["g"],
            "guarantee": proj["guarantee"],
            "iters": proj["iters"],
            "epsilon": proj["epsilon"],
            "duplicate_attempt_rate": round(dup_rate, 4),
            "selection": sel_stats,
        },
    )

    return state


def sleep_loop_interval(state):
    # Adaptive cadence:
    # - default 10s
    # - tighten to 5s when a selected market is in the final 90s
    secs_left = state.get("last_secs_left")
    if secs_left is not None:
        try:
            s = float(secs_left)
            if 0 < s <= 90:
                time.sleep(5.0)
                return
        except Exception:
            pass
    time.sleep(10.0)


if __name__ == "__main__":
    os.makedirs("/home/openclawd/.openclaw/workspace/memory", exist_ok=True)
    state = load_state()
    exec_adapter = build_execution_adapter()
    log("loop start")

    while True:
        try:
            state = run_cycle(state, exec_adapter)
            state["stats"]["cycles"] = int(state["stats"].get("cycles", 0)) + 1
            state["last_cycle_ts"] = iso_now()
            save_state(state)
        except (HTTPError, URLError, TimeoutError) as e:
            log(f"network/API error: {e}")
            append_jsonl(EVENTS_PATH, {"type": "error", "error": str(e)})
        except SystemExit:
            log("loop stopped by kill-switch")
            append_jsonl(EVENTS_PATH, {"type": "stop", "reason": "kill-switch"})
            save_state(state)
            break
        except Exception as e:
            log(f"unexpected error: {type(e).__name__}: {e}")
            append_jsonl(EVENTS_PATH, {"type": "error", "error": f"{type(e).__name__}: {e}"})
        sleep_loop_interval(state)