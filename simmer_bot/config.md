1) Capital / Trade Sizing

• SIM_START_BAL = 10000.0
• MAX_TRADE = 100.0
• Legacy internal ceiling in sizing function.
• SAFE_MAX_ORDER_NOTIONAL = 20
• Active global hard cap on requested order notional.
• MAX_TRADES_PER_DAY = 1000

Practical effect

• Normal order sizing logic can compute higher values, but final request is capped at min(computed_size, 20).

───

2) Market Universe / Discovery

• Assets scanned: BTC / ETH / SOL
• Market types targeted: 5m + 15m Up-or-Down
• Sources used in loop:
• Simmer markets API for active markets
• Binance 1m klines for signal inputs

Selection toggles

• ENABLE_ASSET_DIVERSIFICATION = true (env default 1)
• LOOKAHEAD_5M_SEC = 480
• LOOKAHEAD_15M_SEC = 1200

Selection behavior

• If diversification enabled:
• Prefer 5m over 15m, then prefer least-recently-traded asset.
• Per-market anti-repeat cooldown:
• MARKET_COOLDOWN_SEC = 125

───

3) Signal / Decision Stack

Momentum signal (build_signal)

• Inputs: Binance 1m closes (last 30 bars)
• Features:
• r1, r3, r5 (% changes)
• rolling short volatility estimate
• Produces: side/confidence/reason (ok, weak_or_choppy, no_clear_edge)

Proxy gate (should_execute_projection_proxy)

Blocks if:

• side missing
• confidence < 0.42
• price < 0.08 or > 0.92

Note: cadence logic can still permit attempts depending on timing/state.

Stop-and-trade (theory stack)

• InitFW + Barrier FW -> D, g, and guarantee logic
• Key params:
• ALPHA_STOP = 0.9
• NET_THRESHOLD = 0.02
• FEE_EST = 0.003
• SLIPPAGE_EST = 0.006
• RISK_BUFFER_EST = 0.004

───

4) Spread Gate (Hard Pre-Trade Filter)

• ENABLE_SPREAD_GATE = true
• Base SPREAD_EPSILON = 0.03
• Uses: spread_abs = |mu_yes - market_price|
• If spread_abs < epsilon -> no trade

Adaptive epsilon (enabled)

• ADAPTIVE_SPREAD_EPSILON = true
• SPREAD_EPSILON_MIN = 0.02
• SPREAD_EPSILON_MAX = 0.06
• SPREAD_EPSILON_STEP_UP = 0.005
• SPREAD_EPSILON_STEP_DOWN = 0.002
• SPREAD_EPSILON_RECALC_SEC = 600

Adaptation intent:

• Raise epsilon when execution quality degrades.
• Lower epsilon when execution quality is clean.

───

5) Tail-Price Safety

• EXTREME_PRICE_BLOCK_LOW = 0.08
• Active hard block in cycle:
• If market probability <= 0.08 -> skip (hard_low_price_block)

High-tail hard block is currently not active as an explicit cycle-level block.

───

6) Cadence / Loop Timing

Loop sleep

• Default loop sleep: 10s
• Tightened sleep near close: 5s when 0 < last_secs_left <= 90

Cadence due condition

• cadence_due when >= 60s since last successful trade.

───

7) Risk Controls

RISK_CFG = RiskConfig(

• max_position_per_market = 400.0
• max_gross_exposure = 1500.0
• max_daily_drawdown = -500.0
)

Other guards:

• Daily trade cap (MAX_TRADES_PER_DAY = 1000)
• Kill-switch behavior on drawdown breach

───

8) Execution Layer

• Idempotent submission keys (bucketed)
• Execution adapter with submit/retry handling
• Partial-fill reconciliation enabled
• Event/metrics logging on success/fail/no-trade branches

───

9) State & Telemetry Files

• State: memory/simmer-loop-state.json
• Event log: memory/simmer-events.jsonl
• Metrics log: memory/simmer-metrics.jsonl
• Runtime log: memory/simmer-loop.log
• Status report: reports/status_report.py

───

10) Environment Overrides Supported (key ones)

• ENABLE_ASSET_DIVERSIFICATION
• LOOKAHEAD_5M_SEC
• LOOKAHEAD_15M_SEC
• SAFE_MAX_ORDER_NOTIONAL
• EXTREME_PRICE_BLOCK_LOW
• ENABLE_SPREAD_GATE
• SPREAD_EPSILON
• ADAPTIVE_SPREAD_EPSILON
• SPREAD_EPSILON_MIN/MAX
• SPREAD_EPSILON_STEP_UP/DOWN
• SPREAD_EPSILON_RECALC_SEC

───

11) Current Operating Profile (plain English)

• High-frequency, short-window microstructure bot
• Mostly small tickets (capped at $20 requested notional)
• Adaptive edge selectivity via spread epsilon
• Low-tail entries blocked (<= 0.08) to reduce thin-book cost blowouts
• 24/7-oriented daily headroom (1000 trades/day)