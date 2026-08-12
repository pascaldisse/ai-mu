#!/usr/bin/env bash
# V9-WIRE LEVER 1+2 combo /budget bench (task mandate 2026-07-26).
# Usage: v9wire-lever-budget.sh <label> <async_trace 0|1> <trace_reuse 0|1>
set -u
cd "$(dirname "$0")/../../.."   # repo root
LABEL="${1:?label}"; ASYNC="${2:?async}"; REUSE="${3:?reuse}"
PORT=8431
PROOF=packages/scrying-glass/scratch
BIN=target/release/scrying-glass
LOG="$PROOF/v9wire-budget-$LABEL.log"
WEIGHTS=/Users/pascaldisse/projects/magic-crystal-v9body/packages/scrying-glass/data/rdirect-weights-v9y.bin

env GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_HUD=false GAIA_NATIVE_PORT="$PORT" \
    GAIA_WORLD="$PWD/worlds/naruko" \
    GAIA_NATIVE_WEIGHTS="$WEIGHTS" GAIA_EYE_TEST=1 GAIA_NATIVE_EVIDENCE_SPLIT=1 \
    GAIA_V9N_HITGATE=1 GAIA_V9V_INPUT_CLAMP=0.10 \
    GAIA_NATIVE_ASYNC_TRACE="$ASYNC" GAIA_V9_TRACE_REUSE="$REUSE" \
    "$BIN" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

ok=0
for i in $(seq 1 60); do
  curl -s "http://127.0.0.1:$PORT/pose" >/dev/null 2>&1 && { ok=1; break; }
  sleep 0.5
done
if [ "$ok" != "1" ]; then
  echo "=== $LABEL: SERVER NEVER CAME UP ===" | tee -a "$LOG"
  exit 1
fi

# wait for >=500 frames (poll /budget)
frames=0
for i in $(seq 1 120); do
  frames=$(curl -s "http://127.0.0.1:$PORT/budget" | sed -n 's/.*"frames":\([0-9]*\).*/\1/p')
  [ -n "$frames" ] || frames=0
  if [ "$frames" -ge 500 ]; then break; fi
  sleep 1
done

BUDGET=$(curl -s "http://127.0.0.1:$PORT/budget")
STATE=$(curl -s "http://127.0.0.1:$PORT/state")
echo "=== /budget ($LABEL) frames=$frames ===" | tee -a "$LOG"
echo "$BUDGET" | tee -a "$LOG"
echo "$BUDGET" > "$PROOF/v9wire-budget-$LABEL.json"
echo "=== /state ($LABEL) ===" | tee -a "$LOG"
echo "$STATE" | tee -a "$LOG"
