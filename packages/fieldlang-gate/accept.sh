#!/bin/sh
# ============================================================
# GATE TOOL — NOT THE COMPILER.
# accept.sh: byte-level acceptance of the ARM64 asm fieldc against the
# Rust reference oracle, for the 3 example programs.
#   1. flgate ref   → reference .fldj (frozen journal writer = byte oracle)
#   2. cmp golden   → committed byte snapshot (regression tripwire)
#   3. fieldc + cmp → asm output byte-for-byte vs reference (SKIPPED-
#                     UNVERIFIED while the asm lane has not landed)
#   4. digest ×2    → World::replay of the artifact twice, must match
# Usage: accept.sh [FIELDC=/path/to/fieldc]
# ============================================================
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT="$HERE/out"
GOLD="$HERE/golden"
FIELDC=${FIELDC:-/Users/pascaldisse/projects/mc-fieldlang-asm/packages/fieldlang-asm/fieldc}
FLGATE="$ROOT/target/debug/flgate"
EXAMPLES="01_wave 02_algebra 03_raw"

mkdir -p "$OUT"
cargo build -q -p fieldlang-gate --manifest-path "$ROOT/Cargo.toml"

fail=0
for name in $EXAMPLES; do
    src="$HERE/examples/$name.fld"
    ref="$OUT/$name.ref.fldj"
    "$FLGATE" ref "$src" "$ref" >/dev/null

    if [ -f "$GOLD/$name.fldj" ]; then
        if cmp -s "$GOLD/$name.fldj" "$ref"; then
            echo "GOLDEN  $name  byte-exact"
        else
            echo "GOLDEN  $name  MISMATCH vs committed snapshot"; fail=1
        fi
    else
        echo "GOLDEN  $name  no snapshot (run: cp out/*.ref.fldj golden/)"
    fi

    if [ -x "$FIELDC" ]; then
        asm="$OUT/$name.asm.fldj"
        # AUDIT WORKAROUND (2026-08-01): fieldc driver.s opens output with a
        # bad mode (--wx------) and cannot overwrite it. Reported to asm lane.
        rm -f "$asm"
        if ! "$FIELDC" "$src" "$asm"; then
            echo "ASM     $name  fieldc exit nonzero — UNVERIFIED"; fail=1; continue
        fi
        chmod u+r "$asm" 2>/dev/null || true
        if cmp -s "$ref" "$asm"; then
            echo "ASM     $name  byte-exact vs reference"
        else
            echo "ASM     $name  MISMATCH:"; cmp -l "$ref" "$asm" | head -5; fail=1
        fi
        target="$asm"
    else
        echo "ASM     $name  SKIP — fieldc not built ($FIELDC) — UNVERIFIED"
        target="$ref"
    fi

    d1=$("$FLGATE" digest "$target" | sed 's/.*digest=//')
    d2=$("$FLGATE" digest "$target" | sed 's/.*digest=//')
    if [ "$d1" = "$d2" ]; then
        echo "DIGEST  $name  $d1  (deterministic ×2)"
    else
        echo "DIGEST  $name  DRIFT $d1 != $d2"; fail=1
    fi
done

if [ "$fail" -eq 0 ]; then echo "ACCEPT: PASS"; else echo "ACCEPT: FAIL"; exit 1; fi
