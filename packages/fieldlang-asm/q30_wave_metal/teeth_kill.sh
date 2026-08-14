#!/bin/sh
# teeth_kill.sh — mechanical self-attack on every tooth in gate.sh.
# For each tooth: apply its mutation (single source of truth = the `mut` lines in
# gate.sh), rebuild, run the live GPU gate binary. Verdicts:
#   KILLED          mutation applied AND live run went RED (tooth is non-vacuous)
#   NOOP-VACUOUS    mutation failed to change the file (tooth bites nothing)
#   SURVIVED-VACUOUS mutation applied but the run stayed GREEN (tooth proves nothing)
# Exit 0 only if every tooth is KILLED.
set -u
cd "$(dirname "$0")"
V=../q30_wave/wave_vectors.bin
FAIL=0
mut() {
    n=$1; f=$2; e=$3
    cp "$f" "$f.save"
    perl -0pe "$e" "$f.save" > "$f"
    if cmp -s "$f" "$f.save"; then r=NOOP-VACUOUS
    elif ./build.sh >/dev/null 2>&1 && ./wave_metal_runner q30_wave.metallib "$V" >/dev/null 2>&1; then r=SURVIVED-VACUOUS
    else r=KILLED; fi
    mv "$f.save" "$f"
    printf '%-26s %-16s %s\n' "$n" "$f" "$r"
    [ "$r" = KILLED ] || FAIL=1
}
trap 'for f in wave_q30.metal metal_bridge.s; do test -f "$f.save" && mv "$f.save" "$f"; done' EXIT HUP INT TERM
./build.sh >/dev/null
printf '%-26s %-16s %s\n' TOOTH FILE VERDICT
grep '^mut ' gate.sh > .teeth.list
. ./.teeth.list
rm -f .teeth.list
./build.sh >/dev/null
printf 'kill table: %s\n' "$([ "$FAIL" -eq 0 ] && echo 'ALL KILLED' || echo 'VACUOUS TEETH PRESENT')"
exit "$FAIL"
