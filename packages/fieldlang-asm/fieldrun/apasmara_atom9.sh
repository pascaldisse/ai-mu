#!/bin/sh
# Apasmara atom9: raw FLDJ memory-state adversary. Shell + hand-asm only.
# No fieldc/golden/teeth generator dependency; fixtures are literal FLDJ bytes.
set -eu
cd "$(dirname "$0")"
work="./.apasmara-atom9.$$"
trap 'rm -rf "$work"' EXIT HUP INT TERM
mkdir "$work"
SDK=$(xcrun --show-sdk-path)

./build.sh >/dev/null

header() {
  # FLDJ/v1, 1x1, canonical physical bits, seed=42, n_slots=2.
  printf '\106\114\104\112\001\000\000\000\001\000\000\000\001\000\000\000'
  printf '\000\000\200\077\315\314\314\075\167\276\177\077\000\000\200\077'
  printf '\052\000\000\000\000\000\000\000\000\000\200\077\002\000\000\000'
}
w0one() { printf '\006\000\000\000\000\001\000\000\000\000\000\200\077'; }
w0zero() { printf '\006\000\000\000\000\001\000\000\000\000\000\000\000'; }
w1one() { printf '\006\001\000\000\000\001\000\000\000\000\000\200\077'; }
w1zero() { printf '\006\001\000\000\000\001\000\000\000\000\000\000\000'; }
step() { printf '\003\001\000\000\000'; }
step2() { printf '\003\002\000\000\000'; }
make() { f=$1; shift; header >"$f"; for op in "$@"; do "$op" >>"$f"; done; }
run() { ./fieldrun "$1" "$2"; [ "$(wc -c <"$2" | tr -d ' ')" = 36 ]; }
sha() { shasum -a 256 "$1" | awk '{print $1}'; }

# slot0, slot1, unwritten, repeated write; all raw FLDJ, not fieldc output.
make "$work/a.fldj" w0one w1zero step
make "$work/b.fldj" w0one w1one step
make "$work/z.fldj" step
make "$work/repeat.fldj" w0one w0zero step
make "$work/rotate.fldj" w0one w1zero step2
for n in a b z rotate; do run "$work/$n.fldj" "$work/$n.flro"; done
if ./fieldrun "$work/repeat.fldj" "$work/repeat.flro" >"$work/repeat.log" 2>&1; then rrc=0; else rrc=$?; fi
[ "$rrc" = 27 ] || { cat "$work/repeat.log" >&2; echo "atom9: duplicate slot rc=$rrc, want=27" >&2; exit 1; }
[ ! -e "$work/repeat.flro" ] || { echo 'atom9: duplicate slot wrote output' >&2; exit 1; }
if cmp -s "$work/a.flro" "$work/b.flro"; then echo 'atom9: slot1 input was ignored' >&2; exit 1; fi
printf 'green raw-slot0-slot1-unwritten a=%s b=%s z=%s\n' "$(sha "$work/a.flro")" "$(sha "$work/b.flro")" "$(sha "$work/z.flro")"
printf 'KILLED raw-duplicate-slot rc=%s output=absent\n' "$rrc"
printf 'green raw-two-step-rotation rotate=%s\n' "$(sha "$work/rotate.flro")"

# Mutation 1: slot1 is misrouted to cur; raw slot1 fixture must change.
sed 's|mov x0, x10                    // \[MUT:slot-index\]|mov x0, xzr                    // [MUT:slot-index]|' fieldrun.s >"$work/slot.s"
as -arch arm64 -o "$work/slot.o" "$work/slot.s"
ld -arch arm64 -o "$work/slot" -e _main -lSystem -lobjc -framework Metal -framework Foundation \
  "$work/slot.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$SDK"
"$work/slot" "$work/b.fldj" "$work/slot.flro"
if cmp -s "$work/b.flro" "$work/slot.flro"; then echo 'atom9: slot routing mutant survived' >&2; exit 1; fi
printf 'KILLED raw-slot1-misroute baseline=%s mutant=%s\n' "$(sha "$work/b.flro")" "$(sha "$work/slot.flro")"

# Mutation 2: remove one arm of three-pointer rotation; two ticks must change.
sed 's|str x25, \[x9, #8\]              // \[MUT:rot3\] \[MUT:slot-rotate\]|str x27, [x9, #8]              // [MUT:rot3] [MUT:slot-rotate]|' fieldrun.s >"$work/rot.s"
as -arch arm64 -o "$work/rot.o" "$work/rot.s"
ld -arch arm64 -o "$work/rot" -e _main -lSystem -lobjc -framework Metal -framework Foundation \
  "$work/rot.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$SDK"
"$work/rot" "$work/rotate.fldj" "$work/rot.flro"
if cmp -s "$work/rotate.flro" "$work/rot.flro"; then echo 'atom9: rotation mutant survived' >&2; exit 1; fi
printf 'KILLED raw-three-pointer-rotation baseline=%s mutant=%s\n' "$(sha "$work/rotate.flro")" "$(sha "$work/rot.flro")"

# Mutation 3: every mmap reports MAP_FAILED; loud rc19 and no output are required.
sed 's|bl _mmap|mov x0, #-1|' fieldrun.s >"$work/mmap.s"
as -arch arm64 -o "$work/mmap.o" "$work/mmap.s"
ld -arch arm64 -o "$work/mmap" -e _main -lSystem -lobjc -framework Metal -framework Foundation \
  "$work/mmap.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$SDK"
if "$work/mmap" "$work/a.fldj" "$work/mmap.flro" >"$work/mmap.log" 2>&1; then rc=0; else rc=$?; fi
[ "$rc" = 19 ] || { cat "$work/mmap.log" >&2; echo "atom9: mmap mutant rc=$rc, want=19" >&2; exit 1; }
[ ! -e "$work/mmap.flro" ] || { echo 'atom9: mmap failure wrote output' >&2; exit 1; }
printf 'KILLED mmap-failure-loud rc=%s output=absent\n' "$rc"

# 動的slot tableは全allocationを明示解放する。
if ! nm -u fieldrun | grep -q '_munmap'; then echo 'atom9: missing munmap symbol' >&2; exit 1; fi
printf 'green partial-cleanup=explicit nm-_munmap=present\n'
echo 'gate: Apasmara atom9 raw journal/mutation OK'
