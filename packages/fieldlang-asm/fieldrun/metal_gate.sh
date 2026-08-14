#!/bin/sh
# fieldrun A6 門: --metal backend(実機 GPU)。三経路 FLRO byte/SHA 一致 + no-fallback + metal 変異歯。
# shell のみ(新規 Rust/C/Swift/Python 零)。
set -eu
cd "$(dirname "$0")"

if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi
# fieldrun.s は整数のみ(§11/§12 の走査を継承。SIMD/GPU は backend file に閉じる)。
scan=$(grep -nE '(^|[^[:alnum:]_.])(v[0-9]+\.|[qdshb][0-9]+)([^[:alnum:]_]|$)' fieldrun.s || :)
case "$scan" in '') ;; *) echo "gate: FP/SIMD register in fieldrun.s"; echo "$scan" >&2; exit 1 ;; esac
# metallib path 硬碼禁: fieldrun.s に .metallib 文字列が無い事
if grep -qE '(asciz|ascii).*metallib' fieldrun.s; then echo 'gate: metallib path hardcoded in fieldrun.s' >&2; exit 1; fi

./build.sh >/dev/null
MLIB=${MLIB:-../q30_wave_metal/q30_wave.metallib}
[ -f "$MLIB" ] || (cd ../q30_wave_metal && ./build.sh >/dev/null)
[ -f "$MLIB" ] || { echo "gate: metallib absent: $MLIB" >&2; exit 1; }

work=$(mktemp -d ./metal-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM
SDK=$(xcrun --show-sdk-path)

link() { # link <outbin> <fieldrun.o>
  ld -arch arm64 -o "$1" -e _main -lSystem -lobjc -framework Metal -framework Foundation \
     "$2" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$SDK"
}

# ---- 場: A5 の 5 vector + 大型非一様(GPU thread 多数・vector 本体経路)----
PAYLOAD="1065353216 0 0 0" ./gen_fldj.sh "$work/t1.fldj"
W=4 H=4 STEPN=3 PAYLOAD="1065353216" ./gen_fldj.sh "$work/t2.fldj"
FILL=1157627903 ./gen_fldj.sh "$work/t3.fldj"
mkfield() { # mkfield <w> <h> -> stdout payload
  n=$(( $1 * $2 )); k=0; p=''
  while [ $k -lt $n ]; do p="$p $((1065353216 + k * 1048576))"; k=$((k + 1)); done
  printf '%s' "$p"
}
W=8 H=8 STEPN=3 PAYLOAD="$(mkfield 8 8)" ./gen_fldj.sh "$work/t4.fldj"
W=9 H=9 STEPN=3 PAYLOAD="$(mkfield 9 9)" ./gen_fldj.sh "$work/t5.fldj"
# 大型非一様場(1600 胞 = GPU thread 多数)。値は |v|<2048 に留める為 周期的非一様。
big=''
k=0
while [ $k -lt 1600 ]; do
  big="$big $((1065353216 + (k % 64) * 1048576 + (k / 64) * 65536))"
  k=$((k + 1))
done
W=40 H=40 STEPN=4 PAYLOAD="$big" ./gen_fldj.sh "$work/t6.fldj"

tri() { # tri <name> <fldj>
  ./fieldrun "$2" "$work/$1.s.flro" >/dev/null
  ./fieldrun --neon "$2" "$work/$1.n.flro" >/dev/null
  ./fieldrun --metal "$MLIB" "$2" "$work/$1.m.flro" >"$work/$1.m.log" 2>&1
  a=$(shasum -a 256 "$work/$1.s.flro" | cut -d' ' -f1)
  b=$(shasum -a 256 "$work/$1.n.flro" | cut -d' ' -f1)
  c=$(shasum -a 256 "$work/$1.m.flro" | cut -d' ' -f1)
  if [ "$a" != "$b" ] || [ "$a" != "$c" ]; then
    echo "gate: $1 scalar=$a neon=$b metal=$c MISMATCH" >&2
    od -An -tx1 "$work/$1.s.flro" >&2; od -An -tx1 "$work/$1.m.flro" >&2; exit 1
  fi
  cmp "$work/$1.s.flro" "$work/$1.m.flro" || { echo "gate: $1 byte mismatch" >&2; exit 1; }
  printf 'green   %-22s sha=%s (scalar==neon==metal, byte 一致)\n' "$1" "$a"
}
tri 3c-2x2-one-tick   "$work/t1.fldj"
tri 4x4-3tick         "$work/t2.fldj"
tri saturating-vector "$work/t3.fldj"
tri 8x8-3tick-vecpath "$work/t4.fldj"
tri 9x9-3tick-vecedge "$work/t5.fldj"
tri 40x40-4tick-gpu   "$work/t6.fldj"

want4=33074917e723d60a4434ddf1badb9844faa734beb17204f0b8bdb712c16c3f7f
got4=$(shasum -a 256 "$work/4x4-3tick.m.flro" | cut -d' ' -f1)
[ "$got4" = "$want4" ] || { echo "gate: 4x4-3tick metal sha=$got4 want=$want4" >&2; exit 1; }
printf 'green   %-22s %s\n' '4x4-known-sha(metal)' "$got4"

# GPU 実走の証跡(bridge が fd1 へ raw 出力)
grep -q 'metal command status: 4' "$work/40x40-4tick-gpu.m.log" \
  || { echo 'gate: no GPU completion evidence' >&2; exit 1; }
printf 'green   %-22s %s\n' 'gpu-evidence' "$(grep -m1 'status' "$work/40x40-4tick-gpu.m.log")"

# ---- 歯1: metallib 欠落 -> 非零 exit・出力 file を作らぬ(scalar 結果を返さぬ)----
rm -f "$work/absent.flro"
if ./fieldrun --metal "$work/no-such.metallib" "$work/t4.fldj" "$work/absent.flro" \
     >"$work/absent.log" 2>&1; then arc=0; else arc=$?; fi
[ "$arc" -ne 0 ] || { echo 'gate: metallib-absent tooth SURVIVED (rc=0)' >&2; exit 1; }
[ ! -f "$work/absent.flro" ] || { echo 'gate: metallib-absent produced output (fallback!)' >&2; exit 1; }
printf 'KILLED  %-22s rc=%s no output file (CPU fallback 無し)\n' 'metallib-absent' "$arc"

# ---- 歯2: 既定は scalar のまま(--metal 無しは GPU を踏まぬ)----
./fieldrun "$work/t6.fldj" "$work/def.flro" >"$work/def.log" 2>&1
grep -q 'metal command' "$work/def.log" && { echo 'gate: default touched Metal' >&2; exit 1; }
cmp "$work/def.flro" "$work/40x40-4tick-gpu.s.flro"
printf 'green   %-22s %s\n' 'default-is-scalar' '旗無し= GPU 未接触・scalar 出力'

# ---- 歯3: Metal state 再利用(同一 process 内 複数 init)----
sed 's|bl _fl_wave_metal_init|bl _fl_wave_metal_init\n    ldr x0, [x27, #16]\n    bl _fl_wave_metal_init|' \
  fieldrun.s >"$work/twice.s"
cmp -s "$work/twice.s" fieldrun.s && { echo 'gate: twice-init did not mutate' >&2; exit 1; }
as -arch arm64 -o "$work/twice.o" "$work/twice.s"
link "$work/fr_twice" "$work/twice.o"
"$work/fr_twice" --metal "$MLIB" "$work/t6.fldj" "$work/twice.flro" >"$work/twice.log" 2>&1
cmp "$work/twice.flro" "$work/40x40-4tick-gpu.s.flro" \
  || { echo 'gate: double-init changed output (state reuse broken)' >&2; exit 1; }
printf 'green   %-22s %s\n' 'metal-state-reuse' '複数 init でも scalar と byte 一致'

# ---- 歯4..: fieldrun.s 変異(各々単独で scalar 出力と乖離 or 非零 rc)----
tooth() { # tooth <name> <sed-expr> [vectors...]
  name=$1; expr=$2; shift 2
  sed "$expr" fieldrun.s >"$work/m.s"
  cmp -s "$work/m.s" fieldrun.s && { echo "gate: tooth $name did not mutate" >&2; exit 1; }
  if as -arch arm64 -o "$work/m.o" "$work/m.s" 2>"$work/m.aserr"; then
    link "$work/mfr" "$work/m.o"
    killed=0; rcs=''
    for v in "$@"; do
      lbl=${v%%:*}; f=${v##*:}
      rm -f "$work/m.flro"
      if "$work/mfr" --metal "$MLIB" "$work/$f.fldj" "$work/m.flro" >"$work/m.out" 2>&1; then mrc=0; else mrc=$?; fi
      rcs="$rcs $lbl:rc=$mrc"
      if [ "$mrc" -ne 0 ] || ! cmp -s "$work/m.flro" "$work/$lbl.s.flro"; then killed=1; fi
    done
    [ "$killed" -eq 1 ] || { echo "gate: tooth $name SURVIVED (identical FLRO)" >&2; exit 1; }
    printf 'KILLED  %-22s%s  differs from scalar reference\n' "$name" "$rcs"
  else
    printf 'KILLED  %-22s assembler rejected mutant\n' "$name"
  fi
}
VEC='saturating-vector:t3 8x8-3tick-vecpath:t4 40x40-4tick-gpu:t6'
# reps 改変。reps=2 は「同一入力の再走」∴ out も sat も不変(bridge が dispatch 毎に sat 零化)
# = FLRO から観測不能 → 歯にならぬ(死枝、因つき記録)。観測可能な改変 = reps=0(dispatch 零)。
tooth metal-reps-0       's/mov x5, #1                     \/\/ reps/mov x5, #0  \/\/ reps/' $VEC
# offset 改変。bridge の byte_offset は **staging buffer 内部の置き場**(cur/prev/out 全てに同じく
# 適用され、長さも count*4+offset)∴ 4B 倍数の offset は **不変量**(q30_wave_metal 門が 0/4/28 を
# 同一結果として採点済)。故 「offset 改変が赤」は偽の歯 = 死枝。代りに不変性を緑で採点し、
# 4B 非倍数(3)を歯とする(misaligned = 同一結果を返してはならぬ)。
for off in 4 28; do
  sed "s/mov x4, #0                     \/\/ byte_offset/mov x4, #$off  \/\/ byte_offset/" fieldrun.s >"$work/off.s"
  cmp -s "$work/off.s" fieldrun.s && { echo "gate: offset $off did not mutate" >&2; exit 1; }
  as -arch arm64 -o "$work/off.o" "$work/off.s"
  link "$work/fr_off" "$work/off.o"
  "$work/fr_off" --metal "$MLIB" "$work/t6.fldj" "$work/off.flro" >/dev/null 2>&1
  cmp "$work/off.flro" "$work/40x40-4tick-gpu.s.flro" \
    || { echo "gate: offset=$off broke invariance" >&2; exit 1; }
  printf 'green   %-22s offset=%s scalar と byte 一致(bridge 不変量)\n' 'metal-offset-inv' "$off"
done
tooth metal-offset-3     's/mov x4, #0                     \/\/ byte_offset/mov x4, #3  \/\/ byte_offset/' $VEC
# 失敗時の黙落歯: **metallib 欠落下**で評価する(緑経路では -1 が起きぬ ∴ 検査は死碼に見える)。
# 生存条件 = rc=0 且 scalar と byte 一致(= CPU fallback を黙って返した)。
toothfail() { # toothfail <name> <sed-expr>
  name=$1; expr=$2
  sed "$expr" fieldrun.s >"$work/f.s"
  cmp -s "$work/f.s" fieldrun.s && { echo "gate: tooth $name did not mutate" >&2; exit 1; }
  as -arch arm64 -o "$work/f.o" "$work/f.s"
  link "$work/ffr" "$work/f.o"
  rm -f "$work/f.flro"
  if "$work/ffr" --metal "$work/no-such.metallib" "$work/t6.fldj" "$work/f.flro" >"$work/f.out" 2>&1; then frc=0; else frc=$?; fi
  if [ "$frc" -eq 0 ] && cmp -s "$work/f.flro" "$work/40x40-4tick-gpu.s.flro"; then
    echo "gate: tooth $name SURVIVED (silent CPU-equal result without GPU)" >&2; exit 1
  fi
  if [ -f "$work/f.flro" ]; then ev="output≠scalar"; else ev='no output'; fi
  printf 'KILLED  %-22s rc=%s %s (metallib 欠落下)\n' "$name" "$frc" "$ev"
}
toothfail metal-fail-silent  '/b.eq Lmetal_fail               \/\/ \[MUT:mfail\]/d'
toothfail metal-init-silent  's/cbnz x0, Lmetal_init_fail/nop/'
toothfail metal-both-silent  '/b.eq Lmetal_fail/d
s/cbnz x0, Lmetal_init_fail/nop/'
# 出力 endianness 改変(FLRO magic を BE で書く)
tooth metal-flro-magic-be 's/movk w10, #0x4F52, lsl #16/movk w10, #0x524F, lsl #16/' $VEC

# ---- 歯: dispatch 完了未待ち(bridge の waitUntilCompleted 選択子を潰す)----
sed 's/Lw_wait:         .asciz "waitUntilCompleted"/Lw_wait:         .asciz "waitUntilCompletedX"/' \
  ../q30_wave_metal/metal_bridge.s >"$work/nowait.s"
cmp -s "$work/nowait.s" ../q30_wave_metal/metal_bridge.s && { echo 'gate: nowait did not mutate' >&2; exit 1; }
as -arch arm64 -o "$work/nowait.o" "$work/nowait.s"
ld -arch arm64 -o "$work/fr_nowait" -e _main -lSystem -lobjc -framework Metal -framework Foundation \
   fieldrun.o q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o "$work/nowait.o" -syslibroot "$SDK"
rm -f "$work/nowait.flro"
if "$work/fr_nowait" --metal "$MLIB" "$work/t6.fldj" "$work/nowait.flro" >"$work/nowait.log" 2>&1; then nrc=0; else nrc=$?; fi
if [ "$nrc" -eq 0 ] && cmp -s "$work/nowait.flro" "$work/40x40-4tick-gpu.s.flro"; then
  echo 'gate: dispatch-not-awaited tooth SURVIVED' >&2; tail -3 "$work/nowait.log" >&2; exit 1
fi
printf 'KILLED  %-22s rc=%s %s\n' 'metal-no-wait' "$nrc" "$(head -1 "$work/nowait.log" | tr -d '\n')"

echo 'gate: fieldrun A6 (--metal) OK'
