#!/bin/sh
# fieldrun A4 門: §3c 2x2 一tick byte 一致 + 赤歯(回転/sat/FLRO/alias)。shell のみ。
set -eu
cd "$(dirname "$0")"

# 他言語混入禁 + FP/SIMD 禁(二重走査、A2/A3 の歯を継承)。
if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi
scan=$(grep -nE '(^|[^[:alnum:]_.])(v[0-9]+\.|[qdshb][0-9]+)([^[:alnum:]_]|$)' fieldrun.s || :)
case "$scan" in '') ;; *) echo "gate: FP/SIMD register in fieldrun.s"; echo "$scan" >&2; exit 1 ;; esac
scan2=$(grep -nE '^[[:space:]]*(f(add|sub|mul|div|cvt|mov|neg|abs|cmp)|scvtf|ucvtf|ld[1-4]|st[1-4])' fieldrun.s || :)
case "$scan2" in '') ;; *) echo "gate: FP instruction in fieldrun.s"; echo "$scan2" >&2; exit 1 ;; esac

./build.sh >/dev/null
work=$(mktemp -d ./fr-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM

# ---- 緑1: §3c 2x2 一tick ----
PAYLOAD="1065353216 0 0 0" ./gen_fldj.sh "$work/t1.fldj"
./fieldrun "$work/t1.fldj" "$work/t1.flro"
cells=$(od -An -td4 -j32 "$work/t1.flro" | tr -s ' ' | sed 's/^ //;s/ $//')
hdr=$(od -An -tx1 -N32 "$work/t1.flro" | tr '\n' ' ' | tr -s ' ' | sed 's/^ //;s/ $//')
want_cells='1950460 20972 20972 0'
want_hdr='46 4c 52 4f 00 00 00 00 02 00 00 00 02 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'
[ "$cells" = "$want_cells" ] || { echo "gate: cells '$cells' != '$want_cells'" >&2; exit 1; }
[ "$hdr" = "$want_hdr" ] || { echo "gate: header '$hdr' != '$want_hdr'" >&2; exit 1; }
sz=$(wc -c <"$work/t1.flro" | tr -d ' ')
[ "$sz" -eq 48 ] || { echo "gate: size $sz != 48" >&2; exit 1; }
printf 'green   %-22s cells=[%s] steps=1 sat=0\n' '3c-2x2-one-tick' "$cells"

# ---- 緑2: 多tick(回転の証拠)· 緑3: 飽和(sat 非零) ----
W=4 H=4 STEPN=3 PAYLOAD="1065353216" ./gen_fldj.sh "$work/t2.fldj"
./fieldrun "$work/t2.fldj" "$work/t2.flro"
ref2=$(od -An -tx1 "$work/t2.flro" | tr -s ' ')
printf 'green   %-22s sha=%s\n' '4x4-3tick' "$(shasum -a 256 "$work/t2.flro" | cut -d' ' -f1)"
FILL=1157627903 ./gen_fldj.sh "$work/t3.fldj"    # 2047.99.. 全胞 -> 飽和 4

./fieldrun "$work/t3.fldj" "$work/t3.flro"
sat=$(od -An -td8 -j24 -N8 "$work/t3.flro" | tr -d ' ')
# 独立経路(bc): acc = (2040220160*2147483520+2^29)>>30 = 4080440077 > 2^31-1 ∴ 全 4 胞飽和
[ "$sat" -eq 4 ] || { echo "gate: saturating vector sat=$sat want=4" >&2; exit 1; }
ref3=$(od -An -tx1 "$work/t3.flro" | tr -s ' ')
printf 'green   %-22s sat=%s\n' 'saturating-vector' "$sat"

# ---- payload 拒否は loud(飽和禁) ----
rej() { # rej <name> <want_rc> <payload bits>
  PAYLOAD="$3" ./gen_fldj.sh "$work/r.fldj"
  if ./fieldrun "$work/r.fldj" "$work/r.flro" >"$work/r.out" 2>&1; then rc=0; else rc=$?; fi
  [ "$rc" -eq "$2" ] || { echo "gate: $1 rc=$rc want=$2" >&2; cat "$work/r.out" >&2; exit 1; }
  printf 'KILLED  %-22s rc=%s %s\n' "$1" "$rc" "$(cat "$work/r.out")"
}
rej payload-nan 20 '2143289344'
rej payload-inf 20 '2139095040'
rej payload-over-2048 21 '1157627904'

# rc 歯(A1 の検査を継承)
rcz() { # rcz <name> <want_rc> <env...>
  n=$1; want=$2; shift 2
  env "$@" ./gen_fldj.sh "$work/e.fldj"
  if ./fieldrun "$work/e.fldj" "$work/e.flro" >"$work/e.out" 2>&1; then rc=0; else rc=$?; fi
  [ "$rc" -eq "$want" ] || { echo "gate: $n rc=$rc want=$want" >&2; cat "$work/e.out" >&2; exit 1; }
  printf 'KILLED  %-22s rc=%s %s\n' "$n" "$rc" "$(cat "$work/e.out")"
}
rcz magic 3 MAGIC=1245989959
rcz version 4 VERSION=2
rcz dt-noncanonical 22 DT=1036831948
rcz damping-decimal 22 DAMPING=1065336951
rcz writeraw-slot2 13 WSLOT=2
rcz writeraw-len-ne 15 WLEN=3
rcz nslots-lt2 9 NSLOTS=1
rcz unknown-tag 12 TAIL='\0377'

# ---- 変異歯: 各々単独で赤 ----
tooth() { # tooth <name> <sed-expr> <fldj> <ref-od>
  name=$1; expr=$2; src=$3; ref=$4
  sed "$expr" fieldrun.s >"$work/m.s"
  cmp -s "$work/m.s" fieldrun.s && { echo "gate: tooth $name did not mutate" >&2; exit 1; }
  as -arch arm64 -o "$work/m.o" "$work/m.s"
  ld -arch arm64 -o "$work/m" -e _main -lSystem "$work/m.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o \
     -syslibroot "$(xcrun --show-sdk-path)"
  if "$work/m" "$src" "$work/m.flro" >"$work/m.out" 2>&1; then mrc=0; else mrc=$?; fi
  got=$(od -An -tx1 "$work/m.flro" 2>/dev/null | tr -s ' ' || :)
  if [ "$mrc" -eq 0 ] && [ "$got" = "$ref" ]; then
    echo "gate: tooth $name SURVIVED (identical FLRO)" >&2; exit 1
  fi
  printf 'KILLED  %-22s rc=%s output differs from reference\n' "$name" "$mrc"
}

tooth rotation-2swap  '/\[MUT:rot3\]/d'                            "$work/t2.fldj" "$ref2"
tooth sat-dropped     '/\[MUT:sat\]/d'                             "$work/t3.fldj" "$ref3"
tooth flro-steps-zero 's|str x28, \[x9, #16\].*|str xzr, [x9, #16]|' "$work/t2.fldj" "$ref2"
tooth flro-sat-zero   's|str x19, \[x9, #24\].*|str xzr, [x9, #24]|' "$work/t3.fldj" "$ref3"
tooth flro-magic-be   's|movk w10, #0x4F52, lsl #16.*|movz w10, #0x524F\n    movk w10, #0x464C, lsl #16|' "$work/t1.fldj" "$(od -An -tx1 "$work/t1.flro" | tr -s ' ')"
tooth out-alias-cur   's|mov x3, x27 .*\[MUT:alias\].*|mov x3, x25|' "$work/t2.fldj" "$ref2"

echo 'gate: fieldrun A4 OK'
