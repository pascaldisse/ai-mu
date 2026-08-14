#!/bin/sh
# fieldrun A5 門: --neon backend。scalar と FLRO byte/SHA 一致 + no-fallback + neon 変異歯。
# shell のみ(新規 Rust/C/Swift/Python 零)。
set -eu
cd "$(dirname "$0")"

if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi
# fieldrun.s 自体は整数のみ(§11 の走査を継承)。NEON は ../q30_wave/wave_neon.s に閉じ、
# fieldrun.s は記号呼出のみ ∴ 走査対象は fieldrun.s に限る(除外理由 = 契約 §12)。
scan=$(grep -nE '(^|[^[:alnum:]_.])(v[0-9]+\.|[qdshb][0-9]+)([^[:alnum:]_]|$)' fieldrun.s || :)
case "$scan" in '') ;; *) echo "gate: FP/SIMD register in fieldrun.s"; echo "$scan" >&2; exit 1 ;; esac

./build.sh >/dev/null
work=$(mktemp -d ./neon-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM
SDK=$(xcrun --show-sdk-path)

# ---- 三 vector(前任 A4 の門と同一)+ vector 経路を踏む 8x8 ----
PAYLOAD="1065353216 0 0 0" ./gen_fldj.sh "$work/t1.fldj"
W=4 H=4 STEPN=3 PAYLOAD="1065353216" ./gen_fldj.sh "$work/t2.fldj"
FILL=1157627903 ./gen_fldj.sh "$work/t3.fldj"
# w>=6 ∴ 4胞 vector 経路実走。非一様場(lane 置換が見える様に胞毎異値)。
p4=''
k=0
while [ $k -lt 64 ]; do
  p4="$p4 $((1065353216 + k * 1048576))"   # 1.0, 1.0078125, ... 各胞相異
  k=$((k + 1))
done
W=8 H=8 STEPN=3 PAYLOAD="$p4" ./gen_fldj.sh "$work/t4.fldj"
# w=9: vector 進行 x=1,5 の次 x=9 で x+4==w ∴ vector 境界の off-by-one が露出する幅。
p5=''
k=0
while [ $k -lt 81 ]; do
  p5="$p5 $((1065353216 + k * 1048576))"
  k=$((k + 1))
done
W=9 H=9 STEPN=3 PAYLOAD="$p5" ./gen_fldj.sh "$work/t5.fldj"

cmp_backend() { # cmp_backend <name> <fldj>
  ./fieldrun "$2" "$work/$1.s.flro"
  ./fieldrun --neon "$2" "$work/$1.n.flro"
  a=$(shasum -a 256 "$work/$1.s.flro" | cut -d' ' -f1)
  b=$(shasum -a 256 "$work/$1.n.flro" | cut -d' ' -f1)
  if [ "$a" != "$b" ]; then
    echo "gate: $1 scalar=$a neon=$b MISMATCH" >&2
    od -An -tx1 "$work/$1.s.flro" >&2; od -An -tx1 "$work/$1.n.flro" >&2; exit 1
  fi
  cmp "$work/$1.s.flro" "$work/$1.n.flro" || { echo "gate: $1 byte mismatch" >&2; exit 1; }
  printf 'green   %-22s sha=%s (scalar==neon, byte 一致)\n' "$1" "$a"
}
cmp_backend 3c-2x2-one-tick   "$work/t1.fldj"
cmp_backend 4x4-3tick         "$work/t2.fldj"
cmp_backend saturating-vector "$work/t3.fldj"
cmp_backend 8x8-3tick-vecpath "$work/t4.fldj"
cmp_backend 9x9-3tick-vecedge "$work/t5.fldj"

# 4x4-3tick の scalar SHA は契約既知値と一致せねばならぬ
want4=a85a4cc0ee310770209e5a67834ed7693b159c6130eae7f5afe709b093050a3c
got4=$(shasum -a 256 "$work/4x4-3tick.s.flro" | cut -d' ' -f1)
[ "$got4" = "$want4" ] || { echo "gate: 4x4-3tick scalar sha=$got4 want=$want4" >&2; exit 1; }
printf 'green   %-22s %s\n' '4x4-known-sha' "$got4"

# ---- 既定 = scalar の実証: neon を壊した build でも 旗無しは無傷 ----
sed 's/lsl x4, x4, #29/lsl x4, x4, #28/' ../q30_wave/wave_neon.s >"$work/brk.s"
as -arch arm64 -o "$work/brk.o" "$work/brk.s"
ld -arch arm64 -o "$work/fr_brk" -e _main -lSystem -lobjc -framework Metal -framework Foundation metal_bridge.o fieldrun.o q20_conv_lib.o coef_lib.o \
   wave_scalar.o "$work/brk.o" -syslibroot "$SDK"
"$work/fr_brk" "$work/t4.fldj" "$work/brk_default.flro"
cmp "$work/brk_default.flro" "$work/8x8-3tick-vecpath.s.flro" \
  || { echo 'gate: default backend is not scalar' >&2; exit 1; }
if "$work/fr_brk" --neon "$work/t4.fldj" "$work/brk_neon.flro" >/dev/null 2>&1 \
   && cmp -s "$work/brk_neon.flro" "$work/8x8-3tick-vecpath.s.flro"; then
  echo 'gate: --neon silently produced scalar result (fallback!)' >&2; exit 1
fi
printf 'green   %-22s %s\n' 'default-is-scalar' '壊れた neon を連結しても旗無しは scalar 出力・--neon は乖離'

# ---- no-fallback: neon 記号を潰す -> 赤(非零rc)。scalar 結果を返さぬ事 ----
sed 's/_fl_q30_wave_neon/_fl_q30_wave_neon_GONE/g' ../q30_wave/wave_neon.s >"$work/gone.s"
as -arch arm64 -o "$work/gone.o" "$work/gone.s"
if ld -arch arm64 -o "$work/fr_gone" -e _main -lSystem -lobjc -framework Metal -framework Foundation metal_bridge.o fieldrun.o q20_conv_lib.o coef_lib.o \
     wave_scalar.o "$work/gone.o" -syslibroot "$SDK" >"$work/gone.out" 2>&1; then lrc=0; else lrc=$?; fi
[ "$lrc" -ne 0 ] || { echo 'gate: no-fallback tooth SURVIVED (linked without neon symbol)' >&2; exit 1; }
[ ! -x "$work/fr_gone" ] || { echo 'gate: binary produced despite missing neon symbol' >&2; exit 1; }
printf 'KILLED  %-22s rc=%s %s\n' 'neon-symbol-gone' "$lrc" "$(grep -m1 -i 'undefined' "$work/gone.out" || echo 'link failed')"

# ---- neon 経路 変異歯(各々単独で赤、scalar 出力と乖離)----
tooth() { # tooth <name> <sed-expr>  — t4(8x8)/t5(9x9) の何れかで scalar 出力と乖離せば KILLED
  name=$1; expr=$2
  sed "$expr" ../q30_wave/wave_neon.s >"$work/m.s"
  cmp -s "$work/m.s" ../q30_wave/wave_neon.s && { echo "gate: tooth $name did not mutate" >&2; exit 1; }
  if as -arch arm64 -o "$work/m.o" "$work/m.s" 2>"$work/m.aserr"; then
    ld -arch arm64 -o "$work/mfr" -e _main -lSystem -lobjc -framework Metal -framework Foundation metal_bridge.o fieldrun.o q20_conv_lib.o coef_lib.o \
       wave_scalar.o "$work/m.o" -syslibroot "$SDK"
    killed=0; rcs=''
    for v in 8x8-3tick-vecpath:t4 9x9-3tick-vecedge:t5; do
      lbl=${v%%:*}; f=${v##*:}
      if "$work/mfr" --neon "$work/$f.fldj" "$work/m.flro" >"$work/m.out" 2>&1; then mrc=0; else mrc=$?; fi
      rcs="$rcs $lbl:rc=$mrc"
      if [ "$mrc" -ne 0 ] || ! cmp -s "$work/m.flro" "$work/$lbl.s.flro"; then killed=1; fi
    done
    [ "$killed" -eq 1 ] || { echo "gate: tooth $name SURVIVED (identical FLRO on both vectors)" >&2; exit 1; }
    printf 'KILLED  %-22s%s  output differs from scalar reference\n' "$name" "$rcs"
  else
    printf 'KILLED  %-22s assembler rejected mutant\n' "$name"
  fi
}
# 丸め差: 半加算 2^29 -> 2^28
tooth neon-round-half     's/lsl x4, x4, #29/lsl x4, x4, #28/'
# 末尾要素落ち: 上位2 lane の書出を落とす
tooth neon-tail-drop      '/sqxtn2 v0.4s, v31.2d/d'
# レーン跨ぎ: center 上半抽出の ext を #8 -> #4 へ
tooth neon-lane-cross     's/ext v29.16b, v0.16b, v0.16b, #8/ext v29.16b, v0.16b, v0.16b, #4/'
# vector 境界の off-by-one: x+5<=w を x+4<=w へ(端胞を vector が食う)
tooth neon-bound-off-by-1 's/add x6, x4, #5/add x6, x4, #4/'
# 飽和数の欠落: vector 半の飽和計上を落とす
tooth neon-sat-drop       's/sub x13, x13, x12/nop/'

echo 'gate: fieldrun A5 (--neon) OK'
