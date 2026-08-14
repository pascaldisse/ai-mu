#!/bin/sh
# A7 門: 実 fieldc 生成 journal(32x32・寫 0 1024・歩 200)一本から
#   scalar / --neon / --metal 三経路 FLRO byte・SHA 一致 + 真の時間発展(≥200 step, 初期場≠最終場)。
# shell のみ(新規 Rust/C/Swift/Python 零)。実機 GPU 実走のみ緑。
set -eu
cd "$(dirname "$0")"

if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi
# 独立性証跡: .fld 生成器は runtime/parser 実装を読まぬ。
if grep -vE '^[[:space:]]*#' gen_fld_a7.sh \
   | grep -nE '(fieldrun|fldj_parse|fieldc|q20_conv|coef|wave_|\.s\b|\.fldj)' ; then
  echo 'gate: generator references implementation (independence broken)' >&2; exit 1
fi
printf 'green   %-26s %s\n' 'generator-independence' 'gen_fld_a7.sh の非註釈行に実装参照 零'

./build.sh >/dev/null
MLIB=${MLIB:-../q30_wave_metal/q30_wave.metallib}
[ -f "$MLIB" ] || (cd ../q30_wave_metal && ./build.sh >/dev/null)
[ -f "$MLIB" ] || { echo "gate: metallib absent: $MLIB" >&2; exit 1; }
FIELDC=${FIELDC:-../fieldc}
[ -x "$FIELDC" ] || { echo "gate: fieldc absent: $FIELDC" >&2; exit 1; }
MAXCELLS=${MAXCELLS:-4096}      # 硬碼禁 = 既定 + 引数(既定 16384 を上書き可能な事の実証)

work=$(mktemp -d ./a7-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM

W=${W:-32}; H=${H:-32}; STEPN=${STEPN:-200}
export W H

# ---- 1. .fld 源(shell)→ 2. fieldc → 実 FLDJ ----
STEPN=$STEPN ./gen_fld_a7.sh "$work/a7.fld"
"$FIELDC" "$work/a7.fld" "$work/a7.fldj"
srcsha=$(shasum -a 256 "$work/a7.fld" | cut -d' ' -f1)
jsha=$(shasum -a 256 "$work/a7.fldj" | cut -d' ' -f1)
printf 'green   %-26s fld=%sB fldj=%sB\n' 'fieldc-compile' \
  "$(wc -c <"$work/a7.fld" | tr -d ' ')" "$(wc -c <"$work/a7.fldj" | tr -d ' ')"
printf 'raw     %-26s %s\n' 'sha(a7.fld)'  "$srcsha"
printf 'raw     %-26s %s\n' 'sha(a7.fldj)' "$jsha"
printf 'raw     %-26s ' 'fldj-header(48B)'; od -An -tx1 -N48 "$work/a7.fldj" | tr -s ' \n' ' '; echo
printf 'raw     %-26s %s\n' 'fld-src-head' "$(sed -n 2p "$work/a7.fld")"
printf 'raw     %-26s %s\n' 'fld-src-step' "$(tail -n1 "$work/a7.fld")"

# ---- 3. 三経路実走(同一 journal)----
tri() { # tri <name> <fldj> ; stdout に sha
  ./fieldrun            "$2" "$work/$1.s.flro" "$MAXCELLS" >/dev/null
  ./fieldrun --neon     "$2" "$work/$1.n.flro" "$MAXCELLS" >/dev/null
  ./fieldrun --metal "$MLIB" "$2" "$work/$1.m.flro" "$MAXCELLS" >"$work/$1.m.log" 2>&1
  a=$(shasum -a 256 "$work/$1.s.flro" | cut -d' ' -f1)
  b=$(shasum -a 256 "$work/$1.n.flro" | cut -d' ' -f1)
  c=$(shasum -a 256 "$work/$1.m.flro" | cut -d' ' -f1)
  if [ "$a" != "$b" ] || [ "$a" != "$c" ]; then
    echo "gate: $1 scalar=$a neon=$b metal=$c MISMATCH" >&2; exit 1
  fi
  cmp "$work/$1.s.flro" "$work/$1.n.flro" || { echo "gate: $1 s/n byte mismatch" >&2; exit 1; }
  cmp "$work/$1.s.flro" "$work/$1.m.flro" || { echo "gate: $1 s/m byte mismatch" >&2; exit 1; }
  printf 'green   %-26s sha=%s (scalar==neon==metal, byte 一致)\n' "$1" "$a" >&2
  printf '%s' "$a"
}
sha200=$(tri "real-journal-${W}x${H}-${STEPN}step" "$work/a7.fldj")
base="real-journal-${W}x${H}-${STEPN}step"

# GPU 実走の証跡
grep -q 'metal command status: 4' "$work/$base.m.log" \
  || { echo 'gate: no GPU completion evidence' >&2; cat "$work/$base.m.log" >&2; exit 1; }
printf 'green   %-26s %s\n' 'gpu-evidence' "$(grep -m1 'status' "$work/$base.m.log")"

# ---- 4. ≥200 step · nonzero evolution ----
hdr=$(od -An -tx1 -N32 "$work/$base.s.flro" | tr -s ' \n' ' ')
printf 'raw     %-26s%s\n' 'flro-header' "$hdr"
rd32() { od -An -tu4 -j"$1" -N4 "$2" | tr -d ' '; }
steps=$(rd32 16 "$work/$base.s.flro"); sathi=$(rd32 24 "$work/$base.s.flro")
[ "$steps" -ge 200 ] || { echo "gate: steps=$steps < 200" >&2; exit 1; }
printf 'green   %-26s steps=%s sat=%s (FLRO off16/off24)\n' 'steps>=200' "$steps" "$sathi"

# 0 step 版(同一初期場)= 初期場そのもの。之と 200 step 版を比べる。
STEPN=0 ./gen_fld_a7.sh "$work/a7_0.fld"
"$FIELDC" "$work/a7_0.fld" "$work/a7_0.fldj"
./fieldrun "$work/a7_0.fldj" "$work/step0.flro" "$MAXCELLS" >/dev/null
sha0=$(shasum -a 256 "$work/step0.flro" | cut -d' ' -f1)
[ "$sha0" != "$sha200" ] || { echo 'gate: 0step == 200step (no evolution)' >&2; exit 1; }
ndiff=$(cmp -l "$work/step0.flro" "$work/$base.s.flro" | wc -l | tr -d ' ')
printf 'green   %-26s sha0=%s\n' 'nonzero-evolution' "$sha0"
printf 'green   %-26s sha200=%s diff_bytes=%s/%s\n' 'nonzero-evolution' "$sha200" "$ndiff" \
  "$(wc -c <"$work/$base.s.flro" | tr -d ' ')"
printf 'raw     %-26s%s\n' 'flro0-header' "$(od -An -tx1 -N32 "$work/step0.flro" | tr -s ' \n' ' ')"
printf 'raw     %-26s%s\n' 'cells0-first8'   "$(od -An -tu4 -j32 -N32 "$work/step0.flro" | tr -s ' \n' ' ')"
printf 'raw     %-26s%s\n' 'cells200-first8' "$(od -An -tu4 -j32 -N32 "$work/$base.s.flro" | tr -s ' \n' ' ')"

# ---- 6. 歯: digest が内容に真に依存する事 ----
# 6a. step 数改変(199)
STEPN=199 ./gen_fld_a7.sh "$work/m_step.fld"
"$FIELDC" "$work/m_step.fld" "$work/m_step.fldj"
./fieldrun "$work/m_step.fldj" "$work/m_step.flro" "$MAXCELLS" >/dev/null
s=$(shasum -a 256 "$work/m_step.flro" | cut -d' ' -f1)
[ "$s" != "$sha200" ] || { echo 'gate: tooth step-199 SURVIVED' >&2; exit 1; }
printf 'KILLED  %-26s sha=%s != sha200\n' 'tooth:step-count-199' "$s"

# 6b. 初期場改変(PHASE=1 = 別の非一様場、同 step 数)
PHASE=1 STEPN=$STEPN ./gen_fld_a7.sh "$work/m_field.fld"
cmp -s "$work/a7.fld" "$work/m_field.fld" && { echo 'gate: PHASE had no effect' >&2; exit 1; }
"$FIELDC" "$work/m_field.fld" "$work/m_field.fldj"
./fieldrun "$work/m_field.fldj" "$work/m_field.flro" "$MAXCELLS" >/dev/null
s=$(shasum -a 256 "$work/m_field.flro" | cut -d' ' -f1)
[ "$s" != "$sha200" ] || { echo 'gate: tooth init-field SURVIVED' >&2; exit 1; }
printf 'KILLED  %-26s sha=%s != sha200\n' 'tooth:init-field-phase1' "$s"

# 6c. journal 1 byte 改変(payload 中の 1 byte を反転)→ 出力乖離 或 loud reject
off=${FLIPOFF:-100}
dd if="$work/a7.fldj" of="$work/pre.bin"  bs=1 count=$off              2>/dev/null
dd if="$work/a7.fldj" of="$work/byte.bin" bs=1 skip=$off count=1       2>/dev/null
dd if="$work/a7.fldj" of="$work/post.bin" bs=1 skip=$((off + 1))       2>/dev/null
ob=$(od -An -tu1 "$work/byte.bin" | tr -d ' ')
nb=$(( (ob + 1) % 256 ))
printf '%b' "\\0$(printf '%03o' $nb)" >"$work/nb.bin"
cat "$work/pre.bin" "$work/nb.bin" "$work/post.bin" >"$work/m_1byte.fldj"
[ "$(wc -c <"$work/m_1byte.fldj")" = "$(wc -c <"$work/a7.fldj")" ] || { echo 'gate: flip size drift' >&2; exit 1; }
cmp -s "$work/a7.fldj" "$work/m_1byte.fldj" && { echo 'gate: flip no-op' >&2; exit 1; }
if ./fieldrun "$work/m_1byte.fldj" "$work/m_1byte.flro" "$MAXCELLS" >"$work/m1.log" 2>&1; then
  s=$(shasum -a 256 "$work/m_1byte.flro" | cut -d' ' -f1)
  [ "$s" != "$sha200" ] || { echo 'gate: tooth 1byte SURVIVED' >&2; exit 1; }
  printf 'KILLED  %-26s off=%s %s->%s sha=%s != sha200\n' 'tooth:journal-1byte' "$off" "$ob" "$nb" "$s"
else
  rc=$?
  printf 'KILLED  %-26s off=%s %s->%s rc=%s (loud reject)\n' 'tooth:journal-1byte' "$off" "$ob" "$nb" "${rc:-?}"
fi

# 6d. max_cells 引数が真に効く(硬碼でない): 1024 胞 > 上限 512 → rc=8
rc=0; ./fieldrun "$work/a7.fldj" "$work/x.flro" 512 >"$work/mc.log" 2>&1 || rc=$?
[ "$rc" -ne 0 ] || { echo 'gate: tooth max-cells-arg SURVIVED' >&2; exit 1; }
printf 'KILLED  %-26s rc=%s (max_cells=512 < 1024 胞)\n' 'tooth:max-cells-arg' "$rc"
# 6e. 逆: max_cells=1024 丁度 = 緑(引数が真に上限を上げる事)
./fieldrun "$work/a7.fldj" "$work/mc1024.flro" 1024 >/dev/null
s=$(shasum -a 256 "$work/mc1024.flro" | cut -d' ' -f1)
[ "$s" = "$sha200" ] || { echo 'gate: max_cells=1024 output drift' >&2; exit 1; }
printf 'green   %-26s sha=%s (同一)\n' 'max-cells=1024-exact' "$s"

printf 'gate: fieldrun A7 (real fieldc journal, %sx%s, %s step, 三経路) OK\n' "$W" "$H" "$steps"
