#!/bin/sh
# fieldrun A8 門: §3d 全変異の単独適用 + A1..A7 全歯の一括表。shell のみ(新規 Rust/C/Swift/Python 零)。
# 用: teeth_kill.sh [--new-only]
#   段1 = A8 残余変異(truncation・trailing byte・bad tag・bad len・w*h 32bit・backend 強制失敗)
#   段2 = 既存門(A1..A7)を実走し KILLED/green を採取 → 一枚表
set -eu
cd "$(dirname "$0")"

if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi

./build.sh >/dev/null
work=$(mktemp -d ./a8-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM
SDK=$(xcrun --show-sdk-path)
rows="$work/rows"
: >"$rows"

row() { # row <atom> <tooth> <mutation> <expect> <observed> <verdict>
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >>"$rows"
}

# ---- 材料(全て gen_fldj.sh = 独立生成器)----
./gen_fldj.sh "$work/good.fldj"                      # canonical 2x2
SZ=$(wc -c <"$work/good.fldj" | tr -d ' ')
dd if="$work/good.fldj" of="$work/cut.fldj" bs=1 count=60 >/dev/null 2>&1   # payload 途中で切断
TAIL='\0003' ./gen_fldj.sh "$work/tail.fldj"          # 末尾余剰 1B
TAIL='\0377' ./gen_fldj.sh "$work/tag.fldj"           # 未知 tag 0xFF
WLEN=3 ./gen_fldj.sh "$work/wlen.fldj"                # WriteRaw len != w*h
W=65536 H=65536 WLEN=1 ./gen_fldj.sh "$work/mul.fldj" # w*h = 2^32(32bit なら 0 へ wrap)

# 基準 rc(無変異 fieldrun)
refrc() { if ./fieldrun "$1" "$work/ref.flro" >"$work/ref.log" 2>&1; then echo 0; else echo $?; fi; }
RC_CUT=$(refrc "$work/cut.fldj")
RC_TAIL=$(refrc "$work/tail.fldj")
RC_TAG=$(refrc "$work/tag.fldj")
RC_WLEN=$(refrc "$work/wlen.fldj")
RC_MUL=$(refrc "$work/mul.fldj")

echo "== A8 段1: §3d 残余変異(各々単独適用)=="
printf 'raw     baseline rc: truncation=%s trailing=%s bad-tag=%s bad-len=%s wh-32bit=%s\n' \
  "$RC_CUT" "$RC_TAIL" "$RC_TAG" "$RC_WLEN" "$RC_MUL"

tooth() { # tooth <name> <sed-expr> <fldj> <ref_rc> <mutation-desc> [metal-args...]
  name=$1; expr=$2; src=$3; want=$4; desc=$5; shift 5
  sed "$expr" fieldrun.s >"$work/m.s"
  cmp -s "$work/m.s" fieldrun.s && { echo "gate: tooth $name did not mutate" >&2; exit 1; }
  as -arch arm64 -o "$work/m.o" "$work/m.s"
  ld -arch arm64 -o "$work/m" -e _main -lSystem -lobjc -framework Metal -framework Foundation \
     "$work/m.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$SDK"
  rm -f "$work/m.flro"
  if "$work/m" "$@" "$src" "$work/m.flro" >"$work/m.out" 2>&1; then mrc=0; else mrc=$?; fi
  if [ "$mrc" -eq "$want" ]; then
    printf 'SURVIVED %-24s rc=%s == baseline rc=%s\n' "$name" "$mrc" "$want" >&2
    row A8 "$name" "$desc" "rc!=$want" "rc=$mrc" SURVIVED
    echo "gate: tooth $name SURVIVED" >&2; exit 1
  fi
  printf 'KILLED  %-26s baseline rc=%s -> mutant rc=%s  %s\n' "$name" "$want" "$mrc" "$(head -1 "$work/m.out" 2>/dev/null || :)"
  row A8 "$name" "$desc" "rc!=$want" "rc=$mrc" KILLED
}

tooth truncation-check-removed 's|cmp x15, x12|cmp x15, xzr|' "$work/cut.fldj" "$RC_CUT" \
  'WriteRaw payload 残長検査 cmp x15,x12 -> x15,xzr(常真)'
tooth trailing-byte-check-removed 's|cmp x15, #5|cmp x15, #0|' "$work/tail.fldj" "$RC_TAIL" \
  'Step op 残長検査 cmp x15,#5 -> #0(末尾余剰 1B を素通し)'
tooth bad-tag-check-removed 's|b Lrej_tag|b Ldone|' "$work/tag.fldj" "$RC_TAG" \
  '未知 tag -> Lrej_tag を Ldone へ(黙って正常終了)'
tooth bad-len-check-removed 's|b.ne Lrej_wlen|nop|' "$work/wlen.fldj" "$RC_WLEN" \
  'WriteRaw len==w*h 検査(b.ne Lrej_wlen)除去'
tooth wh-32bit-multiply 's|mul x24, x22, x23|mul w24, w22, w23\n    uxtw x24, w24|' "$work/mul.fldj" "$RC_MUL" \
  'n = w*h を 64bit -> 32bit 乗算へ弱化(65536^2 が 0 へ wrap)'

# backend 強制失敗: metallib 不在で init 失敗を強制。無変異 = rc=24 loud・出力 file 零。
rm -f "$work/bk.flro"
if ./fieldrun --metal "$work/no-such.metallib" "$work/good.fldj" "$work/bk.flro" >"$work/bk.out" 2>&1; then RC_BK=0; else RC_BK=$?; fi
[ "$RC_BK" -eq 24 ] || { echo "gate: forced backend failure rc=$RC_BK want=24" >&2; cat "$work/bk.out" >&2; exit 1; }
[ ! -f "$work/bk.flro" ] || { echo 'gate: backend 失敗なのに出力 file を作った(黙落)' >&2; exit 1; }
printf 'green   %-26s rc=24 出力 file 零 (%s)\n' 'backend-forced-failure' "$(head -1 "$work/bk.out")"
row A8 backend-forced-failure 'metallib 不在で _fl_wave_metal_init 失敗を強制(変異零・入力側)' \
  'rc=24 かつ出力零' "rc=$RC_BK 出力零" green
tooth metal-init-check-removed 's|cbnz x0, Lmetal_init_fail|nop|' "$work/good.fldj" 24 \
  'init 失敗検査除去 -> 黙って CPU へ落ちる経路が無い事の実証' --metal "$work/no-such.metallib"

if [ "${1:-}" = '--new-only' ]; then
  echo 'gate: fieldrun A8 段1 のみ (--new-only)'
  exit 0
fi

# ---- 段2: 既存門 A1..A7 を実走し歯を採取 ----
echo
echo "== A8 段2: 既存門 実走(A1..A7 全歯採取・緑維持検査)=="
if ./gate.sh >"$work/gate.log" 2>&1; then GRC=0; else GRC=$?; fi
if grep -q 'SURVIVED' "$work/gate.log"; then
  echo 'gate: SURVIVED tooth in existing gates' >&2; grep -n SURVIVED "$work/gate.log" >&2; exit 1
fi
[ "$GRC" -eq 0 ] || { echo "gate: gate.sh rc=$GRC" >&2; tail -40 "$work/gate.log" >&2; exit 1; }
sed -n 's/^gate: /  gate: /p' "$work/gate.log"

# 節境界: gate.sh の 'gate: ... OK' 行で atom を進める。
awk '
  BEGIN { atom="A1" }
  /^KILLED[ \t]/ { name=$2; rc=$3; $1=""; $2=""; $3=""; sub(/^[ \t]+/,"",$0);
                   printf "%s\t%s\t%s\t%s\t%s\t%s\n", atom, name, "既存門記載(§8-§14)", "赤", (rc " " $0), "KILLED" ; next }
  /^green[ \t]/  { name=$2; $1=""; $2=""; sub(/^[ \t]+/,"",$0);
                   printf "%s\t%s\t%s\t%s\t%s\t%s\n", atom, name, "不変量/緑点", "緑", $0, "green" ; next }
  /^gate: fldj_parse A1 OK/ { atom="A2"; next }
  /^gate: q20_conv A2 OK/   { atom="A3"; next }
  /^gate: coef A3 OK/       { atom="A4"; next }
  /^gate: fieldrun A4 OK/   { atom="A5"; next }
  /^gate: fieldrun A5/      { atom="A6"; next }
  /^gate: fieldrun A6/      { atom="A7"; next }
' "$work/gate.log" >>"$rows"

echo
echo "== teeth_kill 一括表 (atom / 歯名 / 変異内容 / 期待 / 実測 / 判定) =="
awk -F'\t' '{ printf "%-3s | %-26s | %-62s | %-10s | %-34s | %s\n", $1,$2,$3,$4,$5,$6 }' "$rows"

k=$(awk -F'\t' '$6=="KILLED"' "$rows" | wc -l | tr -d ' ')
g=$(awk -F'\t' '$6=="green"' "$rows" | wc -l | tr -d ' ')
s=$(awk -F'\t' '$6=="SURVIVED"' "$rows" | wc -l | tr -d ' ')
printf '\n合計: KILLED=%s green=%s SURVIVED=%s\n' "$k" "$g" "$s"
[ "$s" -eq 0 ] || { echo 'gate: SURVIVED present' >&2; exit 1; }
echo 'gate: fieldrun A8 (teeth_kill 一括表) OK'
