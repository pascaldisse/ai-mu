#!/bin/sh
# slotstore 実 journal 門: real fieldc(.fld→.fldj)で slot>1 書込・多 slot 書込 → 200 steps。
# 主張: canonical 出力 = slot0 ∴ 異 slot への追加書込は FLRO を一切変えぬ(三経路とも)。
# 併せて mmap 部分確保失敗の注入(rc=19 loud・出力残零・台帳漏零)を実走で示す。
# shell + 既存 ARM64 build のみ。新規 Rust/C/Swift/Python 零。
set -eu
cd "$(dirname "$0")"
FIELDC=${FIELDC:-../fieldc}
METALLIB=${METALLIB:-../q30_wave_metal/q30_wave.metallib}
W=${W:-32}; H=${H:-32}; STEPN=${STEPN:-200}; NSLOTS=${NSLOTS:-4}
max_cells=${MAX_CELLS:-16384}; max_bytes=${MAX_BYTES:-4194304}; max_slots=${MAX_SLOTS:-1024}
[ -x "$FIELDC" ] || { echo "gate: SKIP-ENV fieldc absent: $FIELDC — 検査未実施 rc=3" >&2; exit 3; }
./build.sh >/dev/null
work=$(mktemp -d ./slotstore-real.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM

run() { ./fieldrun "$1" "$2" "$max_cells" "$max_bytes" "$max_slots"; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

# ---- 基準 = 単一 slot0 書込の実 journal(既存 A7 生成器・実装非参照)----
W=$W H=$H NSLOTS=$NSLOTS STEPN=$STEPN ./gen_fld_a7.sh "$work/base.fld"
"$FIELDC" "$work/base.fld" "$work/base.fldj"
run "$work/base.fldj" "$work/base.flro"
base=$(sha "$work/base.flro")
printf 'green   %-26s sha=%s (real fieldc · slot0 のみ)\n' 'real-base' "$base"

# ---- 多 slot journal = 同一源 + slot2/slot3 への 寫(位相ずらし場)----
W=$W H=$H NSLOTS=$NSLOTS STEPN=$STEPN PHASE=7 ./gen_fld_a7.sh "$work/p7.fld"
W=$W H=$H NSLOTS=$NSLOTS STEPN=$STEPN PHASE=13 ./gen_fld_a7.sh "$work/p13.fld"
w2=$(grep '^寫 0 ' "$work/p7.fld" | sed 's/^寫 0 /寫 2 /')
w3=$(grep '^寫 0 ' "$work/p13.fld" | sed 's/^寫 0 /寫 3 /')
awk -v a="$w2" -v b="$w3" '/^歩 /{print a; print b} {print}' "$work/base.fld" >"$work/multi.fld"
grep -c '^寫 ' "$work/multi.fld" >"$work/nwrites"
[ "$(cat "$work/nwrites")" = 3 ] || { echo "gate: multi.fld 寫 行数 異常" >&2; exit 1; }
"$FIELDC" "$work/multi.fld" "$work/multi.fldj"
run "$work/multi.fldj" "$work/multi.flro"
multi=$(sha "$work/multi.flro")
cmp "$work/base.flro" "$work/multi.flro"
steps=$(od -An -tu8 -j16 -N8 "$work/multi.flro" | tr -d ' ')
[ "$steps" = "$STEPN" ] || { echo "gate: steps=$steps != $STEPN" >&2; exit 1; }
printf 'green   %-26s sha=%s steps=%s (slot2/slot3 書込は canonical を変えぬ)\n' 'real-multislot' "$multi" "$steps"

# ---- 三経路(scalar/NEON/Metal)が多 slot journal でも byte 一致 ----
./fieldrun --neon "$work/multi.fldj" "$work/multi.neon.flro" "$max_cells" "$max_bytes" "$max_slots"
cmp "$work/multi.flro" "$work/multi.neon.flro"
printf 'green   %-26s %s\n' 'real-multislot-neon' "$(sha "$work/multi.neon.flro")"
if [ -f "$METALLIB" ]; then
  ./fieldrun --metal "$METALLIB" "$work/multi.fldj" "$work/multi.metal.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/metal.log" 2>&1
  cmp "$work/multi.flro" "$work/multi.metal.flro"
  printf 'green   %-26s %s\n' 'real-multislot-metal' "$(sha "$work/multi.metal.flro")"
  printf 'raw     %-26s %s\n' 'gpu-evidence' "$(grep -i 'metal command status' "$work/metal.log" || echo 'なし')"
else
  echo "UNVERIFIED real-multislot-metal: metallib 不在 $METALLIB" >&2
fi

# ---- mmap 部分確保失敗の注入: slot 一枚の寸を巨大化 → 二枚目以降の確保が必ず失敗。
#      期待 = rc=19 loud · 出力 file 残零 · 台帳(生存 mapping)漏零(rc=28 に非ず = 早期 loud)。
sed 's|str x9, \[sp, #200\].*|movz x9, #0x7FF0, lsl #48\n    str x9, [sp, #200]|' fieldrun.s >"$work/oom.s"
cmp -s fieldrun.s "$work/oom.s" && { echo 'gate: OOM 注入 mutant 不変' >&2; exit 1; }
as -arch arm64 -o "$work/oom.o" "$work/oom.s"
ld -arch arm64 -o "$work/oom" -e _main -lSystem -lobjc -framework Metal -framework Foundation \
   "$work/oom.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$(xcrun --show-sdk-path)"
rm -f "$work/oom.flro"
if "$work/oom" "$work/multi.fldj" "$work/oom.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/oom.log" 2>&1; then orc=0; else orc=$?; fi
[ "$orc" -eq 19 ] || { echo "gate: OOM 注入 rc=$orc != 19 :: $(head -1 "$work/oom.log")" >&2; exit 1; }
[ ! -e "$work/oom.flro" ] || { echo 'gate: 失敗走が出力 file を残した' >&2; exit 1; }
printf 'KILLED  %-26s rc=%s %s (出力残零)\n' 'mmap-partial-alloc-fail' "$orc" "$(head -1 "$work/oom.log")"

# ---- 正常走の mapping 漏零(leaks 可用時のみ・不可用は明示 UNVERIFIED)----
if command -v leaks >/dev/null 2>&1; then
  leaks --atExit -- ./fieldrun "$work/multi.fldj" "$work/leak.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/leaks.log" 2>&1 || {
    cat "$work/leaks.log" >&2; exit 1; }
  grep -E '0 leaks' "$work/leaks.log" >/dev/null || { cat "$work/leaks.log" >&2; exit 1; }
  printf 'green   %-26s leaks --atExit=0 · custody 台帳 rc=0\n' 'no-mapping-leak'
else
  echo 'UNVERIFIED no-mapping-leak: leaks 不在' >&2
fi

echo 'gate: slotstore real-journal (multi-slot 200 step · 三経路 · OOM 注入) OK'
