#!/bin/sh
# slotstore 審門: 動的 slot 表・wire 拒絶・三上限・三buffer意味論・変異歯。
# 用: slotstore_gate.sh [fieldrun path]。shell/ARM64既存buildのみ。
# 建者錨: fieldrun.s に [MUT:slot-cap] [MUT:slot-index] [MUT:slot-zero]
#           [MUT:slot-cleanup] [MUT:slot-rotate] を各一箇所残せ。
set -eu
cd "$(dirname "$0")"
run=${1:-./fieldrun}
max_cells=${MAX_CELLS:-64}
max_bytes=${MAX_BYTES:-1048576}
max_slots=${MAX_SLOTS:-8}
git diff --check
./build.sh >/dev/null
git diff --check
work=$(mktemp -d ./slotstore-gate.XXXXXX)
trap 'rc=$?; rm -rf "$work"; git diff --check; exit "$rc"' EXIT HUP INT TERM

u8() { printf '%b' "\\0$(printf '%03o' "$(( $1 & 255 ))")"; }
u32() { v=$1; i=0; while [ "$i" -lt 4 ]; do u8 "$(( (v >> (8*i)) & 255 ))"; i=$((i+1)); done; }
header() { # header <out> <w> <h> <nslots>
  out=$1; w=$2; h=$3; ns=$4
  MAGIC=1245989958 W="$w" H="$h" NSLOTS="$ns" WLEN=0 STEPN=0 ./gen_fldj.sh "$work/h.fldj"
  dd if="$work/h.fldj" of="$out" bs=1 count=48 >/dev/null 2>&1
}
writeop() { # writeop <slot> <u32 payload...>
  slot=$1; shift; n=$#; u8 6; u32 "$slot"; u32 "$n"; for x in "$@"; do u32 "$x"; done
}
stepop() { u8 3; u32 "$1"; }
invoke() { # invoke <journal> <out>
  "$run" "$1" "$2" "$max_cells" "$max_bytes" "$max_slots"
}
reject() { # reject <name> <journal>
  name=$1; j=$2; rm -f "$work/$name.flro"
  if invoke "$j" "$work/$name.flro" >"$work/$name.log" 2>&1; then rc=0; else rc=$?; fi
  [ "$rc" -ne 0 ] || { echo "SURVIVED $name: accepted malformed input" >&2; exit 1; }
  [ -s "$work/$name.log" ] || { echo "SURVIVED $name: error text empty" >&2; exit 1; }
  [ ! -e "$work/$name.flro" ] || { echo "SURVIVED $name: rejected input left output" >&2; exit 1; }
  printf 'KILLED  %-28s rc=%s %s\n' "$name" "$rc" "$(head -1 "$work/$name.log")"
}

# canonical slot0: old journal 回帰基準。slot2/3 は合法lazy領域、canonical FLROを変えぬ。
header "$work/base.fldj" 2 2 4
{ writeop 0 1065353216 0 0 0; writeop 1 0 0 0 0; stepop 2; } >>"$work/base.fldj"
invoke "$work/base.fldj" "$work/base.flro"
header "$work/foreign.fldj" 2 2 4
{ writeop 0 1065353216 0 0 0; writeop 1 0 0 0 0; writeop 2 1073741824 1073741824 1073741824 1073741824; writeop 3 1056964608 1056964608 1056964608 1056964608; stepop 2; } >>"$work/foreign.fldj"
invoke "$work/foreign.fldj" "$work/foreign.flro"
cmp "$work/base.flro" "$work/foreign.flro"
printf 'green   %-28s slot2/3 accepted; canonical=slot0 byte-identical\n' 'lazy-foreign-slots'

# slot2をslot0/1へ誤配線すれば出力差、zero欠落なら空 journal との差で捕捉。
header "$work/zero.fldj" 2 2 4
stepop 1 >>"$work/zero.fldj"
invoke "$work/zero.fldj" "$work/zero.flro"
zeros=$(od -An -tu4 -j32 "$work/zero.flro" | tr -d ' \n')
[ "$zeros" = 0000 ] || { echo "gate: zero initialization failed: $zeros" >&2; exit 1; }
printf 'green   %-28s all slot buffers start zero\n' 'zero-initialization'

# slot境界: parser と runtime の双方が slot<n_slots を同一に強制。
header "$work/bound-ok.fldj" 2 2 3; { writeop 2 0 0 0 0; stepop 1; } >>"$work/bound-ok.fldj"
./fldj_parse "$work/bound-ok.fldj" "$max_cells" 3 >"$work/bound-ok.parse" 2>&1
invoke "$work/bound-ok.fldj" "$work/bound-ok.flro"
printf 'green   %-28s slot==n_slots-1 accepted (parser+runtime)\n' 'slot-boundary-last'
header "$work/oob.fldj" 2 2 3; { writeop 3 0 0 0 0; stepop 1; } >>"$work/oob.fldj"
if ./fldj_parse "$work/oob.fldj" "$max_cells" 3 >"$work/oob.parse" 2>&1; then prc=0; else prc=$?; fi
[ "$prc" -ne 0 ] || { echo 'SURVIVED parser-slot-equals-nslots' >&2; exit 1; }
reject slot-equals-nslots "$work/oob.fldj"
# 重複slot・payload途中切断・末尾余剰。
header "$work/dup.fldj" 2 2 4; { writeop 2 0 0 0 0; writeop 2 0 0 0 0; stepop 1; } >>"$work/dup.fldj"; reject duplicate-slot "$work/dup.fldj"
cp "$work/foreign.fldj" "$work/cut.fldj"; n=$(wc -c <"$work/cut.fldj" | tr -d ' '); dd if="$work/cut.fldj" of="$work/cut2.fldj" bs=1 count=$((n-2)) >/dev/null 2>&1; reject journal-truncated-payload "$work/cut2.fldj"
cp "$work/base.fldj" "$work/tail.fldj"; u8 127 >>"$work/tail.fldj"; reject journal-trailing-byte "$work/tail.fldj"

# 引数三上限: 皆明白error。値は環境で可変、硬碼せず。
header "$work/slots-cap.fldj" 2 2 "$((max_slots + 1))"; stepop 1 >>"$work/slots-cap.fldj"; reject max-slots "$work/slots-cap.fldj"
old_cells=$max_cells; max_cells=3; reject max-cells "$work/base.fldj"; max_cells=$old_cells
old_bytes=$max_bytes; max_bytes=1; reject max-bytes "$work/base.fldj"; max_bytes=$old_bytes

# u64 product/address 境界: 64bit積とisize/mmap以前に拒絶。headerのみ故巨大割当零。
header "$work/u64-product.fldj" 4294967295 4294967295 2; stepop 1 >>"$work/u64-product.fldj"; reject u64-product-overflow "$work/u64-product.fldj"
header "$work/address-range.fldj" 2147483648 1 2; stepop 1 >>"$work/address-range.fldj"; reject isize-address-range "$work/address-range.fldj"

# source変異: markerを削る/置換→同一functional probeが必ず赤。cleanupはleak検出器無き環境で
# 明示的SKIP禁: ownership marker消失を静的赤にし、実行変異はmacOS leaks利用可能時のみ追加せよ。
mutate() { # mutate <name> <sed expression> <journal>
  name=$1; expr=$2; j=$3
  grep -F "[MUT:$name]" fieldrun.s >/dev/null || { echo "gate: missing mutation marker $name" >&2; exit 1; }
  sed "$expr" fieldrun.s >"$work/$name.s"
  cmp -s fieldrun.s "$work/$name.s" && { echo "gate: mutant unchanged $name" >&2; exit 1; }
  as -arch arm64 -o "$work/$name.o" "$work/$name.s"
  ld -arch arm64 -o "$work/$name" -e _main -lSystem -lobjc -framework Metal -framework Foundation "$work/$name.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$(xcrun --show-sdk-path)"
  if "$work/$name" "$j" "$work/$name.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/$name.log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" -eq 0 ] && cmp -s "$work/$name.flro" "$work/base.flro"; then echo "SURVIVED $name" >&2; exit 1; fi
  printf 'KILLED  %-28s rc=%s functional mismatch/reject\n' "$name" "$rc"
}
# 建者はmarker所在の一命令を、下の唯一一致文字列へ保つ。
mutate slot-cap 's|cmp x[0-9][0-9]*, x[0-9][0-9]*.*\[MUT:slot-cap\]|cmp xzr, xzr // [MUT:slot-cap]|' "$work/slots-cap.fldj"
mutate slot-index 's|.*\[MUT:slot-index\].*|mov x0, xzr // [MUT:slot-index]|' "$work/foreign.fldj"
mutate slot-zero 's|.*\[MUT:slot-zero\].*|nop // [MUT:slot-zero]|' "$work/zero.fldj"
mutate slot-rotate 's|.*\[MUT:slot-rotate\].*|nop // [MUT:slot-rotate]|' "$work/base.fldj"
# mmap部分失敗注入: test hook は第N確保をMAP_FAILED化し、既確保域をcleanupせねばならぬ。
# hook/`leaks` 不在はSKIPでなくFAIL。建者は FIELD_RUN_FAIL_MMAP_AT=N (N>=2)を実装せよ。
if ! command -v leaks >/dev/null 2>&1; then echo 'FAIL mmap-cleanup: leaks unavailable' >&2; exit 1; fi
rm -f "$work/partial.flro"
if FIELD_RUN_FAIL_MMAP_AT=2 "$run" "$work/base.fldj" "$work/partial.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/partial.log" 2>&1; then prc=0; else prc=$?; fi
[ "$prc" -ne 0 ] || { echo 'SURVIVED mmap-partial-failure injection ignored' >&2; exit 1; }
[ ! -e "$work/partial.flro" ] || { echo 'SURVIVED mmap-partial-failure output exists' >&2; exit 1; }
if leaks --atExit -- env FIELD_RUN_FAIL_MMAP_AT=2 "$run" "$work/base.fldj" "$work/partial2.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/partial-leaks.log" 2>&1; then lprc=0; else lprc=$?; fi
[ "$lprc" -ne 0 ] || { echo 'SURVIVED mmap-partial-failure leaks run succeeded' >&2; exit 1; }
grep -E '0 leaks for 0 total leaked bytes|0 leaks' "$work/partial-leaks.log" >/dev/null || { cat "$work/partial-leaks.log" >&2; exit 1; }
printf 'KILLED  %-28s injected allocation #2 rejects; leak=0\n' 'mmap-partial-failure'

# cleanup: marker付きmunmap callを削るmutant。leaks は実行終了時のmmap漏れを別路で観る。
grep -F '[MUT:slot-cleanup]' fieldrun.s >/dev/null || { echo 'gate: missing mutation marker slot-cleanup' >&2; exit 1; }
if command -v leaks >/dev/null 2>&1; then
  leaks --atExit -- "$run" "$work/base.fldj" "$work/leak.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/leaks.log" 2>&1 || { cat "$work/leaks.log" >&2; exit 1; }
  grep -E '0 leaks for 0 total leaked bytes|0 leaks' "$work/leaks.log" >/dev/null || { cat "$work/leaks.log" >&2; exit 1; }
  sed '/\[MUT:slot-cleanup\]/d' fieldrun.s >"$work/slot-cleanup.s"
  cmp -s fieldrun.s "$work/slot-cleanup.s" && { echo 'gate: cleanup mutant unchanged' >&2; exit 1; }
  as -arch arm64 -o "$work/slot-cleanup.o" "$work/slot-cleanup.s"
  ld -arch arm64 -o "$work/slot-cleanup" -e _main -lSystem -lobjc -framework Metal -framework Foundation "$work/slot-cleanup.o" q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o metal_bridge.o -syslibroot "$(xcrun --show-sdk-path)"
  if leaks --atExit -- "$work/slot-cleanup" "$work/base.fldj" "$work/slot-cleanup.flro" "$max_cells" "$max_bytes" "$max_slots" >"$work/slot-cleanup.log" 2>&1; then lrc=0; else lrc=$?; fi
  if [ "$lrc" -eq 0 ] && grep -E '0 leaks for 0 total leaked bytes|0 leaks' "$work/slot-cleanup.log" >/dev/null; then
    echo 'SURVIVED slot-cleanup: deleted munmap leaked nothing' >&2; exit 1
  fi
  printf 'KILLED  %-28s leaks rc=%s\n' 'slot-cleanup' "$lrc"
  printf 'green   %-28s leaks --atExit=0\n' 'mmap-cleanup'
else
  echo 'FAIL mmap-cleanup: leaks unavailable' >&2; exit 1
fi

git diff --check
echo 'gate: slotstore quality OK (SURVIVED=0 SKIP=0)'
