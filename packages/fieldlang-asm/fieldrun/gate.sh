#!/bin/sh
# fieldrun A1 門: fldj_parse(手ARM64)の緑一点 + 赤歯全種。
set -eu
cd "$(dirname "$0")"

# 他言語混入禁(走査器は主題集合の外)。
if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi
case "$(grep -E '(^|[^[:alnum:]_])(cargo|rustc|swiftc)([^[:alnum:]_]|$)' build.sh || :)" in
  '') ;;
  *) echo 'gate: foreign toolchain in build path' >&2; exit 1 ;;
esac

./build.sh
work=$(mktemp -d ./fldj-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM

expect_rc() { # expect_rc <name> <want_rc> <file>
  name=$1; want=$2; f=$3
  if ./fldj_parse "$f" >"$work/$name.out" 2>"$work/$name.err"; then rc=0; else rc=$?; fi
  if [ "$rc" -ne "$want" ]; then
    echo "gate: $name rc=$rc want=$want" >&2; cat "$work/$name.err" >&2; exit 1
  fi
  if [ "$want" -eq 0 ]; then
    printf 'green   %-22s rc=0 %s\n' "$name" "$(cat "$work/$name.out")"
  else
    printf 'KILLED  %-22s rc=%s %s\n' "$name" "$rc" "$(cat "$work/$name.err")"
  fi
}

# --- 緑: canonical 2x2 ---
./gen_fldj.sh "$work/good.fldj"
expect_rc good 0 "$work/good.fldj"
want='fldj ok w=2 h=2 n=4 slots=2 ops=2 steps=1 writes=1'
got=$(cat "$work/good.out")
[ "$got" = "$want" ] || { echo "gate: stdout '$got' != '$want'" >&2; exit 1; }

# 緑其二: 8x8 · 多 op
W=8 H=8 STEPN=200 ./gen_fldj.sh "$work/good8.fldj"
expect_rc good8x8 0 "$work/good8.fldj"

# --- 赤歯 ---
MAGIC=1245989959 ./gen_fldj.sh "$work/m.fldj";      expect_rc magic 3 "$work/m.fldj"
VERSION=2 ./gen_fldj.sh "$work/v.fldj";             expect_rc version 4 "$work/v.fldj"
W=0 WLEN=4 ./gen_fldj.sh "$work/w0.fldj";           expect_rc w-zero 5 "$work/w0.fldj"
H=0 WLEN=4 ./gen_fldj.sh "$work/h0.fldj";           expect_rc h-zero 6 "$work/h0.fldj"
# 32bit 乗算なら w*h=0x100000000 は 0 へ wrap した。u64 故 上限超過で拒否。
W=65536 H=65536 WLEN=1 ./gen_fldj.sh "$work/mul.fldj"; expect_rc wh-32bit-wrap 8 "$work/mul.fldj"
W=200 H=200 WLEN=1 ./gen_fldj.sh "$work/big.fldj";  expect_rc wh-over-arena 8 "$work/big.fldj"
NSLOTS=1 ./gen_fldj.sh "$work/s1.fldj";             expect_rc nslots-lt2 9 "$work/s1.fldj"
NSLOTS=4294967295 ./gen_fldj.sh "$work/sN.fldj";    expect_rc nslots-over 10 "$work/sN.fldj"
DT=1036831948 ./gen_fldj.sh "$work/dt.fldj";        expect_rc dt-noncanonical 11 "$work/dt.fldj"
DAMPING=1065336951 ./gen_fldj.sh "$work/dp.fldj";   expect_rc damping-0.999-decimal 11 "$work/dp.fldj"
C=0 ./gen_fldj.sh "$work/c.fldj";                   expect_rc c-noncanonical 11 "$work/c.fldj"
DX=0 ./gen_fldj.sh "$work/dx.fldj";                 expect_rc dx-noncanonical 11 "$work/dx.fldj"
RANGE=0 ./gen_fldj.sh "$work/rg.fldj";              expect_rc range-noncanonical 11 "$work/rg.fldj"
TAIL='\0377' ./gen_fldj.sh "$work/tag.fldj";         expect_rc unknown-tag 12 "$work/tag.fldj"
TAIL='\0001\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000' ./gen_fldj.sh "$work/seed.fldj"
expect_rc seedatom-tag1 12 "$work/seed.fldj"
WSLOT=2 ./gen_fldj.sh "$work/sl.fldj";              expect_rc writeraw-slot2 13 "$work/sl.fldj"
TAIL='\0003' ./gen_fldj.sh "$work/tr.fldj";          expect_rc trailing-partial-op 14 "$work/tr.fldj"
WLEN=3 ./gen_fldj.sh "$work/wl.fldj";               expect_rc writeraw-len-ne-wh 15 "$work/wl.fldj"
WLEN=5 ./gen_fldj.sh "$work/wl5.fldj";              expect_rc writeraw-len-gt-wh 15 "$work/wl5.fldj"

# 切断(header 内 / op 内)
SZ=$(wc -c <"$work/good.fldj" | tr -d ' ')
dd if="$work/good.fldj" of="$work/short.fldj" bs=1 count=30 >/dev/null 2>&1
expect_rc header-truncated 2 "$work/short.fldj"
dd if="$work/good.fldj" of="$work/cut.fldj" bs=1 count=$((SZ - 3)) >/dev/null 2>&1
expect_rc op-truncated 14 "$work/cut.fldj"
: >"$work/empty.fldj"
expect_rc empty 2 "$work/empty.fldj"
expect_rc missing-file 16 "$work/no-such-file.fldj"

# 上限は引数で上書き可(硬碼禁)の実証: 200x200 は上限を上げれば緑。
W=200 H=200 ./gen_fldj.sh "$work/big2.fldj"
if ./fldj_parse "$work/big2.fldj" 40000 >"$work/big2.out" 2>&1; then
  printf 'green   %-22s rc=0 %s\n' 'max-cells-arg' "$(cat "$work/big2.out")"
else
  echo 'gate: max_cells argument override failed' >&2; cat "$work/big2.out" >&2; exit 1
fi

echo 'gate: fldj_parse A1 OK'

# A2: q20_conv
./q20_gate.sh

# A3: coef
./coef_gate.sh

# A4: fieldrun
./fieldrun_gate.sh

# A5: --neon
./neon_gate.sh

# A6: --metal(実機 GPU)
./metal_gate.sh

# A7: 実 fieldc journal 三経路(≥200 step)
./a7_gate.sh
