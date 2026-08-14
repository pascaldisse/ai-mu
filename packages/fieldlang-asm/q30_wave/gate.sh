#!/bin/sh
set -eu
cd "$(dirname "$0")"
D=b826a11494d9e988ad90cc2db93aceceb77229ae741e028a2a785339751b493e
echo "$D  wave_vectors.bin" | shasum -a 256 -c -
# Forbidden source-name and token scan; the scanner is deliberately outside its subject set.
if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then exit 1; fi
if grep -nEi 'rust|cargo|rustc|[.]rs|clang|swift|python|[.]c' ./*.s CONTRACT.md; then exit 1; fi
./build.sh
./wave_runner all wave_vectors.bin
# corruption tooth: magic byte changed; assembly parser must reject even if invoked directly.
cp wave_vectors.bin wave_vectors.bin.corrupt
printf "X" | dd of=wave_vectors.bin.corrupt bs=1 seek=0 conv=notrunc >/dev/null 2>&1
if ./wave_runner all wave_vectors.bin.corrupt >/dev/null 2>&1; then rm -f wave_vectors.bin.corrupt; exit 1; fi
rm -f wave_vectors.bin.corrupt
printf "%s\n" "mutation fixture-magic rejected=ok"
# 残長 teeth (Kali blocker 1/2): trailing garbage・truncation・dims overflow は必ず赤。
cp wave_vectors.bin wave_vectors.bin.trail
printf 'ZZZZ' >> wave_vectors.bin.trail
if ./wave_runner all wave_vectors.bin.trail >/dev/null 2>&1; then rm -f wave_vectors.bin.trail; exit 1; fi
rm -f wave_vectors.bin.trail
printf "%s\n" "mutation fixture-trailing rejected=ok"
SZ=$(wc -c < wave_vectors.bin)
dd if=wave_vectors.bin of=wave_vectors.bin.trunc bs=1 count=$((SZ-17)) >/dev/null 2>&1
if ./wave_runner all wave_vectors.bin.trunc >/dev/null 2>&1; then rm -f wave_vectors.bin.trunc; exit 1; fi
rm -f wave_vectors.bin.trunc
printf "%s\n" "mutation fixture-truncated rejected=ok"
# w=h=65536: 32bit mul は 0 へ wrap した。u64 checked 故いま拒否。
{ printf 'Q30WAVE2\000'; printf '\002\000ab'; printf '\000\000\001\000\000\000\001\000'; } > wave_vectors.bin.dims
if ./wave_runner all wave_vectors.bin.dims >/dev/null 2>&1; then rm -f wave_vectors.bin.dims; exit 1; fi
rm -f wave_vectors.bin.dims
printf "%s\n" "mutation fixture-dims-overflow rejected=ok"
# w=0 / h=0 拒否。
{ printf 'Q30WAVE2\000'; printf '\002\000ab'; printf '\000\000\000\000\001\000\000\000'; } > wave_vectors.bin.zero
if ./wave_runner all wave_vectors.bin.zero >/dev/null 2>&1; then rm -f wave_vectors.bin.zero; exit 1; fi
rm -f wave_vectors.bin.zero
printf "%s\n" "mutation fixture-zero-dim rejected=ok"
# record 途中切断(record 内 remaining-length)。
{ printf 'Q30WAVE2\000'; printf '\002\000ab'; printf '\002\000\000\000\002\000\000\000'; } > wave_vectors.bin.short
if ./wave_runner all wave_vectors.bin.short >/dev/null 2>&1; then rm -f wave_vectors.bin.short; exit 1; fi
rm -f wave_vectors.bin.short
printf "%s\n" "mutation fixture-record-truncated rejected=ok"
# arena頂 teeth(round11 自攻): 宣言上限 n=0x4000 は真に arena 内でなければならぬ。
# count検査のみ緩めた probe を組立て、n=16384 は緑・n=16385 は赤 を要求。
# 修正前(65536 arena)は n>=16360 で out guard が越境破壊され n=16384 が赤だった。
sed 's/    cmp x10, #138/    cmp x10, x10/' wave_runner.s > bounds_probe.s
as -arch arm64 -o bounds_probe.o bounds_probe.s
ld -arch arm64 -o bounds_probe -e _main -lSystem bounds_probe.o wave_scalar.o wave_neon.o -syslibroot "$(xcrun --show-sdk-path)"
mkcase() { n=$1; b=$((n*4))
  { printf 'Q30WAVE2\000'; printf '\002\000ab'; printf '\001\000\000\000'
    printf "$(printf '\\%03o\\%03o\\%03o\\%03o' $((n&255)) $(((n>>8)&255)) $(((n>>16)&255)) $(((n>>24)&255)))"
    head -c 24 /dev/zero; head -c $((3*b)) /dev/zero; head -c 8 /dev/zero; printf '\000\000'; } > wave_bounds_$n.bin; }
mkcase 16384; mkcase 16385
./bounds_probe all wave_bounds_16384.bin >/dev/null 2>&1 || { rm -f wave_bounds_*.bin bounds_probe*; exit 1; }
if ./bounds_probe all wave_bounds_16385.bin >/dev/null 2>&1; then rm -f wave_bounds_*.bin bounds_probe*; exit 1; fi
rm -f wave_bounds_*.bin bounds_probe bounds_probe.o bounds_probe.s
printf "%s\n" "arena-top n=16384 in-bounds / n=16385 rejected=ok"
./wave_abi_probe
otool -tvV wave_runner > wave-otool.txt
for m in smull.2d saddl.2d saddl2.2d sshll.2d sshll2.2d sqxtn.2s sqxtn2.4s sshr.2d shl.2d xtn.2s cmgt.2d addp.2d dup.2d ld1.4s st1.4s; do grep -qF "$m" wave-otool.txt || exit 1; done
if grep -nE '^[[:space:]]*(b|bl|br|blr)[[:space:]]+_' wave_neon.s | grep -q .; then exit 1; fi
for r in x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28; do if grep -vE '^[[:space:]]*//' wave_scalar.s wave_neon.s | grep -nE "\b$r\b" | grep -q .; then exit 1; fi; done
mut() { n=$1; f=$2; e=$3; cp "$f" "$f.save"; perl -pe "$e" "$f.save" > "$f"; cmp -s "$f" "$f.save" && exit 1; if ./build.sh >/dev/null 2>&1 && ./wave_runner all wave_vectors.bin >/dev/null 2>&1 && ./wave_abi_probe >/dev/null 2>&1 && ! grep -vE '^[[:space:]]*//' wave_scalar.s wave_neon.s | grep -q '\bx18\b'; then mv "$f.save" "$f"; ./build.sh >/dev/null; exit 1; fi; mv "$f.save" "$f"; ./build.sh >/dev/null; printf 'mutation %s rejected=ok\n' "$n"; }
trap 'for f in wave_scalar.s wave_neon.s; do test -f "$f.save" && mv "$f.save" "$f"; done' EXIT HUP INT TERM
mut rounding-half wave_neon.s 'if(!$d&&s/lsl x4, x4, #29/lsl x4, x4, #28/){$d=1}'
mut q-scale wave_neon.s 'if(!$d&&s/sshr v1\.2d, v1\.2d, #30/sshr v1.2d, v1.2d, #29/){$d=1}'
mut lane-order wave_neon.s 'if(!$d&&s/sqxtn v0\.2s, v28\.2d/sqxtn v0.2s, v31.2d/){$d=1}'
mut sat-negative wave_neon.s 'if(!$d&&s/cmgt v2\.2d, v23\.2d, v31\.2d/cmgt v2.2d, v31.2d, v23.2d/){$d=1}'
mut vec-boundary wave_neon.s 'if(!$d&&s/add x6, x4, #5/add x6, x4, #4/){$d=1}'
mut scalar-fallback wave_scalar.s 'if(!$d&&s/_fl_q30_wave_scalar:\n/_fl_q30_wave_scalar:\n    eor x19, x19, #1\n/){$d=1}'
mut x18-tooth wave_scalar.s 'if(!$d&&s/_fl_q30_wave_scalar:\n/_fl_q30_wave_scalar:\n    movz x18, #1\n/){$d=1}'
# 非整列teeth: cur を 16B 境界へ強制丸め。整列入力=恒等故 frozen133 は無傷、
# +1B 経路のみ誤pointerとなる。故に本変異のredは非整列経路が実際に採点される證。
mut unaligned-tooth wave_neon.s 'if(!$d&&s/_fl_q30_wave_neon:\n/_fl_q30_wave_neon:\n    and x1, x1, #-16\n/){$d=1}'
# alias teeth: cur==prev の時のみ prev を狂わす。非alias呼出=恒等故
# 本変異のredは alias 同一基址呼出が実在し採点される證。
mut alias-tooth wave_neon.s 'if(!$d&&s/_fl_q30_wave_neon:\n/_fl_q30_wave_neon:\n    cmp x1, x2\n    b.ne Lat9\n    add x2, x2, #4\nLat9:\n/){$d=1}'
# guard teeth: out 直後(one-past-end)へ一書き。出力本体は正しい儘故、
# 本変異のredは out 周囲 guard のみが検出しうる。
mut out-guard-tooth wave_neon.s 'if(!$d&&s/_fl_q30_wave_neon:\n/_fl_q30_wave_neon:\n    ldrsw x9, [x0, #0]\n    ldrsw x10, [x0, #4]\n    mul x9, x9, x10\n    lsl x9, x9, #2\n    str wzr, [x3, x9]\n/){$d=1}'
# custody teeth: 計算後に入力 cur を破壊。出力は正しい儘故、
# 本変異のredは入力不変性 custody 比較のみが検出しうる。
# 注: 注入点=Ldone(全計算完了後)。故に out 内容も saturation も正しいまま、
# cur のみが破れる。出力比較では検出不能、custody 比較のみが捕らえる。
mut input-custody-tooth wave_neon.s 'if(!$d&&s/^Ldone:\n/Ldone:\n    ldr x9, [sp, #0]\n    str wzr, [x9]\n/){$d=1}'
mut input-custody-scalar-tooth wave_scalar.s 'if(!$d&&s/^Ldone:\n/Ldone:\n    ldr x9, [sp, #0]\n    str wzr, [x9]\n/){$d=1}'
printf '%s\n' 'q30_wave gate: frozen138-Q30WAVE2 scalar+neon aligned+unaligned alias ABI custody guard mutations=ok'
