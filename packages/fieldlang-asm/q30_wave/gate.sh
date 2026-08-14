#!/bin/sh
set -eu
cd "$(dirname "$0")"
D=1973cf60a0cf4b20e2117d64af66ed99a79a3751646d8f3f51837e33648e2397
echo "$D  wave_vectors.bin" | shasum -a 256 -c -
# Forbidden source-name and token scan; the scanner is deliberately outside its subject set.
if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then exit 1; fi
if grep -nEi 'rust|cargo|rustc|[.]rs|clang|swift|python|[.]c' ./*.s CONTRACT.md; then exit 1; fi
./build.sh
./wave_runner all wave_vectors.bin
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
printf '%s\n' 'q30_wave gate: frozen133 scalar+neon aligned+unaligned alias ABI custody guard mutations=ok'
