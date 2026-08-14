#!/bin/sh
set -eu
cd "$(dirname "$0")"
D=1973cf60a0cf4b20e2117d64af66ed99a79a3751646d8f3f51837e33648e2397
echo "$D  wave_vectors.bin" | shasum -a 256 -c -
if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then exit 1; fi
if grep -nEi 'rust|cargo|[.]rs|clang|swift|python|[.]c' ./*.s CONTRACT.md; then exit 1; fi
./build.sh
./wave_runner all wave_vectors.bin
otool -tvV wave_runner > wave-otool.txt
for m in smull.2d saddl.2d saddl2.2d sshll.2d sshll2.2d sqxtn.2s sqxtn2.4s sshr.2d shl.2d xtn.2s cmgt.2d addp.2d dup.2d ld1.4s st1.4s; do grep -qF "	$m	" wave-otool.txt || exit 1; done
if grep -nE '^[[:space:]]*(b|bl|br|blr)[[:space:]]+_' wave_neon.s | grep -q .; then exit 1; fi
for r in x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28; do if grep -vE '^[[:space:]]*//' wave_scalar.s wave_neon.s | grep -nE "\b$r\b" | grep -q .; then exit 1; fi; done
printf '%s\n' 'q30_wave gate: scalar+neon frozen133 aligned=ok; unaligned/alias/ABI mutation=UNVERIFIED'
