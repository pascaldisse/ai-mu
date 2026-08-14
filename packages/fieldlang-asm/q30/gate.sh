#!/bin/sh
set -eu
cd "$(dirname "$0")"
case "$(grep -E '\.(c|cc|cpp|rs|py)$|cargo|rustc|python' build.sh CONTRACT.md *.s || :)" in
  '') ;;
  *) echo 'q30 gate: forbidden non-assembly product path' >&2; exit 1 ;;
esac
./build.sh
./q30_selftest
otool -tvV q30_selftest > q30-otool.txt
grep -q 'smull' q30-otool.txt
grep -q 'smull2' q30-otool.txt
grep -q 'sshr' q30-otool.txt
grep -q 'sqxtn' q30-otool.txt
grep -q 'sqxtn2' q30-otool.txt
rm -f q30-otool.txt
# Mutation tooth: a wrong Q scale must make independent scalar parity fail.
cp decay_neon.s decay_neon.s.gate-save
trap 'mv -f decay_neon.s.gate-save decay_neon.s; rm -f decay_neon.o q30_selftest q30_bench' EXIT HUP INT TERM
perl -pe 'if (!$done && s/#30/#29/) { $done=1 }' decay_neon.s.gate-save > decay_neon.s
./build.sh
if ./q30_selftest; then
  echo 'q30 gate: scale mutation survived' >&2
  exit 1
fi
mv -f decay_neon.s.gate-save decay_neon.s
trap - EXIT HUP INT TERM
./build.sh
printf 'q30 exact: scalar oracle, unaligned, count 0..9, extrema, 100003 deterministic cells=ok\n'
printf 'q30 disassembly: smull/smull2/sshr/sqxtn/sqxtn2=ok\n'
printf 'q30 mutation: Q30->Q29 rejected=ok\n'
printf 'benchmark scalar (32 x 1048576 cells; wall seconds):\n'
/usr/bin/time -p ./q30_bench scalar
printf 'benchmark neon (32 x 1048576 cells; wall seconds):\n'
/usr/bin/time -p ./q30_bench neon
printf 'q30 gate: assembly closure OK (no speed claim)\n'
