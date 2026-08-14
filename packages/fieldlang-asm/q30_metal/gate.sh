#!/bin/sh
set -eu
cd "$(dirname "$0")"
trap 'rm -f *.o decay_q30.air q30_metal_selftest' EXIT HUP INT TERM
case "$(grep -E 'printf|rust|wgpu|cargo' metal_bridge.s q30_metal_selftest.s build.sh || :)" in '') ;; *) echo 'q30 metal gate: forbidden product dependency' >&2; exit 1;; esac
./build.sh
./q30_metal_selftest
otool -L q30_metal_selftest
if otool -L q30_metal_selftest | grep -Ei 'rust|wgpu'; then exit 1; fi
nm -u q30_metal_selftest
if nm -u q30_metal_selftest | grep -Ei 'rust|wgpu'; then exit 1; fi
cp decay_q30.metal decay_q30.metal.save
trap 'mv -f decay_q30.metal.save decay_q30.metal; rm -f *.o decay_q30.air q30_metal_selftest' EXIT HUP INT TERM
perl -0pi -e 's/lo >> 30/lo >> 29/' decay_q30.metal
./build.sh
if ./q30_metal_selftest; then echo 'q30 metal gate: Q30->Q29 mutation survived' >&2; exit 1; fi
mv -f decay_q30.metal.save decay_q30.metal
trap 'rm -f *.o decay_q30.air q30_metal_selftest' EXIT HUP INT TERM
echo 'q30 metal: GPU pipeline, 64 threads, 77 scalar+NEON byte parity, mutation red=ok'
