#!/bin/sh
set -eu
cd "$(dirname "$0")"
SDK=$(xcrun --show-sdk-path)
for source in decay_scalar decay_neon q30_selftest q30_bench; do
  as -arch arm64 -o "$source.o" "$source.s"
done
ld -arch arm64 -o q30_selftest q30_selftest.o decay_scalar.o decay_neon.o -lSystem -syslibroot "$SDK"
ld -arch arm64 -o q30_bench q30_bench.o decay_scalar.o decay_neon.o -lSystem -syslibroot "$SDK"
echo 'built: q30_selftest q30_bench'
