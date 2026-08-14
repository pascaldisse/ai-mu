#!/bin/sh
set -eu
cd "$(dirname "$0")"
SDK=$(xcrun --show-sdk-path)
xcrun metal -c decay_q30.metal -o decay_q30.air
xcrun metallib decay_q30.air -o q30.metallib
as -arch arm64 -o metal_bridge.o metal_bridge.s
as -arch arm64 -o q30_metal_selftest.o q30_metal_selftest.s
as -arch arm64 -o ../q30/decay_scalar.o ../q30/decay_scalar.s
as -arch arm64 -o ../q30/decay_neon.o ../q30/decay_neon.s
ld -arch arm64 -o q30_metal_selftest q30_metal_selftest.o metal_bridge.o ../q30/decay_scalar.o ../q30/decay_neon.o -lSystem -lobjc -framework Metal -framework Foundation -syslibroot "$SDK"
echo 'built: q30_metal_selftest'
