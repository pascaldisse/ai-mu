#!/bin/sh
set -eu
cd "$(dirname "$0")"
SDK=$(xcrun --show-sdk-path)
xcrun metal -c wave_q30.metal -o wave_q30.air
xcrun metallib wave_q30.air -o q30_wave.metallib
as -arch arm64 -o metal_bridge.o metal_bridge.s
as -arch arm64 -o wave_metal_runner.o wave_metal_runner.s
as -arch arm64 -o wave_scalar.o ../q30_wave/wave_scalar.s
as -arch arm64 -o wave_neon.o ../q30_wave/wave_neon.s
ld -arch arm64 -o wave_metal_runner wave_metal_runner.o metal_bridge.o wave_scalar.o wave_neon.o \
   -lSystem -lobjc -framework Metal -framework Foundation -syslibroot "$SDK"
echo 'built: wave_metal_runner'
