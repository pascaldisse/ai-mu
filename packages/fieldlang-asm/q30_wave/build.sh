#!/bin/sh
set -eu
cd "$(dirname "$0")"
as -arch arm64 -o wave_scalar.o wave_scalar.s
as -arch arm64 -o wave_neon.o wave_neon.s
as -arch arm64 -o wave_runner.o wave_runner.s
ld -arch arm64 -o wave_runner -e _main -lSystem wave_runner.o wave_scalar.o wave_neon.o -syslibroot "$(xcrun --show-sdk-path)"
