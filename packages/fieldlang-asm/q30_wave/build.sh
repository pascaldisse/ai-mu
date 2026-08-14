#!/bin/sh
set -eu
cd "$(dirname "$0")"
as -arch arm64 -o wave_scalar.o wave_scalar.s
as -arch arm64 -o wave_neon.o wave_neon.s
as -arch arm64 -o wave_runner.o wave_runner.s
as -arch arm64 -o abi_probe.o abi_probe.s
ld -arch arm64 -o wave_runner -e _main -lSystem wave_runner.o wave_scalar.o wave_neon.o -syslibroot "$(xcrun --show-sdk-path)"
ld -arch arm64 -o wave_abi_probe -e _main -lSystem abi_probe.o wave_scalar.o wave_neon.o -syslibroot "$(xcrun --show-sdk-path)"
