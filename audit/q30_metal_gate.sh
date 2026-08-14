#!/bin/sh
set -eu
cd "$(dirname "$0")"
trap 'rm -f q30_metal_vectors.bin q30_metal_runner' EXIT HUP INT TERM
python3 ./q30_metal_oracle.py --emit q30_metal_vectors.bin
xcrun swiftc -O q30_metal_runner.swift -o q30_metal_runner
./q30_metal_runner --fixture q30_metal_vectors.bin
