#!/bin/sh
set -eu
cd "$(dirname "$0")"
rm -f q30_metal_vectors.bin
python3 ./q30_metal_oracle.py --emit q30_metal_vectors.bin
# Binary is deliberately checked by its producer only here; the Metal runner must decode it independently.
test "$(wc -c < q30_metal_vectors.bin)" -gt 800000
rm -f q30_metal_vectors.bin
