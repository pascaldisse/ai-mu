#!/bin/sh
# fieldrun A1: fldj_parse のみ(手ARM64)。他言語 零。
set -eu
cd "$(dirname "$0")"
as -arch arm64 -o fldj_parse.o fldj_parse.s
ld -arch arm64 -o fldj_parse -e _main -lSystem fldj_parse.o -syslibroot "$(xcrun --show-sdk-path)"
echo "built: fldj_parse"
