#!/bin/sh
# fieldrun A1: fldj_parse のみ(手ARM64)。他言語 零。
set -eu
cd "$(dirname "$0")"
as -arch arm64 -o fldj_parse.o fldj_parse.s
ld -arch arm64 -o fldj_parse -e _main -lSystem fldj_parse.o -syslibroot "$(xcrun --show-sdk-path)"
echo "built: fldj_parse"
as -arch arm64 -o q20_conv.o q20_conv.s
ld -arch arm64 -o q20_conv -e _main -lSystem q20_conv.o -syslibroot "$(xcrun --show-sdk-path)"
echo "built: q20_conv"
as -arch arm64 -o coef.o coef.s
ld -arch arm64 -o coef -e _main -lSystem coef.o -syslibroot "$(xcrun --show-sdk-path)"
echo "built: coef"
