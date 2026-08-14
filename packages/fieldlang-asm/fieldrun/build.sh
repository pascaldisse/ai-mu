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

# A4: fieldrun(scalar)。A2/A3 は _main を改名した lib 版で連結(記号衝突回避)。
sed 's/_main/_q20_libmain/g' q20_conv.s >q20_conv_lib.s
sed 's/_main/_coef_libmain/g' coef.s >coef_lib.s
as -arch arm64 -o q20_conv_lib.o q20_conv_lib.s
as -arch arm64 -o coef_lib.o coef_lib.s
as -arch arm64 -o wave_scalar.o ../q30_wave/wave_scalar.s
as -arch arm64 -o wave_neon.o ../q30_wave/wave_neon.s
as -arch arm64 -o fieldrun.o fieldrun.s
ld -arch arm64 -o fieldrun -e _main -lSystem fieldrun.o q20_conv_lib.o coef_lib.o wave_scalar.o wave_neon.o -syslibroot "$(xcrun --show-sdk-path)"
echo "built: fieldrun"
