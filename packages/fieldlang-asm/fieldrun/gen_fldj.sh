#!/bin/sh
# FLDJ v1 断片生成器(shell のみ・fldj_parse を読まぬ = 独立)。
# 環境変数で各欄を上書き可(歯の材料)。既定 = canonical 2x2。
#   MAGIC VERSION W H C DT DAMPING DX SEED RANGE NSLOTS
#   WSLOT(WriteRaw slot) WLEN(payload cell 数、既定 = W*H) STEPN TAIL
# 用: gen_fldj.sh <out>
set -eu
out=${1:?usage: gen_fldj.sh <out>}

MAGIC=${MAGIC:-1245989958}      # 0x4A444C46 "FLDJ"
VERSION=${VERSION:-1}
W=${W:-2}
H=${H:-2}
C=${C:-1065353216}              # 0x3F800000
DT=${DT:-1036831949}            # 0x3DCCCCCD
DAMPING=${DAMPING:-1065336439}  # 0x3F7FBE77
DX=${DX:-1065353216}
SEED=${SEED:-0}
RANGE=${RANGE:-1065353216}
NSLOTS=${NSLOTS:-2}
WSLOT=${WSLOT:-0}
WLEN=${WLEN:-$((W * H))}
STEPN=${STEPN:-1}
TAIL=${TAIL:-}

u8() { printf '%b' "\\0$(printf '%03o' "$(($1 & 255))")"; }
u32() { v=$1; i=0; while [ $i -lt 4 ]; do u8 $(((v >> (8 * i)) & 255)); i=$((i + 1)); done; }
u64() { v=$1; i=0; while [ $i -lt 8 ]; do u8 $(((v >> (8 * i)) & 255)); i=$((i + 1)); done; }

{
  u32 "$MAGIC"; u32 "$VERSION"; u32 "$W"; u32 "$H"
  u32 "$C"; u32 "$DT"; u32 "$DAMPING"; u32 "$DX"
  u64 "$SEED"; u32 "$RANGE"; u32 "$NSLOTS"
  # op 1: WriteRaw(tag 6)
  u8 6; u32 "$WSLOT"; u32 "$WLEN"
  k=0
  while [ "$k" -lt "$WLEN" ]; do u32 1065353216; k=$((k + 1)); done
  # op 2: Step(tag 3)
  u8 3; u32 "$STEPN"
  [ -n "$TAIL" ] && printf '%b' "$TAIL"
} >"$out"
exit 0
