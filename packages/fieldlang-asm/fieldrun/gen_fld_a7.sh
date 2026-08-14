#!/bin/sh
# A7: .fld 源(gaialang 文)生成器 — shell のみ。
# 独立性: 本器は fieldrun/fldj_parse.s/fieldc の実装を一切読まぬ。
#   参照は ../CONTRACT.md の文法(界/寫/歩)のみ。出力は人可読 .fld 文字列。
# 環境: W H NSLOTS SEED STEPN PHASE(初期場の位相ずらし)
# 用: gen_fld_a7.sh <out.fld>
set -eu
out=${1:?usage: gen_fld_a7.sh <out.fld>}
W=${W:-32}
H=${H:-32}
NSLOTS=${NSLOTS:-2}
SEED=${SEED:-0}
STEPN=${STEPN:-200}
PHASE=${PHASE:-0}

n=$((W * H))

# 非一様場を f32 bit pattern の十進で作る(host 浮動小数 不使用)。
# 指数欄 e と仮数欄 f を整数算で組む ∴ |v| は 2^(e-127) 級、e<=137 に留め |v|<2048 を保証。
# e ∈ [118,137] を巡回、f を胞毎に散らし、符号も交番 → 一様でない実場。
k=0
vals=''
while [ $k -lt $n ]; do
  i=$((k + PHASE))
  e=$((118 + (i * 7 + (i / W) * 3) % 20))     # 118..137 → |v| ∈ [2^-9, 2^10]
  f=$(( (i * 2654435761 + (i / W) * 40503) % 8388608 ))
  s=$(( (i / 5 + i) % 2 ))
  bits=$(( s * 2147483648 + e * 8388608 + f ))
  vals="$vals $bits"
  k=$((k + 1))
done

{
  printf '# A7 実場 journal 源(shell 生成・独立)\n'
  printf '界 %s %s %s %s\n' "$W" "$H" "$NSLOTS" "$SEED"
  printf '寫 0 %s%s\n' "$n" "$vals"
  printf '歩 %s\n' "$STEPN"
} >"$out"
exit 0
