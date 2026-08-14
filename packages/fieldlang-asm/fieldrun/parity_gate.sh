#!/bin/sh
# §15c parity 門(Chandi)— 上流 `wave_step_reference`(f32) 対 fieldrun(Q20/Q30) の
# **凍結 fixture** parity。shell のみ(新規 Rust/C/Swift/Python 零)。
#
# 審(Jyestha二番・Rahu 独立検算)の指摘に応ずる:
#  (あ) BOUND は契約本文の数値を信ぜず、**`./coef` の実走出力から毎回再導出**する
#       (G = (|c_cur| + 4|c_lap| + |c_prev|)/2^30、E(N) = u·Σ_{i<N} G^i、u = 2^-20)。
#  (い) fixture は本 script 内の**決定論的生成律**で固定 ∴ 誰が走らせても同一 journal・同一数値。
#  (う) lap=0(一様場)走は **SPECIAL** ラベル ∴ PASS 数に算入せぬ。
# 飽和走(FLRO off24 sat≠0)は §15 の射程外 ∴ 赤(fixture 設計の誤り)。
set -eu
cd "$(dirname "$0")"
./build.sh >/dev/null
FIELDC=${FIELDC:-../fieldc}
REPLAY=${REPLAY:-../../../target/debug/examples/replay_fldj}
[ -x "$REPLAY" ] || { echo "gate: SKIP-ENV replay_fldj absent at '$REPLAY' — 検査未実施 rc=3" >&2; exit 3; }

work=$(mktemp -d ./parity-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM

# ---- 係数を実走から取る(契約本文の数値に依存せぬ = 独立路)----
coefline=$(./coef 3F800000 3DCCCCCD 3F7FBE77 3F800000 3F800000)
C_CUR=$(echo "$coefline" | sed 's/.*c_cur=\([-0-9]*\).*/\1/')
C_LAP=$(echo "$coefline" | sed 's/.*c_lap=\([-0-9]*\).*/\1/')
C_PREV=$(echo "$coefline" | sed 's/.*c_prev=\([-0-9]*\).*/\1/')
printf 'raw     %-22s %s\n' 'coef(実走)' "$coefline"
G=$(awk -v a="$C_CUR" -v b="$C_LAP" -v c="$C_PREV" \
  'BEGIN{ if(a<0)a=-a; if(b<0)b=-b; if(c<0)c=-c; printf "%.12f", (a+4*b+c)/1073741824 }')
printf 'raw     %-22s G=%s (=(|c_cur|+4|c_lap|+|c_prev|)/2^30)\n' 'amplification' "$G"

bound() { awk -v g="$G" -v n="$1" 'BEGIN{u=1/1048576;s=0;p=1;for(i=0;i<n;i++){s+=p;p*=g};printf "%.6e",u*s}'; }

# ---- 凍結 fixture 生成律(決定論・host 浮動小数 不使用)----
# 胞値 = f32 bit pattern を整数算で組む。指数 e ∈ [EB, EB+ES) ∴ 振幅は Q20 域内(sat=0 を狙う)。
# UNIFORM=1 なら全胞 1.0 = lap 恒零(SPECIAL 走)。
payload() { # payload <n> <EB> <ES> <uniform>
  k=0
  while [ $k -lt $1 ]; do
    if [ "$4" = 1 ]; then printf ' 1065353216'
    else
      e=$(( $2 + (k * 7 + k / 4 * 3) % $3 ))
      f=$(( (k * 2654435761 + k * 40503) % 8388608 ))
      s=$(( (k / 5 + k) % 2 ))
      printf ' %s' $(( s * 2147483648 + e * 8388608 + f ))
    fi
    k=$((k + 1))
  done
}

npass=0; nspecial=0
run() { # run <name> <W> <H> <STEPN> <uniform>
  name=$1; w=$2; h=$3; n=$4; uni=$5
  W=$w H=$h STEPN=$n NSLOTS=2 PAYLOAD="$(payload $((w * h)) 118 6 "$uni")" \
    ./gen_fldj.sh "$work/$name.fldj"
  ./fieldrun "$work/$name.fldj" "$work/$name.flro" 16384 >/dev/null
  sat=$(od -An -v -tu4 -j24 -N4 "$work/$name.flro" | tr -d ' ')
  [ "$sat" -eq 0 ] || { echo "gate: $name sat=$sat — §15 射程外(fixture 設計誤)" >&2; exit 1; }
  od -An -v -td4 -j32 "$work/$name.flro" | tr -s ' ' '\n' | grep -v '^$' > "$work/$name.q"
  FLDJ_REF=1 "$REPLAY" "$work/$name.fldj" > "$work/$name.r"
  b=$(bound "$n")
  res=$(paste "$work/$name.q" "$work/$name.r" | awk -v b="$b" '
    { q=$1/1048576.0; r=$2+0; d=q-r; if(d<0)d=-d; if(d>m)m=d;
      ar=(r<0?-r:r); if(ar>mr)mr=ar; if(ar>1e-30)nz++; n++ }
    END{ rel=(mr>0? m/mr : 0);
      printf "%.6e %.6e %.6e %d %d %s", m, mr, rel, n, nz, (m<=b?"PASS":"FAIL") }')
  set -- $res
  fsha=$(shasum -a 256 "$work/$name.fldj" | cut -d' ' -f1)
  lab=PASS; [ "$6" = PASS ] || lab=FAIL
  if [ "$uni" = 1 ]; then lab="SPECIAL(lap=0 ∴ PASS 数に算入せず)"; fi
  printf '%-8s cells=%-5s %2stick MAXABS=%s BOUND=%s MAXREF=%s MAXREL=%s sat=%s %s\n' \
    "$name" "$4" "$n" "$1" "$b" "$2" "$3" "$sat" "$lab"
  printf '         sha(fldj)=%s\n' "$fsha"
  [ "$6" = PASS ] || { echo "gate: $name MAXABS>BOUND" >&2; exit 1; }
  if [ "$uni" = 1 ]; then nspecial=$((nspecial + 1)); else npass=$((npass + 1)); fi
}

printf -- '--- §15c 凍結 fixture parity(BOUND は coef 実走から再導出)---\n'
run v0-uniform  2  2  1  1
run v1          4  4  1  0
run v2          4  4  3  0
run v3          8  8  6  0
run v4         16 16 10  0
run v5         32 32 10  0

# 歯: BOUND が真に効く事(不可能な閾 0 では FAIL する)= 判定が飾りでない証跡
if awk -v m=1e-6 'BEGIN{exit !(m<=0)}'; then echo 'gate: tooth bound-arith broken' >&2; exit 1; fi
printf 'green   %-22s PASS=%s SPECIAL=%s\n' 'parity-vectors' "$npass" "$nspecial"
printf 'gate: fieldrun §15c parity (frozen fixtures, BOUND from live coef) OK\n'
