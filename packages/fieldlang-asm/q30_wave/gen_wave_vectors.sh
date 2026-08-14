#!/bin/bash
# Q30WAVE2 fixture generator — deterministic shell-only (bash arithmetic, 64-bit exact).
# Independent law implementation: no product code, no external interpreters.
# Emits the frozen corpus byte-for-byte:
#   sha256 = b826a11494d9e988ad90cc2db93aceceb77229ae741e028a2a785339751b493e
# Wire (Q30WAVE2\0): per record LE u16 namelen, name, i32 w, i32 h,
# i64 c_cur,c_lap,c_prev, i32 cur[n], prev[n], want[n], u64 sat; final u16 0.
set -eu
OUT=${1:?usage: gen_wave_vectors.sh <outfile> [edge]}
MODE=${2:-corpus}
: > "$OUT"

I32_MAX=2147483647
I32_MIN=-2147483648
HALF=536870912            # 2^29

BUF=''
flush() { printf '%b' "$BUF" >> "$OUT"; BUF=''; }
# little-endian byte emitters (values taken mod 2^k via masking)
put_u16() { local v=$1; printf -v _o '\\%03o\\%03o' $((v & 255)) $(((v >> 8) & 255)); BUF+=$_o; }
put_i32() { local v=$1; printf -v _o '\\%03o\\%03o\\%03o\\%03o' $((v & 255)) $(((v >> 8) & 255)) $(((v >> 16) & 255)) $(((v >> 24) & 255)); BUF+=$_o; }
put_i64() { local v=$1 i b; for ((i = 0; i < 8; i++)); do b=$((v & 255)); printf -v _o '\\%03o' "$b"; BUF+=$_o; v=$((v >> 8)); done; }
put_str() { local s=$1 i; for ((i = 0; i < ${#s}; i++)); do printf -v _o '\\%03o' "'${s:$i:1}"; BUF+=$_o; done; }

# q30(c,x) = (c*x + 2^29) arithmetic >> 30. |c|<=2^31, |x|<=2^31 -> product <= 2^62: i64-exact.
# bash >> on negatives is arithmetic: floor division by 2^30 as required.

EXTREMA=(0 1 -1 2 -2 3 -3 536870912 -536870912 1073741824 -1073741824
         2147483647 -2147483648 2147483646 -2147483647 1073741825)
# COEFS triples flattened
COEFS=(1073741824 268435456 -1073741824
       2147483647 536870911 -1073741824
       0 0 0
       1073741824 -536870911 1073741824
       -2147483648 536870911 2147483647
       1 -1 1
       2147483647 1 -2147483648)

SEED=$((0xA5A5F00D))
lcg() { # -> RND (signed i32)
    SEED=$(( (1664525 * SEED + 1013904223) & 4294967295 ))
    if (( SEED >= 2147483648 )); then RND=$((SEED - 4294967296)); else RND=$SEED; fi
}

CUR=(); PREV=()
# emit_case name w h cc cl cp   (CUR/PREV already filled, length w*h)
emit_case() {
    local name=$1 w=$2 h=$3 cc=$4 cl=$5 cp=$6
    local n=$((w * h)) sat=0 y x i ym yp xm xp lap acc
    local -a want=()
    for ((y = 0; y < h; y++)); do
        (( ym = (y == 0) ? h - 1 : y - 1 ))
        (( yp = (y == h - 1) ? 0 : y + 1 ))
        for ((x = 0; x < w; x++)); do
            (( xm = (x == 0) ? w - 1 : x - 1 ))
            (( xp = (x == w - 1) ? 0 : x + 1 ))
            i=$((y * w + x))
            lap=$(( CUR[y*w+xm] + CUR[y*w+xp] + CUR[ym*w+x] + CUR[yp*w+x] - 4 * CUR[i] ))
            acc=$(( ((cc * CUR[i] + HALF) >> 30) + ((cl * lap + HALF) >> 30) + ((cp * PREV[i] + HALF) >> 30) ))
            if (( acc > I32_MAX )); then want[i]=$I32_MAX; ((sat += 1))
            elif (( acc < I32_MIN )); then want[i]=$I32_MIN; ((sat += 1))
            else want[i]=$acc; fi
        done
    done
    put_u16 ${#name}; put_str "$name"
    put_i32 "$w"; put_i32 "$h"
    put_i64 "$cc"; put_i64 "$cl"; put_i64 "$cp"
    for ((i = 0; i < n; i++)); do put_i32 "${CUR[i]}"; done
    for ((i = 0; i < n; i++)); do put_i32 "${PREV[i]}"; done
    for ((i = 0; i < n; i++)); do put_i32 "${want[i]}"; done
    put_i64 "$sat"
    flush
}

put_str 'Q30WAVE2'; BUF+='\000'

if [ "$MODE" = edge ]; then
    # 6 beyond-fixture edge cases (1x1 / w=1 / h=1 / saturation / 64x64 / 65x33 tail),
    # coefficients at the i64 boundaries c_cur=±2^31, |c_lap|=2^29.
    CUR=(2147483647); PREV=(-2147483648)
    emit_case 'edge-1x1' 1 1 2147483648 536870912 -2147483648
    CUR=(); PREV=()
    for ((i = 0; i < 7; i++)); do CUR[i]=${EXTREMA[(i * 5 + 7) % 16]}; PREV[i]=${EXTREMA[(i * 3 + 11) % 16]}; done
    emit_case 'edge-w1' 1 7 -2147483648 -536870912 2147483648
    emit_case 'edge-h1' 7 1 2147483648 536870912 -2147483648
    CUR=(); PREV=()
    for ((i = 0; i < 36; i++)); do CUR[i]=$I32_MAX; PREV[i]=$I32_MAX; done
    emit_case 'edge-sat' 6 6 2147483648 536870912 2147483647
    n=4096; CUR=(); PREV=()
    for ((i = 0; i < n; i++)); do lcg; CUR[i]=$RND; done
    for ((i = 0; i < n; i++)); do lcg; PREV[i]=$RND; done
    emit_case 'edge-64x64' 64 64 2147483648 536870912 -2147483648
    n=2145; CUR=(); PREV=()
    for ((i = 0; i < n; i++)); do lcg; CUR[i]=$RND; done
    for ((i = 0; i < n; i++)); do lcg; PREV[i]=$RND; done
    emit_case 'edge-65x33' 65 33 -2147483648 536870912 2147483648
    put_u16 0
    flush
    exit 0
fi

# (1) degenerate dims x 7 coefficient triples; EXTREMA-patterned fields
for wh in '1 1' '1 2' '2 1' '1 5' '5 1' '1 8' '8 1' '2 2' '3 3' '4 4' '5 4' '4 5' '7 3' '3 7'; do
    set -- $wh; w=$1; h=$2; n=$((w * h))
    for ((ci = 0; ci < 7; ci++)); do
        CUR=(); PREV=()
        for ((i = 0; i < n; i++)); do
            CUR[i]=${EXTREMA[(i * 7 + ci) % 16]}
            PREV[i]=${EXTREMA[(i * 11 + ci * 3) % 16]}
        done
        emit_case "dim-${w}x${h}-c${ci}" "$w" "$h" "${COEFS[ci*3]}" "${COEFS[ci*3+1]}" "${COEFS[ci*3+2]}"
    done
done

# (2) lane-tail widths, alternating extremes
for ((w = 1; w <= 12; w++)); do
    h=3; n=$((w * h)); CUR=(); PREV=()
    for ((i = 0; i < n; i++)); do
        if (( i % 2 )); then CUR[i]=$I32_MAX; else CUR[i]=$I32_MIN; fi
        if (( i % 3 )); then PREV[i]=$I32_MIN; else PREV[i]=$I32_MAX; fi
    done
    emit_case "tail-w${w}" "$w" "$h" "${COEFS[0]}" "${COEFS[1]}" "${COEFS[2]}"
done

# (3) saturation planes per coefficient triple
for ((ci = 0; ci < 7; ci++)); do
    n=36; CUR=(); PREV=()
    for ((i = 0; i < n; i++)); do CUR[i]=$I32_MAX; PREV[i]=$I32_MAX; done
    emit_case "sat-hi-c${ci}" 6 6 "${COEFS[ci*3]}" "${COEFS[ci*3+1]}" "${COEFS[ci*3+2]}"
    for ((i = 0; i < n; i++)); do CUR[i]=$I32_MIN; PREV[i]=$I32_MIN; done
    emit_case "sat-lo-c${ci}" 6 6 "${COEFS[ci*3]}" "${COEFS[ci*3+1]}" "${COEFS[ci*3+2]}"
done

# (4) halfway rounding exposure
CUR=(); PREV=()
for ((k = -4; k < 12; k++)); do CUR[k+4]=$((HALF + k)); PREV[k+4]=$((-HALF + k)); done
emit_case 'halfway' 4 4 1073741824 1 1073741824
for ((i = 0; i < 16; i++)); do PREV[i]=${CUR[i]}; CUR[i]=$(( -(HALF + i - 4) )); done
emit_case 'halfway-neg' 4 4 -1073741824 -1 -1073741824

# (5) LCG random boards, seed 0xA5A5F00D
for wh in '17 5' '5 17' '13 13' '64 3' '3 64' '31 9'; do
    set -- $wh; w=$1; h=$2; n=$((w * h)); CUR=(); PREV=()
    for ((i = 0; i < n; i++)); do lcg; CUR[i]=$RND; done
    for ((i = 0; i < n; i++)); do lcg; PREV[i]=$RND; done
    emit_case "rand-${w}x${h}" "$w" "$h" "${COEFS[0]}" "${COEFS[1]}" "${COEFS[2]}"
done

# (6) large digest board, coefficient triple 1
w=128; h=97; n=$((w * h)); CUR=(); PREV=()
for ((i = 0; i < n; i++)); do lcg; CUR[i]=$RND; done
for ((i = 0; i < n; i++)); do lcg; PREV[i]=$RND; done
emit_case 'rand-128x97' 128 97 2147483647 536870911 -1073741824

# (7) V2 i64-coefficient boundary cases
CUR=(1 -1 1073741824 -1073741824); PREV=(0 0 0 0)
emit_case 'i64-cur-pos' 2 2 2147483648 0 0
emit_case 'i64-cur-neg' 2 2 -2147483648 0 0
CUR=(2147483647 -2147483648 2147483647); PREV=(0 0 0)
emit_case 'lap-pos-bound' 3 1 0 536870912 0
emit_case 'lap-neg-bound' 1 3 0 -536870912 0
CUR=(2147483647 -2147483648 1 -1); PREV=(-1 1 -2147483648 2147483647)
emit_case 'term-order-i64' 2 2 2147483648 536870912 -2147483648

put_u16 0
flush
