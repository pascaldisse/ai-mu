#!/bin/sh
# fieldrun A2 門: q20_conv(手ARM64・整数のみ)。契約 §2a の律 + §3a のベクタ + 赤歯。
# shell と asm のみ。Rust/C/Swift/Python 零。
set -eu
cd "$(dirname "$0")"

if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi
# 浮動小数/SIMD レジスタ 使用禁(契約 A2)。
fpscan() { # 註釈・指示子・文字列を除いた命令行のみ走査
  sed -e 's|//.*||' -e '/\.ascii/d' -e '/^[[:space:]]*\./d' q20_conv.s \
    | grep -nE '(^|[ ,[])[vdsqh][0-9]+([^[:alnum:]_]|$)' || :
}
if [ -n "$(fpscan)" ]; then
  echo 'gate: FP/SIMD register used in q20_conv.s' >&2; fpscan >&2; exit 1
fi

./build.sh >/dev/null

work=$(mktemp -d ./q20-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM

# ベクタ表: bits|期待(数値 或 REJECT:rc)|註
#   契約 §3a の 17 行。ERRATUM 行は §3a の bit literal が §2a の律と矛盾する箇所
#   (0x34000000 は 2^-23 であり 2^-21 ではない ∴ q=0.125 → 0)。律を法とし、
#   §3a が意図した 2^-21 = 0x35000000 を別行で足す。
VECTORS='
00000000|0|+0
80000000|0|-0 -> +0
3F800000|1048576|1.0
BF800000|-1048576|-1.0
3F000000|524288|0.5
3DCCCCCD|104858|0.1 f32 ties-away
BDCCCCCD|-104858|-0.1
34000000|0|ERRATUM 2^-23 q=0.125 (§3a は 2^-21 と誤記し 1 を期待)
B4000000|0|ERRATUM 対称
33FFFFFF|0|2^-23 直下
34000001|0|2^-23 直上
447FFFFF|1073741760|1023.99994
C5000000|-2147483648|-2048.0 = i32 min
45000000|REJECT:21|+2048.0 mag==2^31 正 = 表現不可
7F800000|REJECT:20|+Inf
7FC00000|REJECT:20|NaN
00000001|0|最小 subnormal
35000000|1|TIE 2^-21 q=0.5 丁度 -> away -> 1
B5000000|-1|TIE 負側
34FFFFFF|0|TIE 直下 q<0.5
35000001|1|TIE 直上 q>0.5
FF800000|REJECT:20|-Inf
C5000001|REJECT:21|-2048.0 未満 |mag|>2^31
'

run_vectors() { # run_vectors <bin> ; 全一致 0 / 不一致 1(不一致行を出す)
  bin=$1; bad=0
  echo "$VECTORS" | while IFS='|' read -r bits want note; do
    [ -n "${bits:-}" ] || continue
    if out=$("$bin" "$bits" 2>&1); then rc=0; else rc=$?; fi
    case "$want" in
      REJECT:*) exp_rc=${want#REJECT:}
        if [ "$rc" = "$exp_rc" ]; then r=ok; else r=MISMATCH; fi
        got="rc=$rc $out" ;;
      *) if [ "$rc" = 0 ] && [ "$out" = "$want" ]; then r=ok; else r=MISMATCH; fi
        got="rc=$rc $out" ;;
    esac
    printf '%-8s want=%-12s got=%-26s %-7s %s\n' "$bits" "$want" "$got" "$r" "$note"
    [ "$r" = ok ] || echo MISMATCH >>"$WORKMARK"
  done
  [ ! -s "$WORKMARK" ] || bad=1
  return $bad
}

WORKMARK="$work/mismatch"
: >"$WORKMARK"
echo '--- q20_conv 実走: 契約 §3a ベクタ ---'
if run_vectors ./q20_conv; then
  echo 'green   vectors: all match'
else
  echo 'gate: vector mismatch' >&2; exit 1
fi

# --- 赤歯: 変異させたら必ず赤 ---
build_mutant() { # build_mutant <name> <sed-expr>
  m=$work/$1
  sed "$2" q20_conv.s >"$m.s"
  if cmp -s "$m.s" q20_conv.s; then echo "gate: tooth $1 のパッチが当たらぬ" >&2; exit 1; fi
  as -arch arm64 -o "$m.o" "$m.s"
  ld -arch arm64 -o "$m" -e _main -lSystem "$m.o" -syslibroot "$(xcrun --show-sdk-path)" 2>/dev/null
  echo "$m"
}

tooth() { # tooth <name> <sed-expr>
  name=$1
  b=$(build_mutant "$name" "$2")
  : >"$WORKMARK"
  if run_vectors "$b" >"$work/$name.log" 2>&1; then
    echo "gate: tooth $name が緑のまま = 歯無し" >&2; cat "$work/$name.log" >&2; exit 1
  fi
  printf 'KILLED  %-20s 不一致行:\n' "$name"
  grep MISMATCH "$work/$name.log" | sed 's/^/          /'
}

echo '--- 赤歯 ---'
tooth ties-to-even 's|    // \[MUT:ties-away\]|    lsl x11, x5, #1\n    sub x11, x11, #1\n    sub x10, x4, x5\n    and x10, x10, x11\n    cmp x10, x5\n    b.ne 8f\n    tst x6, #1\n    b.eq 8f\n    sub x6, x6, #1\n8:|'
tooth truncate     's|    add x4, x4, x5 .*\[MUT:round-add\]|    nop|'
tooth clamp-no-reject 's|    b.hi Lq_range_err|    b.hi Lq_sign|'
tooth plus-2p31-allowed 's|    cbz w2, Lq_range_err|    nop|'
tooth reject-boundary-tight 's|    b.hi Lq_range_err|    b.hs Lq_range_err|'

: >"$WORKMARK"
echo 'gate: q20_conv A2 OK'
