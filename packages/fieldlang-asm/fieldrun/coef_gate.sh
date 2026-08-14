#!/bin/sh
# fieldrun A3 門: coef(手ARM64・整数のみ)。契約 §2b · §3b · G11。shell と asm のみ。
set -eu
cd "$(dirname "$0")"

if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then
  echo 'gate: foreign source present' >&2; exit 1
fi
# 浮動小数/SIMD レジスタ 使用禁(A2 の走査歯を継承)。
fpscan() {
  sed -e 's|//.*||' -e '/\.ascii/d' -e '/^[[:space:]]*\./d' coef.s \
    | grep -nE '(^|[ ,[])[vdsqh][0-9]+([^[:alnum:]_]|$)' || :
}
if [ -n "$(fpscan)" ]; then
  echo 'gate: FP/SIMD register used in coef.s' >&2; fpscan >&2; exit 1
fi
# 実行時 f32 算 禁: FP 命令(fadd/fmul/fdiv/scvtf/fcvt…)零。
fpinsn() {
  sed -e 's|//.*||' -e '/\.ascii/d' coef.s \
    | grep -nE '(^|[[:space:]])(f(add|sub|mul|div|neg|abs|cvt|cmp|mov)|scvtf|ucvtf)[[:alnum:].]*[[:space:]]' || :
}
if [ -n "$(fpinsn)" ]; then
  echo 'gate: FP instruction in coef.s' >&2; fpinsn >&2; exit 1
fi

./build.sh >/dev/null
work=$(mktemp -d ./coef-gate.XXXXXX)
trap 'rm -rf "$work"' EXIT HUP INT TERM

CANON='3F800000 3DCCCCCD 3F7FBE77 3F800000 3F800000'
WANT='c_cur=2040216832 c_lap=10737419 c_prev=-966475008'

# ベクタ: 引数5|期待(出力 或 REJECT:rc)|註
VECTORS="
$CANON|$WANT|canonical(§2b)
3F800000 3E000000 3F7FBE77 3F800000 3F800000|REJECT:22|dt=0.25 非canonical(§3b)
3F800000 3DCCCCCD 3F7FC077 3F800000 3F800000|REJECT:22|damping 十進0.999 の罠(G2)
00000000 3DCCCCCD 3F7FBE77 3F800000 3F800000|REJECT:22|c=0
3F800000 3DCCCCCD 3F7FBE77 00000000 3F800000|REJECT:22|dx=0
3F800000 3DCCCCCD 3F7FBE77 3F800000 00000000|REJECT:22|range=0
3F800000 3DCCCCCC 3F7FBE77 3F800000 3F800000|REJECT:22|dt 1ulp 下
3F800000 3DCCCCCD 3F7FBE78 3F800000 3F800000|REJECT:22|damping 1ulp 上
"

run_vectors() { # run_vectors <bin> -> 0 全一致
  bin=$1
  : >"$work/mm"
  echo "$VECTORS" | while IFS='|' read -r args want note; do
    [ -n "${args:-}" ] || continue
    # shellcheck disable=SC2086
    if out=$("$bin" $args 2>&1); then rc=0; else rc=$?; fi
    case "$want" in
      REJECT:*) exp=${want#REJECT:}
        if [ "$rc" = "$exp" ]; then r=ok; else r=MISMATCH; fi ;;
      *) if [ "$rc" = 0 ] && [ "$out" = "$want" ]; then r=ok; else r=MISMATCH; fi ;;
    esac
    printf '%-12s want=%-18s got=%-46s %-8s %s\n' "${args%% *}…" "$want" "rc=$rc $out" "$r" "$note"
    [ "$r" = ok ] || echo MISMATCH >>"$work/mm"
  done
  [ ! -s "$work/mm" ]
}

echo '--- coef 実走: §2b/§3b ベクタ ---'
if run_vectors ./coef; then echo 'green   vectors: all match'
else echo 'gate: vector mismatch' >&2; exit 1; fi

# 不変式(G11)の実測: canonical 経路が |c_lap| <= 2^29 検査を通っている事を
# 「c_lap を 2^29 超へ変異させたら rc=23 で落ちる」で示す。
build_mutant() { # <name> <sed-expr>
  m=$work/$1
  sed "$2" coef.s >"$m.s"
  if cmp -s "$m.s" coef.s; then echo "gate: tooth $1 のパッチが当たらぬ" >&2; exit 1; fi
  as -arch arm64 -o "$m.o" "$m.s"
  ld -arch arm64 -o "$m" -e _main -lSystem "$m.o" -syslibroot "$(xcrun --show-sdk-path)" 2>/dev/null
  echo "$m"
}
tooth() { # <name> <sed-expr>
  name=$1
  b=$(build_mutant "$name" "$2")
  if run_vectors "$b" >"$work/$name.log" 2>&1; then
    echo "gate: tooth $name が緑のまま = 歯無し" >&2; cat "$work/$name.log" >&2; exit 1
  fi
  printf 'KILLED  %-24s 不一致行:\n' "$name"
  grep MISMATCH "$work/$name.log" | sed 's/^/          /'
}

echo '--- 赤歯 ---'
# 誤 frac 読み(旧 G2 経路 0.998969…)起こしの係数 = c_cur 2040220160 = 0x799B4A00(骸)
tooth c_cur-misread-damping 's|movz x6, #0x3D00|movz x6, #0x4A00|'
tooth c_prev-misread-damping 's|movz x8, #0x3D00|movz x8, #0x49C0|'
tooth c_lap-truncate        's|movz x7, #0xD70B|movz x7, #0xD70A|'
tooth c_prev-sign           's|    neg x8, x8|    nop|'
tooth dt-check-removed      's|    movk w4, #0x3DCC, lsl #16|    ldr w4, [x0, #4]|'
tooth damping-check-removed 's|    movk w5, #0x3F7F, lsl #16|    ldr w5, [x0, #8]|'
tooth range-check-removed   's|    ldr w2, \[x0, #16\]|    mov w2, w3|'
tooth invariant-removed     's|    b.hi Lcf_inv .*|    nop|;s|movk x7, #0x00A3, lsl #16|movk x7, #0x2000, lsl #16|'

# 不変式が生きている実証: c_lap を 2^29 超にした版は rc=23(loud)。
b=$(build_mutant lap-over-2p29 's|movk x7, #0x00A3, lsl #16|movk x7, #0x2000, lsl #16|')
# shellcheck disable=SC2086
if $b $CANON >"$work/inv.out" 2>&1; then rc=0; else rc=$?; fi
if [ "$rc" -ne 23 ]; then
  echo "gate: 不変式検査が働かぬ rc=$rc" >&2; cat "$work/inv.out" >&2; exit 1
fi
printf 'KILLED  %-24s rc=23 %s\n' 'lap-over-2p29(invariant)' "$(cat "$work/inv.out")"

echo 'gate: coef A3 OK'
