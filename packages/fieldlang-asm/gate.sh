#!/bin/sh
# fieldlang-asm real integration gate: hand-asm build + fieldc only.
set -eu

cd "$(dirname "$0")"
FIELDC=${FIELDC:-./fieldc}
work=$(mktemp -d "./fieldc-gate.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

case "$(grep -E '(^|[^[:alnum:]_])(cargo|rustc)([^[:alnum:]_]|$)' build.sh || :)" in
  '') ;;
  *) echo 'gate: Rust toolchain found in build path' >&2; exit 1 ;;
esac

./build.sh

check_example() {
  name=$1
  golden=$2
  first="$work/$name.a.fldj"
  second="$work/$name.b.fldj"

  "$FIELDC" "examples/$name.fld" "$first"
  "$FIELDC" "examples/$name.fld" "$second"
  cmp "$first" "$second"
  digest=$(shasum -a 256 "$first" | awk '{print $1}')
  [ "$digest" = "$golden" ]
  printf 'golden %s sha256=%s parity=ok\n' "$name" "$digest"
}

check_reject() {
  name=$1
  input=$2
  printf '%s' "$input" >"$work/$name.fld"
  if "$FIELDC" "$work/$name.fld" "$work/$name.fldj" >"$work/$name.out" 2>"$work/$name.err"; then
    echo "gate: invalid $name accepted" >&2
    exit 1
  else
    rc=$?
  fi
  [ "$rc" -eq 1 ] || { echo "gate: invalid $name rc=$rc, expected 1" >&2; exit 1; }
  [ ! -e "$work/$name.fldj" ]
  printf 'negative %s rc=1 rejected=ok\n' "$name"
}

# Size-boundary closure: every input is either accepted or explicitly rejected.
# BUFSZ = 16MiB, mirrored from driver.s.
BUFSZ=16777216

fill() { # fill BYTES -> stdout, 'x' repeated
  head -c "$1" /dev/zero | tr '\0' 'x'
}

check_size_boundary() {
  # reference: the same program with no padding
  printf '\347\250\256 0 7\n' >"$work/small.fld"   # 種 0 7
  proglen=$(wc -c <"$work/small.fld" | tr -d ' ')
  "$FIELDC" "$work/small.fld" "$work/small.fldj"
  ref=$(shasum -a 256 "$work/small.fldj" | awk '{print $1}')

  # exactly BUFSZ: program + comment padding. Must be ACCEPTED, rc 0,
  # and semantically identical to the unpadded program.
  { cat "$work/small.fld"; printf '#'; fill $((BUFSZ - proglen - 2)); printf '\n'; } >"$work/exact.fld"
  size=$(wc -c <"$work/exact.fld" | tr -d ' ')
  [ "$size" -eq "$BUFSZ" ] || { echo "gate: exact fixture is $size, want $BUFSZ" >&2; exit 1; }
  "$FIELDC" "$work/exact.fld" "$work/exact.fldj"
  got=$(shasum -a 256 "$work/exact.fldj" | awk '{print $1}')
  [ "$got" = "$ref" ] || { echo "gate: exact-BUFSZ digest $got != $ref" >&2; exit 1; }
  perm=$(ls -l "$work/exact.fldj" | cut -c1-10)
  [ "$perm" = '-rw-r--r--' ] || { echo "gate: output perms $perm, want -rw-r--r--" >&2; exit 1; }
  printf 'boundary exact-BUFSZ rc=0 sha256=%s perm=%s parity=ok\n' "$got" "$perm"

  # BUFSZ+1 and 17MiB: must be REJECTED, no output file created.
  { cat "$work/exact.fld"; printf '\n'; } >"$work/over.fld"
  check_oversize over
  { printf '#'; fill $((17 * 1024 * 1024 - 1)); } >"$work/big.fld"
  check_oversize big

  # A pre-existing output file must NOT be truncated when input is rejected.
  printf 'PRESERVE' >"$work/keep.fldj"
  if "$FIELDC" "$work/big.fld" "$work/keep.fldj" >/dev/null 2>&1; then
    echo 'gate: oversize accepted on second output path' >&2; exit 1
  fi
  [ "$(cat "$work/keep.fldj")" = 'PRESERVE' ] || { echo 'gate: rejected run truncated existing output' >&2; exit 1; }
  printf 'boundary existing-output preserved=ok\n'
}

check_shape_pair() {
  # mismatch: n=3 != w*h=4 -> rc 1, no output file
  printf '界 2 2 16 42\n寫 0 3 1 2 3\n' >"$work/shape_bad.fld"
  if "$FIELDC" "$work/shape_bad.fld" "$work/shape_bad.fldj" 2>"$work/shape_bad.err"; then
    echo 'gate: 寫 n!=w*h accepted' >&2; exit 1
  else
    rc=$?
  fi
  [ "$rc" -eq 1 ] || { echo "gate: shape mismatch rc=$rc, expected 1" >&2; exit 1; }
  [ ! -e "$work/shape_bad.fldj" ] || { echo 'gate: shape mismatch created output' >&2; exit 1; }
  # match: n=4 == w*h=4 -> rc 0, output written
  printf '界 2 2 16 42\n寫 0 4 1 2 3 4\n' >"$work/shape_ok.fld"
  "$FIELDC" "$work/shape_ok.fld" "$work/shape_ok.fldj" || {
    echo 'gate: full-plane 寫 rejected' >&2; exit 1; }
  [ -s "$work/shape_ok.fldj" ] || { echo 'gate: full-plane 寫 wrote nothing' >&2; exit 1; }
  printf 'shape 寫 n!=w*h rc=1 rejected=ok · n==w*h rc=0 accepted=ok\n'
}

check_oversize() {
  name=$1
  if "$FIELDC" "$work/$name.fld" "$work/$name.fldj" >"$work/$name.out" 2>"$work/$name.err"; then
    echo "gate: oversize $name accepted" >&2; exit 1
  else
    rc=$?
  fi
  [ "$rc" -ne 0 ] || { echo "gate: oversize $name rc=0" >&2; exit 1; }
  [ ! -e "$work/$name.fldj" ] || { echo "gate: oversize $name created output" >&2; exit 1; }
  grep -q 'too large' "$work/$name.err" || { echo "gate: oversize $name lacks explicit message" >&2; exit 1; }
  printf 'boundary %s rc=%s rejected=ok (no output file)\n' "$name" "$rc"
}

check_example ex1 d89a5df92da79f2d435a4383bfde59e3f63f81c57ab4204aae2d2e4aa7b30b52
check_example ex2 7ef5548e6cb9ae3519a05bc0db40a7c2542d2e072045627f759f9ffdec131ba2
check_example ex3 2412d691e6b85bb9dab5f5e90e78f829501e4641128bdcc28788f36137179a5d
check_reject syntax 'not-a-field-program\n'
check_reject range '種 4294967296 1\n'
check_reject missing_arg '種 1\n'
check_reject extra_arg '歩 1 2\n'
check_reject invalid_byte "$(printf '\377')"
# D1 producer side: 寫 n MUST equal w*h (full plane) — mismatch is rejected here,
# not left for the consumer to panic on.
check_shape_pair
check_size_boundary

echo 'gate: fieldc assembly closure OK'
