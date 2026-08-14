#!/bin/sh
# fieldlang-asm real integration gate: hand-asm build + fieldc only.
set -eu

cd "$(dirname "$0")"
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

  ./fieldc "examples/$name.fld" "$first"
  ./fieldc "examples/$name.fld" "$second"
  cmp "$first" "$second"
  digest=$(shasum -a 256 "$first" | awk '{print $1}')
  [ "$digest" = "$golden" ]
  printf 'golden %s sha256=%s parity=ok\n' "$name" "$digest"
}

check_reject() {
  name=$1
  input=$2
  printf '%s' "$input" >"$work/$name.fld"
  if ./fieldc "$work/$name.fld" "$work/$name.fldj" >"$work/$name.out" 2>"$work/$name.err"; then
    echo "gate: invalid $name accepted" >&2
    exit 1
  fi
  [ ! -e "$work/$name.fldj" ]
  printf 'negative %s rejected=ok\n' "$name"
}

check_example ex1 d89a5df92da79f2d435a4383bfde59e3f63f81c57ab4204aae2d2e4aa7b30b52
check_example ex2 7ef5548e6cb9ae3519a05bc0db40a7c2542d2e072045627f759f9ffdec131ba2
check_example ex3 2412d691e6b85bb9dab5f5e90e78f829501e4641128bdcc28788f36137179a5d
check_reject malformed 'not-a-field-program\n'
check_reject range '種 4294967296 1\n'
