#!/bin/sh
# fieldlang-asm build: real lexer+emit+driver -> fieldc
# stubs remain for gate-only harness builds (not linked into fieldc)
set -e
cd "$(dirname "$0")"
SDK="$(xcrun --show-sdk-path)"
as -arch arm64 -o driver.o driver.s
as -arch arm64 -o lexer.o lexer.s
as -arch arm64 -o emit.o emit.s
ld -arch arm64 -o fieldc driver.o lexer.o emit.o -lSystem -syslibroot "$SDK"
echo "built: fieldc"
