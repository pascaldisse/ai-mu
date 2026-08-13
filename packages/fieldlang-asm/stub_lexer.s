// GATE-ONLY STUB — real lexer.s owned by lexer lane. DO NOT SHIP.
// _fl_lex: ignores src, returns fixed token stream: 歩 1, EOF → 3 records.
.text
.globl _fl_lex
.p2align 2
_fl_lex:
  mov x9, #3                // kind 3 = 歩
  str x9, [x2, #0]
  mov x9, #0
  str x9, [x2, #8]
  mov x9, #7                // kind 7 = INT
  str x9, [x2, #16]
  mov x9, #1                // value 1
  str x9, [x2, #24]
  mov x9, #0                // EOF
  str x9, [x2, #32]
  str x9, [x2, #40]
  mov x0, #3
  ret
