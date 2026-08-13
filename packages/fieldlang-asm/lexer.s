// fieldlang-asm lexer.s — _fl_lex per CONTRACT v1 (frozen)
// arm64 macOS, AAPCS64. x0=src, x1=len, x2=out buf, x3=cap (records)
// ret x0 = token count incl EOF record; <0 = -(line<<8|code) 1=badchar 2=overflow 3=buffull
// token record 16B { u64 kind, u64 value }
// glyphs (UTF-8 3-byte, LE-packed b0|b1<<8|b2<<16):
//   種 e7a8ae=0x00AEA8E7 k1 · 撃 e69283=0x008392E6 k2 · 歩 e6ada9=0x00A9ADE6 k3
//   縛 e7b89b=0x009BB8E7 k4 · 束 e69d9f=0x009F9DE6 k5 · 寫 e5afab=0x00ABAFE5 k6
//   界 e7958c=0x008C95E7 k9
//   觀 e8a780=0x0080A7E8 k10 (V1 reservation: probe/render, no journal tag — CONTRACT §V1)

.text
.globl _fl_lex
.p2align 2
_fl_lex:
    mov     x9, x0              // cur
    add     x10, x0, x1         // end
    mov     x11, x2             // out cur
    add     x12, x2, x3, lsl #4 // out end
    mov     x13, #1             // line
    mov     x14, #0             // count
Lloop:
    cmp     x9, x10
    b.hs    Leof
    ldrb    w4, [x9]
    cmp     w4, #0x0A           // \n
    b.eq    Lnl
    cmp     w4, #0x20           // space
    b.eq    Lskip1
    cmp     w4, #0x09           // \t
    b.eq    Lskip1
    cmp     w4, #0x0D           // \r
    b.eq    Lskip1
    cmp     w4, #0x23           // #
    b.eq    Lcomment
    sub     w5, w4, #0x30
    cmp     w5, #9
    b.ls    Lint
    cmp     w4, #0xE0
    b.hs    Lglyph
Lbad:
    mov     x0, #1
    b       Lerr
Lnl:
    add     x13, x13, #1
Lskip1:
    add     x9, x9, #1
    b       Lloop
Lcomment:
    add     x9, x9, #1
    cmp     x9, x10
    b.hs    Leof
    ldrb    w4, [x9]
    cmp     w4, #0x0A
    b.ne    Lcomment
    b       Lloop               // \n counted next iter
Lint:                           // x5 = first digit
    mov     x6, #0              // value
    mov     x7, #10
Lint_loop:
    umulh   x8, x6, x7
    cbnz    x8, Lovf
    mul     x6, x6, x7
    adds    x6, x6, x5
    b.cs    Lovf
    add     x9, x9, #1
    cmp     x9, x10
    b.hs    Lint_done
    ldrb    w4, [x9]
    sub     w5, w4, #0x30
    cmp     w5, #9
    b.ls    Lint_loop
Lint_done:
    mov     x4, #7              // INT
    b       Lemit
Lovf:
    mov     x0, #2
    b       Lerr
Lglyph:
    add     x5, x9, #3
    cmp     x5, x10
    b.hi    Lbad
    ldrb    w5, [x9, #1]
    ldrb    w6, [x9, #2]
    orr     w4, w4, w5, lsl #8
    orr     w4, w4, w6, lsl #16
    movz    w7, #0xA8E7
    movk    w7, #0x00AE, lsl #16    // 種
    cmp     w4, w7
    mov     x8, #1
    b.eq    Lgm
    movz    w7, #0x92E6
    movk    w7, #0x0083, lsl #16    // 撃
    cmp     w4, w7
    mov     x8, #2
    b.eq    Lgm
    movz    w7, #0xADE6
    movk    w7, #0x00A9, lsl #16    // 歩
    cmp     w4, w7
    mov     x8, #3
    b.eq    Lgm
    movz    w7, #0xB8E7
    movk    w7, #0x009B, lsl #16    // 縛
    cmp     w4, w7
    mov     x8, #4
    b.eq    Lgm
    movz    w7, #0x9DE6
    movk    w7, #0x009F, lsl #16    // 束
    cmp     w4, w7
    mov     x8, #5
    b.eq    Lgm
    movz    w7, #0xAFE5
    movk    w7, #0x00AB, lsl #16    // 寫
    cmp     w4, w7
    mov     x8, #6
    b.eq    Lgm
    movz    w7, #0x95E7
    movk    w7, #0x008C, lsl #16    // 界
    cmp     w4, w7
    mov     x8, #9
    b.eq    Lgm
    movz    w7, #0xA7E8
    movk    w7, #0x0080, lsl #16    // 觀 (V1 reservation kind, no journal tag)
    cmp     w4, w7
    mov     x8, #10
    b.eq    Lgm
    b       Lbad
Lgm:
    add     x9, x9, #3
    mov     x4, x8
    mov     x6, #0
Lemit:                          // x4=kind x6=value
    cmp     x11, x12
    b.hs    Lfull
    stp     x4, x6, [x11], #16
    add     x14, x14, #1
    b       Lloop
Lfull:
    mov     x0, #3
Lerr:
    orr     x0, x0, x13, lsl #8
    neg     x0, x0
    ret
Leof:
    cmp     x11, x12
    b.hs    Lfull
    stp     xzr, xzr, [x11]
    add     x0, x14, #1
    ret
