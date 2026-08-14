.text
.p2align 2
.globl _main
.extern _fl_q30_decay_scalar
.extern _fl_q30_decay_neon
.extern _fl_metal_init
.extern _fl_metal_print_device
.extern _fl_q30_decay_metal
// Independent comparison: scalar oracle and NEON receive byte-identical copies.
_main:
    adrp x0, Lmetallib@PAGE
    add x0, x0, Lmetallib@PAGEOFF
    bl _fl_metal_init
    cbnz x0, Lgpu_fail
    bl _fl_metal_print_device
    mov x19, #0                         // failure flag
    adrp x20, Lcoeffs@PAGE
    add x20, x20, Lcoeffs@PAGEOFF
    mov x21, #13
Lcoef:
    cbz x21, Lrandom
    ldr w22, [x20], #4
    mov x23, #0
    // First seven coefficients cover count 0..9; six negative adversaries
    // run the extrema/tail count only: 70 + 6 + random = exactly 77 cases.
    cmp x21, #6
    b.hi Ln
    mov x23, #9
Ln:
    cmp x23, #10
    b.eq Lnextcoef
    adrp x0, Lleft@PAGE
    add x0, x0, Lleft@PAGEOFF
    add x0, x0, #1                      // intentional unaligned buffer
    adrp x1, Lright@PAGE
    add x1, x1, Lright@PAGEOFF
    add x1, x1, #1
    adrp x5, Lmetal@PAGE
    add x5, x5, Lmetal@PAGEOFF
    add x5, x5, #1
    mov x2, #0
Lfillsmall:
    cmp x2, #10
    b.eq Lrunsmall
    adrp x3, Lvalues@PAGE
    add x3, x3, Lvalues@PAGEOFF
    ldr w4, [x3, x2, lsl #2]
    str w4, [x0, x2, lsl #2]
    str w4, [x1, x2, lsl #2]
    str w4, [x5, x2, lsl #2]
    add x2, x2, #1
    b Lfillsmall
Lrunsmall:
    mov x24, x1
    mov x1, x23
    mov w2, w22
    bl _fl_q30_decay_scalar
    mov x25, x0
    mov x0, x24
    mov x1, x23
    mov w2, w22
    bl _fl_q30_decay_neon
    cmp x0, x25
    b.ne Lcountfail
    adrp x0, Lmetal@PAGE
    add x0, x0, Lmetal@PAGEOFF
    add x0, x0, #1
    mov x1, x23
    mov w2, w22
    bl _fl_q30_decay_metal
    cmp x0, x25
    b.ne Lcountfail
    adrp x0, Lleft@PAGE
    add x0, x0, Lleft@PAGEOFF
    add x0, x0, #1
    adrp x1, Lright@PAGE
    add x1, x1, Lright@PAGEOFF
    add x1, x1, #1
    adrp x5, Lmetal@PAGE
    add x5, x5, Lmetal@PAGEOFF
    add x5, x5, #1
    mov x2, x23
Lcmpsmall:
    cbz x2, Lnextn
    ldr w3, [x0], #4
    ldr w4, [x1], #4
    ldr w6, [x5], #4
    cmp w3, w4
    b.ne Lfail
    cmp w3, w6
    b.ne Lfail
    sub x2, x2, #1
    b Lcmpsmall
Lnextn:
    add x23, x23, #1
    b Ln
Lnextcoef:
    sub x21, x21, #1
    b Lcoef
Lrandom:
    adrp x0, Lleft@PAGE
    add x0, x0, Lleft@PAGEOFF
    add x0, x0, #1
    adrp x1, Lright@PAGE
    add x1, x1, Lright@PAGEOFF
    add x1, x1, #1
    adrp x5, Lmetal@PAGE
    add x5, x5, Lmetal@PAGEOFF
    add x5, x5, #1
    mov w3, #0x1234
    movk w3, #0x5678, lsl #16
    mov x2, #0x86a3
    movk x2, #1, lsl #16
Lfillrandom:
    cbz x2, Lrunrandom
    // LCG: deterministic, full-width test cells.
    mov w4, #0x4e6d
    movk w4, #0x1966, lsl #16
    madd w3, w3, w4, wzr
    mov w4, #0xf35f
    movk w4, #0x3c6e, lsl #16
    add w3, w3, w4
    str w3, [x0], #4
    str w3, [x1], #4
    str w3, [x5], #4
    sub x2, x2, #1
    b Lfillrandom
Lrunrandom:
    adrp x0, Lleft@PAGE
    add x0, x0, Lleft@PAGEOFF
    add x0, x0, #1
    mov x1, #0x86a3
    movk x1, #1, lsl #16
    mov w2, #0xffff
    movk w2, #0x7fff, lsl #16
    bl _fl_q30_decay_scalar
    mov x25, x0
    adrp x0, Lright@PAGE
    add x0, x0, Lright@PAGEOFF
    add x0, x0, #1
    mov x1, #0x86a3
    movk x1, #1, lsl #16
    mov w2, #0xffff
    movk w2, #0x7fff, lsl #16
    bl _fl_q30_decay_neon
    cmp x0, x25
    b.ne Lcountfail
    adrp x0, Lmetal@PAGE
    add x0, x0, Lmetal@PAGEOFF
    add x0, x0, #1
    mov x1, #0x86a3
    movk x1, #1, lsl #16
    mov w2, #0xffff
    movk w2, #0x7fff, lsl #16
    bl _fl_q30_decay_metal
    cmp x0, x25
    b.ne Lcountfail
    adrp x0, Lleft@PAGE
    add x0, x0, Lleft@PAGEOFF
    add x0, x0, #1
    adrp x1, Lright@PAGE
    add x1, x1, Lright@PAGEOFF
    add x1, x1, #1
    adrp x5, Lmetal@PAGE
    add x5, x5, Lmetal@PAGEOFF
    add x5, x5, #1
    mov x2, #0x86a3
    movk x2, #1, lsl #16
Lcmprandom:
    cbz x2, Lok
    ldr w3, [x0], #4
    ldr w4, [x1], #4
    ldr w6, [x5], #4
    cmp w3, w4
    b.ne Lfail
    cmp w3, w6
    b.ne Lfail
    sub x2, x2, #1
    b Lcmprandom
Lcountfail:
    mov x0, #2
    b Lexit
Lgpu_fail:
    mov x0, #4
    b Lexit
Lfail:
    mov x0, #3
    b Lexit
Lok:
    mov x0, #0
Lexit:
    bl _exit
.data
.p2align 2
Lmetallib: .asciz "q30.metallib"
Lcoeffs: .word 0, 1, -1, 0x40000000, -0x40000000, 0x7fffffff, 0x80000000, -2, -0x20000000, -0x40000000, 0x80000000, 0x7fffffff, -1
Lvalues: .word 0, 1, -1, 0x40000000, 0xc0000000, 0x7fffffff, 0x80000000, 0x20000000, 0xe0000000, 0x3fffffff
.bss
.p2align 4
Lleft: .skip 400020
Lright: .skip 400020
Lmetal: .skip 400020
