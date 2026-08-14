.text
.p2align 2
.globl _main
.extern _fl_q30_decay_scalar
.extern _fl_q30_decay_neon
// q30_bench scalar|neon: 32 * 1Mi cells. Wall timing belongs to shell gate.
_main:
    ldr x9, [x1, #8]
    ldrb w9, [x9]
    adrp x20, Lcells@PAGE
    add x20, x20, Lcells@PAGEOFF
    mov x21, #32
Lrepeat:
    mov x0, x20
    mov x1, #1048576
    mov w2, #0xffff
    movk w2, #0x3fff, lsl #16
    cmp w9, #'s'
    b.ne Lneon
    bl _fl_q30_decay_scalar
    b Lnext
Lneon:
    bl _fl_q30_decay_neon
Lnext:
    subs x21, x21, #1
    b.ne Lrepeat
    mov x0, #0
    bl _exit
.bss
.p2align 4
Lcells: .skip 4194304
