// Q30WAVE1 hand ARM64 gate runner. libSystem: open/read/close/write/exit only.
// Bounded static arenas; validates every frozen case through scalar + NEON.
// x18/x19..x28 untouched; stack is 16-byte aligned at every external call.
.subsections_via_symbols
.p2align 3
.globl _main
.extern _open
.extern _read
.extern _close
.extern _write
.extern _exit
.extern _fl_q30_wave_scalar
.extern _fl_q30_wave_neon

_main:
    sub sp, sp, #128
    // argv[2] is vector pathname; gate supplies it.  argc < 3 is failure.
    cmp w0, #3
    b.lt Lfail
    ldr x0, [x1, #16]
    mov x1, #0
    bl _open
    cmp w0, #0
    b.lt Lfail
    str x0, [sp, #0]
    adrp x1, _filebuf@PAGE
    add x1, x1, _filebuf@PAGEOFF
    mov x2, #0x40000
    bl _read
    cmp x0, #11
    b.lt Lfail
    str x0, [sp, #8]
    ldr x0, [sp, #0]
    bl _close
    adrp x9, _filebuf@PAGE
    add x9, x9, _filebuf@PAGEOFF
    // magic Q30WAVE1\0
    ldr x10, [x9]
    movz x11, #0x3351
    movk x11, #0x5730, lsl #16
    movk x11, #0x5641, lsl #32
    movk x11, #0x3145, lsl #48
    cmp x10, x11
    b.ne Lfail
    ldrb w10, [x9, #8]
    cbnz w10, Lfail
    add x9, x9, #9
    str x9, [sp, #16]              // parse cursor
    mov x10, #0
    str x10, [sp, #24]             // cases
Lcase:
    ldr x9, [sp, #16]
    ldrh w10, [x9], #2
    cbz w10, Lok
    add x9, x9, x10                 // skip name
    ldr w11, [x9], #4               // w
    ldr w12, [x9], #4               // h
    cmp w11, #0
    b.le Lfail
    cmp w12, #0
    b.le Lfail
    mul w13, w11, w12
    cmp w13, #0x4000                // arena: 16384 i32 max
    b.hi Lfail
    adrp x14, _cfg@PAGE
    add x14, x14, _cfg@PAGEOFF
    str w11, [x14]
    str w12, [x14, #4]
    ldrsw x15, [x9], #4
    str x15, [x14, #8]
    ldrsw x15, [x9], #4
    str x15, [x14, #16]
    ldrsw x15, [x9], #4
    str x15, [x14, #24]
    uxtw x15, w13
    lsl x15, x15, #2
    str x15, [sp, #32]             // bytes
    // copy cur, prev into bounded aligned arenas; retain expected pointer
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    bl Lcopy
    adrp x0, _prevbuf@PAGE
    add x0, x0, _prevbuf@PAGEOFF
    ldr x15, [sp, #32]
    bl Lcopy
    str x9, [sp, #40]               // expected vector
    ldr x15, [sp, #32]
    add x9, x9, x15
    ldr x10, [x9], #8               // expected sat
    str x10, [sp, #48]
    str x9, [sp, #16]
    // scalar(cfg,cur,prev,out)
    adrp x0, _cfg@PAGE
    add x0, x0, _cfg@PAGEOFF
    adrp x1, _curbuf@PAGE
    add x1, x1, _curbuf@PAGEOFF
    adrp x2, _prevbuf@PAGE
    add x2, x2, _prevbuf@PAGEOFF
    adrp x3, _outbuf@PAGE
    add x3, x3, _outbuf@PAGEOFF
    bl _fl_q30_wave_scalar
    ldr x10, [sp, #48]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    ldr x1, [sp, #40]
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    // NEON identical input/output comparison.
    adrp x0, _cfg@PAGE
    add x0, x0, _cfg@PAGEOFF
    adrp x1, _curbuf@PAGE
    add x1, x1, _curbuf@PAGEOFF
    adrp x2, _prevbuf@PAGE
    add x2, x2, _prevbuf@PAGEOFF
    adrp x3, _outbuf@PAGE
    add x3, x3, _outbuf@PAGEOFF
    bl _fl_q30_wave_neon
    ldr x10, [sp, #48]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    ldr x1, [sp, #40]
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    ldr x10, [sp, #24]
    add x10, x10, #1
    str x10, [sp, #24]
    b Lcase
// x0=dst, x9=src, x15=count; preserves parser source advanced.
Lcopy:
    cbz x15, 2f
1:  ldrb w10, [x9], #1
    strb w10, [x0], #1
    subs x15, x15, #1
    b.ne 1b
2:  ret
// x0,x1 buffers; x2 bytes -> w0 mismatch boolean
Lcmp:
    cbz x2, 4f
3:  ldrb w9, [x0], #1
    ldrb w10, [x1], #1
    cmp w9, w10
    b.ne 5f
    subs x2, x2, #1
    b.ne 3b
4:  mov w0, #0
    ret
5:  mov w0, #1
    ret
Lok:
    ldr x10, [sp, #24]
    cmp x10, #133
    b.ne Lfail
    adrp x1, Lokmsg@PAGE
    add x1, x1, Lokmsg@PAGEOFF
    mov x0, #1
    mov x2, #21
    bl _write
    mov x0, #0
    b Lexit
Lfail:
    adrp x1, Lbadmsg@PAGE
    add x1, x1, Lbadmsg@PAGEOFF
    mov x0, #2
    mov x2, #22
    bl _write
    mov x0, #1
Lexit:
    bl _exit

.section __DATA,__bss
.p2align 4
_filebuf: .space 262144
_curbuf:  .space 65536
_prevbuf: .space 65536
_outbuf:  .space 65536
_cfg:     .space 32
.section __TEXT,__cstring,cstring_literals
Lokmsg: .asciz "wave_runner: 133 ok\n"
Lbadmsg: .asciz "wave_runner: failure\n"
