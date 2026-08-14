.text
.p2align 2
// _fl_q30_wave_scalar(cfg, cur, prev, out) -> saturation count.
// x18 untouched; x19-x28 untouched.
.globl _fl_q30_wave_scalar
_fl_q30_wave_scalar:
    sub sp, sp, #160
    stp x29, x30, [sp, #144]
    add x29, sp, #144
    str x1, [sp, #0]
    str x2, [sp, #8]
    str x3, [sp, #16]
    ldrsw x4, [x0, #0]
    str x4, [sp, #24]
    ldrsw x4, [x0, #4]
    str x4, [sp, #32]
    ldr x4, [x0, #8]
    str x4, [sp, #40]               // c_cur i64
    ldr x4, [x0, #16]
    str x4, [sp, #48]               // c_lap i64
    ldr x4, [x0, #24]
    str x4, [sp, #56]               // c_prev i64
    mov x4, #1
    lsl x4, x4, #29
    str x4, [sp, #64]
    mov x4, #1
    lsl x4, x4, #31
    sub x4, x4, #1
    str x4, [sp, #72]
    mov x4, #-1
    lsl x4, x4, #31
    str x4, [sp, #80]
    str xzr, [sp, #88]              // saturation count
    str xzr, [sp, #96]              // y
Ly:
    ldr x4, [sp, #96]
    ldr x5, [sp, #32]
    cmp x4, x5
    b.eq Ldone
    str xzr, [sp, #104]             // x
Lx:
    ldr x4, [sp, #104]
    ldr x5, [sp, #24]
    cmp x4, x5
    b.eq Lynext
    // i=y*w+x
    ldr x6, [sp, #96]
    madd x7, x6, x5, x4
    ldr x8, [sp, #0]
    ldrsw x9, [x8, x7, lsl #2]      // c0
    // horizontal contribution
    cbz x4, Lxmwrap
    sub x10, x4, #1
    b Lxmok
Lxmwrap:
    sub x10, x5, #1
Lxmok:
    add x11, x4, #1
    cmp x11, x5
    b.ne Lxpok
    mov x11, #0
Lxpok:
    madd x10, x6, x5, x10
    madd x11, x6, x5, x11
    ldrsw x12, [x8, x10, lsl #2]
    ldrsw x13, [x8, x11, lsl #2]
    add x12, x12, x13
    // vertical contribution
    cbz x6, Lymwrap
    sub x10, x6, #1
    b Lymok
Lymwrap:
    ldr x10, [sp, #32]
    sub x10, x10, #1
Lymok:
    add x11, x6, #1
    ldr x14, [sp, #32]
    cmp x11, x14
    b.ne Lypok
    mov x11, #0
Lypok:
    madd x10, x10, x5, x4
    madd x11, x11, x5, x4
    ldrsw x13, [x8, x10, lsl #2]
    add x12, x12, x13
    ldrsw x13, [x8, x11, lsl #2]
    add x12, x12, x13
    lsl x13, x9, #2
    sub x12, x12, x13               // lap i64, max=2^34-4
    // primitive: one Q30 lap multiplication; each term +half then asr.
    ldr x13, [sp, #40]
    mul x13, x13, x9
    ldr x14, [sp, #64]
    add x13, x13, x14
    asr x13, x13, #30
    ldr x15, [sp, #48]
    mul x15, x15, x12
    add x15, x15, x14
    asr x15, x15, #30
    ldr x0, [sp, #8]
    ldrsw x0, [x0, x7, lsl #2]
    ldr x1, [sp, #56]
    mul x0, x1, x0
    add x0, x0, x14
    asr x0, x0, #30
    add x13, x13, x15
    add x13, x13, x0
    ldr x14, [sp, #72]
    cmp x13, x14
    b.gt Lhi
    ldr x14, [sp, #80]
    cmp x13, x14
    b.lt Llo
    ldr x0, [sp, #16]
    str w13, [x0, x7, lsl #2]
    b Lnext
Lhi:
    ldr x0, [sp, #16]
    str w14, [x0, x7, lsl #2]
    b Lsat
Llo:
    ldr x0, [sp, #16]
    str w14, [x0, x7, lsl #2]
Lsat:
    ldr x0, [sp, #88]
    add x0, x0, #1
    str x0, [sp, #88]
Lnext:
    ldr x0, [sp, #104]
    add x0, x0, #1
    str x0, [sp, #104]
    b Lx
Lynext:
    ldr x0, [sp, #96]
    add x0, x0, #1
    str x0, [sp, #96]
    b Ly
Ldone:
    ldr x0, [sp, #88]
    ldp x29, x30, [sp, #144]
    add sp, sp, #160
    ret
