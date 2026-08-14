.text
.p2align 2
// _fl_q30_decay_neon(cells, count, coeff) -> saturations
// ARM64 NEON: four independent signed i32 cells per vector; x0 unaligned valid.
.globl _fl_q30_decay_neon
_fl_q30_decay_neon:
    mov x8, #0
    mov x9, #1
    lsl x9, x9, #29
    mov x10, #1
    lsl x10, x10, #31
    sub x10, x10, #1
    mov x11, #-1
    lsl x11, x11, #31
    dup v1.4s, w2
    dup v4.2d, x9
    dup v5.2d, x10
    dup v6.2d, x11
Lneon_four:
    cmp x1, #4
    b.lo Lneon_tail
    ld1 {v0.4s}, [x0]
    smull v2.2d, v0.2s, v1.2s
    smull2 v3.2d, v0.4s, v1.4s
    add v2.2d, v2.2d, v4.2d
    add v3.2d, v3.2d, v4.2d
    sshr v2.2d, v2.2d, #30
    sshr v3.2d, v3.2d, #30
    cmgt v7.2d, v2.2d, v5.2d
    cmgt v16.2d, v6.2d, v2.2d
    orr v7.16b, v7.16b, v16.16b
    addp d7, v7.2d
    fmov x12, d7
    neg x12, x12
    add x8, x8, x12
    cmgt v7.2d, v3.2d, v5.2d
    cmgt v16.2d, v6.2d, v3.2d
    orr v7.16b, v7.16b, v16.16b
    addp d7, v7.2d
    fmov x12, d7
    neg x12, x12
    add x8, x8, x12
    // v2/v3 already contain rounded Q30 values; saturating narrow only.
    sqxtn v0.2s, v2.2d
    sqxtn2 v0.4s, v3.2d
    st1 {v0.4s}, [x0]
    add x0, x0, #16
    sub x1, x1, #4
    b Lneon_four
Lneon_tail:
    cbz x1, Lneon_done
    ldrsw x3, [x0]
    smull x12, w2, w3
    add x12, x12, x9
    asr x12, x12, #30
    cmp x12, x10
    b.gt Lneon_hi
    cmp x12, x11
    b.lt Lneon_lo
    str w12, [x0]
    b Lneon_next
Lneon_hi:
    str w10, [x0]
    add x8, x8, #1
    b Lneon_next
Lneon_lo:
    str w11, [x0]
    add x8, x8, #1
Lneon_next:
    add x0, x0, #4
    sub x1, x1, #1
    b Lneon_tail
Lneon_done:
    mov x0, x8
    ret
