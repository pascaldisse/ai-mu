.text
.p2align 2
// _fl_q30_decay_scalar(cells, count, coeff) -> saturations
// x0 may be unaligned; updates cells in place.
.globl _fl_q30_decay_scalar
_fl_q30_decay_scalar:
    mov x8, #0
    mov x9, #1
    lsl x9, x9, #29                 // Q30 round bias
    mov x10, #1
    lsl x10, x10, #31
    sub x10, x10, #1                // INT32_MAX
    mov x11, #-1
    lsl x11, x11, #31               // INT32_MIN as i64
Lscalar_loop:
    cbz x1, Lscalar_done
    ldrsw x3, [x0]
    smull x4, w2, w3
    add x4, x4, x9
    asr x4, x4, #30
    cmp x4, x10
    b.gt Lscalar_hi
    cmp x4, x11
    b.lt Lscalar_lo
    str w4, [x0]
    b Lscalar_next
Lscalar_hi:
    str w10, [x0]
    add x8, x8, #1
    b Lscalar_next
Lscalar_lo:
    str w11, [x0]
    add x8, x8, #1
Lscalar_next:
    add x0, x0, #4
    sub x1, x1, #1
    b Lscalar_loop
Lscalar_done:
    mov x0, x8
    ret
