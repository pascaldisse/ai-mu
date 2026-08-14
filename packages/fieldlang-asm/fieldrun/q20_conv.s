// q20_conv — f32 bit pattern -> Q20 i32(手ARM64、整数演算のみ)。
// 契約: fieldrun/CONTRACT.md §2a(G4) · §3a · §4 A2。
// 浮動小数/SIMD レジスタ 一切不使用(v/d/s/h/b レジスタ零、FP 命令零)。
// 用: q20_conv <hexbits>      例: q20_conv 3F800000 -> 1048576
// rc: 0 成功(stdout 十進一行) · 20 非有限(NaN/±Inf) · 21 範囲外(|mag|>2^31 或 +2^31)
//     17 用法/hex 不正。飽和禁 = 拒否は loud。
// 公開記号: _q20_from_bits(x0=u32 bits) -> x0=Q20 値(符号付), x1=0 成功/非零 rc
.subsections_via_symbols
.p2align 2
.globl _main
.globl _q20_from_bits
.extern _write
.extern _exit

// ---- _q20_from_bits: x0 = bits ; 戻 x0 = q, x1 = err ----
// 使用: w2=s w3=e w4=sig/f w5=half w6=mag w7=2^31 w8=n x9=S
_q20_from_bits:
    lsr w2, w0, #31                // s
    ubfx w3, w0, #23, #8           // e
    and w4, w0, #0x7fffff          // f
    uxtw x4, w4
    cmp w3, #0xff
    b.eq Lq_nonfinite
    cbnz w3, Lq_normal
    // e == 0
    cbz x4, Lq_zero                // ±0 -> +0
    mov x9, #-149                  // E = -126-23
    b Lq_have
Lq_normal:
    movz x5, #0x80, lsl #16        // 2^23
    add x4, x4, x5                 // sig = f + 2^23
    uxtw x9, w3
    sub x9, x9, #150               // E = e-127-23
Lq_have:
    add x9, x9, #20                // S = E+20
    cmp x9, #0
    b.lt Lq_frac
    // S >= 0: 小数部無し。sig >= 1 ゆえ S>=32 は必ず mag > 2^31。
    cmp x9, #32
    b.ge Lq_range_err
    lsl x6, x4, x9
    b Lq_range
Lq_frac:
    neg x8, x9                     // n = -S >= 1
    cmp x8, #64
    b.ge Lq_zero                   // sig < 2^24 ∴ 全て丸め落ち
    mov x5, #1
    sub x10, x8, #1
    lsl x5, x5, x10                // half = 1 << (n-1)
    add x4, x4, x5                 // ties-AWAY-from-zero  [MUT:round-add]
    lsr x6, x4, x8                 // mag
    // [MUT:ties-away]
Lq_range:
    movz x7, #0x8000, lsl #16      // 2^31
    cmp x6, x7
    b.hi Lq_range_err              // |mag| > 2^31 -> loud reject(飽和禁)
    b.lo Lq_sign
    cbz w2, Lq_range_err           // +2^31 は i32 に無い
Lq_sign:
    cbz w2, Lq_ok
    neg x6, x6
Lq_ok:
    mov x0, x6
    mov x1, #0
    ret
Lq_zero:
    mov x0, #0
    mov x1, #0
    ret
Lq_nonfinite:
    mov x0, #0
    mov x1, #20
    ret
Lq_range_err:
    mov x0, #0
    mov x1, #21
    ret

// ---- _main ----
_main:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    cmp w0, #2
    b.lt Lusage
    ldr x0, [x1, #8]
    bl Lparse_hex                  // -> x0 值, x1 ok
    cbz x1, Lusage
    bl _q20_from_bits
    cbnz x1, Lreject
    bl Lprint_dec
    mov x0, #0
    bl _exit

Lreject:
    mov x19, x1
    cmp x19, #20
    b.ne Lrej_range
    adrp x1, Lmsg_nf@PAGE
    add x1, x1, Lmsg_nf@PAGEOFF
    mov x2, #23
    b Lrej_out
Lrej_range:
    adrp x1, Lmsg_rg@PAGE
    add x1, x1, Lmsg_rg@PAGEOFF
    mov x2, #18
Lrej_out:
    mov x0, #2
    bl _write
    mov x0, x19
    bl _exit

Lusage:
    adrp x1, Lmsg_use@PAGE
    add x1, x1, Lmsg_use@PAGEOFF
    mov x2, #26
    mov x0, #2
    bl _write
    mov x0, #17
    bl _exit

// ---- Lparse_hex: x0 = C 文字列 -> x0 值, x1 = 1 成功 ----
Lparse_hex:
    mov x3, x0
    mov x0, #0
    mov x4, #0                     // 桁数
    ldrb w5, [x3]
    cbz w5, Lhx_bad
    // 任意の "0x" 前置
    cmp w5, #'0'
    b.ne Lhx_loop
    ldrb w6, [x3, #1]
    cmp w6, #'x'
    b.eq Lhx_skip
    cmp w6, #'X'
    b.ne Lhx_loop
Lhx_skip:
    add x3, x3, #2
Lhx_loop:
    ldrb w5, [x3]
    cbz w5, Lhx_done
    cmp w5, #'0'
    b.lt Lhx_bad
    cmp w5, #'9'
    b.gt Lhx_alpha
    sub w5, w5, #'0'
    b Lhx_acc
Lhx_alpha:
    orr w5, w5, #0x20              // 小文字化
    cmp w5, #'a'
    b.lt Lhx_bad
    cmp w5, #'f'
    b.gt Lhx_bad
    sub w5, w5, #'a'
    add w5, w5, #10
Lhx_acc:
    lsl x0, x0, #4
    add x0, x0, w5, uxtb
    add x4, x4, #1
    cmp x4, #8
    b.gt Lhx_bad
    add x3, x3, #1
    b Lhx_loop
Lhx_done:
    cbz x4, Lhx_bad
    mov x1, #1
    ret
Lhx_bad:
    mov x0, #0
    mov x1, #0
    ret

// ---- Lprint_dec: x0 = 符号付 64bit -> stdout 十進 + 改行 ----
Lprint_dec:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    add x4, sp, #56                // 書込末端(下向き)
    mov w5, #10
    strb w5, [x4, #-1]!
    mov x6, #0                     // 負旗
    cmp x0, #0
    b.ge Ldp_abs
    mov x6, #1
    neg x0, x0
Ldp_abs:
    mov x7, #10
Ldp_loop:
    udiv x8, x0, x7
    msub x9, x8, x7, x0            // 余り
    add w9, w9, #'0'
    strb w9, [x4, #-1]!
    mov x0, x8
    cbnz x0, Ldp_loop
    cbz x6, Ldp_out
    mov w9, #'-'
    strb w9, [x4, #-1]!
Ldp_out:
    add x2, sp, #56
    sub x2, x2, x4                 // 長さ
    mov x1, x4
    mov x0, #1
    bl _write
    ldp x29, x30, [sp], #64
    ret

.section __TEXT,__const
Lmsg_nf:  .ascii "q20: reject non-finite\n"
Lmsg_rg:  .ascii "q20: reject range\n"
Lmsg_use: .ascii "usage: q20_conv <hexbits>\n"
