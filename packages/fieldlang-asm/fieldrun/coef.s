// coef — canonical header(5値 f32 bits)-> Q30 係数三定数(手ARM64、整数演算のみ)。
// 契約: fieldrun/CONTRACT.md §2b · §3b · §4 A3 · §5(導出)· G11(不変式)。
// 浮動小数/SIMD レジスタ 一切不使用。実行時 f32 算 禁 = bit 一致検査 + 即値定数。
// 用: coef <c> <dt> <damping> <dx> <range>   (各 hex bits)
//     -> stdout "c_cur=2040216832 c_lap=10737419 c_prev=-966475008"
// 骸(A9-4 Ashwin 訂正): 旧値 c_cur=2040220160 · c_prev=-966478272 = **誤**。
//   因 = damping 0x3F7FBE77 の frac 誤読(8371319、真=8371831)→ 0.998969… 経路。
//   真: sig=16760439 ∴ dd=13408351/2^27=0.09989999979734420776。
// rc: 0 成功 · 22 非 canonical header(loud reject)· 23 不変式違反(|c_lap|>2^29)
//     17 用法/hex 不正。
// 公開記号: _coef_from_header(x0 = 5*u32 LE 配列, x1 = 3*i64 出力域)
//           -> x0 = 0 成功 / 非零 rc。出力 [0]=c_cur [1]=c_lap [2]=c_prev。
.subsections_via_symbols
.p2align 2
.globl _main
.globl _coef_from_header
.extern _write
.extern _exit

// canonical bits(§2b): c=0x3F800000 dt=0x3DCCCCCD damping=0x3F7FBE77
//                      dx=0x3F800000 range=0x3F800000
// ---- _coef_from_header ----
_coef_from_header:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    mov x9, x1                     // 出力域
    // c
    ldr w2, [x0, #0]
    movz w3, #0x0000
    movk w3, #0x3F80, lsl #16      // w3 = 0x3F800000
    cmp w2, w3
    b.ne Lcf_reject
    // dt = 0x3DCCCCCD
    ldr w2, [x0, #4]
    movz w4, #0xCCCD
    movk w4, #0x3DCC, lsl #16
    cmp w2, w4
    b.ne Lcf_reject
    // damping = 0x3F7FBE77
    ldr w2, [x0, #8]
    movz w5, #0xBE77
    movk w5, #0x3F7F, lsl #16
    cmp w2, w5
    b.ne Lcf_reject
    // dx
    ldr w2, [x0, #12]
    cmp w2, w3
    b.ne Lcf_reject
    // range
    ldr w2, [x0, #16]
    cmp w2, w3
    b.ne Lcf_reject

    // canonical ∴ 固定定数(§5 の手算経路で導出済、実行時算術零)
    movz x6, #0x3D00               // c_cur = 2040216832 = 0x799B3D00
    movk x6, #0x799B, lsl #16
    movz x7, #0xD70B               // c_lap = 10737419 = 0x00A3D70B
    movk x7, #0x00A3, lsl #16
    movz x8, #0x3D00               // 966475008 = 0x399B3D00
    movk x8, #0x399B, lsl #16
    neg x8, x8                     // c_prev = -966475008

    // 不変式(G11 · §2b): |c_lap| <= 2^29 を実測検査
    cmp x7, #0
    b.ge Lcf_lapabs
    neg x10, x7
    b Lcf_lapchk
Lcf_lapabs:
    mov x10, x7
Lcf_lapchk:
    movz x11, #0x0000
    movk x11, #0x2000, lsl #16     // 2^29 = 0x20000000
    cmp x10, x11
    b.hi Lcf_inv                   // [MUT:lap-invariant]

    str x6, [x9, #0]
    str x7, [x9, #8]
    str x8, [x9, #16]
    mov x0, #0
    ldp x29, x30, [sp], #32
    ret
Lcf_reject:
    mov x0, #22
    ldp x29, x30, [sp], #32
    ret
Lcf_inv:
    mov x0, #23
    ldp x29, x30, [sp], #32
    ret

// ---- _main ----
_main:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    cmp w0, #6
    b.ne Lusage
    mov x19, x1                    // argv
    add x20, sp, #48               // 5*u32 域
    mov x21, #0
Lm_loop:
    add x22, x21, #1
    ldr x0, [x19, x22, lsl #3]
    bl Lparse_hex
    cbz x1, Lusage
    str w0, [x20, x21, lsl #2]
    add x21, x21, #1
    cmp x21, #5
    b.lt Lm_loop

    mov x0, x20
    add x1, sp, #72                // 3*i64 出力域(24B)
    bl _coef_from_header
    cbnz x0, Lreject

    adrp x1, Lmsg_cc@PAGE
    add x1, x1, Lmsg_cc@PAGEOFF
    mov x2, #6
    mov x0, #1
    bl _write
    ldr x0, [sp, #72]
    bl Lprint_dec_nonl
    adrp x1, Lmsg_cl@PAGE
    add x1, x1, Lmsg_cl@PAGEOFF
    mov x2, #7
    mov x0, #1
    bl _write
    ldr x0, [sp, #80]
    bl Lprint_dec_nonl
    adrp x1, Lmsg_cp@PAGE
    add x1, x1, Lmsg_cp@PAGEOFF
    mov x2, #8
    mov x0, #1
    bl _write
    ldr x0, [sp, #88]
    bl Lprint_dec
    mov x0, #0
    bl _exit

Lreject:
    mov x19, x0
    cmp x19, #22
    b.ne Lrej_inv
    adrp x1, Lmsg_nc@PAGE
    add x1, x1, Lmsg_nc@PAGEOFF
    mov x2, #34
    b Lrej_out
Lrej_inv:
    adrp x1, Lmsg_iv@PAGE
    add x1, x1, Lmsg_iv@PAGEOFF
    mov x2, #29
Lrej_out:
    mov x0, #2
    bl _write
    mov x0, x19
    bl _exit

Lusage:
    adrp x1, Lmsg_use@PAGE
    add x1, x1, Lmsg_use@PAGEOFF
    mov x2, #44
    mov x0, #2
    bl _write
    mov x0, #17
    bl _exit

// ---- Lparse_hex: x0 = C 文字列 -> x0 值, x1 = 1 成功 ----
Lparse_hex:
    mov x3, x0
    mov x0, #0
    mov x4, #0
    ldrb w5, [x3]
    cbz w5, Lhx_bad
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
    orr w5, w5, #0x20
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

// ---- Lprint_dec / Lprint_dec_nonl: x0 = 符号付 64bit ----
Lprint_dec:
    mov x12, #1
    b Ldp_enter
Lprint_dec_nonl:
    mov x12, #0
Ldp_enter:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    add x4, sp, #56
    cbz x12, Ldp_abs
    mov w5, #10
    strb w5, [x4, #-1]!
Ldp_abs:
    mov x6, #0
    cmp x0, #0
    b.ge Ldp_pos
    mov x6, #1
    neg x0, x0
Ldp_pos:
    mov x7, #10
Ldp_loop:
    udiv x8, x0, x7
    msub x9, x8, x7, x0
    add w9, w9, #'0'
    strb w9, [x4, #-1]!
    mov x0, x8
    cbnz x0, Ldp_loop
    cbz x6, Ldp_out
    mov w9, #'-'
    strb w9, [x4, #-1]!
Ldp_out:
    add x2, sp, #56
    sub x2, x2, x4
    mov x1, x4
    mov x0, #1
    bl _write
    ldp x29, x30, [sp], #64
    ret

.section __TEXT,__const
Lmsg_cc:  .ascii "c_cur="
Lmsg_cl:  .ascii " c_lap="
Lmsg_cp:  .ascii " c_prev="
Lmsg_nc:  .ascii "coef: reject non-canonical header\n"
Lmsg_iv:  .ascii "coef: invariant |c_lap|>2^29\n"
Lmsg_use: .ascii "usage: coef <c> <dt> <damping> <dx> <range>\n"
