.text
.p2align 2
// _fl_q30_wave_neon(cfg, cur, prev, out) -> saturation count.
// 真NEON: 内部4胞をvector(2d二半)で処理。外部scalar実装への委譲=無(b/bl 零)。
// 端(x=0, x=w-1, 4未満の残)のみ胞毎scalar経路(本file内、外部呼出無)。
// x18 untouched; x19-x28 untouched。v0-v7,v16-v31のみ使用。
// 係数 i64: |c_cur|,|c_prev| <= 2^31 故 c = ca + cb (各 i32 に収まる二半) へ分割。
// lap は i64(|lap| <= 2^34-4) 故 lap = (lap>>4)*16 + (lap&15) へ分割し smull で積む。
.globl _fl_q30_wave_neon
_fl_q30_wave_neon:
    sub sp, sp, #160
    stp x29, x30, [sp, #144]
    add x29, sp, #144
    str x1, [sp, #0]
    str x2, [sp, #8]
    str x3, [sp, #16]
    ldrsw x4, [x0, #0]
    str x4, [sp, #24]               // width
    ldrsw x4, [x0, #4]
    str x4, [sp, #32]               // height
    ldr x4, [x0, #8]
    str x4, [sp, #40]               // c_cur i64
    ldr x4, [x0, #16]
    str x4, [sp, #48]               // c_lap i64
    ldr x4, [x0, #24]
    str x4, [sp, #56]               // c_prev i64
    mov x4, #1
    lsl x4, x4, #29
    str x4, [sp, #64]               // half = 2^29
    dup v21.2d, x4
    mov x4, #1
    lsl x4, x4, #31
    sub x4, x4, #1
    str x4, [sp, #72]               // i32 max
    dup v22.2d, x4
    mov x4, #-1
    lsl x4, x4, #31
    str x4, [sp, #80]               // i32 min
    dup v23.2d, x4
    mov x4, #15
    dup v24.2d, x4
    str xzr, [sp, #88]              // saturation count
    str xzr, [sp, #96]              // y
    // 係数分割: ca = c asr 1, cb = c - ca (共に i32 域)
    ldr x4, [sp, #40]
    asr x5, x4, #1
    sub x6, x4, x5
    dup v16.4s, w5
    dup v17.4s, w6
    ldr x4, [sp, #56]
    asr x5, x4, #1
    sub x6, x4, x5
    dup v19.4s, w5
    dup v20.4s, w6
    ldr x4, [sp, #48]
    dup v18.4s, w4                  // |c_lap| <= 2^29 故 i32 に収まる
Ly:
    ldr x4, [sp, #96]
    ldr x5, [sp, #32]
    cmp x4, x5
    b.eq Ldone
    ldr x6, [sp, #24]
    // ym*w
    cbz x4, Lnymwrap
    sub x7, x4, #1
    b Lnymok
Lnymwrap:
    sub x7, x5, #1
Lnymok:
    mul x7, x7, x6
    str x7, [sp, #112]
    // yp*w
    add x7, x4, #1
    cmp x7, x5
    b.ne Lnypok
    mov x7, #0
Lnypok:
    mul x7, x7, x6
    str x7, [sp, #120]
    str xzr, [sp, #104]             // x
Lx:
    ldr x4, [sp, #104]
    ldr x5, [sp, #24]
    cmp x4, x5
    b.eq Lynext
    // vector 条件: x>=1 かつ x+3 <= w-2  <=>  x+5 <= w
    cmp x4, #1
    b.lt Lcell
    add x6, x4, #5
    cmp x6, x5
    b.gt Lcell
    // ---- 4胞 vector 経路 ----
    ldr x6, [sp, #96]
    madd x7, x6, x5, x4             // i = y*w + x
    ldr x8, [sp, #0]
    add x9, x8, x7, lsl #2
    ld1 {v0.4s}, [x9]               // center
    sub x10, x9, #4
    ld1 {v1.4s}, [x10]              // left
    add x10, x9, #4
    ld1 {v2.4s}, [x10]              // right
    ldr x11, [sp, #112]
    add x11, x11, x4
    add x10, x8, x11, lsl #2
    ld1 {v3.4s}, [x10]              // up
    ldr x11, [sp, #120]
    add x11, x11, x4
    add x10, x8, x11, lsl #2
    ld1 {v4.4s}, [x10]              // down
    ldr x10, [sp, #8]
    add x10, x10, x7, lsl #2
    ld1 {v5.4s}, [x10]              // prev
    // lap(lo 2 lanes) -> v6, lap(hi 2 lanes) -> v26; いずれも i64
    saddl v6.2d, v1.2s, v2.2s
    saddl v7.2d, v3.2s, v4.2s
    add v6.2d, v6.2d, v7.2d
    sxtl v7.2d, v0.2s
    shl v7.2d, v7.2d, #2
    sub v6.2d, v6.2d, v7.2d
    saddl2 v26.2d, v1.4s, v2.4s
    saddl2 v27.2d, v3.4s, v4.4s
    add v26.2d, v26.2d, v27.2d
    sxtl2 v27.2d, v0.4s
    shl v27.2d, v27.2d, #2
    sub v26.2d, v26.2d, v27.2d
    ext v29.16b, v0.16b, v0.16b, #8 // center 上2 lane
    ext v30.16b, v5.16b, v5.16b, #8 // prev 上2 lane
    mov x13, #0                     // 飽和数(本反復)
    // ---- lo 半 (vc=v0, vp=v5, vlap=v6) ----
    smull v1.2d, v0.2s, v16.2s
    smull v2.2d, v0.2s, v17.2s
    add v1.2d, v1.2d, v2.2d
    add v1.2d, v1.2d, v21.2d
    sshr v1.2d, v1.2d, #30
    smull v2.2d, v5.2s, v19.2s
    smull v3.2d, v5.2s, v20.2s
    add v2.2d, v2.2d, v3.2d
    add v2.2d, v2.2d, v21.2d
    sshr v2.2d, v2.2d, #30
    sshr v3.2d, v6.2d, #4
    and v4.16b, v6.16b, v24.16b
    xtn v3.2s, v3.2d
    xtn v4.2s, v4.2d
    smull v3.2d, v3.2s, v18.2s
    shl v3.2d, v3.2d, #4
    smull v4.2d, v4.2s, v18.2s
    add v3.2d, v3.2d, v4.2d
    add v3.2d, v3.2d, v21.2d
    sshr v3.2d, v3.2d, #30
    add v31.2d, v1.2d, v2.2d
    add v31.2d, v31.2d, v3.2d
    cmgt v1.2d, v31.2d, v22.2d
    cmgt v2.2d, v23.2d, v31.2d
    orr v1.16b, v1.16b, v2.16b
    addp d1, v1.2d
    fmov x12, d1
    sub x13, x13, x12
    mov v28.16b, v31.16b            // acc lo 保存
    // ---- hi 半 (vc=v29, vp=v30, vlap=v26) ----
    smull v1.2d, v29.2s, v16.2s
    smull v2.2d, v29.2s, v17.2s
    add v1.2d, v1.2d, v2.2d
    add v1.2d, v1.2d, v21.2d
    sshr v1.2d, v1.2d, #30
    smull v2.2d, v30.2s, v19.2s
    smull v3.2d, v30.2s, v20.2s
    add v2.2d, v2.2d, v3.2d
    add v2.2d, v2.2d, v21.2d
    sshr v2.2d, v2.2d, #30
    sshr v3.2d, v26.2d, #4
    and v4.16b, v26.16b, v24.16b
    xtn v3.2s, v3.2d
    xtn v4.2s, v4.2d
    smull v3.2d, v3.2s, v18.2s
    shl v3.2d, v3.2d, #4
    smull v4.2d, v4.2s, v18.2s
    add v3.2d, v3.2d, v4.2d
    add v3.2d, v3.2d, v21.2d
    sshr v3.2d, v3.2d, #30
    add v31.2d, v1.2d, v2.2d
    add v31.2d, v31.2d, v3.2d
    cmgt v1.2d, v31.2d, v22.2d
    cmgt v2.2d, v23.2d, v31.2d
    orr v1.16b, v1.16b, v2.16b
    addp d1, v1.2d
    fmov x12, d1
    sub x13, x13, x12
    // 書出: acc lo/hi を i32 飽和で束ねる
    sqxtn v0.2s, v28.2d
    sqxtn2 v0.4s, v31.2d
    ldr x10, [sp, #16]
    add x10, x10, x7, lsl #2
    st1 {v0.4s}, [x10]
    ldr x12, [sp, #88]
    add x12, x12, x13
    str x12, [sp, #88]
    ldr x4, [sp, #104]
    add x4, x4, #4
    str x4, [sp, #104]
    b Lx
Lcell:
    // ---- 端胞 scalar 経路(本file内) ----
    ldr x6, [sp, #96]
    madd x7, x6, x5, x4
    ldr x8, [sp, #0]
    ldrsw x9, [x8, x7, lsl #2]      // c0
    cbz x4, Lcxmwrap
    sub x10, x4, #1
    b Lcxmok
Lcxmwrap:
    sub x10, x5, #1
Lcxmok:
    add x11, x4, #1
    cmp x11, x5
    b.ne Lcxpok
    mov x11, #0
Lcxpok:
    madd x10, x6, x5, x10
    madd x11, x6, x5, x11
    ldrsw x12, [x8, x10, lsl #2]
    ldrsw x13, [x8, x11, lsl #2]
    add x12, x12, x13
    ldr x10, [sp, #112]
    add x10, x10, x4
    ldr x11, [sp, #120]
    add x11, x11, x4
    ldrsw x13, [x8, x10, lsl #2]
    add x12, x12, x13
    ldrsw x13, [x8, x11, lsl #2]
    add x12, x12, x13
    lsl x13, x9, #2
    sub x12, x12, x13               // lap i64
    ldr x13, [sp, #40]
    mul x13, x13, x9
    ldr x14, [sp, #64]
    add x13, x13, x14
    asr x13, x13, #30
    ldr x15, [sp, #48]
    mul x15, x15, x12
    add x15, x15, x14
    asr x15, x15, #30
    ldr x16, [sp, #8]
    ldrsw x16, [x16, x7, lsl #2]
    ldr x17, [sp, #56]
    mul x16, x17, x16
    add x16, x16, x14
    asr x16, x16, #30
    add x13, x13, x15
    add x13, x13, x16
    ldr x14, [sp, #72]
    cmp x13, x14
    b.gt Lchi
    ldr x14, [sp, #80]
    cmp x13, x14
    b.lt Lclo
    ldr x16, [sp, #16]
    str w13, [x16, x7, lsl #2]
    b Lcnext
Lchi:
    ldr x16, [sp, #16]
    str w14, [x16, x7, lsl #2]
    b Lcsat
Lclo:
    ldr x16, [sp, #16]
    str w14, [x16, x7, lsl #2]
Lcsat:
    ldr x16, [sp, #88]
    add x16, x16, #1
    str x16, [sp, #88]
Lcnext:
    ldr x16, [sp, #104]
    add x16, x16, #1
    str x16, [sp, #104]
    b Lx
Lynext:
    ldr x4, [sp, #96]
    add x4, x4, #1
    str x4, [sp, #96]
    b Ly
Ldone:
    ldr x0, [sp, #88]
    ldp x29, x30, [sp, #144]
    add sp, sp, #160
    ret
