// fieldrun — FLDJ v1 実行体(scalar backend)。手ARM64、整数のみ。
// 契約: fieldrun/CONTRACT.md §2b §2c §2d §3c · §4 A4。
// 用: fieldrun <in.fldj> <out.flro> [max_cells=16384]
// 記号: _coef_from_header(A3) · _q20_from_bits(A2) · _fl_q30_wave_scalar(q30_wave)
// rc: 0 成功 · 2 header短 · 3 magic · 4 version · 5 w==0 · 6 h==0 · 7 w*h溢
//     8 w*h>上限 · 9 n_slots<2 · 10 n_slots>上限 · 12 未知tag · 13 slot>=2
//     14 op切断 · 15 len!=w*h · 16 open/read · 17 用法 · 18 出力書込失敗
//     20 非有限payload · 21 範囲外payload · 22 非canonical · 23 不変式
// FP/SIMD レジスタ・FP 命令 零(門が二重走査で強制)。
.subsections_via_symbols
.p2align 2
.globl _main
.extern _open
.extern _read
.extern _write
.extern _close
.extern _exit
.extern _coef_from_header
.extern _q20_from_bits
.extern _fl_q30_wave_scalar

// 局所域(sp 基準、160B):
//   0..31  cfg{w i32,h i32,c_cur i64,c_lap i64,c_prev i64}
//   32     step 残
//   40     out fd
//   48     max_cells
//   56     payload cursor
//   64..87 coef 出力 3*i64
//   88..107 hdr5(5*u32 LE)
//   112    write 先 ptr
//   120    k
_main:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    sub sp, sp, #160
    mov x28, x0                    // argc
    mov x27, x1                    // argv
    cmp w28, #3
    b.lt Lusage
    mov x9, #16384                 // 既定 max_cells(硬碼禁 = 既定 + 引数)
    str x9, [sp, #48]
    cmp w28, #4
    b.lt Largs_done
    ldr x0, [x27, #24]
    bl Lparse_dec
    cbz x1, Lusage
    cbz x0, Lusage
    str x0, [sp, #48]
Largs_done:
    // 出力 path を退避(x27 は後で scratch へ転用される)
    ldr x9, [x27, #16]
    adrp x10, _argv_out@PAGE
    add x10, x10, _argv_out@PAGEOFF
    str x9, [x10]
    // ---- 入力読込 ----
    ldr x0, [x27, #8]
    mov x1, #0
    bl _open
    cmp w0, #0
    b.lt Lopenfail
    mov x19, x0
    adrp x1, _filebuf@PAGE
    add x1, x1, _filebuf@PAGEOFF
    movz x2, #0x40, lsl #16
    mov x0, x19
    bl _read
    mov x21, x0
    mov x0, x19
    bl _close
    cmp x21, #0
    b.lt Lopenfail
    movz x2, #0x40, lsl #16
    cmp x21, x2
    b.ge Lopenfail

    adrp x19, _filebuf@PAGE
    add x19, x19, _filebuf@PAGEOFF
    add x20, x19, x21              // end
    cmp x21, #48
    b.lt Lrej_short

    ldr w9, [x19]
    movz w10, #0x4C46
    movk w10, #0x4A44, lsl #16
    cmp w9, w10
    b.ne Lrej_magic
    ldr w9, [x19, #4]
    cmp w9, #1
    b.ne Lrej_version
    ldr w22, [x19, #8]
    cbz w22, Lrej_w
    ldr w23, [x19, #12]
    cbz w23, Lrej_h
    uxtw x22, w22
    uxtw x23, w23
    umulh x9, x22, x23
    cbnz x9, Lrej_mul
    mul x24, x22, x23              // n
    ldr x9, [sp, #48]
    cmp x24, x9
    b.hi Lrej_big
    mov x9, #16384                 // arena 容量(領域実寸)
    cmp x24, x9
    b.hi Lrej_big
    ldr w9, [x19, #44]
    uxtw x9, w9
    cmp x9, #2
    b.lo Lrej_slots_lo
    mov x10, #1024
    cmp x9, x10
    b.hi Lrej_slots_hi

    // ---- 係数(§2b, A3 の ABI) ----
    ldr w9, [x19, #16]
    str w9, [sp, #88]
    ldr w9, [x19, #20]
    str w9, [sp, #92]
    ldr w9, [x19, #24]
    str w9, [sp, #96]
    ldr w9, [x19, #28]
    str w9, [sp, #100]
    ldr w9, [x19, #40]
    str w9, [sp, #104]
    add x0, sp, #88
    add x1, sp, #64
    bl _coef_from_header
    cbnz x0, Lreject               // 22/23 を loud 伝播
    // cfg 組立(§G8: w,h @0,4 ; c_cur,c_lap,c_prev @8,16,24)
    str w22, [sp, #0]
    str w23, [sp, #4]
    ldr x9, [sp, #64]
    str x9, [sp, #8]
    ldr x9, [sp, #72]
    str x9, [sp, #16]
    ldr x9, [sp, #80]
    str x9, [sp, #24]

    // ---- 三 buffer(§2d)、全域 0 ----
    adrp x25, _bufA@PAGE
    add x25, x25, _bufA@PAGEOFF    // cur
    adrp x26, _bufB@PAGE
    add x26, x26, _bufB@PAGEOFF    // prev
    adrp x27, _bufC@PAGE
    add x27, x27, _bufC@PAGEOFF    // scratch

    add x21, x19, #48              // op cursor
    mov x19, #0                    // sat 累計
    mov x28, #0                    // steps 累計
Lop:
    cmp x21, x20
    b.eq Ldone
    sub x15, x20, x21
    ldrb w9, [x21]
    cmp w9, #3
    b.eq Lop_step
    cmp w9, #6
    b.eq Lop_write
    b Lrej_tag

Lop_write:
    cmp x15, #9
    b.lo Lrej_trunc
    ldr w10, [x21, #1]
    uxtw x10, w10
    cmp x10, #2
    b.hs Lrej_slot
    ldr w11, [x21, #5]
    uxtw x11, w11
    cmp x11, x24
    b.ne Lrej_wlen
    lsl x12, x11, #2
    add x12, x12, #9
    cmp x15, x12
    b.lo Lrej_trunc
    // 宛先 = slot0 -> cur, slot1 -> prev
    cbz x10, Lw_dst0
    mov x13, x26
    b Lw_have
Lw_dst0:
    mov x13, x25
Lw_have:
    str x13, [sp, #112]
    add x14, x21, #9
    str x14, [sp, #56]
    mov x14, #0
    str x14, [sp, #120]
Lw_loop:
    ldr x14, [sp, #120]
    cmp x14, x24
    b.hs Lw_end
    ldr x11, [sp, #56]
    ldr w0, [x11, x14, lsl #2]
    bl _q20_from_bits
    cbnz x1, Lq_err                // reject を loud 伝播(飽和禁)
    ldr x13, [sp, #112]
    ldr x14, [sp, #120]
    str w0, [x13, x14, lsl #2]
    add x14, x14, #1
    str x14, [sp, #120]
    b Lw_loop
Lw_end:
    lsl x12, x24, #2
    add x12, x12, #9
    add x21, x21, x12
    b Lop

Lop_step:
    cmp x15, #5
    b.lo Lrej_trunc
    ldr w10, [x21, #1]
    uxtw x10, w10
    add x21, x21, #5
    str x10, [sp, #32]
Lstep_tick:
    ldr x10, [sp, #32]
    cbz x10, Lop
    sub x10, x10, #1
    str x10, [sp, #32]
    mov x0, sp                     // cfg
    mov x1, x25                    // cur
    mov x2, x26                    // prev
    mov x3, x27                    // out = scratch(alias 禁) [MUT:alias]
    bl _fl_q30_wave_scalar
    add x19, x19, x0               // sat 累計 [MUT:sat]
    // 三者回転: (cur,prev,scratch) <- (scratch,cur,prev)
    mov x9, x27
    mov x27, x26                   // [MUT:rot3]
    mov x26, x25                   // [MUT:rot3]
    mov x25, x9
    add x28, x28, #1               // steps
    b Lstep_tick

Ldone:
    // ---- FLRO v0 書出(§2c) ----
    adrp x9, _flro@PAGE
    add x9, x9, _flro@PAGEOFF
    movz w10, #0x4C46
    movk w10, #0x4F52, lsl #16     // "FLRO" LE = 0x4F524C46
    str w10, [x9, #0]
    str wzr, [x9, #4]              // version 0
    str w22, [x9, #8]
    str w23, [x9, #12]
    str x28, [x9, #16]             // steps u64 [MUT:steps]
    str x19, [x9, #24]             // sat u64 [MUT:satfield]
    adrp x0, _argv_out@PAGE
    add x0, x0, _argv_out@PAGEOFF
    ldr x0, [x0]
    movz x1, #0x601                // O_WRONLY|O_CREAT|O_TRUNC
    sub sp, sp, #16                // 可変引数 mode は stack(arm64 macOS ABI)
    mov x9, #420                   // 0644
    str x9, [sp]
    bl _open
    add sp, sp, #16
    cmp w0, #0
    b.lt Lwritefail
    mov x20, x0
    mov x0, x20
    adrp x1, _flro@PAGE
    add x1, x1, _flro@PAGEOFF
    mov x2, #32
    bl _write
    cmp x0, #32
    b.ne Lwritefail
    mov x0, x20
    mov x1, x25                    // 最終 slot0 = cur
    lsl x2, x24, #2
    bl _write
    lsl x2, x24, #2
    cmp x0, x2
    b.ne Lwritefail
    mov x0, x20
    bl _close
    mov x0, #0
    bl _exit

Lq_err:
    mov x0, x1
    b Lreject
Lrej_short:     mov x0, #2
    b Lreject
Lrej_magic:     mov x0, #3
    b Lreject
Lrej_version:   mov x0, #4
    b Lreject
Lrej_w:         mov x0, #5
    b Lreject
Lrej_h:         mov x0, #6
    b Lreject
Lrej_mul:       mov x0, #7
    b Lreject
Lrej_big:       mov x0, #8
    b Lreject
Lrej_slots_lo:  mov x0, #9
    b Lreject
Lrej_slots_hi:  mov x0, #10
    b Lreject
Lrej_tag:       mov x0, #12
    b Lreject
Lrej_slot:      mov x0, #13
    b Lreject
Lrej_trunc:     mov x0, #14
    b Lreject
Lrej_wlen:      mov x0, #15
    b Lreject
Lopenfail:      mov x0, #16
    b Lreject
Lusage:         mov x0, #17
    b Lreject
Lwritefail:     mov x0, #18
    b Lreject

Lreject:
    mov x19, x0
    adrp x0, Lrej_head@PAGE
    add x0, x0, Lrej_head@PAGEOFF
    mov x1, #Lrej_head_len
    bl Lapp
    mov x0, x19
    bl Lapp_num
    adrp x0, Ls_nl@PAGE
    add x0, x0, Ls_nl@PAGEOFF
    mov x1, #1
    bl Lapp
    mov x0, #2
    bl Lflush
    mov x0, x19
    bl _exit

// ---- 補助(A1 と同型) ----
Lapp:
    adrp x2, _outlen@PAGE
    add x2, x2, _outlen@PAGEOFF
    ldr x3, [x2]
    adrp x4, _outline@PAGE
    add x4, x4, _outline@PAGEOFF
Lapp_l:
    cbz x1, Lapp_done
    cmp x3, #255
    b.hs Lapp_done
    ldrb w5, [x0], #1
    strb w5, [x4, x3]
    add x3, x3, #1
    sub x1, x1, #1
    b Lapp_l
Lapp_done:
    str x3, [x2]
    ret

Lapp_num:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    bl Lu64toa
    bl Lapp
    ldp x29, x30, [sp], #16
    ret

Lu64toa:
    adrp x2, _numbuf@PAGE
    add x2, x2, _numbuf@PAGEOFF
    add x2, x2, #32
    mov x3, #0
    mov x5, #10
Lit_l:
    udiv x4, x0, x5
    msub x6, x4, x5, x0
    add w6, w6, #48
    sub x2, x2, #1
    strb w6, [x2]
    add x3, x3, #1
    mov x0, x4
    cbnz x0, Lit_l
    mov x0, x2
    mov x1, x3
    ret

Lflush:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    adrp x2, _outlen@PAGE
    add x2, x2, _outlen@PAGEOFF
    ldr x2, [x2]
    adrp x1, _outline@PAGE
    add x1, x1, _outline@PAGEOFF
    bl _write
    adrp x2, _outlen@PAGE
    add x2, x2, _outlen@PAGEOFF
    mov x3, #0
    str x3, [x2]
    ldp x29, x30, [sp], #16
    ret

Lparse_dec:
    mov x2, #0
    mov x3, #0
    mov x7, #10
Lpd_l:
    ldrb w4, [x0], #1
    cbz w4, Lpd_end
    sub w5, w4, #48
    cmp w5, #9
    b.hi Lpd_bad
    mul x2, x2, x7
    uxtw x5, w5
    add x2, x2, x5
    add x3, x3, #1
    cmp x3, #18
    b.hi Lpd_bad
    b Lpd_l
Lpd_end:
    cbz x3, Lpd_bad
    mov x0, x2
    mov x1, #1
    ret
Lpd_bad:
    mov x0, #0
    mov x1, #0
    ret

.section __DATA,__bss
.p2align 4
_filebuf:  .space 4194304
_bufA:     .space 65536
_bufB:     .space 65536
_bufC:     .space 65536
_flro:     .space 32
_outline:  .space 256
_numbuf:   .space 32
_outlen:   .space 8
_argv_out: .space 8

.section __TEXT,__const
.p2align 2
Ls_nl:      .ascii "\n"
Lrej_head:  .ascii "fieldrun reject code="
.set Lrej_head_len, . - Lrej_head
