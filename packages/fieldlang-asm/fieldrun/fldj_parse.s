// fldj_parse — FLDJ v1 header(48B LE) + op stream 解析(手ARM64)。
// 契約: fieldrun/CONTRACT.md §2e(G7) · §4 A1。実行せぬ、走査のみ。
// libSystem: open/read/close/write/exit のみ。
// 用: fldj_parse <path.fldj> [max_cells] [max_slots]
//   既定 max_cells=16384 · max_slots=1024(硬碼禁 = 既定値 + 引数上書)。
// 成功 rc=0 + stdout 一行。 拒否 = 検査毎に別 rc(下表)+ stderr 一行。
//   2 header短 · 3 magic · 4 version · 5 w==0 · 6 h==0 · 7 w*h溢 · 8 w*h>上限
//   9 n_slots<2 · 10 n_slots>上限 · 11 物理5値 非canonical · 12 未知tag
//   13 slot>=2 · 14 op切断/末尾余剰 · 15 WriteRaw len != w*h
//   16 open/read 失敗 · 17 用法
.subsections_via_symbols
.p2align 2
.globl _main
.extern _open
.extern _read
.extern _close
.extern _write
.extern _exit

_main:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x28, x0                    // argc
    mov x19, x1                    // argv
    cmp w28, #2
    b.lt Lusage
    movz x25, #0x4000              // 既定 max cells
    movz x26, #1024                // 既定 max slots
    cmp w28, #3
    b.lt Largs_done
    ldr x0, [x19, #16]
    bl Lparse_dec
    cbz x1, Lusage
    cbz x0, Lusage
    mov x25, x0
    cmp w28, #4
    b.lt Largs_done
    ldr x0, [x19, #24]
    bl Lparse_dec
    cbz x1, Lusage
    cbz x0, Lusage
    mov x26, x0
Largs_done:
    ldr x0, [x19, #8]
    mov x1, #0
    bl _open
    cmp w0, #0
    b.lt Lopenfail
    mov x20, x0                    // fd
    adrp x1, _filebuf@PAGE
    add x1, x1, _filebuf@PAGEOFF
    movz x2, #0x40, lsl #16        // 4 MiB arena
    bl _read
    mov x21, x0
    mov x0, x20
    bl _close
    cmp x21, #0
    b.lt Lopenfail
    // arena 一杯 = 読切れ疑い。厳に拒否。
    movz x2, #0x40, lsl #16
    cmp x21, x2
    b.ge Lopenfail

    adrp x19, _filebuf@PAGE
    add x19, x19, _filebuf@PAGEOFF
    add x20, x19, x21              // file end
    cmp x21, #48
    b.lt Lrej_short

    // ---- header 検査(各項 別経路) ----
    ldr w9, [x19]                  // magic
    movz w10, #0x4C46
    movk w10, #0x4A44, lsl #16     // "FLDJ" LE
    cmp w9, w10
    b.ne Lrej_magic
    ldr w9, [x19, #4]              // version
    cmp w9, #1
    b.ne Lrej_version
    ldr w22, [x19, #8]             // w
    cbz w22, Lrej_w
    ldr w23, [x19, #12]            // h
    cbz w23, Lrej_h
    uxtw x22, w22
    uxtw x23, w23
    umulh x9, x22, x23             // u64 checked product
    cbnz x9, Lrej_mul
    mul x24, x22, x23              // n = w*h
    cmp x24, x25
    b.hi Lrej_big
    ldr w9, [x19, #44]             // n_slots
    uxtw x27, w9
    cmp x27, #2
    b.lo Lrej_slots_lo
    cmp x27, x26
    b.hi Lrej_slots_hi
    // 物理5値 = canonical bit 一致(数値比較禁)。seed(off32,8B)= 無視。
    ldr w9, [x19, #16]             // c   = 0x3F800000
    movz w10, #0x0000
    movk w10, #0x3F80, lsl #16
    cmp w9, w10
    b.ne Lrej_params
    ldr w9, [x19, #20]             // dt  = 0x3DCCCCCD
    movz w10, #0xCCCD
    movk w10, #0x3DCC, lsl #16
    cmp w9, w10
    b.ne Lrej_params
    ldr w9, [x19, #24]             // damping = 0x3F7FBE77
    movz w10, #0xBE77
    movk w10, #0x3F7F, lsl #16
    cmp w9, w10
    b.ne Lrej_params
    ldr w9, [x19, #28]             // dx  = 0x3F800000
    movz w10, #0x0000
    movk w10, #0x3F80, lsl #16
    cmp w9, w10
    b.ne Lrej_params
    ldr w9, [x19, #40]             // range = 0x3F800000
    movz w10, #0x0000
    movk w10, #0x3F80, lsl #16
    cmp w9, w10
    b.ne Lrej_params

    // ---- op stream 走査 ----
    add x21, x19, #48              // cursor
    mov x12, #0                    // ops
    mov x13, #0                    // steps 総和
    mov x14, #0                    // WriteRaw 数
Lop:
    cmp x21, x20
    b.eq Lok
    sub x15, x20, x21              // remaining
    // tag = 1B
    ldrb w9, [x21]
    cmp w9, #3
    b.eq Lop_step
    cmp w9, #6
    b.eq Lop_write
    b Lrej_tag
Lop_step:
    cmp x15, #5
    b.lo Lrej_trunc
    ldr w10, [x21, #1]             // count u32
    uxtw x10, w10
    add x13, x13, x10
    add x21, x21, #5
    add x12, x12, #1
    b Lop
Lop_write:
    cmp x15, #9
    b.lo Lrej_trunc
    ldr w10, [x21, #1]             // slot u32
    uxtw x10, w10
    cmp x10, #2
    b.hs Lrej_slot
    ldr w11, [x21, #5]             // len u32(cell 数)
    uxtw x11, w11
    // 上流 D1 の穴を此処で閉ざす: payload 長は必ず n == w*h。
    cmp x11, x24
    b.ne Lrej_wlen
    lsl x11, x11, #2               // bytes = 4n
    add x11, x11, #9
    cmp x15, x11
    b.lo Lrej_trunc
    add x21, x21, x11
    add x12, x12, #1
    add x14, x14, #1
    b Lop

Lok:
    // "fldj ok w=W h=H n=N slots=S ops=O steps=T writes=R\n"
    adrp x0, Lok_head@PAGE
    add x0, x0, Lok_head@PAGEOFF
    mov x1, #Lok_head_len
    bl Lapp
    mov x0, x22
    bl Lapp_num
    adrp x0, Ls_h@PAGE
    add x0, x0, Ls_h@PAGEOFF
    mov x1, #Ls_h_len
    bl Lapp
    mov x0, x23
    bl Lapp_num
    adrp x0, Ls_n@PAGE
    add x0, x0, Ls_n@PAGEOFF
    mov x1, #Ls_n_len
    bl Lapp
    mov x0, x24
    bl Lapp_num
    adrp x0, Ls_slots@PAGE
    add x0, x0, Ls_slots@PAGEOFF
    mov x1, #Ls_slots_len
    bl Lapp
    mov x0, x27
    bl Lapp_num
    adrp x0, Ls_ops@PAGE
    add x0, x0, Ls_ops@PAGEOFF
    mov x1, #Ls_ops_len
    bl Lapp
    mov x0, x12
    bl Lapp_num
    adrp x0, Ls_steps@PAGE
    add x0, x0, Ls_steps@PAGEOFF
    mov x1, #Ls_steps_len
    bl Lapp
    mov x0, x13
    bl Lapp_num
    adrp x0, Ls_writes@PAGE
    add x0, x0, Ls_writes@PAGEOFF
    mov x1, #Ls_writes_len
    bl Lapp
    mov x0, x14
    bl Lapp_num
    adrp x0, Ls_nl@PAGE
    add x0, x0, Ls_nl@PAGEOFF
    mov x1, #1
    bl Lapp
    mov x0, #1
    bl Lflush
    mov x0, #0
    b Lexit

// ---- 拒否経路(各々 別 rc) ----
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
Lrej_params:    mov x0, #11
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
Lexit:
    bl _exit

// ---- 補助 ----
// Lapp(x0=ptr, x1=len): _outline へ追記(上限 256、超過分は捨てる)。
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

// Lapp_num(x0=u64): 十進化して追記。
Lapp_num:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    bl Lu64toa
    bl Lapp
    ldp x29, x30, [sp], #16
    ret

// Lu64toa(x0=val) -> x0=ptr, x1=len(_numbuf 内、後方構築)。
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

// Lflush(x0=fd): _outline を書出し、長さを 0 へ戻す。
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

// Lparse_dec(x0=NUL終端十進) -> x0=値, x1=1可/0否。
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
_outline:  .space 256
_numbuf:   .space 32
_outlen:   .space 8

.section __TEXT,__const
.p2align 2
Lok_head:   .ascii "fldj ok w="
.set Lok_head_len, . - Lok_head
Ls_h:       .ascii " h="
.set Ls_h_len, . - Ls_h
Ls_n:       .ascii " n="
.set Ls_n_len, . - Ls_n
Ls_slots:   .ascii " slots="
.set Ls_slots_len, . - Ls_slots
Ls_ops:     .ascii " ops="
.set Ls_ops_len, . - Ls_ops
Ls_steps:   .ascii " steps="
.set Ls_steps_len, . - Ls_steps
Ls_writes:  .ascii " writes="
.set Ls_writes_len, . - Ls_writes
Ls_nl:      .ascii "\n"
Lrej_head:  .ascii "fldj reject code="
.set Lrej_head_len, . - Lrej_head
