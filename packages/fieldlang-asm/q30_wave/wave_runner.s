// Q30WAVE2 hand ARM64 gate runner. libSystem: open/read/close/write/exit only.
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
    // magic Q30WAVE2\0
    ldr x10, [x9]
    movz x11, #0x3351
    movk x11, #0x5730, lsl #16
    movk x11, #0x5641, lsl #32
    movk x11, #0x3245, lsl #48
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
    ldr x15, [x9], #8
    str x15, [x14, #8]
    ldr x15, [x9], #8
    str x15, [x14, #16]
    ldr x15, [x9], #8
    str x15, [x14, #24]
    uxtw x15, w13
    lsl x15, x15, #2
    str x15, [sp, #32]             // bytes
    // out周囲 guard: 下位=_outbuf+32(out基址の直下), 上位=_outbuf+65+bytes(+1B経路の直上)。
    // out基址=_outbuf+64 故、下方走査も上方越境も 0xA5 破壊として現れる。
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #32
    bl Lgfill
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #65
    ldr x15, [sp, #32]
    add x0, x0, x15
    bl Lgfill
    ldr x15, [sp, #32]
    // copy cur, prev into bounded aligned arenas; retain vector offsets.
    str x9, [sp, #56]
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    bl Lcopy
    str x9, [sp, #64]
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
    // custody: cur/prev 全bytesを独立複写。入力不変契約を呼出毎に検す。
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    adrp x1, _prevbuf@PAGE
    add x1, x1, _prevbuf@PAGEOFF
    ldr x2, [sp, #32]
    bl Lcust_save
    // scalar(cfg,cur,prev,out)
    adrp x0, _cfg@PAGE
    add x0, x0, _cfg@PAGEOFF
    adrp x1, _curbuf@PAGE
    add x1, x1, _curbuf@PAGEOFF
    adrp x2, _prevbuf@PAGE
    add x2, x2, _prevbuf@PAGEOFF
    adrp x3, _outbuf@PAGE
    add x3, x3, _outbuf@PAGEOFF
    add x3, x3, #64
    bl _fl_q30_wave_scalar
    ldr x10, [sp, #48]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #64
    ldr x1, [sp, #40]
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    adrp x1, _prevbuf@PAGE
    add x1, x1, _prevbuf@PAGEOFF
    ldr x2, [sp, #32]
    bl Lcust_chk
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
    add x3, x3, #64
    bl _fl_q30_wave_neon
    ldr x10, [sp, #48]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #64
    ldr x1, [sp, #40]
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    adrp x1, _prevbuf@PAGE
    add x1, x1, _prevbuf@PAGEOFF
    ldr x2, [sp, #32]
    bl Lcust_chk
    cbnz w0, Lfail
    // Repeat both implementations at deliberately +1-byte addresses.
    ldr x9, [sp, #56]
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    add x0, x0, #1
    ldr x15, [sp, #32]
    bl Lcopy
    ldr x9, [sp, #64]
    adrp x0, _prevbuf@PAGE
    add x0, x0, _prevbuf@PAGEOFF
    add x0, x0, #1
    ldr x15, [sp, #32]
    bl Lcopy
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    add x0, x0, #1
    adrp x1, _prevbuf@PAGE
    add x1, x1, _prevbuf@PAGEOFF
    add x1, x1, #1
    ldr x2, [sp, #32]
    bl Lcust_save
    adrp x0, _cfg@PAGE
    add x0, x0, _cfg@PAGEOFF
    adrp x1, _curbuf@PAGE
    add x1, x1, _curbuf@PAGEOFF
    add x1, x1, #1
    adrp x2, _prevbuf@PAGE
    add x2, x2, _prevbuf@PAGEOFF
    add x2, x2, #1
    adrp x3, _outbuf@PAGE
    add x3, x3, _outbuf@PAGEOFF
    add x3, x3, #64
    add x3, x3, #1
    bl _fl_q30_wave_scalar
    ldr x10, [sp, #48]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #64
    add x0, x0, #1
    ldr x1, [sp, #40]
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    add x0, x0, #1
    adrp x1, _prevbuf@PAGE
    add x1, x1, _prevbuf@PAGEOFF
    add x1, x1, #1
    ldr x2, [sp, #32]
    bl Lcust_chk
    cbnz w0, Lfail
    adrp x0, _cfg@PAGE
    add x0, x0, _cfg@PAGEOFF
    adrp x1, _curbuf@PAGE
    add x1, x1, _curbuf@PAGEOFF
    add x1, x1, #1
    adrp x2, _prevbuf@PAGE
    add x2, x2, _prevbuf@PAGEOFF
    add x2, x2, #1
    adrp x3, _outbuf@PAGE
    add x3, x3, _outbuf@PAGEOFF
    add x3, x3, #64
    add x3, x3, #1
    bl _fl_q30_wave_neon
    ldr x10, [sp, #48]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #64
    add x0, x0, #1
    ldr x1, [sp, #40]
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    // alias live: curとprevを同一bufferへ束ね、scalarを基準にNEONを差分照合。
    // 凍結期待値はprev別buffer前提故、基準=scalar実走。
    adrp x0, _curbuf@PAGE
    add x0, x0, _curbuf@PAGEOFF
    add x0, x0, #1
    adrp x1, _prevbuf@PAGE
    add x1, x1, _prevbuf@PAGEOFF
    add x1, x1, #1
    ldr x2, [sp, #32]
    bl Lcust_chk
    cbnz w0, Lfail
    ldr x9, [sp, #56]
    adrp x0, _aliasbuf@PAGE
    add x0, x0, _aliasbuf@PAGEOFF
    ldr x15, [sp, #32]
    bl Lcopy
    adrp x0, _aliasbuf@PAGE
    add x0, x0, _aliasbuf@PAGEOFF
    mov x1, x0
    ldr x2, [sp, #32]
    bl Lcust_save
    adrp x0, _cfg@PAGE
    add x0, x0, _cfg@PAGEOFF
    adrp x1, _aliasbuf@PAGE
    add x1, x1, _aliasbuf@PAGEOFF
    mov x2, x1
    adrp x3, _outbuf@PAGE
    add x3, x3, _outbuf@PAGEOFF
    add x3, x3, #64
    bl _fl_q30_wave_scalar
    str x0, [sp, #72]               // alias 基準 sat
    adrp x0, _aliasbuf@PAGE
    add x0, x0, _aliasbuf@PAGEOFF
    mov x1, x0
    ldr x2, [sp, #32]
    bl Lcust_chk
    cbnz w0, Lfail
    adrp x9, _outbuf@PAGE
    add x9, x9, _outbuf@PAGEOFF
    add x9, x9, #64
    adrp x0, _aliasref@PAGE
    add x0, x0, _aliasref@PAGEOFF
    ldr x15, [sp, #32]
    bl Lcopy
    ldr x9, [sp, #56]
    adrp x0, _aliasbuf@PAGE
    add x0, x0, _aliasbuf@PAGEOFF
    ldr x15, [sp, #32]
    bl Lcopy
    adrp x0, _cfg@PAGE
    add x0, x0, _cfg@PAGEOFF
    adrp x1, _aliasbuf@PAGE
    add x1, x1, _aliasbuf@PAGEOFF
    mov x2, x1
    adrp x3, _outbuf@PAGE
    add x3, x3, _outbuf@PAGEOFF
    add x3, x3, #64
    bl _fl_q30_wave_neon
    ldr x10, [sp, #72]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #64
    adrp x1, _aliasref@PAGE
    add x1, x1, _aliasref@PAGEOFF
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    adrp x0, _aliasbuf@PAGE
    add x0, x0, _aliasbuf@PAGEOFF
    mov x1, x0
    ldr x2, [sp, #32]
    bl Lcust_chk
    cbnz w0, Lfail
    // guard 健全確認: 全呼出(aligned/+1B/alias, scalar+NEON)後に一度。
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #32
    bl Lgchk
    cbnz w0, Lfail
    adrp x0, _outbuf@PAGE
    add x0, x0, _outbuf@PAGEOFF
    add x0, x0, #65
    ldr x15, [sp, #32]
    add x0, x0, x15
    bl Lgchk
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
// x0=guard址。2 32バイトを 0xA5 で敷く。
Lgfill:
    mov x11, #32
    mov w12, #0xA5
6:  strb w12, [x0], #1
    subs x11, x11, #1
    b.ne 6b
    ret
// x0=guard址 -> w0=破壊 boolean
Lgchk:
    mov x11, #32
    mov w12, #0xA5
7:  ldrb w13, [x0], #1
    cmp w13, w12
    b.ne 8f
    subs x11, x11, #1
    b.ne 7b
    mov w0, #0
    ret
8:  mov w0, #1
    ret
// x0=cur, x1=prev, x2=bytes -> 独立 custody 領域へ複写。
Lcust_save:
    mov x11, x2
    adrp x12, _custcur@PAGE
    add x12, x12, _custcur@PAGEOFF
    cbz x11, 11f
10: ldrb w13, [x0], #1
    strb w13, [x12], #1
    subs x11, x11, #1
    b.ne 10b
11: mov x11, x2
    adrp x12, _custprev@PAGE
    add x12, x12, _custprev@PAGEOFF
    cbz x11, 13f
12: ldrb w13, [x1], #1
    strb w13, [x12], #1
    subs x11, x11, #1
    b.ne 12b
13: ret
// x0=cur, x1=prev, x2=bytes -> w0=入力改変 boolean
Lcust_chk:
    mov x11, x2
    adrp x12, _custcur@PAGE
    add x12, x12, _custcur@PAGEOFF
    cbz x11, 15f
14: ldrb w13, [x0], #1
    ldrb w14, [x12], #1
    cmp w13, w14
    b.ne 17f
    subs x11, x11, #1
    b.ne 14b
15: mov x11, x2
    adrp x12, _custprev@PAGE
    add x12, x12, _custprev@PAGEOFF
    cbz x11, 16f
18: ldrb w13, [x1], #1
    ldrb w14, [x12], #1
    cmp w13, w14
    b.ne 17f
    subs x11, x11, #1
    b.ne 18b
16: mov w0, #0
    ret
17: mov w0, #1
    ret
Lok:
    ldr x10, [sp, #24]
    cmp x10, #138
    b.ne Lfail
    adrp x1, Lokmsg@PAGE
    add x1, x1, Lokmsg@PAGEOFF
    mov x0, #1
    mov x2, #29
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
_aliasbuf: .space 65536
_aliasref: .space 65536
_custcur:  .space 65536
_custprev: .space 65536
_cfg:     .space 32
.section __TEXT,__cstring,cstring_literals
Lokmsg: .asciz "wave_runner: 138 Q30WAVE2 ok\n"
Lbadmsg: .asciz "wave_runner: failure\n"
