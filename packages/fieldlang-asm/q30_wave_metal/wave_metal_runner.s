// Q30WAVE2 Metal gate runner — hand ARM64 only. libSystem: open/read/close/write/exit.
// Usage: wave_metal_runner <q30_wave.metallib> <wave_vectors.bin>
// Independent strict decode (remaining-length checked per record and per field,
// u64 checked width*height, trailing bytes forbidden), then for every record:
//   scalar -> byte+sat compare vs frozen want
//   NEON   -> byte+sat compare vs frozen want
//   Metal reps=2 offset 0  -> byte+sat compare (saturation reset per dispatch scored)
//   Metal reps=1 offset 4  -> byte+sat compare (encoder byte offset scored)
//   Metal reps=1 offset 28 -> byte+sat compare
//   Metal alias cur==prev  -> compared against live scalar alias reference
// out is fenced with 0xA5 guards on both sides; cur/prev custody is compared
// byte-for-byte after every implementation call. After the corpus: host checks
// count==0 -> 0 (no dispatch), negative dims -> -1, u32-overflowing product -> -1.
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
.extern _fl_q30_wave_metal
.extern _fl_wave_metal_init
.extern _fl_wave_metal_print_device
.extern _fl_wput

_main:
    sub sp, sp, #128
    cmp w0, #3
    b.lt Lfail
    mov x19, x1                     // argv (x19 callee-saved: restored via exit only)
    ldr x0, [x19, #8]
    bl _fl_wave_metal_init
    cbnz x0, Lfail
    bl _fl_wave_metal_print_device
    ldr x0, [x19, #16]
    mov x1, #0
    bl _open
    cmp w0, #0
    b.lt Lfail
    str x0, [sp, #0]
    adrp x1, _wfilebuf@PAGE
    add x1, x1, _wfilebuf@PAGEOFF
    mov x2, #0x40000
    bl _read
    cmp x0, #11
    b.lt Lfail
    mov x11, #0x40000
    cmp x0, x11
    b.ge Lfail                      // arena-full read = suspected truncation
    str x0, [sp, #8]
    ldr x0, [sp, #0]
    bl _close
    adrp x9, _wfilebuf@PAGE
    add x9, x9, _wfilebuf@PAGEOFF
    ldr x11, [sp, #8]
    add x11, x9, x11
    str x11, [sp, #80]              // strict file end
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
    str x9, [sp, #16]
    str xzr, [sp, #24]              // case counter
Lcase:
    ldr x9, [sp, #16]
    mov x17, #2
    bl Lneed
    ldrh w10, [x9], #2
    cbz w10, Lok
    uxtw x17, w10
    add x17, x17, #8
    bl Lneed
    add x9, x9, x10                 // skip name
    ldr w11, [x9], #4               // w
    ldr w12, [x9], #4               // h
    cmp w11, #0
    b.le Lfail
    cmp w12, #0
    b.le Lfail
    uxtw x16, w11
    uxtw x17, w12
    umulh x13, x16, x17
    cbnz x13, Lfail
    mul x13, x16, x17
    cbz x13, Lfail
    cmp x13, #0x4000                // arena: 16384 i32 max
    b.hi Lfail
    lsl x17, x13, #2
    add x17, x17, x17, lsl #1
    add x17, x17, #32               // 3 fields + 24B coeffs + 8B sat
    bl Lneed
    adrp x14, _wcfg@PAGE
    add x14, x14, _wcfg@PAGEOFF
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
    str x15, [sp, #32]              // bytes
    // out guards: below at _wout+32, above at _wout+64+bytes
    adrp x0, _wout@PAGE
    add x0, x0, _wout@PAGEOFF
    add x0, x0, #32
    bl Lgfill
    adrp x0, _wout@PAGE
    add x0, x0, _wout@PAGEOFF
    add x0, x0, #64
    ldr x15, [sp, #32]
    add x0, x0, x15
    bl Lgfill
    ldr x15, [sp, #32]
    str x9, [sp, #56]
    adrp x0, _wcur@PAGE
    add x0, x0, _wcur@PAGEOFF
    bl Lcopy
    str x9, [sp, #64]
    adrp x0, _wprev@PAGE
    add x0, x0, _wprev@PAGEOFF
    ldr x15, [sp, #32]
    bl Lcopy
    str x9, [sp, #40]               // frozen want
    ldr x15, [sp, #32]
    add x9, x9, x15
    ldr x10, [x9], #8
    str x10, [sp, #48]              // frozen sat
    str x9, [sp, #16]
    // scalar
    bl Lcust_save
    bl Lcall_scalar
    bl Lcheck_frozen
    bl Lcust_chk
    cbnz w0, Lfail
    // NEON
    bl Lcall_neon
    bl Lcheck_frozen
    bl Lcust_chk
    cbnz w0, Lfail
    // Metal: reps=2 offset 0 (repeated dispatch + per-dispatch sat reset scored)
    mov x4, #0
    mov x5, #2
    bl Lcall_metal
    bl Lcheck_frozen
    bl Lcust_chk
    cbnz w0, Lfail
    // Metal: offset 4
    mov x4, #4
    mov x5, #1
    bl Lcall_metal
    bl Lcheck_frozen
    // Metal: offset 28
    mov x4, #28
    mov x5, #1
    bl Lcall_metal
    bl Lcheck_frozen
    bl Lcust_chk
    cbnz w0, Lfail
    // alias cur==prev: live scalar reference, then Metal must match it exactly
    ldr x9, [sp, #56]
    adrp x0, _walias@PAGE
    add x0, x0, _walias@PAGEOFF
    ldr x15, [sp, #32]
    bl Lcopy
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    adrp x1, _walias@PAGE
    add x1, x1, _walias@PAGEOFF
    mov x2, x1
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    bl _fl_q30_wave_scalar
    str x0, [sp, #72]               // alias reference sat
    adrp x9, _wout@PAGE
    add x9, x9, _wout@PAGEOFF
    add x9, x9, #64
    adrp x0, _waliasref@PAGE
    add x0, x0, _waliasref@PAGEOFF
    ldr x15, [sp, #32]
    bl Lcopy
    bl Lscrub_out
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    adrp x1, _walias@PAGE
    add x1, x1, _walias@PAGEOFF
    mov x2, x1
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    mov x4, #0
    mov x5, #1
    bl _fl_q30_wave_metal
    ldr x10, [sp, #72]
    cmp x0, x10
    b.ne Lfail
    adrp x0, _wout@PAGE
    add x0, x0, _wout@PAGEOFF
    add x0, x0, #64
    adrp x1, _waliasref@PAGE
    add x1, x1, _waliasref@PAGEOFF
    ldr x2, [sp, #32]
    bl Lcmp
    cbnz w0, Lfail
    // guards intact after every implementation call of this record
    adrp x0, _wout@PAGE
    add x0, x0, _wout@PAGEOFF
    add x0, x0, #32
    bl Lgchk
    cbnz w0, Lfail
    adrp x0, _wout@PAGE
    add x0, x0, _wout@PAGEOFF
    add x0, x0, #64
    ldr x15, [sp, #32]
    add x0, x0, x15
    bl Lgchk
    cbnz w0, Lfail
    ldr x10, [sp, #24]
    add x10, x10, #1
    str x10, [sp, #24]
    b Lcase

// ---- helpers (runner-internal calling conventions noted per label) --------
// x9=cursor, x17=needed bytes -> Lfail on overrun. Clobbers x16/x17.
Lneed:
    ldr x16, [sp, #80]
    add x17, x9, x17
    cmp x17, x16
    b.hi Lfail
    ret
// x0=dst, x9=src(advances), x15=bytes
Lcopy:
    cbz x15, 2f
1:  ldrb w10, [x9], #1
    strb w10, [x0], #1
    subs x15, x15, #1
    b.ne 1b
2:  ret
// x0,x1,x2=bytes -> w0 mismatch flag
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
// x0=guard base: 32 bytes 0xA5
Lgfill:
    mov x11, #32
    mov w12, #0xA5
6:  strb w12, [x0], #1
    subs x11, x11, #1
    b.ne 6b
    ret
// x0=guard base -> w0 corruption flag
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
// custody: snapshot cur/prev arenas (bytes from [sp,#32+16] of caller frame? no —
// runner keeps a single frame; [sp,#32] is live here because helpers do not push).
Lcust_save:
    ldr x2, [sp, #32]
    adrp x0, _wcur@PAGE
    add x0, x0, _wcur@PAGEOFF
    adrp x12, _wcustcur@PAGE
    add x12, x12, _wcustcur@PAGEOFF
    mov x11, x2
    cbz x11, 11f
10: ldrb w13, [x0], #1
    strb w13, [x12], #1
    subs x11, x11, #1
    b.ne 10b
11: adrp x1, _wprev@PAGE
    add x1, x1, _wprev@PAGEOFF
    adrp x12, _wcustprev@PAGE
    add x12, x12, _wcustprev@PAGEOFF
    mov x11, x2
    cbz x11, 13f
12: ldrb w13, [x1], #1
    strb w13, [x12], #1
    subs x11, x11, #1
    b.ne 12b
13: ret
// -> w0 custody violation flag
Lcust_chk:
    ldr x2, [sp, #32]
    adrp x0, _wcur@PAGE
    add x0, x0, _wcur@PAGEOFF
    adrp x12, _wcustcur@PAGE
    add x12, x12, _wcustcur@PAGEOFF
    mov x11, x2
    cbz x11, 15f
14: ldrb w13, [x0], #1
    ldrb w14, [x12], #1
    cmp w13, w14
    b.ne 17f
    subs x11, x11, #1
    b.ne 14b
15: adrp x1, _wprev@PAGE
    add x1, x1, _wprev@PAGEOFF
    adrp x12, _wcustprev@PAGE
    add x12, x12, _wcustprev@PAGEOFF
    mov x11, x2
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
// scrub out body to 0xC3 so a stale previous result cannot satisfy the compare
Lscrub_out:
    adrp x0, _wout@PAGE
    add x0, x0, _wout@PAGEOFF
    add x0, x0, #64
    ldr x11, [sp, #32]
    mov w12, #0xC3
    cbz x11, 20f
19: strb w12, [x0], #1
    subs x11, x11, #1
    b.ne 19b
20: ret
// call wrappers: each scrubs out, then invokes with cfg/cur/prev/out.
Lcall_scalar:
    mov x7, x30
    bl Lscrub_out
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    adrp x1, _wcur@PAGE
    add x1, x1, _wcur@PAGEOFF
    adrp x2, _wprev@PAGE
    add x2, x2, _wprev@PAGEOFF
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    mov x30, x7
    b _fl_q30_wave_scalar
Lcall_neon:
    mov x7, x30
    bl Lscrub_out
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    adrp x1, _wcur@PAGE
    add x1, x1, _wcur@PAGEOFF
    adrp x2, _wprev@PAGE
    add x2, x2, _wprev@PAGEOFF
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    mov x30, x7
    b _fl_q30_wave_neon
// x4=offset, x5=reps preserved into the call
Lcall_metal:
    mov x7, x30
    mov x15, x4
    mov x16, x5
    bl Lscrub_out
    mov x4, x15
    mov x5, x16
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    adrp x1, _wcur@PAGE
    add x1, x1, _wcur@PAGEOFF
    adrp x2, _wprev@PAGE
    add x2, x2, _wprev@PAGEOFF
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    mov x30, x7
    b _fl_q30_wave_metal
// x0=sat from implementation -> frozen want+sat comparison (Lfail on mismatch)
Lcheck_frozen:
    ldr x10, [sp, #48]
    cmp x0, x10
    b.ne Lfail
    mov x7, x30
    adrp x0, _wout@PAGE
    add x0, x0, _wout@PAGEOFF
    add x0, x0, #64
    ldr x1, [sp, #40]
    ldr x2, [sp, #32]
    bl Lcmp
    mov x30, x7
    cbnz w0, Lfail
    ret

Lok:
    ldr x16, [sp, #80]
    cmp x9, x16
    b.ne Lfail                      // trailing bytes forbidden
    ldr x10, [sp, #24]
    cmp x10, #138
    b.ne Lfail
    // host adversarial checks (no fixture needed):
    // count==0 -> 0 without any dispatch or write
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    str wzr, [x0]
    mov w1, #5
    str w1, [x0, #4]
    adrp x1, _wcur@PAGE
    add x1, x1, _wcur@PAGEOFF
    adrp x2, _wprev@PAGE
    add x2, x2, _wprev@PAGEOFF
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    mov x4, #0
    mov x5, #1
    bl _fl_q30_wave_metal
    cbnz x0, Lfail
    // negative width -> -1
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    mov w1, #-1
    str w1, [x0]
    mov w1, #4
    str w1, [x0, #4]
    adrp x1, _wcur@PAGE
    add x1, x1, _wcur@PAGEOFF
    adrp x2, _wprev@PAGE
    add x2, x2, _wprev@PAGEOFF
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    mov x4, #0
    mov x5, #1
    bl _fl_q30_wave_metal
    cmn x0, #1
    b.ne Lfail
    // u32-overflowing product 65536*65536 -> -1 (host checked u64 count)
    adrp x0, _wcfg@PAGE
    add x0, x0, _wcfg@PAGEOFF
    mov w1, #0x10000
    str w1, [x0]
    str w1, [x0, #4]
    adrp x1, _wcur@PAGE
    add x1, x1, _wcur@PAGEOFF
    adrp x2, _wprev@PAGE
    add x2, x2, _wprev@PAGEOFF
    adrp x3, _wout@PAGE
    add x3, x3, _wout@PAGEOFF
    add x3, x3, #64
    mov x4, #0
    mov x5, #1
    bl _fl_q30_wave_metal
    cmn x0, #1
    b.ne Lfail
    adrp x0, Lokmsg@PAGE
    add x0, x0, Lokmsg@PAGEOFF
    bl _fl_wput
    mov x0, #0
    b Lexit
Lfail:
    adrp x0, Lbadmsg@PAGE
    add x0, x0, Lbadmsg@PAGEOFF
    bl _fl_wput
    mov x0, #1
Lexit:
    bl _exit

.section __DATA,__bss
.p2align 4
_wfilebuf: .space 262144
_wcur:     .space 65536
_wprev:    .space 65536
_wout:     .space 65792
_walias:   .space 65536
_waliasref: .space 65536
_wcustcur:  .space 65536
_wcustprev: .space 65536
_wcfg:     .space 32
.section __TEXT,__cstring,cstring_literals
Lokmsg: .asciz "wave_metal_runner: 138 Q30WAVE2 scalar=neon=metal ok (offsets 0/4/28, alias, reps, count0/neg/overflow)\n"
Lbadmsg: .asciz "wave_metal_runner: failure\n"
