// emit.s — fieldlang-asm V0 (narigo lane)
// _fl_emit per CONTRACT.md v1 (frozen). arm64 macOS, hand asm.
// x0=token ptr (16B records {u64 kind,u64 value})
// x1=token count · x2=out buf · x3=cap bytes
// x4=header params ptr 40B {u32 w,h; u32 c,dt,damp,dx; u64 seed; u32 range; u32 n_slots}
// ret x0 = bytes written; <0 = -(code): 1=syntax 2=buffull 4=arg-range

.global _fl_emit
.align 2

_fl_emit:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    mov     x19, x0             // token base
    mov     x20, x1             // token count
    mov     x21, x2             // out base
    mov     x22, x3             // cap
    mov     x23, x4             // params
    mov     x24, #0             // token index

    // overridable header fields from params
    ldr     w25, [x23]          // w
    ldr     w26, [x23, #4]      // h
    ldr     x27, [x23, #24]     // seed
    ldr     w28, [x23, #36]     // n_slots

    cbz     x20, Lsyntax        // lexer always emits EOF record
    ldr     x5, [x19]           // kind of token 0
    cmp     x5, #9              // 界?
    b.ne    Lhdr
    mov     x24, #1
    bl      Lget_u32
    mov     w25, w0             // w
    bl      Lget_u32
    mov     w26, w0             // h
    bl      Lget_u32
    mov     w28, w0             // n_slots
    bl      Lget_int
    mov     x27, x0             // seed (full u64)

Lhdr:
    cmp     x22, #48
    b.lo    Lbuffull
    mov     w9, #0x4C46         // "FL"
    movk    w9, #0x4A44, lsl #16 // "DJ" → 0x4A444C46 LE
    str     w9, [x21]
    mov     w9, #1
    str     w9, [x21, #4]       // version
    str     w25, [x21, #8]      // w
    str     w26, [x21, #12]     // h
    ldr     w9, [x23, #8]
    str     w9, [x21, #16]      // c bits
    ldr     w9, [x23, #12]
    str     w9, [x21, #20]      // dt bits
    ldr     w9, [x23, #16]
    str     w9, [x21, #24]      // damping bits
    ldr     w9, [x23, #20]
    str     w9, [x21, #28]      // dx bits
    str     x27, [x21, #32]     // seed
    ldr     w9, [x23, #32]
    str     w9, [x21, #40]      // range bits
    str     w28, [x21, #44]     // n_slots
    mov     x8, #48             // out offset

Lloop:
    cmp     x24, x20
    b.hs    Lsyntax             // ran out with no EOF
    lsl     x9, x24, #4
    add     x9, x19, x9
    ldr     x10, [x9]           // kind
    add     x24, x24, #1
    cbz     x10, Ldone          // EOF
    cmp     x10, #6
    b.hi    Lsyntax             // INT(7) or 界(9) at op position
    // emit tag byte
    add     x11, x8, #1
    cmp     x11, x22
    b.hi    Lbuffull
    strb    w10, [x21, x8]
    mov     x8, x11
    cmp     x10, #1
    b.eq    Ltag1
    cmp     x10, #3
    b.eq    Ltag_one
    cmp     x10, #5
    b.hs    Lvar                // 5=束, 6=寫: same shape {u32, u32 len, len×u32}
    // 2=撃, 4=縛: three u32
Ltag3:
    bl      Lget_u32
    bl      Lput_u32
    bl      Lget_u32
    bl      Lput_u32
    bl      Lget_u32
    bl      Lput_u32
    b       Lloop

Ltag1:                          // 種 {slot u32, seed u64}
    bl      Lget_u32
    bl      Lput_u32
    bl      Lget_int            // seed: full u64, no range check
    add     x10, x8, #8
    cmp     x10, x22
    b.hi    Lbuffull
    str     x0, [x21, x8]
    mov     x8, x10
    b       Lloop

Ltag_one:                       // 歩 {count u32}
    bl      Lget_u32
    bl      Lput_u32
    b       Lloop

Lvar:                           // 束/寫 {u32, u32 n, n×u32}
    bl      Lget_u32
    bl      Lput_u32
    bl      Lget_u32
    mov     x12, x0             // n
    bl      Lput_u32
    cbz     x12, Lloop
Lvar_loop:
    bl      Lget_u32
    bl      Lput_u32
    subs    x12, x12, #1
    b.ne    Lvar_loop
    b       Lloop

// --- leaf subroutines (sp untouched; error branches restore via Lret) ---

Lget_int:                       // → x0 = u64 value; INT token required
    cmp     x24, x20
    b.hs    Lsyntax
    lsl     x9, x24, #4
    add     x9, x19, x9
    ldr     x10, [x9]
    cmp     x10, #7
    b.ne    Lsyntax
    ldr     x0, [x9, #8]
    add     x24, x24, #1
    ret

Lget_u32:                       // → x0 = value, must fit u32
    cmp     x24, x20
    b.hs    Lsyntax
    lsl     x9, x24, #4
    add     x9, x19, x9
    ldr     x10, [x9]
    cmp     x10, #7
    b.ne    Lsyntax
    ldr     x0, [x9, #8]
    lsr     x9, x0, #32
    cbnz    x9, Lrange
    add     x24, x24, #1
    ret

Lput_u32:                       // write w0 LE at out offset, bounds-checked
    add     x10, x8, #4
    cmp     x10, x22
    b.hi    Lbuffull
    str     w0, [x21, x8]
    mov     x8, x10
    ret

Ldone:
    mov     x0, x8
    b       Lret
Lsyntax:
    mov     x0, #-1
    b       Lret
Lbuffull:
    mov     x0, #-2
    b       Lret
Lrange:
    mov     x0, #-4
Lret:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     x29, x30, [sp], #96
    ret
