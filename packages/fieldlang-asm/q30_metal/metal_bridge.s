// Hand-written ARM64 host bridge to Metal. No C/ObjC/Swift/Rust source exists
// on this path: every Objective-C message below is an explicit objc_msgSend
// call emitted by hand. See CONTRACT.md.
.text
.p2align 2

.globl _fl_metal_init
.globl _fl_q30_decay_metal
.globl _fl_metal_print_device
.globl _fl_q30_metal_bench
.globl _fl_put
.globl _fl_putdec

// ---- tiny output helpers (write(2), no libc formatting) -------------------
// _fl_put(x0 = cstr)
_fl_put:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]
    mov x19, x0
    mov x1, #0
Lput_len:
    ldrb w2, [x19, x1]
    cbz w2, Lput_go
    add x1, x1, #1
    b Lput_len
Lput_go:
    mov x2, x1
    mov x1, x19
    mov x0, #1
    bl _write
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// _fl_putdec(x0 = signed value)
_fl_putdec:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0
    add x20, sp, #56             // write digits backwards from here
    strb wzr, [x20]
    mov x2, #0
    cmp x19, #0
    b.ge Lputdec_abs
    mov x2, #1
    neg x19, x19
Lputdec_abs:
    mov x3, x2                   // negative flag
    mov x4, #10
Lputdec_loop:
    udiv x5, x19, x4
    msub x6, x5, x4, x19
    add w6, w6, #48
    sub x20, x20, #1
    strb w6, [x20]
    mov x19, x5
    cbnz x19, Lputdec_loop
    cbz x3, Lputdec_out
    mov w6, #45
    sub x20, x20, #1
    strb w6, [x20]
Lputdec_out:
    mov x0, x20
    bl _fl_put
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// ---- init -----------------------------------------------------------------
// _fl_metal_init(x0 = metallib path cstr) -> x0 = 0 ok, 1 fail
_fl_metal_init:
    stp x29, x30, [sp, #-112]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x20, x0
    bl _objc_autoreleasePoolPush
    mov x24, x0
    bl _MTLCreateSystemDefaultDevice
    cbz x0, Linit_fail
    mov x19, x0
    adrp x1, _g_dev@PAGE
    str x19, [x1, _g_dev@PAGEOFF]
    // NSString *p = [NSString stringWithUTF8String:path]
    mov x0, x20
    bl Lmake_nsstring
    mov x21, x0
    str xzr, [sp, #96]
    // lib = [dev newLibraryWithFile:p error:&err]
    adrp x0, Ls_newlib@PAGE
    add x0, x0, Ls_newlib@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    mov x2, x21
    add x3, sp, #96
    bl _objc_msgSend
    cbz x0, Linit_fail
    mov x21, x0
    // fn = [lib newFunctionWithName:@"q30_decay"]
    adrp x0, Ls_kernelname@PAGE
    add x0, x0, Ls_kernelname@PAGEOFF
    bl Lmake_nsstring
    mov x22, x0
    adrp x0, Ls_newfn@PAGE
    add x0, x0, Ls_newfn@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x21
    mov x2, x22
    bl _objc_msgSend
    cbz x0, Linit_fail
    mov x22, x0
    // pso = [dev newComputePipelineStateWithFunction:fn error:&err]
    adrp x0, Ls_newpso@PAGE
    add x0, x0, Ls_newpso@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    mov x2, x22
    add x3, sp, #96
    bl _objc_msgSend
    cbz x0, Linit_fail
    adrp x1, _g_pso@PAGE
    str x0, [x1, _g_pso@PAGEOFF]
    // queue = [dev newCommandQueue]
    adrp x0, Ls_newqueue@PAGE
    add x0, x0, Ls_newqueue@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    cbz x0, Linit_fail
    adrp x1, _g_queue@PAGE
    str x0, [x1, _g_queue@PAGEOFF]
    mov x0, x24
    bl _objc_autoreleasePoolPop
    mov x0, #0
    b Linit_out
Linit_fail:
    mov x0, x24
    bl _objc_autoreleasePoolPop
    mov x0, #1
Linit_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x29, x30, [sp], #112
    ret

// Lmake_nsstring(x0 = cstr) -> x0 = NSString*
Lmake_nsstring:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0
    adrp x0, Ls_NSString@PAGE
    add x0, x0, Ls_NSString@PAGEOFF
    bl _objc_getClass
    mov x20, x0
    adrp x0, Ls_strutf8@PAGE
    add x0, x0, Ls_strutf8@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x20
    mov x2, x19
    bl _objc_msgSend
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// ---- device evidence ------------------------------------------------------
// _fl_metal_print_device() : raw device name + supported GPU families
_fl_metal_print_device:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    bl _objc_autoreleasePoolPush
    mov x22, x0
    adrp x0, _g_dev@PAGE
    ldr x19, [x0, _g_dev@PAGEOFF]
    cbz x19, Lpd_out
    adrp x0, Lm_devname@PAGE
    add x0, x0, Lm_devname@PAGEOFF
    bl _fl_put
    adrp x0, Ls_name@PAGE
    add x0, x0, Ls_name@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    mov x20, x0
    adrp x0, Ls_utf8string@PAGE
    add x0, x0, Ls_utf8string@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x20
    bl _objc_msgSend
    bl _fl_put
    adrp x0, Lm_nl@PAGE
    add x0, x0, Lm_nl@PAGEOFF
    bl _fl_put
    // families
    adrp x0, Lm_fam@PAGE
    add x0, x0, Lm_fam@PAGEOFF
    bl _fl_put
    adrp x0, Ls_supports@PAGE
    add x0, x0, Ls_supports@PAGEOFF
    bl _sel_registerName
    mov x21, x0
    adrp x20, Lfamilies@PAGE
    add x20, x20, Lfamilies@PAGEOFF
Lpd_loop:
    ldr w2, [x20]
    cbz w2, Lpd_done
    mov x0, x19
    mov x1, x21
    sxtw x2, w2
    bl _objc_msgSend
    cbz w0, Lpd_next
    ldr w0, [x20]
    sxtw x0, w0
    bl _fl_putdec
    adrp x0, Lm_sp@PAGE
    add x0, x0, Lm_sp@PAGEOFF
    bl _fl_put
Lpd_next:
    add x20, x20, #4
    b Lpd_loop
Lpd_done:
    adrp x0, Lm_nl@PAGE
    add x0, x0, Lm_nl@PAGEOFF
    bl _fl_put
Lpd_out:
    mov x0, x22
    bl _objc_autoreleasePoolPop
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x29, x30, [sp], #64
    ret

// ---- product path ---------------------------------------------------------
// _fl_q30_decay_metal(x0 = cells, x1 = count, w2 = coeff) -> x0 = saturations
// returns -1 if the GPU path could not complete (there is no CPU fallback).
_fl_q30_decay_metal:
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0                    // cells
    mov x20, x1                    // count
    str w2, [sp, #96]              // params.coeff
    str w1, [sp, #100]             // params.count
    cbz x20, Ldm_zero
    bl _objc_autoreleasePoolPush
    mov x28, x0
    adrp x0, _g_dev@PAGE
    ldr x21, [x0, _g_dev@PAGEOFF]
    cbz x21, Ldm_fail
    // cellbuf = [dev newBufferWithBytes:cells length:count*4 options:0]
    adrp x0, Ls_newbufbytes@PAGE
    add x0, x0, Ls_newbufbytes@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x21
    mov x2, x19
    lsl x3, x20, #2
    mov x4, #0
    bl _objc_msgSend
    cbz x0, Ldm_fail
    mov x22, x0
    // satbuf = [dev newBufferWithLength:4 options:0]
    adrp x0, Ls_newbuflen@PAGE
    add x0, x0, Ls_newbuflen@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x21
    mov x2, #4
    mov x3, #0
    bl _objc_msgSend
    cbz x0, Ldm_fail
    mov x23, x0
    // zero the saturation counter through its shared contents pointer
    mov x0, x23
    bl Lcontents
    mov x24, x0
    str wzr, [x24]
    mov x0, x22
    mov x1, x23
    mov x2, x20
    add x3, sp, #96
    bl Ldispatch_once             // -> x0 = 0 ok / 1 fail
    cbnz x0, Ldm_fail
    // copy the field back out of the shared buffer
    mov x0, x22
    bl Lcontents
    mov x1, x20
Ldm_copy:
    cbz x1, Ldm_copied
    ldr w2, [x0], #4
    str w2, [x19], #4
    sub x1, x1, #1
    b Ldm_copy
Ldm_copied:
    ldr w25, [x24]
    mov x0, x28
    bl _objc_autoreleasePoolPop
    mov x0, x25
    b Ldm_out
Ldm_zero:
    mov x0, #0
    b Ldm_out
Ldm_fail:
    mov x0, x28
    bl _objc_autoreleasePoolPop
    mov x0, #-1
Ldm_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #160
    ret

// Lcontents(x0 = MTLBuffer) -> x0 = void*
Lcontents:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]
    mov x19, x0
    adrp x0, Ls_contents@PAGE
    add x0, x0, Ls_contents@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// Ldispatch_once(x0 = cellbuf, x1 = satbuf, x2 = count, x3 = params*)
//   -> x0 = 0 ok / 1 fail. Real queue, real encoder, real completion check.
Ldispatch_once:
    stp x29, x30, [sp, #-128]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    adrp x0, _g_queue@PAGE
    ldr x23, [x0, _g_queue@PAGEOFF]
    cbz x23, Ldo_fail
    adrp x0, Ls_cmdbuf@PAGE
    add x0, x0, Ls_cmdbuf@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x23
    bl _objc_msgSend
    cbz x0, Ldo_fail
    mov x23, x0                    // command buffer
    adrp x0, Ls_encoder@PAGE
    add x0, x0, Ls_encoder@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x23
    bl _objc_msgSend
    cbz x0, Ldo_fail
    mov x24, x0                    // encoder
    // [enc setComputePipelineState:pso]
    adrp x0, Ls_setpso@PAGE
    add x0, x0, Ls_setpso@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    adrp x2, _g_pso@PAGE
    ldr x2, [x2, _g_pso@PAGEOFF]
    mov x0, x24
    bl _objc_msgSend
    // setBuffer:offset:atIndex:
    adrp x0, Ls_setbuf@PAGE
    add x0, x0, Ls_setbuf@PAGEOFF
    bl _sel_registerName
    mov x25, x0
    mov x0, x24
    mov x1, x25
    mov x2, x19
    mov x3, #0
    mov x4, #0
    bl _objc_msgSend
    mov x0, x24
    mov x1, x25
    mov x2, x20
    mov x3, #0
    mov x4, #1
    bl _objc_msgSend
    // setBytes:length:atIndex: (params)
    adrp x0, Ls_setbytes@PAGE
    add x0, x0, Ls_setbytes@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x24
    mov x2, x22
    mov x3, #8
    mov x4, #2
    bl _objc_msgSend
    // ObjC ABI passes each MTLSize aggregate indirectly (x2/x3 are pointers).
    // Values here are (count,1,1) and exactly (64,1,1).
    str x21, [sp, #80]
    mov x4, #1
    str x4, [sp, #88]
    str x4, [sp, #96]
    mov x5, #64
    str x5, [sp, #104]
    str x4, [sp, #112]
    str x4, [sp, #120]
    adrp x0, Ls_dispatch@PAGE
    add x0, x0, Ls_dispatch@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x24
    add x2, sp, #80
    add x3, sp, #104
    bl _objc_msgSend
    // [enc endEncoding]
    adrp x0, Ls_endenc@PAGE
    add x0, x0, Ls_endenc@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x24
    bl _objc_msgSend
    // [cmd commit]; [cmd waitUntilCompleted]; status must be Completed(4)
    adrp x0, Ls_commit@PAGE
    add x0, x0, Ls_commit@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x23
    bl _objc_msgSend
    adrp x0, Ls_wait@PAGE
    add x0, x0, Ls_wait@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x23
    bl _objc_msgSend
    adrp x0, Ls_status@PAGE
    add x0, x0, Ls_status@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x23
    bl _objc_msgSend
    cmp x0, #4
    b.ne Ldo_fail
    // Success is audit-visible only after the runtime's error property is nil.
    adrp x0, Ls_error@PAGE
    add x0, x0, Ls_error@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x23
    bl _objc_msgSend
    cbnz x0, Ldo_fail
    adrp x0, Lm_status4@PAGE
    add x0, x0, Lm_status4@PAGEOFF
    bl _fl_put
    adrp x0, Lm_errornil@PAGE
    add x0, x0, Lm_errornil@PAGEOFF
    bl _fl_put
    mov x0, #0
    b Ldo_out
Ldo_fail:
    mov x0, #1
Ldo_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x29, x30, [sp], #128
    ret

// _fl_q30_metal_bench(x0 = cells, x1 = count, w2 = coeff, x3 = reps)
//   -> x0 = 0 ok / 1 fail. The field buffer stays GPU-resident across all
//   repetitions: one upload, `reps` dispatches, one readback.
_fl_q30_metal_bench:
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0
    mov x20, x1
    str w2, [sp, #96]
    str w1, [sp, #100]
    mov x26, x3
    bl _objc_autoreleasePoolPush
    mov x28, x0
    adrp x0, _g_dev@PAGE
    ldr x21, [x0, _g_dev@PAGEOFF]
    cbz x21, Lbn_fail
    adrp x0, Ls_newbufbytes@PAGE
    add x0, x0, Ls_newbufbytes@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x21
    mov x2, x19
    lsl x3, x20, #2
    mov x4, #0
    bl _objc_msgSend
    cbz x0, Lbn_fail
    mov x22, x0
    adrp x0, Ls_newbuflen@PAGE
    add x0, x0, Ls_newbuflen@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x21
    mov x2, #4
    mov x3, #0
    bl _objc_msgSend
    cbz x0, Lbn_fail
    mov x23, x0
    mov x0, x23
    bl Lcontents
    str wzr, [x0]
Lbn_loop:
    cbz x26, Lbn_done
    mov x0, x22
    mov x1, x23
    mov x2, x20
    add x3, sp, #96
    bl Ldispatch_once
    cbnz x0, Lbn_fail
    sub x26, x26, #1
    b Lbn_loop
Lbn_done:
    mov x0, x22
    bl Lcontents
    mov x1, x20
Lbn_copy:
    cbz x1, Lbn_copied
    ldr w2, [x0], #4
    str w2, [x19], #4
    sub x1, x1, #1
    b Lbn_copy
Lbn_copied:
    mov x0, x28
    bl _objc_autoreleasePoolPop
    mov x0, #0
    b Lbn_out
Lbn_fail:
    mov x0, x28
    bl _objc_autoreleasePoolPop
    mov x0, #1
Lbn_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #160
    ret

.data
.p2align 3
.globl _g_dev
.globl _g_queue
.globl _g_pso
_g_dev:   .quad 0
_g_queue: .quad 0
_g_pso:   .quad 0
Lfamilies: .word 1001,1002,1003,1004,1005,1006,1007,1008,1009,3001,3002,3003,5001,0

.section __TEXT,__cstring,cstring_literals
Ls_NSString:     .asciz "NSString"
Ls_strutf8:      .asciz "stringWithUTF8String:"
Ls_newlib:       .asciz "newLibraryWithFile:error:"
Ls_newfn:        .asciz "newFunctionWithName:"
Ls_newpso:       .asciz "newComputePipelineStateWithFunction:error:"
Ls_newqueue:     .asciz "newCommandQueue"
Ls_newbufbytes:  .asciz "newBufferWithBytes:length:options:"
Ls_newbuflen:    .asciz "newBufferWithLength:options:"
Ls_contents:     .asciz "contents"
Ls_cmdbuf:       .asciz "commandBuffer"
Ls_encoder:      .asciz "computeCommandEncoder"
Ls_setpso:       .asciz "setComputePipelineState:"
Ls_setbuf:       .asciz "setBuffer:offset:atIndex:"
Ls_setbytes:     .asciz "setBytes:length:atIndex:"
Ls_dispatch:     .asciz "dispatchThreads:threadsPerThreadgroup:"
Ls_endenc:       .asciz "endEncoding"
Ls_commit:       .asciz "commit"
Ls_wait:         .asciz "waitUntilCompleted"
Ls_status:       .asciz "status"
Ls_error:        .asciz "error"
Ls_name:         .asciz "name"
Ls_utf8string:   .asciz "UTF8String"
Ls_supports:     .asciz "supportsFamily:"
Ls_kernelname:   .asciz "q30_decay"
Lm_devname:      .asciz "metal device: "
Lm_fam:          .asciz "metal families supported: "
Lm_nl:           .asciz "\n"
Lm_sp:           .asciz " "
Lm_status4:      .asciz "metal command status: 4\n"
Lm_errornil:     .asciz "metal command error: nil\n"
