// Q30 wave hand-written ARM64 host bridge to Metal (kernel q30_wave).
// No Rust/C/ObjC-source/Swift on this path: every Objective-C message below is
// an explicit hand-emitted objc_msgSend call. See CONTRACT.md.
// Modeled on the accepted decay bridge (../q30_metal/metal_bridge.s).
.text
.p2align 2

.globl _fl_wave_metal_init
.globl _fl_wave_metal_print_device
.globl _fl_q30_wave_metal
.globl _fl_wput
.globl _fl_wputdec

// ---- tiny output helpers (write(2), no libc formatting) -------------------
// _fl_wput(x0 = cstr)
_fl_wput:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]
    mov x19, x0
    mov x1, #0
Lwput_len:
    ldrb w2, [x19, x1]
    cbz w2, Lwput_go
    add x1, x1, #1
    b Lwput_len
Lwput_go:
    mov x2, x1
    mov x1, x19
    mov x0, #1
    bl _write
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// _fl_wputdec(x0 = signed value)
_fl_wputdec:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0
    add x20, sp, #56
    strb wzr, [x20]
    mov x2, #0
    cmp x19, #0
    b.ge Lwpd_abs
    mov x2, #1
    neg x19, x19
Lwpd_abs:
    mov x3, x2
    mov x4, #10
Lwpd_loop:
    udiv x5, x19, x4
    msub x6, x5, x4, x19
    add w6, w6, #48
    sub x20, x20, #1
    strb w6, [x20]
    mov x19, x5
    cbnz x19, Lwpd_loop
    cbz x3, Lwpd_out
    mov w6, #45
    sub x20, x20, #1
    strb w6, [x20]
Lwpd_out:
    mov x0, x20
    bl _fl_wput
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// ---- init -----------------------------------------------------------------
// _fl_wave_metal_init(x0 = metallib path cstr) -> x0 = 0 ok, 1 fail
// Builds device, library, q30_wave pipeline state, command queue.
_fl_wave_metal_init:
    stp x29, x30, [sp, #-112]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x20, x0
    bl _objc_autoreleasePoolPush
    mov x24, x0
    bl _MTLCreateSystemDefaultDevice
    cbz x0, Lwi_fail
    mov x19, x0
    adrp x1, _gw_dev@PAGE
    str x19, [x1, _gw_dev@PAGEOFF]
    mov x0, x20
    bl Lwv_nsstring
    mov x21, x0
    str xzr, [sp, #96]
    // lib = [dev newLibraryWithFile:p error:&err]
    adrp x0, Lw_newlib@PAGE
    add x0, x0, Lw_newlib@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    mov x2, x21
    add x3, sp, #96
    bl _objc_msgSend
    cbz x0, Lwi_libfail
    mov x21, x0
    // fn = [lib newFunctionWithName:@"q30_wave"]
    adrp x0, Lw_kernelname@PAGE
    add x0, x0, Lw_kernelname@PAGEOFF
    bl Lwv_nsstring
    mov x22, x0
    adrp x0, Lw_newfn@PAGE
    add x0, x0, Lw_newfn@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x21
    mov x2, x22
    bl _objc_msgSend
    cbz x0, Lwi_fail
    mov x22, x0
    // pso = [dev newComputePipelineStateWithFunction:fn error:&err]
    adrp x0, Lw_newpso@PAGE
    add x0, x0, Lw_newpso@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    mov x2, x22
    add x3, sp, #96
    bl _objc_msgSend
    cbz x0, Lwi_fail
    adrp x1, _gw_pso@PAGE
    str x0, [x1, _gw_pso@PAGEOFF]
    // queue = [dev newCommandQueue]
    adrp x0, Lw_newqueue@PAGE
    add x0, x0, Lw_newqueue@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    cbz x0, Lwi_fail
    adrp x1, _gw_queue@PAGE
    str x0, [x1, _gw_queue@PAGEOFF]
    mov x0, x24
    bl _objc_autoreleasePoolPop
    mov x0, #0
    b Lwi_out
Lwi_libfail:
    // NSError evidence: [[err localizedDescription] UTF8String] to fd 1.
    ldr x0, [sp, #96]
    cbz x0, Lwi_fail
    mov x19, x0
    adrp x0, Lwm_nserr@PAGE
    add x0, x0, Lwm_nserr@PAGEOFF
    bl _fl_wput
    adrp x0, Lw_locdesc@PAGE
    add x0, x0, Lw_locdesc@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    cbz x0, Lwi_fail
    mov x19, x0
    adrp x0, Lw_utf8string@PAGE
    add x0, x0, Lw_utf8string@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    cbz x0, Lwi_fail
    bl _fl_wput
    adrp x0, Lwm_nl@PAGE
    add x0, x0, Lwm_nl@PAGEOFF
    bl _fl_wput
Lwi_fail:
    mov x0, x24
    bl _objc_autoreleasePoolPop
    mov x0, #1
Lwi_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x29, x30, [sp], #112
    ret

// Lwv_nsstring(x0 = cstr) -> x0 = NSString*
Lwv_nsstring:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0
    adrp x0, Lw_NSString@PAGE
    add x0, x0, Lw_NSString@PAGEOFF
    bl _objc_getClass
    mov x20, x0
    adrp x0, Lw_strutf8@PAGE
    add x0, x0, Lw_strutf8@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x20
    mov x2, x19
    bl _objc_msgSend
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// ---- device evidence ------------------------------------------------------
_fl_wave_metal_print_device:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]
    bl _objc_autoreleasePoolPush
    mov x21, x0
    adrp x0, _gw_dev@PAGE
    ldr x19, [x0, _gw_dev@PAGEOFF]
    cbz x19, Lwpdv_out
    adrp x0, Lwm_devname@PAGE
    add x0, x0, Lwm_devname@PAGEOFF
    bl _fl_wput
    adrp x0, Lw_name@PAGE
    add x0, x0, Lw_name@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    mov x20, x0
    adrp x0, Lw_utf8string@PAGE
    add x0, x0, Lw_utf8string@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x20
    bl _objc_msgSend
    bl _fl_wput
    adrp x0, Lwm_nl@PAGE
    add x0, x0, Lwm_nl@PAGEOFF
    bl _fl_wput
Lwpdv_out:
    mov x0, x21
    bl _objc_autoreleasePoolPop
    ldp x19, x20, [sp, #16]
    ldr x21, [sp, #32]
    ldp x29, x30, [sp], #48
    ret

// ---- product path ---------------------------------------------------------
// _fl_q30_wave_metal(x0=cfg, x1=cur, x2=prev, x3=out, x4=byte_offset, x5=reps)
//   cfg = { i32 width; i32 height; i64 c_cur; i64 c_lap; i64 c_prev } (offsets 0,4,8,16,24)
//   byte_offset = encoder offset applied to buffers 0/1/2 (multiple of 4)
//   reps >= 1: buffers created once, `reps` real dispatches, saturation counter
//   reset before every dispatch, one readback. -> x0 = saturations, -1 = GPU failure.
//   width==0 or height==0: no read, no write, returns 0.
//   Host-side checked u64 count = width*height; negative dims or a product that
//   leaves u32 range are rejected (-1) before any Metal call.
// Frame: [16..80]=x19..x28, [96..127]=Params(32B), [128]=outbuf, [136]=satbuf,
//        [144]=satptr, [152]=out dst, [160]=cur src, [168]=prev src
_fl_q30_wave_metal:
    stp x29, x30, [sp, #-192]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    str x3, [sp, #152]
    str x1, [sp, #160]
    str x2, [sp, #168]
    mov x22, x4                    // byte offset
    mov x23, x5                    // reps
    // load + check dims; u64 checked count
    ldrsw x9, [x0, #0]             // width
    ldrsw x10, [x0, #4]            // height
    tbnz x9, #63, Lwm_bad
    tbnz x10, #63, Lwm_bad
    cbz x9, Lwm_zero
    cbz x10, Lwm_zero
    mul x21, x9, x10               // count (both <= 2^31-1: product fits u64)
    lsr x11, x21, #32
    cbnz x11, Lwm_bad              // product must stay in u32 range
    // Params (32 bytes): u32 w, u32 h, i64 c_cur, i64 c_lap, i64 c_prev (lo/hi pairs, LE)
    str w9, [sp, #96]
    str w10, [sp, #100]
    ldr x11, [x0, #8]
    str x11, [sp, #104]
    ldr x11, [x0, #16]
    str x11, [sp, #112]
    ldr x11, [x0, #24]
    str x11, [sp, #120]
    cmp x23, #1
    b.lt Lwm_bad
    bl _objc_autoreleasePoolPush
    mov x28, x0
    adrp x0, _gw_dev@PAGE
    ldr x24, [x0, _gw_dev@PAGEOFF]
    cbz x24, Lwm_fail
    // data length = count*4 + offset
    lsl x27, x21, #2
    add x27, x27, x22
    // curbuf = [dev newBufferWithLength:len options:0(shared)]; copy cur at +offset
    mov x0, x24
    mov x2, x27
    bl Lwv_newbuf
    cbz x0, Lwm_fail
    mov x25, x0
    bl Lwv_contents
    add x0, x0, x22
    ldr x1, [sp, #160]
    lsl x2, x21, #2
    bl Lwv_copy
    // prevbuf
    mov x0, x24
    mov x2, x27
    bl Lwv_newbuf
    cbz x0, Lwm_fail
    mov x26, x0
    bl Lwv_contents
    add x0, x0, x22
    ldr x1, [sp, #168]
    lsl x2, x21, #2
    bl Lwv_copy
    // outbuf
    mov x0, x24
    mov x2, x27
    bl Lwv_newbuf
    cbz x0, Lwm_fail
    str x0, [sp, #128]
    // satbuf (4 bytes)
    mov x0, x24
    mov x2, #4
    bl Lwv_newbuf
    cbz x0, Lwm_fail
    str x0, [sp, #136]
    bl Lwv_contents
    str x0, [sp, #144]
    // reps dispatches; saturation counter reset before each one.
Lwm_reploop:
    cbz x23, Lwm_read
    ldr x9, [sp, #144]
    str wzr, [x9]                  // sat reset per dispatch
    mov x0, x25
    mov x1, x26
    ldr x2, [sp, #128]
    ldr x3, [sp, #136]
    mov x4, x22
    mov x5, x21
    add x6, sp, #96
    bl Lwv_dispatch
    cbnz x0, Lwm_fail
    sub x23, x23, #1
    b Lwm_reploop
Lwm_read:
    // copy out field back (bytewise; dst alignment not assumed)
    ldr x0, [sp, #128]
    bl Lwv_contents
    add x1, x0, x22
    ldr x0, [sp, #152]
    lsl x2, x21, #2
    bl Lwv_copy
    ldr x9, [sp, #144]
    ldr w19, [x9]
    mov x0, x28
    bl _objc_autoreleasePoolPop
    mov x0, x19
    b Lwm_out
Lwm_zero:
    mov x0, #0
    b Lwm_out
Lwm_bad:
    mov x0, #-1
    b Lwm_out
Lwm_fail:
    mov x0, x28
    bl _objc_autoreleasePoolPop
    mov x0, #-1
Lwm_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #192
    ret

// Lwv_newbuf(x0 = dev, x2 = length) -> x0 = MTLBuffer (storage shared, options 0)
Lwv_newbuf:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0
    mov x20, x2
    adrp x0, Lw_newbuflen@PAGE
    add x0, x0, Lw_newbuflen@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    mov x2, x20
    mov x3, #0
    bl _objc_msgSend
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// Lwv_contents(x0 = MTLBuffer) -> x0 = void*
Lwv_contents:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]
    mov x19, x0
    adrp x0, Lw_contents@PAGE
    add x0, x0, Lw_contents@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x19
    bl _objc_msgSend
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// Lwv_copy(x0=dst, x1=src, x2=bytes) — bytewise, no alignment assumption
Lwv_copy:
    cbz x2, 2f
1:  ldrb w9, [x1], #1
    strb w9, [x0], #1
    subs x2, x2, #1
    b.ne 1b
2:  ret

// Lwv_dispatch(x0=curbuf, x1=prevbuf, x2=outbuf, x3=satbuf, x4=offset,
//              x5=count, x6=params*) -> x0 = 0 ok / 1 fail
// Real queue, real encoder; command status must be 4 and error nil.
Lwv_dispatch:
    stp x29, x30, [sp, #-144]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, x6
    adrp x0, _gw_queue@PAGE
    ldr x26, [x0, _gw_queue@PAGEOFF]
    cbz x26, Lwd_fail
    adrp x0, Lw_cmdbuf@PAGE
    add x0, x0, Lw_cmdbuf@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x26
    bl _objc_msgSend
    cbz x0, Lwd_fail
    mov x26, x0                    // command buffer
    adrp x0, Lw_encoder@PAGE
    add x0, x0, Lw_encoder@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x26
    bl _objc_msgSend
    cbz x0, Lwd_fail
    mov x27, x0                    // encoder
    adrp x0, Lw_setpso@PAGE
    add x0, x0, Lw_setpso@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    adrp x2, _gw_pso@PAGE
    ldr x2, [x2, _gw_pso@PAGEOFF]
    mov x0, x27
    bl _objc_msgSend
    // setBuffer:offset:atIndex: — cur@0, prev@1, out@2 (all at byte offset), sat@3
    adrp x0, Lw_setbuf@PAGE
    add x0, x0, Lw_setbuf@PAGEOFF
    bl _sel_registerName
    mov x28, x0
    mov x0, x27
    mov x1, x28
    mov x2, x19
    mov x3, x23
    mov x4, #0
    bl _objc_msgSend
    mov x0, x27
    mov x1, x28
    mov x2, x20
    mov x3, x23
    mov x4, #1
    bl _objc_msgSend
    mov x0, x27
    mov x1, x28
    mov x2, x21
    mov x3, x23
    mov x4, #2
    bl _objc_msgSend
    mov x0, x27
    mov x1, x28
    mov x2, x22
    mov x3, #0
    mov x4, #3
    bl _objc_msgSend
    // setBytes:length:atIndex: — Params, 32 bytes, index 4
    adrp x0, Lw_setbytes@PAGE
    add x0, x0, Lw_setbytes@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x27
    mov x2, x25
    mov x3, #32
    mov x4, #4
    bl _objc_msgSend
    // MTLSize aggregates pass indirectly (x2/x3 = pointers): (count,1,1) tpg (64,1,1)
    str x24, [sp, #96]
    mov x4, #1
    str x4, [sp, #104]
    str x4, [sp, #112]
    mov x5, #64
    str x5, [sp, #120]
    str x4, [sp, #128]
    str x4, [sp, #136]
    adrp x0, Lw_dispatch@PAGE
    add x0, x0, Lw_dispatch@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x27
    add x2, sp, #96
    add x3, sp, #120
    bl _objc_msgSend
    adrp x0, Lw_endenc@PAGE
    add x0, x0, Lw_endenc@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x27
    bl _objc_msgSend
    // commit; waitUntilCompleted; status must be Completed(4); error must be nil
    adrp x0, Lw_commit@PAGE
    add x0, x0, Lw_commit@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x26
    bl _objc_msgSend
    adrp x0, Lw_wait@PAGE
    add x0, x0, Lw_wait@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x26
    bl _objc_msgSend
    adrp x0, Lw_status@PAGE
    add x0, x0, Lw_status@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x26
    bl _objc_msgSend
    cmp x0, #4
    b.ne Lwd_fail
    adrp x0, Lw_error@PAGE
    add x0, x0, Lw_error@PAGEOFF
    bl _sel_registerName
    mov x1, x0
    mov x0, x26
    bl _objc_msgSend
    cbnz x0, Lwd_fail
    adrp x0, Lwm_status4@PAGE
    add x0, x0, Lwm_status4@PAGEOFF
    bl _fl_wput
    adrp x0, Lwm_errornil@PAGE
    add x0, x0, Lwm_errornil@PAGEOFF
    bl _fl_wput
    mov x0, #0
    b Lwd_out
Lwd_fail:
    mov x0, #1
Lwd_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #144
    ret

.data
.p2align 3
.globl _gw_dev
.globl _gw_queue
.globl _gw_pso
_gw_dev:   .quad 0
_gw_queue: .quad 0
_gw_pso:   .quad 0

.section __TEXT,__cstring,cstring_literals
Lw_NSString:     .asciz "NSString"
Lw_strutf8:      .asciz "stringWithUTF8String:"
Lw_newlib:       .asciz "newLibraryWithFile:error:"
Lw_newfn:        .asciz "newFunctionWithName:"
Lw_newpso:       .asciz "newComputePipelineStateWithFunction:error:"
Lw_newqueue:     .asciz "newCommandQueue"
Lw_newbuflen:    .asciz "newBufferWithLength:options:"
Lw_contents:     .asciz "contents"
Lw_cmdbuf:       .asciz "commandBuffer"
Lw_encoder:      .asciz "computeCommandEncoder"
Lw_setpso:       .asciz "setComputePipelineState:"
Lw_setbuf:       .asciz "setBuffer:offset:atIndex:"
Lw_setbytes:     .asciz "setBytes:length:atIndex:"
Lw_dispatch:     .asciz "dispatchThreads:threadsPerThreadgroup:"
Lw_endenc:       .asciz "endEncoding"
Lw_commit:       .asciz "commit"
Lw_wait:         .asciz "waitUntilCompleted"
Lw_status:       .asciz "status"
Lw_error:        .asciz "error"
Lw_locdesc:      .asciz "localizedDescription"
Lw_name:         .asciz "name"
Lw_utf8string:   .asciz "UTF8String"
Lw_kernelname:   .asciz "q30_wave"
Lwm_devname:     .asciz "metal device: "
Lwm_nserr:       .asciz "metal NSError: "
Lwm_nl:          .asciz "\n"
Lwm_status4:     .asciz "metal command status: 4\n"
Lwm_errornil:    .asciz "metal command error: nil\n"
