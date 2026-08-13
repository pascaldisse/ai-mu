// GATE-ONLY STUB — real emit.s owned by emit lane. DO NOT SHIP.
// _fl_emit: ignores tokens, writes 48B header from x4 struct + one Step{1} op (tag3).
// returns 53 bytes.
.text
.globl _fl_emit
.p2align 2
_fl_emit:
  mov x9, x2
  // magic "FLDJ" + version 1
  mov w10, #0x4C46
  movk w10, #0x4A44, lsl #16
  str w10, [x9], #4
  mov w10, #1
  str w10, [x9], #4
  // w,h,c,dt,damp,dx from struct (offsets 0..20)
  ldr w10, [x4, #0] 
  str w10, [x9], #4
  ldr w10, [x4, #4] 
  str w10, [x9], #4
  ldr w10, [x4, #8] 
  str w10, [x9], #4
  ldr w10, [x4, #12]
  str w10, [x9], #4
  ldr w10, [x4, #16]
  str w10, [x9], #4
  ldr w10, [x4, #20]
  str w10, [x9], #4
  // seed u64 @24, range @32, n_slots @36
  ldr x10, [x4, #24]
  str x10, [x9], #8
  ldr w10, [x4, #32]
  str w10, [x9], #4
  ldr w10, [x4, #36]
  str w10, [x9], #4
  // op: tag 3 Step{count=1}
  mov w10, #3
  strb w10, [x9], #1
  mov w10, #1
  str w10, [x9], #4
  sub x0, x9, x2
  ret
