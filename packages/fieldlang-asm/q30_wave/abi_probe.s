// Live ABI probe: x19-x28 + SP values; LR is proven by return to this code.
// x18 is platform-reserved: non-write is enforced by gate static scan + mutation red.
.subsections_via_symbols
.globl _main
.extern _fl_q30_wave_scalar
.extern _fl_q30_wave_neon
.extern _write
.extern _exit
_main:
 sub sp,sp,#128
 stp x19,x20,[sp,#0]
 stp x21,x22,[sp,#16]
 stp x23,x24,[sp,#32]
 stp x25,x26,[sp,#48]
 stp x27,x28,[sp,#64]
 movz x19,#0x1919
 movz x20,#0x2020
 movz x21,#0x2121
 movz x22,#0x2222
 movz x23,#0x2323
 movz x24,#0x2424
 movz x25,#0x2525
 movz x26,#0x2626
 movz x27,#0x2727
 movz x28,#0x2828
 mov x8,sp
 str x8,[sp,#80]
 adrp x0,_pcfg@PAGE
 add x0,x0,_pcfg@PAGEOFF
 adrp x1,_pcur@PAGE
 add x1,x1,_pcur@PAGEOFF
 adrp x2,_pprv@PAGE
 add x2,x2,_pprv@PAGEOFF
 adrp x3,_pout@PAGE
 add x3,x3,_pout@PAGEOFF
 bl _fl_q30_wave_scalar
 bl Lcheck
 adrp x0,_pcfg@PAGE
 add x0,x0,_pcfg@PAGEOFF
 adrp x1,_pcur@PAGE
 add x1,x1,_pcur@PAGEOFF
 adrp x2,_pprv@PAGE
 add x2,x2,_pprv@PAGEOFF
 adrp x3,_pout@PAGE
 add x3,x3,_pout@PAGEOFF
 bl _fl_q30_wave_neon
 bl Lcheck
 adrp x1,Lok@PAGE
 add x1,x1,Lok@PAGEOFF
 mov x0,#1
 mov x2,#13
 bl _write
 mov x0,#0
 b Lexit
Lcheck:
 movz x9,#0x1919
 cmp x19,x9
 b.ne Lbad
 movz x9,#0x2020
 cmp x20,x9
 b.ne Lbad
 movz x9,#0x2121
 cmp x21,x9
 b.ne Lbad
 movz x9,#0x2222
 cmp x22,x9
 b.ne Lbad
 movz x9,#0x2323
 cmp x23,x9
 b.ne Lbad
 movz x9,#0x2424
 cmp x24,x9
 b.ne Lbad
 movz x9,#0x2525
 cmp x25,x9
 b.ne Lbad
 movz x9,#0x2626
 cmp x26,x9
 b.ne Lbad
 movz x9,#0x2727
 cmp x27,x9
 b.ne Lbad
 movz x9,#0x2828
 cmp x28,x9
 b.ne Lbad
 mov x9,sp
 ldr x8,[sp,#80]
 cmp x8,x9
 b.ne Lbad
 ret
Lbad:
 adrp x1,Lbadmsg@PAGE
 add x1,x1,Lbadmsg@PAGEOFF
 mov x0,#2
 mov x2,#15
 bl _write
 mov x0,#1
Lexit:
 ldp x19,x20,[sp,#0]
 ldp x21,x22,[sp,#16]
 ldp x23,x24,[sp,#32]
 ldp x25,x26,[sp,#48]
 ldp x27,x28,[sp,#64]
 add sp,sp,#128
 bl _exit
.section __DATA,__data
.p2align 3
// 8x8: NEON内部vector経路(width>=6要)を live sentinel 検査下に入れる。
// 1x1 では端scalar経路のみ走り vector loop の register 破壊を live 検出できぬ。
_pcfg: .long 8
 .long 8
 .quad 1073741824
 .quad 268435456
 .quad 536870912
.p2align 4
_pcur: .fill 64, 4, 305419896
.p2align 4
_pprv: .fill 64, 4, 187182
.p2align 4
_pout: .fill 64, 4, 0
.section __TEXT,__cstring,cstring_literals
Lok: .asciz "abi probe ok\n"
Lbadmsg: .asciz "abi probe bad\n"
