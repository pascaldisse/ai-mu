// fieldlang-asm driver.s — arm64 macOS, hand-written asm
// CONTRACT v1. Syscall choice: libSystem thin calls (_open/_read/_write/_close/_exit) — DECLARED.
// usage: fieldc in.fld out.fldj
// bufs: 16MiB src / 16MiB tokens / 16MiB out (.bss)

.equ BUFSZ,   16777216      // 16 MiB
.equ TOKCAP,  1048576       // records = 16MiB/16
.equ O_RDONLY, 0x0000
.equ O_WRTRC,  0x0601       // O_WRONLY|O_CREAT|O_TRUNC (macOS)

.data
.p2align 3
// CONTRACT defaults struct (40B): {u32 w,h; u32 c,dt,damp,dx bits; u64 seed; u32 range; u32 n_slots}
_fl_defaults:
  .long 64                  // w
  .long 64                  // h
  .long 0x3F800000          // c    = 1.0
  .long 0x3DCCCCCD          // dt   = 0.1
  .long 0x3F7FBE77          // damp = 0.999
  .long 0x3F800000          // dx   = 1.0
  .quad 42                  // seed
  .long 0x3F800000          // range = 1.0
  .long 16                  // n_slots

msg_usage: .ascii "usage: fieldc in.fld out.fldj\n"
.equ msg_usage_len, . - msg_usage
msg_open:  .ascii "fieldc: cannot open input\n"
.equ msg_open_len, . - msg_open
msg_read:  .ascii "fieldc: read error\n"
.equ msg_read_len, . - msg_read
msg_lex:   .ascii "fieldc: lex error\n"
.equ msg_lex_len, . - msg_lex
msg_emit:  .ascii "fieldc: emit error\n"
.equ msg_emit_len, . - msg_emit
msg_out:   .ascii "fieldc: cannot write output\n"
.equ msg_out_len, . - msg_out

.bss
.p2align 4
_src_buf: .space BUFSZ
_tok_buf: .space BUFSZ
_out_buf: .space BUFSZ

.text
.globl _main
.p2align 2
_main:
  stp x29, x30, [sp, #-64]!
  mov x29, sp
  stp x19, x20, [sp, #16]   // x19=argv, x20=src len
  stp x21, x22, [sp, #32]   // x21=fd, x22=out bytes

  // argc check
  cmp x0, #3
  b.ne Lusage
  mov x19, x1

  // open input
  ldr x0, [x19, #8]         // argv[1]
  mov w1, #O_RDONLY
  mov w2, #0
  bl _open
  cmp w0, #0
  b.lt Lerr_open
  mov w21, w0

  // read loop into _src_buf
  mov x20, #0               // total
Lread_loop:
  mov w0, w21
  adrp x1, _src_buf@PAGE
  add  x1, x1, _src_buf@PAGEOFF
  add  x1, x1, x20
  mov  x2, #BUFSZ
  sub  x2, x2, x20
  cbz  x2, Lread_done       // buf full
  bl _read
  cmp x0, #0
  b.lt Lerr_read
  b.eq Lread_done
  add x20, x20, x0
  b Lread_loop
Lread_done:
  mov w0, w21
  bl _close

  // lex
  adrp x0, _src_buf@PAGE
  add  x0, x0, _src_buf@PAGEOFF
  mov  x1, x20
  adrp x2, _tok_buf@PAGE
  add  x2, x2, _tok_buf@PAGEOFF
  mov  x3, #TOKCAP
  bl _fl_lex
  cmp x0, #0
  b.lt Lerr_lex

  // emit
  mov  x1, x0               // token count
  adrp x0, _tok_buf@PAGE
  add  x0, x0, _tok_buf@PAGEOFF
  adrp x2, _out_buf@PAGE
  add  x2, x2, _out_buf@PAGEOFF
  mov  x3, #BUFSZ
  adrp x4, _fl_defaults@PAGE
  add  x4, x4, _fl_defaults@PAGEOFF
  bl _fl_emit
  cmp x0, #0
  b.lt Lerr_emit
  mov x22, x0

  // open output
  ldr x0, [x19, #16]        // argv[2]
  mov w1, #O_WRTRC
  mov w2, #0x1A4           // 0644
  // Darwin arm64 variadic ABI: _open's mode argument is stack-passed.
  sub sp, sp, #16
  str w2, [sp]
  bl _open
  add sp, sp, #16
  cmp w0, #0
  b.lt Lerr_out
  mov w21, w0

  // write loop
  mov x20, #0
Lwrite_loop:
  cmp x20, x22
  b.ge Lwrite_done
  mov w0, w21
  adrp x1, _out_buf@PAGE
  add  x1, x1, _out_buf@PAGEOFF
  add  x1, x1, x20
  sub  x2, x22, x20
  bl _write
  cmp x0, #0
  b.le Lerr_out
  add x20, x20, x0
  b Lwrite_loop
Lwrite_done:
  mov w0, w21
  bl _close

  mov w0, #0
  bl _exit

// error paths: stderr short msg + exit 1
.macro ERRPATH label, msg, len
\label:
  adrp x1, \msg@PAGE
  add  x1, x1, \msg@PAGEOFF
  mov  x2, #\len
  b Lfail
.endm
ERRPATH Lusage,    msg_usage, msg_usage_len
ERRPATH Lerr_open, msg_open,  msg_open_len
ERRPATH Lerr_read, msg_read,  msg_read_len
ERRPATH Lerr_lex,  msg_lex,   msg_lex_len
ERRPATH Lerr_emit, msg_emit,  msg_emit_len
ERRPATH Lerr_out,  msg_out,   msg_out_len

Lfail:
  mov w0, #2                // stderr
  bl _write
  mov w0, #1
  bl _exit
