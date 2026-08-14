# fieldlang-asm CONTRACT v1 (narigo archon, frozen for V0)
Layer purity: gaialang source · hand asm · ARM64 output. No Rust/C in compiler.
Target: journal v1 .fldj (packages/field/src/journal.rs, FROZEN):
  magic "FLDJ"=0x4A444C46 LE u32 · version u32=1 · w u32 · h u32 ·
  c,dt,damping,dx f32-BITS u32 · seed u64 · range f32-BITS u32 · n_slots u32
  (header=48B) then ops: tag u8 + payload LE:
  1 SeedAtom{slot u32, seed u64} · 2 Excite{x u32,y u32,amp_bits u32} ·
  3 Step{count u32} · 4 Bind{dst,a,b u32} · 5 Bundle{dst u32,len u32,srcs u32..} ·
  6 WriteRaw{slot u32,len u32,bits u32..}

## Language (line-based, gaialang glyphs, decimal u64 integer args only)
  界 w h n_slots seed          (optional, once, first — world header; else defaults)
  種 slot seed                 → tag1
  撃 x y amp_bits              → tag2   (amp given AS f32 bit pattern, decimal)
  歩 count                     → tag3
  縛 dst a b                   → tag4
  束 dst n src1..srcn          → tag5
  寫 slot n b1..bn             → tag6  (n MUST equal w*h — full plane only.
                                 emit rejects n != w*h with code 5 = shape.)
  # comment to end of line. Whitespace/newlines separate. UTF-8.

## Defaults (driver-owned, doc here = law)
  w=64 h=64 n_slots=16 seed=42
  c=0x3F800000(1.0) dt=0x3DCCCCCD(0.1) damping=0x3F7FBE77(0.999)
  dx=0x3F800000(1.0) range=0x3F800000(1.0)
  LAW: THE BITS ARE NORMATIVE. The decimal in parentheses is a short human
  rendering of the bits, never a source. Implementations MUST copy the u32 bit
  pattern; they MUST NOT re-derive a constant by parsing/rounding the decimal in
  some other precision. Exact values: 0x3F7FBE77 = 0.99900001287460327148 (= the
  f32 nearest to 0.999, so the "0.999" annotation is CORRECT), 0x3DCCCCCD = 0.1
  nearest, 0x3F800000 = 1.0 exact.
  (G2 RETRACTED here: the fieldrun lane claimed 0x3F7FBE77 = 0.998969495… and that
  f32(0.999) = 0x3F7FC077. Both false — frac(0x3F7FBE77) = 0x7FBE77 = 8371831,
  not 8371319; 0x3F7FC077 = 0.99903053045272827148. See fieldrun/CONTRACT.md §17.)

## Token record = 16B: { u64 kind, u64 value }
  kind: 0=EOF 1=種 2=撃 3=歩 4=縛 5=束 6=寫 7=INT(value=u64) 9=界
        10=觀 (V1 reserved; no journal tag — emit MUST reject as syntax err)
  Glyph kinds 1..6 == journal op tag. Lexer skips whitespace + #comments.

## ABI (AAPCS64, exported symbols, each file standalone .s, no data sharing)
  lexer.s : _fl_lex(x0=src ptr, x1=src len, x2=out token buf, x3=cap in RECORDS)
            → x0 = token count (incl. final EOF record) ; x0<0 = error:
              -(line<<8 | code), code 1=bad char 2=overflow 3=buffull. line from 1.
  emit.s  : _fl_emit(x0=token ptr, x1=token count, x2=out byte buf, x3=cap bytes,
            x4=header params ptr — struct 40B {u32 w,h; u32 c,dt,damp,dx bits;
            u64 seed; u32 range; u32 n_slots})
            → x0 = bytes written ; x0<0 = error: -(code), 1=syntax 2=buffull
              4=arg-range(slot/x/y ≥ u32 impossible: values>u32 max → err).
            emit reads 界 token (if present, must be first) to OVERRIDE the
            x4 struct fields w,h,n_slots,seed before writing header.
  driver.s: _main. usage: fieldc in.fld out.fldj. mmap/read whole file,
            fixed bufs (16MiB src / 16MiB tokens / 16MiB out ok), call lex,
            emit (x4 → defaults struct above), write out, exit 0; on error
            write short msg to stderr, exit 1. libSystem thin calls permitted
            (open/read/write/close/exit via _open etc.) OR raw svc — declare.
  Build: as -arch arm64 · ld with -lSystem -syslibroot $(xcrun --show-sdk-path).
  Each lane: own .s + own tiny standalone test harness permitted (test-only C
  or shell OK, marked gate-tool). Integration = archon merges three .s.

## V1 RESERVATION (2026-08-01, Pascal+Nyari — grammar knows, V0 compiler ignores)
Field form V1: 4D W×H×D×C, C=64 default · store d=16384 · hot N≤4096.
Grammar reserve:
  界 extends → 界 w h d ch n_slots seed  (spatial 3D + channels; ch=64 default;
    all IRON params w/ defaults, no hardcodes). V0: 2-arg-form stays valid.
  觀 = probe/render: channel collapse → light (read-only, no journal tag).
V0 compiler targets 2D journal v1 unchanged. Emit layer B swap point (§dict
doc) also covers journal v2 when 4D lands — layer A records stay stable.
