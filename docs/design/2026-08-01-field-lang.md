# FIELD LANGUAGE V0 — design (field/lang, 2026-08-01, FINAL per CONTRACT v1)

Status: language spec FINAL, aligned to packages/fieldlang-asm/CONTRACT.md v1
(frozen, narigo archon, field/lang-asm@7347a0cf). Implementation = ARM64 asm
lanes (lexer.s/emit.s/driver.s), NOT this worktree. This worktree owns:
language design (this file) · REPL draft · acceptance gate (packages/
fieldlang-gate — GATE TOOL, not the compiler).

Layer purity decree (主令改 2026-08-01): gaialang source · hand asm · ARM64
output. No Rust/C in the COMPILER. Rust field core (`World::replay`) remains
the execution/determinism gate — temporarily permitted, explicitly not
compiler. Byte target = journal v1 .fldj 真 (packages/field/src/journal.rs
FROZEN): compiler = text→bytes, no floats computed, lex+emit only.

## 0. Alphabet — one char, one meaning (CONTRACT v1 grammar)

| glyph | meaning | args (all decimal u64) | → bytes (journal tag) |
|---|---|---|---|
| 界 | world header | `w h n_slots seed` — optional, once, FIRST line | header override (no op) |
| 種 | seed atom | `slot seed` | tag 1: slot u32, seed u64 |
| 撃 | excite | `x y amp_bits` — amp AS f32 BIT PATTERN, decimal | tag 2: x,y,amp_bits u32 |
| 歩 | step | `count` | tag 3: count u32 |
| 縛 | bind | `dst a b` | tag 4: dst,a,b u32 |
| 束 | bundle | `dst n src1..srcn` — counted | tag 5: dst u32, len u32, srcs u32.. |
| 寫 | write raw | `slot n b1..bn` — counted, bits decimal | tag 6: slot u32, len u32, bits u32.. |
| 觀 | probe | `key_slot top_k` | **NO BYTES — not in compiler v0** |

觀 = read-only (`Store::probe(&self,…)` 真 store.rs; journal.rs has NO probe
tag 真). It lives at the REPL/runner layer (REPL doc §0), never in a journal —
a read has nothing to pass through the mutation door. fieldc v0 grammar =
the 7 byte-emitting glyphs only; a 觀 line in a .fld file = lexer bad char.

`#` = comment to end of line. Whitespace/newlines separate. UTF-8.

Defaults (driver-owned law, CONTRACT §Defaults; DISTINCT from field crate
defaults — compiler defaults, not FieldConfig/WaveParams defaults):
`w=64 h=64 n_slots=16 seed=42` · c=0x3F800000(1.0) dt=0x3DCCCCCD(0.1)
damping=0x3F7FBE77(0.999) dx=0x3F800000(1.0) range=0x3F800000(1.0).
界 overrides w,h,n_slots,seed only; wave params have NO source spelling in v0.
Courant of defaults = 0.1 ≤ 1/√2 真 stable.

 Corpses (killed branches, kept visible):
- **Rust lexer/parser/compiler** — killed by purity decree 主令改 2026-08-01
  BEFORE FIRST LINE: no Rust compiler code ever existed; nothing to exhume.
  Burial = this note. (packages/fieldlang-gate is the sanctioned GATE TOOL
  oracle, marked not-the-compiler; it is a test instrument, not a compiler.)
- English keywords — killed by 主令正 2026-08-01. CJK = the ONLY spelling.
- 書 for WriteRaw — killed; 寫 (REPL draft proposal) locked.
- 束/寫 as varargs-to-EOL — killed by CONTRACT v1 counted form (`dst n srcs…`).
- `[ ]` bracket lists — killed earlier, stayed dead.
- Float literals in v0 source (`撃 … 1.0`) — killed: decimal bit patterns
  only, compiler computes nothing. Float sugar = future REPL-side layer.
- `:new` as batch-file header — killed: 界 is the language header. `:new`
  survives only as REPL session meta (REPL doc §2), different layer.
- 歩 optional count (default 1) — killed: CONTRACT form takes count.
- 波/位 split header glyphs, `.field` extension (→ `.fld` per driver usage
  `fieldc in.fld out.fldj`) — killed.
- 界 RESURRECTED note: an earlier draft of this doc killed 界/波/位; CONTRACT
  v1 revived 界 in narrower form (w h n_slots seed, no wave params).

## 1. Byte-level acceptance contract (the audit target)

FLDJ journal v1 真 journal.rs, restated from CONTRACT v1:
```
header 48B LE: magic "FLDJ"=0x4A444C46 u32 · version u32=1 · w u32 · h u32 ·
  c,dt,damping,dx f32-bits u32 ×4 · seed u64 · range f32-bits u32 · n_slots u32
ops: tag u8 + payload LE:
  1: slot u32, seed u64
  2: x u32, y u32, amp_bits u32
  3: count u32
  4: dst u32, a u32, b u32
  5: dst u32, len u32, srcs u32×len
  6: slot u32, len u32, bits u32×len
```
Compiler-internal token ABI (CONTRACT §ABI, audit-relevant for lane merges):
16B records `{u64 kind, u64 value}`; kinds 1..6 == journal tags, 7=INT,
9=界, 0=EOF. lexer err `-(line<<8|code)` (1 bad char, 2 overflow, 3 buffull);
emit err `-(code)` (1 syntax, 2 buffull, 4 arg>u32). Acceptance compares
OUTPUT BYTES only; ABI conformance is lane-integration's concern.

Semantic laws NOT enforced by fieldc v0 (syntactic + u32 range only, CONTRACT
§ABI error set): targets ≥ RESERVED_SLOTS(2) 真, x<W y<H, 寫 n==d=W*H 真
(world.apply asserts), list counts honest. Violations = journaled garbage or
replay assert — gate-visible. Programmer's law until a semantic layer exists.

## 2. Grammar (EBNF, CONTRACT v1 form)

```ebnf
program = { line } ;
line    = header | op | comment | empty ;
header  = "界" uint uint uint uint ;          (* w h n_slots seed — once, first *)
op      = "種" uint uint                      (* slot, seed            → tag 1 *)
        | "撃" uint uint uint                 (* x, y, amp_bits        → tag 2 *)
        | "歩" uint                           (* count                 → tag 3 *)
        | "縛" uint uint uint                 (* dst, a, b             → tag 4 *)
        | "束" uint uint uint { uint }        (* dst, n, src×n         → tag 5 *)
        | "寫" uint uint uint { uint } ;      (* slot, n, bits×n       → tag 6 *)
comment = "#" , { ? any char ? } ;
uint    = digit , { digit } ;                 (* decimal, ≤ u64 max; emit: ≤ u32 max *)
```

Lexer: skips whitespace + comments; UTF-8 CJK glyphs = single characters;
any other char = bad char (code 1). Statement = one line (driver/emit treat
newlines as separators; counted lists make line boundaries recoverable).

## 3. Execution + determinism

`Journal::read_all` 真 → `(cfg, params, n_slots, ops)` → `World::replay` 真 =
fresh world + apply all = bit-exact. state = f(op sequence) 真 ENTROPY law.
Digest = `world.digest()` = store fnv1a over f32 bit patterns ^ mix64(
step_index) 真 {lib,store,world}.rs — canonical gate digest. Same bytes →
same digest, always; acceptance runs replay twice per artifact.

## 4. Examples (the 3 acceptance programs, packages/fieldlang-gate/examples/)

E1 `01_wave.fld` — first strike, pure wave (界 撃 歩):
```
# 初撃 — a strike rings the plane
界 32 32 2 7
撃 16 16 1065353216
歩 60
```
(1065353216 = 0x3F800000 = 1.0f32 bits)

E2 `02_algebra.fld` — atom algebra (界 種 縛 束):
```
# 原子代数 — two atoms, bind, bundle
界 16 16 6 42
種 2 42
種 3 7
縛 4 2 3
束 5 2 2 3
```

E3 `03_raw.fld` — raw row + strike + plane into record (界 寫 種 縛 撃 歩 束):
```
# 生書 — raw bits, wave steps, record absorbs the live plane
界 4 4 5 99
寫 2 16 1056964608 3204448256 1048576000 3196059648 0 0 0 0 1065353216 3212836864 1056964608 3204448256 1036831949 1045220557 1050253722 1053609165
種 3 99
縛 4 2 3
撃 2 2 1061158912
歩 8
束 4 2 4 0
```
(寫 row = 0.5 -0.5 0.25 -0.25 0 0 0 0 1 -1 0.5 -0.5 0.1 0.2 0.3 0.4 as f32
bits, decimal; 1061158912 = 0.75f32 bits; d=4×4=16 ✓ L-law 寫 n==d honored.)

Coverage: all 6 journaled ops + 界 across the three files.

## 5. Non-goals v0 (dead on arrival, listed once)

variables/labels · loops · expressions · macros · float literals · English
spellings · 觀 in files (REPL layer) · wave-param source spelling · semantic
checks in fieldc · pixel render in-language · auto-generated seeds.

## 6. V1 RESERVATION (2026-08-01, Pascal+Nyari — grammar knows, V0 ignores)

Field form V1: 4D W×H×D×C, C=64 default · store d=16384 · hot N≤4096.
Grammar reserved now, compiler later (mirrors CONTRACT.md §V1 RESERVATION):
- 界 extends → `界 w h d ch n_slots seed` (spatial 3D + channels; ch=64
  default; all IRON params with defaults, no hardcodes). V0 form
  `界 w h n_slots seed` stays valid (= d, ch defaulted).
- 觀 = channel collapse → light (read-only, NO journal tag — the V0 law that
  觀 emits no bytes carries forward unchanged).
V0 compiler targets the 2D journal v1 unchanged; §0–§5 stay law.

## 7. Pointers

- CONTRACT (frozen law for impl): ../mc-fieldlang-asm/packages/fieldlang-asm/CONTRACT.md
- REPL session layer (`:new :load :save :digest :render :quit`, prompt 場>,
  觀 home layer): docs/design/2026-08-01-field-repl.md — interactive sugar
  there (varargs, optional count, float amps) is NOT the batch grammar.
- Acceptance gate: packages/fieldlang-gate (GATE TOOL — reference .fldj
  generator via frozen Rust journal + byte-compare script + replay digests).
- Journal gap: `Journal::open_append` missing on field-unified 真 (REPL §2) —
  note only; field crate signatures frozen.
- GENESIS.md referenced by original task: absent from both worktrees
  (find -iname empty) — contract doc + CONTRACT.md used. UNVERIFIED.
