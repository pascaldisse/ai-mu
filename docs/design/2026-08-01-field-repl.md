# FIELD LANGUAGE REPL — design draft, 2026-08-01

Design doc ONLY. No impl here. Sources: GENESIS.md (V2 decree, root of
magic-crystal) · packages/field/src/{journal,world,store}.rs @field-unified
a4c9f1b6 (`git show field-unified:path`) · docs/design/2026-08-01-field-guide.md
· docs/design/2026-08-01-field-contract.md. Tags: 真=read directly off frozen
source this session · 假=proposed here, unconfirmed, needs Pascal/archon sign-off.
Branch owner = fette-katze lane (field/lang) — this file does not commit.

**SPEC CHANGE (steered mid-draft)**: field-language keywords = single CJK
chars, not English words. 種=seed 撃=excite 歩=step 縛=bind 束=bundle 觀=probe.
Locked below; propagates through every section.

---

## 0. Two vocabularies, not one

Collapsing "REPL commands" into one bag was the wrong frame. Two distinct
layers, different mutability contract:

- **FIELD LANGUAGE proper** — the CJK op-keywords. Each (except 觀) maps 1:1
  onto a `journal::Op` variant 真 (frozen enum: SeedAtom/Excite/Step/Bind/
  Bundle/WriteRaw). Typing one of these = proposing a mutation. Every
  successful parse gets journal.append()'d AND world.apply()'d, same
  transaction, no daylight between them.
- **REPL session meta** — ASCII, colon-prefixed (`:load :save :digest :new
  :render :quit`). Tooling around the language, not IN it: file I/O, queries,
  process control. NEVER appended to the journal. This split is load-bearing
  §3.

觀 (probe) sits on the CJK side lexically but semantically it is a QUERY —
`Store::probe(&self,...)` 真 takes `&self`, never mutates, has NO `Op` tag in
journal.rs at all. So 觀 is the one field-language keyword that never touches
the journal. Not an inconsistency — it's the language surface correctly
mirroring `World::apply` being "the ONLY mutation door" 真 (world.rs doc
comment): a read has nothing to pass through that door.

WriteRaw has no assigned CJK char in the steer (only 6 were given, covering
5 Op variants + probe). 假 proposed here: **寫** (write) for `WriteRaw`, e.g.
`寫 3 <f32 f32 ...>`.
**LOCKED 2026-08-01**: 寫 confirmed — adopted by fieldlang v0
(docs/design/2026-08-01-field-lang.md §0, packages/fieldlang). Batch `.field`
files = REPL scripts (`:new` meta + CJK op lines, 束 varargs, 歩 default 1)
per that doc; grammars aligned this session.

---

## 1. REPL loop — text → Op → World, incrementally

```
loop:
  read line from stdin (or script/pipe)
  parse line → Cmd::FieldOp(op) | Cmd::Meta(cmd) | Cmd::ParseError(msg)
  match:
    FieldOp(op) if op ≠ 觀:
        journal.append(&op)      # RAM buffer only, 真 journal.rs law: append()
                                  # NEVER touches disk, hot-path safe
        world.apply(&op)         # 真 world.rs: the ONLY mutation door
        print short ack: op echo + step_index + digest-prefix
    FieldOp(觀 args):
        results = store.probe(key, top_k)   # read-only, NOT appended
        print results
    Meta(cmd):  → §2/§5 (session control, never journaled)
    ParseError(msg):
        print msg. NO append, NO apply. World and journal untouched.
```

**Core invariant (the whole point of this REPL)**: after processing the
first *i* field-ops typed (skipping 觀/meta/errors), live `world` state ==
`World::replay(cfg, params, n_slots, &ops[0..i])` 真 (world.rs `replay` sig:
fresh World + apply all). The REPL keeps this true incrementally — each
keystroke-line does ONE `apply()` on the live world rather than replaying
the whole prefix from scratch every time. Cheaper, and behaviorally
identical to full replay by induction on `apply` being deterministic and
side-effect-free beyond `&mut self` — 記 (follows directly from
`World::apply`'s frozen match-only body, no hidden state elsewhere).

Perf note 假: incremental apply is O(1) amortized per line vs O(i) for
full-prefix replay each time — an actual REPL MUST do incremental, replay is
for `:load`/session-restore only (§2), never for the interactive hot loop.

Parse errors are invisible to the journal by construction — a session
replayed later reproduces exactly the *effective* (accepted) commands.
Determinism of the parser itself (same string → same parse result, always)
is a free property of pure string parsing — no clock/rand in the parser.

---

## 2. Journal-as-session

Central reframe: **the append-only op log IS the session.** Not a save-file
bolted onto a running process — the log is the primary artifact; live world
state is a derived, disposable projection of it (cache, in the DB sense).

- Boot: `:new <w> <h> <slots> [params]` 假 → fresh `World::new` 真 + fresh
  `Journal::create` 真 (writes 48-byte header immediately, per journal.rs
  binary format v1). Empty op log = empty session, valid.
- Type field-ops → RAM-buffered append + live apply, lockstep (§1).
- `:save [path]` → `Journal::flush()` 真 (the ONLY disk write, off-cadence,
  explicit) — durable session on SSD, cold-storage law honored.
- `:load <path>` → `Journal::read_all(path)` 真 → `(cfg, params, n_slots,
  ops)` → `World::replay(...)` 真 → live world = restored state, bit-exact.
  Session then CONTINUES: new field-ops typed after `:load` append past the
  loaded prefix — same log, longer.

**Gap, 假 flagged — genuinely needs W3/journal owner**: `journal.rs` today
真 exposes only `create()` (writes header, fails if reopening) and
`read_all()` (whole-file read, no write path). There is no
`Journal::open_append(path)` — read+validate existing header, seek to EOF,
allow further `append`/`flush`. Without it, "`:load` then keep typing then
`:save`" has nowhere durable to flush TO (flush() writes via `append(true)`
`OpenOptions` against `self.path`, but `Journal` only gets a `path` from
`create()` — a `Journal` built by `:load` has no live buffer/path object at
all, just the parsed tuple). Needed extension, not yet built:
`Journal::open_append(path) -> io::Result<Self>` — validate magic/version/
cfg/params/n_slots match the live World, position for append, return a
`Journal` whose `flush()` behaves identically to one continuous
create→append→flush session. New determinism gate once it lands: two-phase
session (`ops[0..5]` saved, reloaded, `ops[5..10]` appended, saved) must
produce a journal file BYTE-IDENTICAL to one continuous session typing all
10 ops then saving once. Not measured, not built — UNVERIFIED, §6.

---

## 3. Determinism guarantees

ENTROPY.md law 真 (quoted in field-guide.md): state = f(seed); no `rand`, no
clock in state; all f32 persisted as bit patterns. REPL generalizes the
"seed" to "the accepted op sequence" — no new law needed, existing gates
(determinism.rs, 5/5 green 真 per field-guide.md §Invariants) already cover
exactly this shape (`World::replay` on a `Vec<Op>`); the REPL is just
ANOTHER PRODUCER of that same `Vec<Op>`, nothing more.

- Two REPL sessions fed the IDENTICAL accepted command sequence (typed live,
  piped from a script, or `:load`ed) → bit-identical `digest()` at every
  step — independent of real wall-clock time between keystrokes. `Step{count}`
  is a tick COUNT, never a duration; no `Instant` anywhere in `Op` or `World`
  state 真 (only bench.rs, outside the state path, is allowed `Instant` per
  field-guide.md).
- Non-determinism risk to actively guard in the parser 假: any field-op
  argument that is itself a "randomness" (seed values in 種/撃) MUST be a
  literal the user typed or piped — the REPL must NEVER auto-generate a
  fresh seed on its own clock (e.g. no `種 2 auto` sugar that reads
  `SystemTime`). If a "random seed" convenience is ever wanted, it must be
  an explicit, journaled, PRE-materialized literal (roll it once, print it,
  user can retype it) — never silent.
- Decimal literals (e.g. 撃's amplitude) parse via `f32::from_str` (stdlib,
  deterministic for a given string) then immediately convert `.to_bits()`
  before packing into `Op::Excite{amp_bits}` 真 — the TEXT itself, not just
  the resulting `Op`, is a deterministic function into bits. No float
  round-tripping ambiguity survives past the parse step.
- 觀 (probe) never touches the journal (§0) so it is CORRECTLY absent from
  the replay/determinism surface — a session with different 觀 queries
  interleaved at different points still journals to the identical `Op`
  sequence and replays identically. Determinism gate 假 to add: probing
  between two ops must not perturb `digest()` (trivially true since probe
  is `&self`, but worth a REPL-level regression test once built).

---

## 4. OS-door architecture

GENESIS.md 真 GOAL ∞: "runs as an OPERATING SYSTEM... Core = pure field +
pure Rust/own-lang; app shell = thin, swappable." This REPL is that
boundary's first concrete instance — the load-bearing design choice is that
it adds ZERO field-logic of its own.

```
   text (CJK ops + ASCII meta)
        │  parse
        ▼
   Cmd::FieldOp(Op) ────────► journal.append + world.apply   (§1)
   Cmd::Meta          ──────► file I/O / query glue           (§2, §5)
        │
        ▼
   packages/field (core, pure Rust)
     World::apply(&Op)   — the ONLY mutation door, frozen 真
     Store::probe(&self) — the ONLY non-journaled read, frozen 真
     Journal::{create,append,flush,read_all} — cold backing 真
```

- **Core** (`packages/field`): no windowing, no stdin, no CJK/text
  anywhere — pure `Op`-in, `Slice`/`u64`/`Vec<u8>`-out. Already this
  minimal 真 by construction of the frozen contract (journal-contract.md
  §Ownership: "lib.rs/ops.rs/store.rs FROZEN").
- **REPL shell** (new, thin, this doc's proposal): owns parsing (CJK glyph →
  `Op`), printing (digest/probe/render feedback), and the file glue for
  `:load`/`:save`/`:new`. That's the ENTIRE shell surface. No physics, no
  FFT, no store internals leak into it.
- **Swappability, why it matters**: because the mutation vocabulary is
  fixed and small (6 `Op` variants, frozen per contract 真), every future
  shell speaks the SAME alphabet into the SAME core with zero core changes:
  a GUI (mouse-drag → 撃/Excite), a network peer (the journal's own binary
  wire format — header+tag+payload — is already serialization-shaped, a
  candidate RPC frame 假), a batch script (a `.txt` file of CJK lines piped
  into this same REPL, no separate "scripting mode" needed), and eventually
  — per GENESIS's own "literal EL1 deferred, never forgotten" — an OS-level
  input handler where keystrokes arrive from a kernel line discipline
  instead of a libc stdin. None of that requires touching `packages/field`.
- **GOAL 1 mapping** 假: GENESIS's boot sequence "Boot → plane → speak/
  strike → world condenses" maps onto `:new` (boot the plane/World) →
  撃 (strike) → 種/縛/束 accreting store slots (world condenses). This REPL,
  headless, already IS the GOAL-1 app's command skeleton — a GUI later
  wraps it without the core crate changing, which is exactly what keeps
  GOAL ∞ (OS door) open rather than foreclosed by GOAL 1 (Mac app) choices.

---

## 5. Command sketch

Field language (CJK, journaled unless noted) — args positional, whitespace-
separated, ints/floats as literals:

| glyph | args | → | Op / call | journaled? |
|---|---|---|---|---|
| 種 | `slot seed` | `SeedAtom{slot,seed}` | store[slot]=seeded_atom(seed) | yes |
| 撃 | `x y amp` | `Excite{x,y,amp_bits}` | strike slots 0/1 (plane cur/prev) | yes |
| 歩 | `[count=1]` | `Step{count}` | count× tick() | yes |
| 縛 | `dst a b` | `Bind{dst,a,b}` | store[dst]=bind(store[a],store[b]) | yes |
| 束 | `dst src...` | `Bundle{dst,srcs}` | store[dst]=Σ store[srcs] | yes |
| 觀 | `key_slot top_k` | — | `Store::probe` read | **no** |
| 寫 | `slot bits...` | `WriteRaw{slot,data_bits}` | raw row write | yes (LOCKED, see §0) |

注 (2026-08-01, batch divergence): the interactive forms above are REPL
sugar. The BATCH compiler grammar (CONTRACT v1, fieldc/ARM64 asm) is
stricter: counted lists (`束 dst n srcs…`, `寫 slot n bits…`), 歩 count
required, amp/raw values as DECIMAL f32 BIT PATTERNS (no float literals),
world header = glyph 界 (`w h n_slots seed`), not `:new`. Batch law =
../mc-fieldlang-asm/packages/fieldlang-asm/CONTRACT.md; language design =
2026-08-01-field-lang.md. REPL float/varargs sugar, if built, desugars
to the batch forms before journal append.

REPL meta (ASCII, colon-prefixed, never journaled):

| cmd | args | effect |
|---|---|---|
| `:new` | `w h n_slots` | fresh `World::new` + fresh `Journal::create` |
| `:load` | `path` | `Journal::read_all` → `World::replay`, session continues from here |
| `:save` | `[path]` | `Journal::flush()` — only disk write |
| `:digest` | — | print `world.digest()` (hex) |
| `:render` | `[path]` | dump `world.render_png()` to path |
| `:quit` / `:q` | — | exit; 假 proposal: warn if unsaved buffered ops pending |

Example transcript 假 (illustrative, not run — no impl exists yet):

```
場> :new 32 32 16
world booted  cfg=32x32 slots=16  digest=0000000000000000
場> 種 2 42
種 slot=2 seed=42        step=0   digest=3f2a9c1b0e77d4a5
場> 撃 8 8 0.5
撃 x=8 y=8 amp=0.5        step=0   digest=91bd7702aa451fc0
場> 歩 100
歩 count=100              step=100 digest=5c0e441adf9b2231
場> 縛 3 0 2
縛 dst=3 a=0 b=2          step=100 digest=a17f0c93be442d18
場> 束 4 2 3
束 dst=4 srcs=[2,3]       step=100 digest=6d2ee81bc0559a04
場> 觀 2 3
觀 key=slot2 top3: (4,0.87) (3,0.52) (2,1.00)
場> :digest
digest=6d2ee81bc0559a04
場> :save session.fldj
saved → session.fldj (48B header + 5 ops)
場> :quit
```

Prompt glyph 假 proposal: `場` (field) — matches project's CJK naming
register already in use (mu, kami packages) rather than importing a
Western `>>>` convention.

---

## 6. Open questions / UNVERIFIED

- ~~寫/WriteRaw glyph unconfirmed~~ RESOLVED 2026-08-01: 寫 locked by
  fieldlang v0 (see §0). Remaining growth question (control flow, named
  slots) still open, last bullet.
- `Journal::open_append(path)` does not exist on `field-unified` 真 — needed
  before `:load` → type more → `:save` is anything but a dead end. No owner
  assigned yet; natural fit for whoever holds W3/journal on the next wave.
- Nothing in this doc has been built or run — every number/behavior is
  design intent (假) except direct quotes of frozen source (真, cited
  inline). No cargo build, no REPL binary exists in this worktree.
- Wire-format-as-RPC-frame idea (§4, journal binary format doubling as a
  network protocol) is a passing note, not evaluated for security/framing
  correctness — flag before anyone builds a network shell on it.
- CJK glyph collision with future language growth (e.g. control flow,
  named slots) not addressed — this draft only covers the 6(+1) primitive
  ops named by the steer.
