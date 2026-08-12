# FIELD adversary review — Nyari's clone, 2026-08-01

Target: FIELD.md (One-Field Decree) + AI-MU.md. V0 atoms under attack:
(a) `field` = VSA/HRR d=8192 bind/bundle/probe, mmap-backed store.
(b) `plane` = 512² wave-eq slice → PNG.
No diplomacy. Corpses kept. 真=measured 記=remembered 假=assumed.

---

## 1. Do the two atoms compose into ONE field, or two modules relabeled?

VERDICT 假→真: as SPEC'D they are TWO MODULES with a PNG seam = the exact banned sin.
Composable ONLY if d + op-set unified at V0. Do it now or rebuild.

因 the two atoms are different math objects:
- `field` VSA store → bag of high-d vectors · ops global content-addressed
  (bind/bundle/probe) · NO geometry · addressing = similarity.
- `plane` wave-eq → 512² grid · state = (amplitude,velocity)/cell · ops LOCAL
  5-point stencil (∂²u/∂t²=c²∇²u) · geometry-native · NO symbolic content.
→ no shared representation. bundle(v,w)=v+w lives in 8192 near-orthogonal random
  dirs; superpose(wave)=interference lives in 262144 spatially-correlated cells.
  "both are addition" = false friend; incompatible spaces.

非 hand-wave — there IS one real unification, and it's forced by the HRR op itself:
→ HRR bind = circular convolution = multiply by a CIRCULANT matrix =
  diagonalized in Fourier basis = a shift-invariant filter = ONE step of a linear
  field PDE. The wave propagator IS an HRR bind by a fixed kernel.
故 the plane is NOT a second module; the plane = the spatial marginal of the field,
  and field evolution = a fixed convolution atom applied in the SAME space symbols
  live in. render-to-PNG = reshape vector → W×H, amplitude→pixel. No renderer module.

定 CONSTRUCTIVE PATH (V0, non-negotiable):
- pick d = W·H (128²=16384 near the 8192 target, or 512²=262144).
- a "hypervector" IS a field slice (not an abstract 8192 blob).
- bind = FFT-domain multiply (= convolution = wave-propagate).
- bundle = superpose = interference.
- probe = correlation = matched filter = render/readout.
→ ONE space, ONE op-set (FFT-friendly), hosts symbols AND the plane AND (Q2)
  compiled Tracr structure. Two atoms collapse to one.

死 branch — d=8192 with i.i.d. Gaussian random atoms + separate 512² hand-stencil:
  reason — random basis has NO geometry, cannot host a wave plane, HARD-CODES the
  seam. This is the default naive build and it is the sin. Kill on sight.
定 atoms = localized wavelets/Gabors OR Fourier modes; binding in FFT domain.

---

## 2. VSA to host compiled (Tracr) AND learned (MuZero) — right, or dead end?
## What unifies a hypervector store with transformer weights — concretely?

VERDICT: RIGHT — but only under the Tracr-superposition reading of "VSA".
Classic frozen-atom HRR = dead end. The decree's "superposition = the linker" line
is CORRECT and is the strongest part of the whole plan.

因 Tracr packs many variables into one residual stream by assigning each symbol a
  near-orthogonal SUBSPACE; attention/MLP read/write those subspaces.
→ that IS VSA: symbols = near-orthogonal directions, bind/bundle = residual-stream
  writes, probe = residual-stream reads. VSA and a transformer residual stream are
  the SAME object viewed twice. 故 "classes/symbols → nearly-orthogonal feature
  directions" is literally true, not metaphor.

CONCRETE BRIDGE (hypervector store ↔ transformer weights):
- field state = residual stream = a hypervector (真: same tensor).
- every VSA op = a matmul by a STRUCTURED matrix:
    bind (circular conv) = CIRCULANT matrix.
    bundle (+) = addition.
    probe (⟨·,k⟩) = a projection = a row of a weight matrix.
- transformer weight = a GENERAL (dense, learned) matrix. circulant ⊂ dense.
故 ONE tensor type (a weight acting on the field vector), TWO ways to fill it:
    compile (Tracr) → weights CONSTRAINED circulant/permutation/near-orthogonal:
      exact, no data.
    learn (Mu) → the SAME weights RELAXED to dense, trained by gradient.
→ compiled = a reachable POINT in the learned weight space. Crystal starts AT the
  compiled point and keeps descending. This is exactly FIELD.md "a compiled field is
  differentiable → the world can later learn." That claim is CORRECT and mechanized.

死 branch — classic VSA as the literal substrate (MAP/HRR, fixed random codebook,
  cleanup memory):
  reason A (learning): frozen random atoms CANNOT learn new features by gradient →
    kills the Mu half outright.
  reason B (capacity): bundling K items into d dims degrades SNR ~ sqrt(d/K);
    d=8192 holds ~hundreds–low-thousands of superposed items before probe fails.
    A world does NOT fit in one bundled hypervector. (记: HRR capacity bound.)
→ correction: atoms must be a LEARNABLE embedding table; the store must be MANY
  vectors, an N×d mmap MATRIX, not one vector.

定 HONEST SUBSTRATE (say it without mysticism):
- field = an N×d matrix, mmap-backed = a PERSISTENT KV store.
- content-addressed probe over slots = ATTENTION.
- make projections learnable = a TRANSFORMER whose KV-cache lives on disk.
- keep projections circulant/near-orthogonal = VSA/compiled.
→ "hypervector store" and "transformer" are the SAME object; VSA is the compiled
  corner of the transformer weight space. NOT a dead end — provided "VSA" means
  "structured-matmul view of a residual stream," never "frozen-atom cleanup memory."

死 branch — "the field is ONE vector": reason — capacity (2B). Field = N×d matrix.

---

## 3. Ring-zero claim — what does aiwnios/TempleOS-fork buy vs macOS process + mmap + Metal?

VERDICT 真: the ring-zero claim as written is FALSE on Apple Silicon. Name it now,
amend the law. aiwnios buys a HolyC JIT + Terry-soul REPL — NOT metal, NOT ring 0.

MEASURED (this repo + aiwnios tree):
- `file ~/projects/aiwnios/aiwnios` → POSIX shell launcher. Runtime links SDL2.
  README: "aarch64 ... MacOS" as a HOSTED target beside Linux/FreeBSD.
→ aiwnios on Apple Silicon = a userspace process on Darwin/XNU drawing via SDL2.
  It is EL0, sandboxed, exactly like any other app. It does NOT own cores/GPU/ANE/RAM.

HARD FACT (state honestly, hint confirmed): Apple Silicon will NOT yield ring 0.
- exception levels: EL0 user · EL1 kernel(macOS) · EL2 hypervisor · EL3 secure monitor.
- third-party code runs EL0. EL1 is locked behind a signed boot chain (iBoot/SEP),
  KTRR/CTRR lock kernel text, PPL/SPRR protect page tables, PAC everywhere.
- ring 0 / EL1 only via (a) Apple-signed kext/DriverKit (deprecated, sandboxed,
  still not "the metal"), or (b) a kernel exploit = jailbreak-class, needs SIP/AMFI
  off, breaks every update — NOT a foundation for a sovereign OS.
- Asahi gets bare metal ONLY by REPLACING macOS (own EL1 after m1n1 EL2 shim) →
  loses Metal, loses ANE (no Apple GPU/ANE drivers). 
故 THE REAL CONSTRAINT: you cannot have Apple's accelerators AND ring 0 on the same
  booted machine. Metal+ANE (via macOS userspace) XOR bare metal (via Asahi). Never both.
  The sovereign-OS-at-ring-zero goal and the 120fps-via-ANE/GPU goal are MUTUALLY
  EXCLUSIVE on this hardware.

what aiwnios ACTUALLY buys vs plain macOS process + mmap + Metal:
+ HolyC JIT + DolDoc REPL + self-contained image + Terry aesthetic = real value for
  the "boot → speak to the plane" soul-layer.
− ring 0: no. − direct metal: no. − own ANE/GPU/RAM: no.
− on accelerators it is STRICTLY WORSE than a native macOS app: SDL2-through-
  userspace has LESS access to Metal/ANE than a native process calling MPS/CoreML/Metal.
  ANE is reachable ONLY via CoreML from signed userspace — proven by THIS repo's
  ane-spike (CoreML ComputeUnit.ALL, fp16 parity PASS). aiwnios cannot touch the ANE;
  a plain macOS process already does (measured, golden.json).
− "SSD-as-extended-memory / mmap swap": that is just mmap of a file — a normal macOS
  process does it identically, no ring 0, no aiwnios needed. And it FAILS the frame
  budget (see §4). 

定 AMENDMENT: redefine "ring zero" = "single-address-space sovereign RUNTIME": ONE
  native process, owns its memory image, no OS services in the hot loop (thread QoS
  USER_INTERACTIVE + wired hot memory + direct Metal + direct CoreML/ANE + mmap SSD
  for COLD only). Delivers ~99% of the intent WITH the accelerators. Keep aiwnios as
  the REPL/soul-of-Terry that talks to that runtime — NOT as the substrate under it.
死 branch — literal ring 0 with Metal+ANE retained: reason — physically contradictory
  on Apple Silicon. Corpse kept.

---

## 4. 120fps floor — which part physically cannot meet it, needs amendment.

120fps = 8.33 ms/frame TOTAL. Walk each piece.

PASS (not the problem):
- 512² wave-eq explicit stencil: 262144 cells × few flops → <0.1 ms on GPU, fine on
  CPU SIMD. Plane meets 120fps trivially. 真.
- VSA probe over a RESIDENT (RAM) store: N×8192 MACs on GPU/ANE — fits if working set
  in RAM. Fine.

死 BREAK #1 — SSD mmap-swap in the hot per-frame loop:
  reason (measured physics): SSD random 4KB read ~50–100µs/page. d=8192 fp32 vector =
  32KB = 8 pages. Cold-scan 10k vectors = 80k faults × ~50µs = ~4 s/frame = ~500× over
  the 8.33ms budget. Sequential BW ~5–7 GB/s buys only ~40–58 MB/frame, and random
  4KB collapses that to IOPS-bound ms-scale. The FIELD.md "SSD is fast enough for
  mmap swap" clause is the FALSE one.
  定 AMENDMENT: HOT field RESIDENT + WIRED in RAM (M-series ≤128GB unified). SSD =
  COLD backing only: checkpoint/journal per ENTROPY.md (seed+journal save), cold slots
  streamed AHEAD of need, NEVER demand-paged inside a frame. Then RAM-resident set hits
  120fps.

死 BREAK #2 — learned MuZero rollout WITH per-frame search in the hot loop:
  reason: ANE ~15–30 TOPS fp16 on M-series → 8.33ms ≈ 125–250 Gops/frame budget →
  caps a SINGLE forward pass to ~tens-of-M params low-precision; MCTS does MANY evals
  per step → blows it. "one flow through the field per frame" taken literally with a
  large learned model + search in-loop cannot hit the floor.
  定 AMENDMENT (the law change to write): 120fps is the floor for the INTERACTIVE
  FIELD TICK (wave + resident-VSA readout + render). LEARNING and PLANNING are
  AMORTIZED BELOW it, off the render cadence: interaction = ONE amortized-policy
  forward pass (no per-frame MCTS); training + search run async at lower Hz. Decouple
  the cadences or the floor is a lie.

SUMMARY of what breaks vs holds:
- wave plane: holds. resident VSA probe: holds. render/readout: holds.
- SSD-paging-in-frame: breaks → RAM-resident, SSD cold-only.
- learned-model+search-in-frame: breaks → amortize below render cadence.
故 the "everything = one flow through the field EVERY tick" ideal is what needs the
  amendment: the RENDER field ticks at 120; learning/search do not.

---

## Corpses (kept, with reasons)
死 two atoms, different d + different ops → hard-codes the banned seam. Unify at V0.
死 classic frozen-atom HRR as substrate → can't learn (frozen), can't hold a world
   (SNR ~sqrt(d/K)).
死 field = ONE vector → capacity; field = N×d mmap matrix = persistent attention store.
死 literal ring 0 + Metal + ANE on Apple Silicon → mutually exclusive; EL1 locked.
死 SSD mmap-swap in the frame loop → ~500× over budget; SSD = cold backing only.
死 per-frame MuZero MCTS at 120fps → ANE TOPS + search depth; amortize below cadence.

## What HOLDS (keep, build on)
真 superposition = the linker: VSA = Tracr residual-stream superposition = the same
   object. Strongest load-bearing claim in FIELD.md.
真 compiled ⊂ learned: circulant ⊂ dense → compiled field differentiable → "crystal
   that starts dreaming" is mechanized, not mysticism.
真 ONE field is achievable — as an N×d mmap matrix with FFT-domain structured/learned
   matmuls, where the wave plane is a fixed-kernel bind and render is a probe. Build
   THIS, not two atoms.

## One-line order to the swarm
Unify `field` and `plane` NOW: single d = a flattened spatial grid, single op-set =
FFT-domain bind/bundle/probe, plane = fixed-kernel bind + reshape-probe render, store
= N×d resident-RAM matrix with SSD cold-journal. Drop "ring 0"; ship a wired-memory
native Metal+CoreML runtime with aiwnios as the REPL skin. Split cadence: field@120,
learning amortized. Anything else rebuilds a module and calls it a field.
