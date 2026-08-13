# FIELD engine guide — W7 docs, 2026-08-01

Base: field-unified @7eb6775b, this worktree field/docs. Source of truth =
packages/field/src/*.rs (frozen signatures, docs/design/2026-08-01-field-contract.md).
真=measured this session · 記=derived/from other docs · 假=assumed/extrapolated.

---

## 1. Architecture overview

ONE substrate, no VSA/plane seam (forced by docs/research/2026-08-01-field-adversary.md).

- **d = W·H**. A hypervector IS a field slice: W×H grid, row-major flat `Vec<f32>`.
  `FieldConfig{width,height}`, both MUST be powers of two ≥2 (radix-2 FFT).
- **ONE op-set, FFT-domain** (ops.rs):
  - `bind(a,b) = ifft(fft(a) ∘ fft(b))` — 2D circular convolution.
  - `unbind(a,key) = ifft(fft(a) ∘ conj(fft(key)))` — correlation / matched filter.
  - `bundle(slices) = Σ` — plain superposition, no normalization.
  - `similarity(a,b)` — cosine, zero-norm guard → 0.0.
- **Plane-as-bind** (plane.rs): the damped-wave FD stencil, regrouped, IS one bind
  by a FIXED kernel: `next = ifft(fft(cur)∘K1) + a_prev·prev`. K1 built analytically
  from the stencil's Fourier symbol — no image-of-a-stencil FFT needed. Boundary =
  TORUS (periodic wrap; V1 law — plane-v0's Dirichlet was V0 scaffold only).
- **N×d resident store** (store.rs): the field is NOT one vector — capacity decays
  ~1/√K when K items share one bundle (§4). Store = `Vec<f32>` row-major N×d,
  RAM-resident by construction, no demand-paging. Slots 0/1 reserved = plane
  cur/prev; `probe()` scans free slots (≥`RESERVED_SLOTS`=2) only.
- **Cold journal** (journal.rs): SSD = backing ONLY. `append()` buffers in RAM,
  `flush()` is the only disk write, called off-cadence — never inside the tick
  hot path. Replay (`World::replay`) = re-derive state bit-exact from seed+ops.
- **Render** = probe of the spatial marginal: reshape flat vector → W×H, amplitude
  → pixel (plane::to_gray/encode_png). No separate renderer module.
- **Determinism law** (ENTROPY.md): state = f(seed). No `rand` crate, no clock in
  state, all `f32` persisted as bit patterns (`to_bits`/`from_bits`), digests via
  FNV-1a over bit-pattern bytes.

```
World { cfg, params: WaveParams, fft: Fft2, kernel: WaveKernel, store: Store, step_index }
  .apply(Op)   — the ONLY mutation door (journal replay = for op { world.apply(op) })
  .tick()      — ONE wave_step on slots 0/1, the 120fps hot path
  .render_png()— plane::encode_png(cur())
  .digest()    — store.digest() ^ mix64(step_index)
```

---

## 2. Per-module API summary

### core (W1) — fft.rs, atoms.rs, lib.rs geometry
- `FieldConfig::new(w,h)` — asserts pow2 ≥2. `.d()` = w*h. `Default` = 128×128.
- `Slice{cfg,data}` — `zeros`, `from_data` (asserts len==d), `idx(x,y)`, `norm()`,
  `digest()` (FNV-1a over f32 LE bit bytes, row-major).
- `C32{re,im}` — sovereign minimal complex, no external crate.
- `mix64(u64)->u64` SplitMix64 finalizer · `seed_unit(seed,tag)->f32 ∈[-1,1]` ·
  `fnv1a(&[u8])->u64` — canonical determinism primitives, used everywhere.
- `naive_dft2`/`naive_idft2` — O(d²) TEST ORACLE ONLY, +i-sign inverse, 1/d norm.
- `Fft2::new(cfg)` precomputes bit-reversal perms + twiddle tables once.
  `.forward(&[f32])->Vec<C32>` unnormalized, layout `spec[ky*W+kx]`.
  `.inverse(&[C32])->Vec<f32>` 1/d-normalized, `inverse(forward(x))==x`.
  Iterative radix-2 DIT, rows then columns (separable 2D), interior-mutability
  scratch buffer so `&self`-only signatures hold.
- `atoms::seeded_atom(f,cfg,seed)->Slice` — unit-magnitude spectrum, random
  phase from `mix64` only, conjugate symmetry enforced (self-conjugate DC/Nyquist
  bins = ±1 from seed). Guarantees: `‖atom‖==1`; `unbind(bind(a,x),a)≈x` EXACT
  inverse (not approximate — unit-magnitude spectrum property); cross-seed
  `|similarity|` small.
- ops.rs (archon-written, thin, semantics frozen): `bind`, `unbind`, `bundle`,
  `similarity` — see §1.

### plane (W2) — plane.rs
- `WaveParams{c,dt,damping,dx,seed,range}`, `Default`: c=1.0 dt=0.5 damping=0.02
  dx=1.0 seed=0 range=0.06. `.courant()=c·dt/dx`. `.stable()` ⟺ courant ≤ 1/√2 (CFL).
- `WaveKernel::new(cfg,&params)` — analytic `K1(kx,ky)=(2-dd)+k·L̂(kx,ky)`,
  `L̂ = 2cos(2π kx/W)+2cos(2π ky/H)-4`, `a_prev = -(1-dd)`, `dd=damping·dt`,
  `k=courant²`. Built directly in Fourier space, no stencil-image FFT.
- `wave_step(f,&kernel,cur,prev)->Slice` — ONE bind by K1 + `a_prev·prev`.
- `wave_step_reference(&params,cur,prev)->Slice` — direct 5-point periodic-wrap
  stencil; PARITY ORACLE for `wave_step` (gate: max|diff| ≤1e-3 over ≥200 steps).
- `excite(cur,prev,x,y,amplitude,seed)` — Gaussian strike σ=2 r=8, displacement
  (both cur AND prev += amp·g), jitter=`1+0.1·seed_unit(seed,(x<<32)|y)`, TORUS wrap.
- `energy(&params,cur,prev)->f64` — kinetic(cur-prev)+potential proxy.
- `has_nonfinite(&Slice)->bool`.
- `to_gray(&Slice,range,&mut Vec<u8>)` / `encode_png(&Slice,range)->Vec<u8>` —
  deterministic quantisation (plane-v0-identical), 8-bit grayscale PNG (png crate).

### journal (W3) — journal.rs, store.rs
- `Op` enum (the ONLY mutation alphabet): `SeedAtom{slot,seed}` ·
  `Excite{x,y,amp_bits}` · `Step{count}` · `Bind{dst,a,b}` · `Bundle{dst,srcs}` ·
  `WriteRaw{slot,data_bits}`. All f32 payloads carried as `u32` bit patterns.
- Binary format v1 LE: magic `"FLDJ"`, version=1, w,h u32, params (c,dt,damping,dx
  as f32-bits u32 · seed u64 · range f32-bits u32), n_slots u32 — header = 48
  bytes exactly. Then ops: tag u8 + payload; truncated trailing op → `io::Error`.
- `Journal::create(path,cfg,&params,n_slots)->io::Result<Self>` writes header
  immediately. `.append(&Op)` buffers RAM ONLY (hot-path safe, never touches
  disk). `.flush()->io::Result<()>` — the only disk write, off-cadence.
- `Journal::read_all(path)->io::Result<(FieldConfig,WaveParams,usize,Vec<Op>)>` —
  bit-exact roundtrip of header + full op sequence.
- `store::RESERVED_SLOTS = 2` (slot0=wave cur, slot1=wave prev).
- `Store::new(cfg,n_slots)` asserts `n_slots≥RESERVED_SLOTS`; `row`/`row_mut`/
  `write`/`read` — plain `Vec<f32>` N×d, resident by construction.
- `Store::probe(key,top_k)->Vec<(usize,f32)>` — cosine similarity, scans slots
  `RESERVED_SLOTS..` only, sorted desc, deterministic tie-break (lower slot
  index first).
- `Store::digest()->u64` — FNV-1a over all rows' f32 bit patterns.

### spike (W4) — tools/field-spike-mlx/ (Python/MLX, NOT Rust — measurement only)
- `spike.py` / `results.json`: MLX vs numpy-CPU timing on Apple M1 Pro for (1)
  `bind` = fft2→cmul→ifft2 fp32 and (2) `probe` = matmul key(1×d)@store(N×d)ᵀ,
  d=16384. NOT wired into the `field` crate — separate feasibility probe.
- **Measured 記** (docs/research/2026-08-01-field-metal-spike.md):
  bind 128²: MLX 0.42ms / numpy 0.30ms — both PASS (budget 8.33ms).
  bind 512²: MLX 0.98ms / numpy 6.67ms — both PASS.
  probe N=4096,d=16384: MLX 2.75ms PASS / numpy 10.38ms FAIL.
  probe **N≤4096** is the safe operating point on this hardware; probe N=16384:
  MLX 11.41ms FAIL, numpy 44.77ms FAIL — probe scales ~N·d and dominates cost,
  not bind. Fused tick (bind128+bind512+probe N16384, `mx.compile`): MLX 11.46ms
  FAIL, headroom 0.73× (needs ~27% cut or smaller N/d).
  **Verdict**: GPU tick fits 120fps only if resident probe store ≤ N≈4096 rows
  at d=16384 (probe 2.75ms + both binds ≈1.4ms ≈ 4.15ms, ~2× headroom).

### determ (W5) — tests/determinism.rs (gate-only, no src ownership)
Fixture 32×32, n_slots=16, op sequence SeedAtom×8 + Excite×3 + Step{100} +
Bind + Bundle. Five gates, ALL PASS 真 (this session, `cargo test --test
determinism`, 0.11s):
1. live-apply same ops on two independent `World`s → digest bit-identical.
2. Journal record→flush→read_all→`World::replay` → digest == live digest
   (bit-exact replay = ENTROPY.md core claim).
3. `render_png()` bytes identical across two independent live runs.
4. digest DIFFERS when seed differs (anti-vacuous — catches degenerate all-zero
   passes).
5. `Slice::digest` stable: same `seeded_atom` seed twice → same digest.

### world (W6) — world.rs, src/bin/bench.rs
- `World::new(cfg,params,n_slots)` builds `Fft2`, `WaveKernel`, `Store`.
- `.apply(&Op)` — the single mutation door, one match arm per `Op` variant,
  semantics per journal.rs doc (SeedAtom→atoms::seeded_atom, Excite→plane::excite
  on slots 0/1, Step→N×tick(), Bind/Bundle→ops:: writes to dst, WriteRaw→raw bits).
- `.tick()` — ONE `wave_step` on slots 0/1: `slot1←old cur, slot0←next,
  step_index+=1`. HOT PATH target: 120fps floor, zero I/O, zero journal calls.
  Known cost: 3 d-sized `Slice` allocs/tick forced by the frozen
  `wave_step(&Slice,&Slice)->Slice` signature (no scratch slot in `World`) —
  flagged in-source as a post-merge fusion candidate.
- `.replay(cfg,params,n_slots,&[Op])->World` — fresh world + apply all ops.
- `.digest()->u64` = `store.digest() ^ mix64(step_index)`.
- `.render_png()->Vec<u8>` = `plane::encode_png(cur(), params.range)`.
- `field-bench` bin: argv `--width(128) --height(128) --slots(4096)
  --probe-keys(1) --frames(1000) --render-every(12)`; loop = tick+probe+render;
  reports ms/frame mean/p50/p99 vs 8.33ms (120fps) floor.
  **Measured 真** (this session, `cargo run --release`, this CPU, radix-2 Fft2,
  no SIMD/GPU path in the Rust crate):
  | config | mean ms/frame | p99 | vs 8.33ms |
  |---|---|---|---|
  | W128 H128 slots=4096 (bench defaults) | 88.67 | 124.05 | **FAIL** |
  | W128 H128 slots=256 | 5.92 | 7.56 | PASS |
  | W64 H64 slots=4096 | 20.47 | 23.51 | **FAIL** |
  CPU-only build FAILS the 120fps floor by ~10× at bench defaults; only fits
  at small N (≈256 @ d=16384) or smaller d. Consistent DIRECTION with the MLX
  spike (probe cost dominates, scales with N·d) but the Rust crate has no
  MLX/Metal/SIMD backend wired in — the spike numbers are NOT this crate's
  numbers. See open questions §5.

### capacity (W8) — tests/capacity.rs, docs/research/2026-08-01-field-capacity.md
Measures adversary §2 bound in-substrate: bundling K items into d dims degrades
SNR ~√(d/K); the N×d store (not bigger bundles) is the fix. Real `ops::{bundle,
similarity}`; atom generation via a TEST-LOCAL helper (atoms.rs was `todo!()`
when this test was authored) pinned to lib `naive_idft2` within 1e-6
(`fast_idft_equiv_naive` test) — see open questions §5 re: re-validation now
that atoms.rs is real.
**Measured 真** @ cfg 64×64, d=4096, 256 fresh distractors/K
(`cargo test --test capacity`, 3/3 pass, 14.1s):
| K | member μ | mean|dist| μ | margin | accuracy |
|---|---|---|---|---|
| 1 | 1.000000 | 0.013100 | 0.986900 | 100.0% |
| 4 | 0.496594 | 0.012951 | 0.483644 | 100.0% |
| 16 | 0.252563 | 0.012101 | 0.240462 | 100.0% |
| 64 | 0.126587 | 0.013503 | 0.113084 | 100.0% |
| 256 | 0.061475 | 0.012763 | 0.048711 | 82.0% |

記 model: member μ ≈ 1.004/√K (fit within 1.5%); distractor floor FLAT ≈0.8/√d
(K-independent — cross-term std = 1/√d, not 1/√(d/K)); retrieval cliff
K*≈d/(2·ln N), measured cliff falls in (64,369) at d=4096,N=256 — matches
100%@K=64 / 82%@K=256. Store slot budget rule 記: `K ≤ d/22` per slot (half the
cliff, margin) → d=4096: use K≤64 (100%, 6× headroom); d=16384: use K≤256
(假, extrapolated — NOT independently measured at d=16384 in this session).

---

## 3. Invariants + gates table

### Invariants (cross-module, checked by construction or by gate)
| invariant | enforced by | verified |
|---|---|---|
| state = f(seed); no `rand`, no clock in state | atoms.rs mix64-only, WaveParams has no time field | 真 determinism.rs 5/5 |
| f32 persisted as bit patterns (never raw float bytes) | journal op payloads = u32 bits, digests = FNV1a(to_bits) | 真 store_journal.rs 7/7 |
| bind/unbind exact inverse for atoms (not approximate) | atoms.rs unit-magnitude spectrum guarantee | 真 core.rs `unbind_bind_recovers_x` (rel≤1e-3) |
| torus/periodic wrap boundary (V1, not Dirichlet) | plane.rs `excite` modulo wrap, `wave_step_reference` periodic stencil | 真 plane_parity.rs 6/6 |
| plane FFT-bind ≡ direct stencil (parity) | K1 spectrum construction vs `wave_step_reference` | 真 ≤1e-3 @32×32,64×64 (200 steps); 128²/512² deferred — see §5 |
| reserved slots 0/1 = plane cur/prev; probe skips them | `store::RESERVED_SLOTS`, `Store::probe` range | 真 store_journal.rs `probe_topk_...`, world.rs `probe_scans_free_slots_only...` |
| journal `append` never touches disk; `flush` is the only write | Journal internal buffering | 真 `journal_append_buffers_ram_only_flush_writes` |
| journal replay bit-exact == live apply | `World::replay` + digest | 真 determinism.rs `journal_replay_matches_live_digest` |
| `World::apply` is the ONLY mutation door | single match, no other `pub` mutator on World | 記 by construction (no other `&mut self` writer in world.rs) |
| capacity degrades ~1/√K per bundle; N×d store is the fix, not bigger bundles | capacity.rs measurement | 真 @d=4096; 假 extrapolated d=16384 |
| tick hot path: zero I/O, zero journal flush, no mmap | `tick()` body — read/write store rows + wave_step only | 記 by construction; NOT perf-verified to hit 120fps (see below) |

### Gates (contract §Gates, this session's run)
| gate | command | result |
|---|---|---|
| build | `cargo build -p field` | 真 PASS (4.74s clean, 1 pre-existing dead_code warn) |
| build+tests | `cargo build -p field --tests` | 真 PASS |
| core (W1) | `cargo test -p field --test core` | 真 8/8 PASS |
| plane parity (W2) | `cargo test -p field --test plane_parity` | 真 6/6 PASS |
| journal+store (W3) | `cargo test -p field --test store_journal` | 真 7/7 PASS |
| determinism (W5) | `cargo test -p field --test determinism` | 真 5/5 PASS |
| world (W6) | `cargo test -p field --test world` | 真 11/11 PASS |
| capacity (W8) | `cargo test -p field --test capacity` | 真 3/3 PASS (14.1s) |
| **total** | `cargo test -p field` | **真 40/40 PASS** |
| 120fps floor bench (W6) | `field-bench` (release, defaults) | 真 **FAIL** — 88.67ms mean vs 8.33ms floor (see §2 world, §5) |
| MLX/Metal spike (W4) | `tools/field-spike-mlx/spike.py` (Python, separate) | 真 mixed — bind PASS both sizes; probe PASS only at N≤4096; fused tick FAIL (0.73× headroom) |
| bun | n/a — no TS touched in this worktree | — |

---

## 4. Open questions

- **CPU perf gap, unresolved**: `field-bench` on this machine (release, radix-2
  Fft2, no SIMD/GPU) FAILS 120fps by ~10× at bench defaults (W128 H128
  slots=4096 → 88.67ms). The MLX spike (tools/field-spike-mlx) shows the SAME
  qualitative bottleneck (probe scales N·d, dominates) but is a separate
  Python/MLX-only measurement never wired into the `field` crate itself. Is a
  Metal/MLX-backed `Fft2`/`Store::probe` path planned for the Rust crate, or
  does the 120fps target get revised (smaller N/d, amortized/batched probe,
  lower fps floor)? Contract doesn't assign this — no wave owns it yet.
- **capacity.rs test-local atom drift**: authored while atoms.rs was
  `todo!()`, pinned to `naive_idft2` within 1e-6 via `fast_idft_equiv_naive`.
  atoms.rs is now real (merged, W1 landed) — has capacity.rs been re-run/
  cross-checked against the REAL `atoms::seeded_atom` to confirm the two
  atom constructions actually agree, or does the duplicate test-local
  function remain unreconciled dead-weight / a drift risk if atoms.rs changes?
- **plane_parity full-size deferred**: contract + plane_parity.rs both flag
  128×128/512×512 parity as "post-merge on fast FFT". The SAME radix-2 `Fft2`
  now runs those sizes fine functionally (used in this session's bench run)
  — but no `parity_128x128`/`parity_512x512` test exists yet. Should W2's
  parity gate be extended to those sizes now that fast FFT has landed?
- **store slot-budget rule at d=16384**: `K≤d/22` is measured ONLY at d=4096
  (64×64); the d=16384 (128×128) figure is 假 extrapolated from the
  K-independent noise-floor model, not independently run. Worth a real
  capacity.rs sweep at 128×128 before any world-capacity planning leans on it.
- **`World::tick` alloc**: 3 d-sized `Slice` allocs/tick, flagged in-source as
  fusable post-merge (frozen `wave_step` signature forces cur/prev/next
  Slices). This session's bench numbers suggest FFT+probe compute dominates,
  not allocation — fusing the copies alone likely will NOT close the 88ms→
  8ms gap at N=4096,d=16384 on CPU. Worth confirming with a profile before
  investing in the fusion.
- **W7 contract wording vs actual deliverable**: contract §Ownership lists W7
  as "FIELD.md amendments · HANDOFF.md log entry"; this session's task instead
  specified this standalone guide file (docs/design/2026-08-01-field-guide.md).
  Neither FIELD.md nor HANDOFF.md was touched here — flag for whoever owns
  merge-up whether those amendments are still expected separately.
