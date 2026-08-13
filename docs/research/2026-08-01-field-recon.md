# FIELD RECON (08-01, whale-flash lane) — tracr / Apple-Silicon compute / MuZero loops
Three questions: (1) tracr+RASP expressiveness & KV-store, (2) M1 Pro compute
surface (MLX, ANE, f16 TFLOPs), (3) minimal MuZero-style build loop + 120Hz
world-model sketch. RESEARCH ONLY — no code touched. All claims cited.

## 1 · Tracr / RASP — what compiles, hard limits, KV lookup

WHAT: tracr = TRAnsformer Compiler for RASP. Input: RASP program (Weiss et al.
2021 — DSL over sequences, ops = selectors (attention) + elementwise/sequence
ops), vocab set, `max_seq_len` (compile-time), causal flag, `mlp_exactness`
(default 100). Output: weights of a STANDARD decoder-only transformer (no layer
norm), assembled in 6 steps; selectors → attention patterns, ops → MLPs;
residual-stream dims fixed per program. Compiled models always expect a BOS
token (all-False selector rows need a target, else softmax semantics break).
src: github.com/google-deepmind/tracr · arxiv.org/abs/2301.05062 ·
github.com/srush/raspy (interactive RASP).

HARD LIMITS (compiler/architecture):
- max_seq_len is COMPILE-TIME; weights are length-specific (non-uniform
  programs) → no length generalization; new length = recompile.
- NO data-dependent control flow, NO loops (single pass, fixed depth). RASP-L
  (Lindner et al., used by arxiv.org/abs/2310.16028) adds bounded loops for
  length-generalizable programs — but still no data-dependent iteration.
- Values live in a finite vocab; arbitrary R→R ops approximated via MLP with
  exactness knob (mlp_exactness) — exactness costs width/depth.
- Selectors are Boolean; attention is hard (softargmax-style).
- Model size grows with: #values (residual width), program length (depth),
  vocab (embeddings), max_seq_len (positional). No free lunch.
- tracr targets Haiku decoder-only w/o layernorm; other impls = redo step 6.
- Caveat: RASP is "too expressive" — contains programs representable but not
  learnable from scratch (arxiv.org/abs/2310.16028) → compiled ≠ trained.

KV STORE / DB LOOKUP IN RASP: YES for in-context lookup (associative recall —
canonical transformer capability; induction-head/factual-recall literature:
arxiv.org/abs/2412.06538, "Remember" task in associative-memory work). Pattern:
selector = key-equality match over the in-sequence table rows → attend to value
column (same mechanism as tracr sort/hist examples). Bounds:
- table must be IN the input sequence → capacity = max_seq_len;
- fixed table can be baked into attention weights at compile time (tracr does
  exactly this for selectors) — bounded by residual width, immutable;
- NO mutable store: no writes, no updates, no persistence across sequences
  (state = context window only) → a persistent/mutable KV database is NOT
  expressible; a static lookup is.

## 2 · Apple Silicon compute for a learned field engine (M1 Pro)

MLX (ml-explore/mlx): NumPy-like array framework, Metal backend, Apple-only.
- UNIFIED MEMORY: weights/activations/KV-cache in one physical pool; ops
  dispatch CPU↔GPU with zero copies (github.com/ml-explore/mlx).
- LAZY EVAL: graph built dynamically, materialized on demand; mx.compile fuses
  subgraphs (batches GPU dispatches) — big deal for small-net overhead.
- MMAP: mlx-lm can wire model weights + KV cache to physical RAM (macOS 15+)
  → memory-mapped weights for huge fields, cold-start lazy page-in.
- Devices: CPU + GPU ONLY. NO ANE support (ml-explore/mlx#18 — "MLX won't
  support ANE unless something changes"). Quantization 4/8-bit native.

NEURAL ENGINE (ANE): 16 cores on every M1→M5 (same block, base Air = Max);
M1/M1 Pro/M1 Max = 11 TOPS (vendor INT8 metric; fp16 ≈ half-rate, quoted
"11 TFLOPS F16" informally — treat as ~5.5-11 fp16, ambiguous).
src: hollance/neural-engine (docs/supported-devices.md: M1 Pro 16 cores 11
TOPS) · reddit r/Applesilicon ANE specs (M1 11 / M2 15.8 / M3 35 / M4 38 TOPS).

REACHABILITY — the real answer: PUBLIC = CoreML only (mlprogram, fp16,
compute_units=ALL; CoreML decides ANE/GPU split; conv-heavy fp16 ops land on
ANE; no dispatch control, no IOSurface layout control). NO public direct
framework (stackoverflow 69983492). DIRECT = private AppleNeuralEngine.framework
via reverse-engineering, real & working in 2026:
- maderix/ANE — direct ANE training via _ANEInMemoryModel/_ANERequest/
  _ANEIOSurfaceObject; discovered 40+ private classes; bypasses CoreML.
- skyfallsin/ane.cpp + apple-neural-engine-field-guide — 4B/9B LLMs end-to-end
  on ANE, GPU/CPU idle; M3 Max: 11.66 tok/s (int8, 4B), 28.62 tok/s across 4
  streams. Key hardware facts (pradeep.md 2026-03-30):
  * weights are BAKED INTO kernels at compile time (compile cache) — ANE never
    streams weights from DRAM; only activations flow (IOSurface + DMA).
  * per-dispatch cost ≈ 119µs FIXED + bytes/78GB/s → DISPATCH-BOUND, not
    compute-bound; "nowhere near saturating M1's 11 TOPS".
  * min spatial width 32 (W-lane) → single-token decode wastes width; prefill
    packs 4 lanes → ~2× throughput.
  * ANE does matmul/conv well; sequential stuff (norm, attention) → CPU bounce.
  * private API = fragile (breaks across macOS updates), App-Store-banned.
- ANE disabled for third-party on AVP; macOS private path currently works.
- maderix M4 reverse-eng notes: conv 3× faster than matmul on ANE; bypassing
  CoreML = 2-4× more throughput (dispatch control).

REALISTIC f16 TFLOPs, M1 Pro:
- GPU: 14-16 cores, ~5.2 TFLOPS fp32 peak (cpu-monkey/flopper); fp16 ≈ 2×
  peak (~10 TFLOPS theoretical) — GPU dual-issue fp16. MEASURED: arxiv
  2502.05317 SGEMM peak M1 (8-core) = 1.36 TFLOPS fp32 → M1 Pro 16-core ≈
  2.7 fp32 / ~5 fp16 realistic dense matmul; Metal tiled SGEMM ~3.8 TFLOPS
  fp32 on M1 Max (bkvogel/metal_performance_testing) → Pro ≈ 2.5.
  VERDICT: call it 2.5-3 fp32 · 5-6 fp16 TFLOPS sustained, ~40-60% of peak.
- ANE: 11 TOPS INT8 → fp16 ~5.5-11; but unreachable without CoreML or private
  API, and dispatch overhead (119µs) swamps small models.
- CPU AMX: 4×4/8×8 tiles, no precision distinction; Accelerate BLAS/vDSP
  auto-use; ~1.4 fp32 TFLOPS class (measured M1 CPU peak 1.36).
- MEMORY: M1 Pro 200 GB/s (256-bit LPDDR5); M1 base 68 GB/s.
- MLX per-op GPU overhead on M1 Pro is HUNDREDS of µs (mlx-benchmark: ReLU
  0.46ms, Gather 3.15ms, Linear 9.57ms at bench size) → a 20-op network per
  tick ≈ several ms UNLESS fused (mx.compile / one Metal kernel) or batched.
  src: github.com/TristanBilot/mlx-benchmark (M1 Pro table, mlx 0.5.0).

ENGINE VERDICT: GPU via MLX/Metal = the sane path (5-6 f16 TFLOPs, unified
mem, mmap, fusion). ANE = only if shipping via CoreML or willing to ride
private API; its win is POWER (many parallel small streams ~ battery-friendly),
not latency. 120Hz loop must fuse the whole net into 1-2 dispatches.

## 3 · MuZero-style self-play / adversarial build loops

MINIMAL RECIPE (Schrittwieser et al. 2019 + AlphaZero):
1. representation net h = repr(obs) — encodes state → latent.
2. dynamics net (s', r) = dyn(s, a) — learned transition + reward in latent.
3. prediction net (p, v) = pred(s) — policy + value heads.
4. MCTS over latent states: root = initial_inference; children via
   recurrent_inference (dyn+pred); UCB selection (PUCT), visit-count policy;
   value = mix of leaf v and trajectory rewards.
5. Self-play: policy = MCTS visit distribution (+ temp), store trajectories.
6. Train on replay: targets = MCTS policy + observed rewards + bootstrapped
   values; losses = policy CE + value MSE + reward CE (julian.ac blog,
   muzero-general wiki).
7. No rules needed: model learned end-to-end — key MuZero trick (vs AlphaZero
   which needs a perfect simulator).
src: arxiv.org/abs/1911.08265 (MuZero) · deepmind.google blog 2020-12-23 ·
github.com/werner-duvaud/muzero-general (commented reference impl, games:
ttt/connect4/othello/chess/atari) · github.com/rlglab/minizero (ToG: AlphaZero
vs MuZero on Go/Othello/Atari, efficient C++ core) ·
julian.ac/blog/2020/12/22/muzero-intuition (losses explained).

ADVERSARIAL/build-loop variant: same loop with two roles (builder vs
adversary) = minimax self-play; MuZero handles it unchanged (it's just a game
where opponent = policy). For a "build" game, dynamics net must learn the
placement/update rule of the field — this is where a learned FIELD engine
earns its keep: dynamics net = the world model.

TINY WORLD-MODEL PRECEDENTS (small, real-time-friendly):
- David Ha & Schmidhuber, "World Models" 2018 (arxiv.org/abs/1803.10122;
  github.com/hardmaru/worldmodels-experiments): VAE (z=32) + MDN-RNN (LSTM
  256) + linear controller (CMA-ES); drives CarRacing from imagination; the
  canonical minimal learned-world-model, CPU-trainable.
- werner-duvaud/muzero-general: ttt/connect4 nets are tiny CNNs/MLPs — the
  minimal MuZero you can actually train on a laptop.
- rlglab/minizero: C++ core, Go/Othello/Atari at competitive speed — the
  efficiency reference for self-play throughput.
- COUNTER-DATA: big learned-world-model video gens (Genie, GameGAN ~20fps,
  WorldPlay/RELIC/HY-World 25fps DiT) are NO guidance for 120Hz — different
  regime entirely (latent-small >> pixel-diffusion).

120Hz SKETCH on M1 Pro (8.33ms/frame budget):
- Shape: small grid field (e.g., 64×64 cells, channels=2-8) or sparse-latent
  state; latent d=16-64; nets = 2-4 conv layers (repr) + MLP/conv-GRU dyn +
  2-head MLP pred; 50k-1M params total.
- Compute per forward: ~1M params → 2 MFLOP fp16 → GPU ~1-2µs compute at
  5 TFLOPS; ANE fused ~1ms/dispatch; GPU per-op overhead ~0.3-1ms → MUST fuse
  (mx.compile / single Metal kernel / CoreML mlprogram).
- MCTS at 120Hz: NO full search. Options:
  (a) 0-sim rollout: 1 fused forward per tick (repr+dyn+pred) → trivially
      120Hz, GPU <0.5ms; adversary = second net, 2 forwards. ← realistic.
  (b) 8-32 sims batched into one dynamics call (batch dim = sims) → still
      single fused graph, few ms; 60-120Hz plausible with 8-16 sims on GPU.
  (c) ANE: batch W-lane ×4 (min width 32), ~4-8 dispatches fused ≈ 0.5-1ms
      per batched step → fits but wastes width; real win = low power
      background streams, not latency.
- Budget rule: keep TOTAL dispatches ≤ 4-8 per tick; anything looped per-sim
  per-node (Python/MLX eager) kills 120Hz — fuse or batch.
- Cache: mlx mmap weights + persistent compiled graph → cold-start page-in,
  zero-copy CPU↔GPU for field I/O.

## UNVERIFIED / caveats
- M1 ANE fp16 TFLOP exact number is vendor-ambiguous (11 TOPS = INT8; fp16
  half-rate assumed ~5.5) — no first-party fp16 figure; treat as ≤ GPU fp16.
- Private-ANE path (maderix/ane.cpp) verified working as of 2026-03 but is
  unsupported; OS updates can break it.
- 120Hz numbers are BUDGET MATH from cited benchmarks, not measured on this
  machine — needs a spike before committing (Metal/MLX fused microbench).
