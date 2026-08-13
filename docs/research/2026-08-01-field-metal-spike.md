# FIELD METAL/MLX SPIKE (08-01) — measured, Apple M1 Pro

Measurement only, no Rust touched. Branch `field/spike` off `field-unified`
(worktree `~/projects/magic-crystal-field-spike`). Script:
`tools/field-spike-mlx/spike.py`, raw dump: `tools/field-spike-mlx/results.json`.
Answers docs/research/2026-08-01-field-recon.md §2 + adversary verdict §4:
does the field-engine's per-tick primitive set fit the 120fps budget
(8.33ms/frame) on MLX/Metal on this machine?

## Device / env

| field | value |
|---|---|
| CPU | Apple M1 Pro |
| RAM | 16 GiB (17179869184 bytes) |
| macOS | 26.5.2 |
| Python | 3.9.6 (venv, `tools/field-spike-mlx/.venv`) |
| mlx | 0.29.3 |
| numpy | 2.0.2 |
| seeds | `mx.random.seed(0)`, `np.random.seed(0)` |
| protocol | 20 warmup iters (discarded) + 200 measured iters per test |
| budget | 8.3333 ms/frame (120 fps) |

## Results

### (1) bind = fft2 → complex mul → ifft2, fp32

| test | mean ms | p50 ms | p99 ms | vs 8.33ms |
|---|---|---|---|---|
| MLX 128×128 | 0.4185 | 0.3951 | 0.8024 | PASS |
| MLX 512×512 | 0.9764 | 0.9622 | 1.1279 | PASS |
| numpy CPU 128×128 | 0.3041 | 0.2925 | 0.4215 | PASS |
| numpy CPU 512×512 | 6.6717 | 6.6652 | 7.2691 | PASS |

Both sizes fit the budget on MLX; MLX is ~6.8x faster than numpy at 512² (the
size that matters — 128² is cheap on either device). numpy is marginally
faster than MLX at 128² (dispatch overhead dominates at this size on GPU).

### (2) probe = matmul key(1×d) @ store(N×d).T, d=16384

| test | mean ms | p50 ms | p99 ms | vs 8.33ms |
|---|---|---|---|---|
| MLX N=4096 | 2.7542 | 2.7539 | 3.0252 | PASS |
| MLX N=16384 | 11.4139 | 11.3890 | 11.9692 | **FAIL** |
| numpy CPU N=4096 | 10.3774 | 10.1603 | 14.9796 | FAIL |
| numpy CPU N=16384 | 44.7729 | 43.3659 | 61.4680 | FAIL |

MLX is 3.8x–3.9x faster than numpy CPU at equal N, consistent with unified-
memory zero-copy dispatch to the GPU (per recon doc §2). But N=16384 alone
already overruns the whole 120fps frame budget on MLX (11.41ms mean vs
8.33ms) — the probe step scales linearly with N and d, and at N=16384,
d=16384 the matmul is 16384×16384 = 268M multiply-adds against a
1×16384 key, i.e. ~4.4 GFLOP per probe; that's the dominant cost, not the
bind step.

### (3) fused tick = bind(128²) + bind(512²) + probe(N=16384, d=16384), `mx.compile`

| test | mean ms | p50 ms | p99 ms | verdict | headroom |
|---|---|---|---|---|---|
| MLX compiled | 11.4554 | 11.3730 | 12.7278 | **FAIL** | 0.73x |

`mx.compile` fusion does not rescue this — the fused tick's mean (11.46ms) is
close to the probe-alone number (11.41ms), i.e. probe dominates the tick and
compile fusion barely changes probe's cost (it's already one big op, matmul,
nothing to fuse across). Headroom factor = budget/measured = 0.73×, meaning
the fused tick takes ~1.37x the available frame time — it does NOT fit.

## Verdict

**GPU tick does NOT fit the 120fps (8.33ms) budget on this machine at
N=16384, d=16384.** Root cause isolated: the `probe` matmul at full store
size (N=16384) alone consumes 11.41ms — already over budget before adding
either bind. `bind` at both 128² and 512² is cheap (≤1ms each, PASS
individually) and not the bottleneck.

Headroom factor for fused tick: **0.73×** (need to cut compute by ~27% or shrink
N/d to fit one frame at 120fps).

Where budget is met: probe at N=4096 (2.75ms) + both binds (0.42+0.98ms)
would total ~4.15ms — well within 8.33ms, ~2× headroom. So the field engine
fits 120fps on this hardware only if the resident probe store is kept at
N≈4096 rows (at d=16384), not N=16384. Scaling N to 16384 rows needs either:
lower fps target, smaller d, batching probe over multiple frames, or a
cheaper approximate lookup (recon doc §1's static-lookup framing suggests a
capacity/latency trade is architecturally expected, not a surprise here).

## UNVERIFIED

None — every test above executed and produced real numbers (see
`tools/field-spike-mlx/results.json` for full JSON, all fields present, no
mocked/faked entries). Install (`pip install mlx numpy` in venv) and run both
succeeded cleanly, no errors caught.
