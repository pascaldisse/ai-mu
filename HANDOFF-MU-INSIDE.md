# HANDOFF — branch `mu-inside` (内側学習: Mu learns the world from inside), 2026-08-01, @opus5

worktree `~/projects/mc-mu-inside` · base `mu-world@8fe6b54d` (= `field-unified@a4c9f1b6` ⊕ `mu-v0@e0d3a6cb`)
venv `~/projects/magic-crystal-mu-v0/.venv/bin/python` · build `cargo build --release -p field --bin mu-serve`
主樹 (`magic-crystal`@main) 不觸.

**Merge-ready = YES for the interface. NO for any learning claim.** Read 未驗 below before quoting a number.

## SHA chain (two lanes share this branch — see 注意)
| SHA | lane | content |
|---|---|---|
| `42a6efb8` | reward | atom-1: `--reward world\|surprise\|progress` |
| `5e402e12` | reward | atom-1 gates GI-1..4 + `MU-INSIDE.md` |
| `3006d4b7` | audit | AUDIT-INSIDE-R1 (@kimi): 全点 HOLDS |
| `dad5d08c` | reward | atom-2: GI-5 held-out gate · `surprise_rel` · residual decoding |
| `94b2e92f` `7f709b6c` `fbc94cd6` | window | atom-1 観測窓: `PATCH`/`POKEXY`, patch predictor, `GATE-WINDOW.txt` |
| `79f1aaea` | reward | atom-2 段2: 1200-tick verdicts |

注意: two lanes committed into this ONE worktree concurrently. At 22:46:15 the window lane rebuilt
`target/release/mu-serve` **mid-run** of my 1200-tick table → table spanned two binaries. Re-run
pinned to a copied binary (`/tmp/mu-serve-pin-79f1aaea`) reproduced **every deterministic figure
bit-identically** ⇒ contamination benign (the added commands are additive; RESET/ACT unchanged).
Verified, not assumed. **Law for the next lane: one worktree, one lane.**

## API 1 — reward mode (python side, `mu/online.py`)
```
--reward world         probe energy at a world-declared target site   [extrinsic TASK]
--reward surprise      mu's own model error, in-graph, stop_gradient  [intrinsic]
--reward surprise_rel  clip(err / max(energy, 1e-4), 0, 10)           [intrinsic, shout-exploit closed]
--reward progress      per-action ema_slow-ema_fast of that error     [intrinsic, learning progress]
--no-residual          disable residual (obs+delta) decoding
```
Laws baked in: reward is always `stop_gradient` (else the optimizer minimises its own objective) ·
running-RMS scaling is an in-graph `scale` input · **direction decides the denominator**: normalising a
MAXIMISED error punishes shouting; normalising a MINIMISED quality signal rewards it.

## API 2 — world door (rust side, `packages/field/src/bin/mu_serve.rs` ⇄ `mu/world_env.py`)
One subprocess = one world. Line commands in, fixed binary frames out.
```
RESET <seed>            -> frame   godseed: state = f(seed)
ACT <a>                 -> frame   excite ring site a (0..7), tick
STEP <n>                -> frame   tick n times, no excitation
POKEXY <x> <y> [steps]  -> frame   excite ARBITRARY (x,y)            [window lane]
PATCH <x> <y> <k>       -> frame   READ-ONLY k×k window, no mutation [window lane]
VIEW3 [at k] / WHEN     -> 3D render by replay / observer axis       [inherited mu-world]
frame = u64 digest | u64 step_index | f32 reward | f32[2·d] obs (cur, then prev)
```
`obs = [cur, prev]` IS the complete plane state (2nd-order wave). Nothing synthetic is invented
python-side; `world_env.py` decides nothing, it carries bytes.

## Gates — `mu/GATE-INSIDE.txt` (1200 ticks/mode) · `mu/GATE-WINDOW.txt` (window lane)
```
GI-1 determinism       PASS   same seed -> bit-identical log, intrinsic reward included
GI-2 world-sourced     PASS   seed 0 vs 7 err streams differ
GI-3 err_rel falls     FAIL   intrinsic modes flat at 1200 ticks (atom-1 PASS was a 200-tick artifact)
GI-4 ms/tick      UNVERIFIED  best window load1=3.18: mean 9.25 p99 12.46 vs 8.33 — load never <2.0
GI-5 held-out          PASS   frozen-init 0.0490 vs trained 0.0418-0.0436 = x0.854-0.889 (weak, control drifts)
GI-6 shout-exploit     PASS   energy growth world +6.7 / surprise +2.5 / progress +5.3 / surprise_rel −3.0
GI-7 vs persistence    FAIL   x0.987-0.999 (need <=0.95) — the model IS the null model + ~0.1-1.3%
GW-3 (window lane)     FAIL   learning-exists claim FORBIDDEN there too
```
**Green-only summary is not available and must not be manufactured**: GI-1/2/5/6 green, GI-3/GI-7 red,
GI-4 unverified. The interface is merge-ready; the science is not finished.

## 未驗 (do not quote these as facts)
1. **ms/tick vs the 8.33ms 120fps budget** — load1 ran 3.2 → 92 all session, never <2.0. As measured
   ~9.3ms mean ⇒ *leans over budget*, but that is a load-contaminated reading, not a verdict.
2. **"Mu learns the world"** — FORBIDDEN as a claim. GI-7: the trained net ≈ the persistence null
   ("next frame = this frame"). GI-5 shows a real but small edge over untrained weights (~13% on
   identical held-out frames).
3. **surprise vs progress** — decided as *neither*: both ≈ uniform action selection at 1200 ticks
   (H 2.059 / 2.048 vs ln8 = 2.079). Intrinsic reward as implemented does **not** shape behaviour.
4. `world` mode's lower err_rel is **confounded** (it collapses onto one action ⇒ narrow state
   distribution ⇒ trivially predictable). Not evidence of a better model.
5. GI-5's control line is **not flat** (frozen halves 0.0765→0.0216) — only the same-frames
   trained/frozen ratio is admissible, never the trend.
6. Dispatch counts / GPU trace: still structural estimates (inherited UNVERIFIED from mu-v0).

## UNIVERSE-ALPHA intake procedure
No `universe-alpha` branch/ref/dir exists on disk at this writing (checked live: `git branch -a`,
`git log --all --grep`, `ls ~/projects`). Written against this branch's own interface:
1. `git worktree add <dir> -b <your-branch> mu-inside` — do **not** work inside `~/projects/mc-mu-inside`
   (two lanes already share it; a third guarantees a false verdict).
2. `cargo build --release -p field --bin mu-serve`; **copy the binary aside and pin it** if you will
   measure anything (`cp target/release/mu-serve /tmp/mu-serve-pin-$(git rev-parse --short HEAD)`,
   set `world_env.DEFAULT_BIN`). A concurrent rebuild silently swaps the world under a running table.
3. Embed: `from world_env import WorldEnv` — that is the whole coupling surface. `WorldEnv(seed=s)`,
   `.reset(seed)`, `.step(a)`, `.tick(n)`, `.patch(x,y,k)`, `.view3(path)`, `.when()`.
   For a *different* universe: implement the same 4-line command protocol in your own binary and
   point `bin_path` at it — nothing above `world_env.py` knows what the world is made of.
4. Learning loop: `online.run(steps, seed, reward_mode=...)` returns `(log_lines, summary, obs)`;
   `summary` carries `err_hist / rel_hist / energy_hist / persist_hist / net / init_params`.
5. Gates: `python mu/gate_inside.py --steps N --out <file>`. Re-run them against your world before
   inheriting any verdict — every gate above is measured on the FFT wave plane, not on yours.
6. Inherit the laws, not just the code: real-field-only · RENDER法 (3D always) · 4D block (past =
   replay, never stored frames) · behaviour-gate over loss-gate · 未測 = UNVERIFIED.

## Next (in order, for whoever takes it)
1. Beat the persistence null (GI-7) — the one thing that would make "learns the world" sayable.
   Candidates: multi-step rollout loss (predict k ticks, not 1) · larger latent · per-pixel weighting
   by |Δ| so the loss stops being dominated by the static background.
2. An intrinsic reward that actually moves the action histogram — current three do not (GI-3).
3. GI-4 on a genuinely idle machine (load1 < 2.0), one run, no other lane building.

## atom-2r closure (2026-08-02, Bhairava lane) — GW-3R: FAIL, 死枝 CONFIRMED
Head `80b0888b` · gates `mu/GATE-WINDOW-R2.txt` (raw) · code `mu/gate_window_r2.py`,
`mu/window_eval.py`, `NormWindowPredictor` in `mu/window_nets.py`.
- **死枝: scale-invariance move** (RMS-norm net 512×2 + scale-free training error, ONE
  move, replay sealed). Cause: online training on the seed-4 trajectory OVERFITS it —
  held-out (seed 11) err_rel 0.565→0.944, i.e. WORSE than its own random init.
  Persistence null unbeaten: ratio 1.68 (need ≤0.95). Old net same protocol: 0.857,
  also never beats persistence. **再試行禁 (decreed): no further scale/normalisation
  attacks on GW-3.**
- **正果 (only one, unexaggerated)**: stationary held-out harness gives an EXACTLY flat
  lr=0 control ((b) PASS, bit-identical checkpoints) — metric attribution is now clean;
  any future GW-3R FAIL is the model's, not the measurement's.
- 学習実在主張=禁 stands, on both the online gate (GW-3) and the held-out gate (GW-3R).
- Next-lane candidates (LISTED ONLY — none executed, none claimed):
  1. multi-step rollout loss (predict k ticks, not 1) — HANDOFF §Next(1), still untried.
  2. train across MANY seeds (the overfit-to-one-trajectory cause directly suggests it).
  3. architecture with an explicit shift prior (4/5 actions translate the window;
     a conv/shift-equivariant net could exploit what an MLP must memorise).
  4. per-pixel |Δ| loss weighting — untried here (was NOT part of the sealed move).

## HANDOFF — Bhairava lane closure (2026-08-02, lifetime law)
Reached: `78992d8a` corpse salvage (norm net + held-out harness, replay SEALED) ·
`80b0888b` GW-3R gate: FAIL, 死枝 scale-invariance (held-out 0.565→0.944, persistence
1.68x unbeaten) — 再試行禁 decreed · `a4d8ea20` closure section above (next-lane
candidates listed only) · `068e7ccf` docs/audit two-lane incident record (facts only).
残atom / next lane: pick ONE candidate from the atom-2r closure list (multi-seed
training is the one the measured failure cause points at); run it through
`mu/gate_window_r2.py` unchanged — the (b) exact-flat control is the asset, keep it.
未驗: all perf numbers (GW-4/GI-4) remain load-contaminated (load1 2.6–126 all session);
学習実在主張=禁 on GW-3 and GW-3R both. One worktree one lane — see docs/audit/.
