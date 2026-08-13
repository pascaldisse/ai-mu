# MU-INSIDE — 内側学習 (mu learns the world from inside), atom-1, 2026-08-01, @opus5

worktree `~/projects/mc-mu-inside` · branch `mu-inside` (from `mu-world@8fe6b54d`) · 主樹不觸
venv `~/projects/magic-crystal-mu-v0/.venv/bin/python` · build `cargo build --release -p field --bin mu-serve`

## 現状査 (live, off disk — NOT from the atom brief)
The brief's three terms were not all open. Measured before writing code:
- 観測 = field probe — **already built**: `packages/field/src/bin/mu_serve.rs` (world door,
  binary frame `digest|step_index|reward|f32[2·d]`) + `mu/world_env.py` (bridge). Obs = `[cur,prev]`
  = the plane's complete second-order state. Gates G2/G4 green on `mu-world`.
- 行動 = ch excitation — **already built**: 8 ring poke sites, amp declared world-side.
- 報酬 = **the only genuinely open term.** ⇒ this atom is the reward, nothing else.
- heartbeat 0.91ms (`mu-v0@09a1773e`) = **STALE as a live premise**: same loop re-measured
  9-21ms in later sessions under load. Load has been 8-77 all day → any ms number today is
  load-contaminated. Marked UNVERIFIED, not re-quoted.
- tracr bridge (`tracr-bridge@86800acd`, HRR bind = circulant matmul = FFT multiply,
  compile→relax→learn) — **untouched by this atom**; it is a compiled-⊂-learned proof, not a
  reward source. No wire made, no wire claimed.

## The design question, stated exactly
Old reward = probe energy in an 8×8 patch at a **declared target site** (`mu_serve.rs:120`).
That is a TASK handed IN from outside ("shout here"). Optimal policy = collapse onto one
action — measured: action-2 share 84/200, coverage H=1.758 vs ln8=2.079. Mu learning that is
mu learning the world's *scoreboard*, not the world. 内側学習 requires the objective to be a
function of mu's own relation to the world, not of a site the world nominates.

## Proposed & implemented (`--reward {world|surprise|progress}`)
| mode | reward | rationale | risk (kept, not hidden) |
|---|---|---|---|
| `world` | probe energy at target site | baseline, unchanged path | task, not world |
| `surprise` | mu's own world-model error, in-graph, `stop_gradient` | seek the unpredictable | noisy-TV: rewards irreducible noise / loudest poke |
| `progress` | per-action `ema_slow−ema_fast` of that error | seek what is still IMPROVABLE (Oudeyer/Kaplan LP) | 1-tick-delayed, per-action granularity only |

Two laws that shaped it:
- reward = `stop_gradient(err)` — otherwise minimising slice loss minimises its own reward
  (the optimizer would eat the objective).
- reward stays **absolute** MSE; only the *diagnostic* is normalised. A reward divided by field
  energy is a reward mu can inflate by shouting.
- cost: **zero extra forward** — `decoded_next` was already computed for `loss_slice`.

## Gates (`python mu/gate_inside.py --steps 200`, output `mu/GATE-INSIDE.txt`)
- **GI-1 determinism PASS** — same seed → bit-identical 60-step log, intrinsic reward included.
- **GI-2 world-sourced PASS** — seed 0 vs 7 err streams differ (reward is not python fiction).
- **GI-3 behaviour PASS** — histograms differ (world→act2 84 · surprise→near-uniform H=2.052 ·
  progress→act4 77); intrinsic coverage > task coverage; scale-free error falls in all three
  (world .910→.837 · surprise .924→.890 · progress .929→.823).
- **GI-4 perf UNVERIFIED** — worst mean 9.46ms / p99 20.7ms vs 8.33ms budget at load1=9.6.
  FAIL *as measured*, but the machine was never idle. Rerun idle before believing either verdict.

## Open / honest
- `err_rel ≈ 0.83–0.93` ⇒ the world model explains only ~10–17% of the next frame. Learning is
  *directional*, not *good*. Do not call mu a world model yet.
- `surprise` produced near-chance action selection (H=2.052 vs ln8=2.079) — i.e. it barely
  discriminates; `progress` commits to a site like the task reward does but a **different** one.
  Which is right is UNDECIDED on 200 steps.
- GI-3's error criterion changed mid-atom: absolute MSE *rises* in all modes (field gets louder
  as mu pokes it), so it cannot answer "is the model improving". Switched to scale-free
  `err_rel`. Stated because it looks like gate-shopping — the absolute number is still logged.

---

# atom-2 (audit R1 homework) — 2026-08-01, 1200 ticks/mode

gate `mu/GATE-INSIDE.txt` (steps=1200, load1=12.7) · code `dad5d08c`

## ① GI-5 — held-out fixed trajectory, promoted to a formal gate — PASS (weak)
Trained net vs **frozen-init twin**, ONE identical 200-forced-action rollout, unseen seed 123,
learning OFF ⇒ both confounds removed at once (policy can't pick easy states; denominator is
bit-identical between the two nets). Init weights snapshotted in `run()` before any gradient.
Result: frozen 0.0490 · all four modes 0.0418–0.0436 = **x0.854–0.889** (≈13% of the frame that
init does not explain). **Control line is NOT flat** (frozen halves 0.0765→0.0216) — the held-out
field itself gets louder, so only the same-frames trained/frozen RATIO is admissible here, not the
trend. Printed as its own line rather than assumed.

## ② surprise mirror-exploit — confirmed, closed — GI-6 PASS
Audit was right and my atom-1 wording had the direction wrong. **Direction decides**: normalising
an error that is MAXIMISED (surprise) *punishes* shouting; normalising a quality signal that is
MINIMISED (progress/err_rel) *rewards* it. Same denominator, opposite verdict — no inconsistency.
`--reward surprise_rel = clip(err / max(energy, 1e-4), 0, 10)`; the **floor closes the reverse
exploit** (go silent → denominator→0 → infinite reward).
Measured field-energy growth 1st→2nd half: world **+6.7%** · surprise **+2.5%** · progress **+5.3%**
· **surprise_rel −3.0%**. The countermeasure holds and the exploit was real but mild at 200-tick
scale (it was the *task* reward that inflated the field most: +36% at 200 ticks).

## ③ surprise vs progress @1200 ticks — VERDICT: neither. Both fail to shape behaviour.
| mode | histogram H (ln8=2.079) | err_rel 1st→2nd half |
|---|---|---|
| world | **0.854** (act2 = 972/1200) | .0493→.0403 |
| surprise | 2.059 | .0851→.0845 |
| surprise_rel | 2.003 | .0827→.0836 |
| progress | 2.048 | .0847→.0834 |
At 200 ticks `progress` looked like it committed to a site (act4 77/200) — **that did not survive
1200 ticks**; it is chance. The intrinsic modes are indistinguishable from uniform sampling.
`world`'s lower err_rel is **confounded**: collapsing onto one action narrows the state
distribution, which is trivially easier to predict — it is not a better world model (GI-5 says all
four are equal to 3 decimals).

## ④ err_rel plateau .82–.84 — broken numerically, NOT substantively
Residual decoding (`obs + decoder(s2)`) drops err_rel **.82 → .04–.08 (10–20×)**. But the new
GI-7 says why: the persistence null ("next frame = this frame") scores **x0.987–0.999** of it.
⇒ the residual net is the null model plus ~0.1–1.3%. **GI-7 FAIL** at the 0.95 margin, deliberately
tightened from the sign-only test that this change would have passed by 0.0001.
Honest consequence: the atom-1 number (.83) and the atom-2 number (.08) are *the same model
quality*; only the parameterisation changed. Learning is still directional-not-good.

## ⑤ GI-4 — UNVERIFIED continues (no idle window existed)
load1 across the session: 9.6 → 45.6 → 12.7 → **92.3**. Never below 2.0. Measured 15.1–16.5ms mean
/ 41.8–48.9ms p99 at load 12.7 — load-contaminated, quoted as neither PASS nor FAIL.

## GI-3 at 1200 ticks: PASS → **FAIL**
`err_rel_falls` holds for `world` only; the three intrinsic modes are flat (±0.5%). The atom-1 PASS
was a 200-tick artifact. Reported as a regression of my own claim, not re-thresholded.

## 死枝 (corpses)
- 死 rebuild obs/action bridge for this atom — reason: exists and is gated on `mu-world`; the
  atom brief was written one lane behind the disk.
- 死 separate compiled forward to compute surprise — reason: `decoded_next` already in-graph;
  a second conv pass would have cost ~1 tick budget for nothing.
- 死 reward normalised by field energy — reason: mu can inflate the denominator (shout → look
  predictable-per-unit-energy). Kept only as a diagnostic.
- 死 quoting the 0.91ms heartbeat as live — reason: a stored measurement is an expired fact;
  the same loop measured 21ms in a later session.
- 死 (atom-2) "progress commits to a site" — reason: 200-tick observation, gone at 1200 (H=2.048 ≈ uniform).
- 死 (atom-2) residual decoding as the answer to the plateau — reason: it buys the persistence null,
  not dynamics (GI-7 x0.999). Kept in the code as the correct parameterisation, demoted as a claim.
- 死 (atom-2) sign-only "beats the null" test — reason: a residual net passes it by predicting zero delta.
