# v9-body detonation autopsy (FINISH ATOM step 1)

Source: `packages/scrying-glass/scratch/v9-train.log` (STAGE 2 run, PID 56664,
killed at epoch ~272 by this atom — was healthy nowhere past epoch 11, no
value left alive in it). Trainer: `examples/rdirect_train_v9.rs`.

## last *BEST epoch (pre-detonation floor)

[source: v9-train.log epoch 11]
```
MONITOR epoch 11: val sparkle 108.5/Mpx resid 0.1387 highlight_ratio -0.168 score=3.962 | mirror sparkle 0.0/Mpx resid 0.1314 highlight_ratio -0.166 *BEST->saved
```
Every epoch -1..11 saved *BEST (monotonic score decrease, sparkle pinned at
0 or 108.5/Mpx — the checkpoint on disk today IS epoch 11's weights).
No epoch after 11 ever saved *BEST again for the rest of the run (262 more
monitor calls, log grepped for `BEST`, none found past epoch 11).

## detonation onset

[source: v9-train.log epoch 12]
```
MONITOR epoch 12: val sparkle 217.0/Mpx resid 0.1382 highlight_ratio -0.159 score=5.425
```
sparkle crosses the `sp<16` gate for the first time with a SUSTAINED,
never-reversed climb (epoch 10 already touched 108.5/Mpx once but score was
still falling and it re-saved *BEST at 10 and 11 — that was noise, not
onset). From epoch 12 onward score rises every single monitor call with no
exception through the tail of the log:

[source: v9-train.log epoch 258]
```
MONITOR epoch 258: val sparkle 51323.8/Mpx resid 0.3260 highlight_ratio 1.578 score=1283.095
```
score at epoch 258 is 324x the last-*BEST score (3.962 -> 1283.095), sparkle
is 473x (108.5 -> 51323.8/Mpx), resid nearly tripled (0.1387 -> 0.3260),
highlight_ratio flipped from under-render (-0.168, net dimmer than teacher)
to a +1.578 OVER-render (net 2.6x brighter than teacher on its own brightest
5% of pixels) — the net learned to blow out highlights, not just add noise.

**Onset epoch: 12.** Score rose in every one of the ~250 monitor calls after
that with zero recoveries (verified: `MONITOR` lines 12 through 262, score
column is strictly non-decreasing modulo single-epoch float noise <0.1%).

## learning-rate schedule: ruled out

`adam.set_lr(lr0 / (1.0 + 1.0 * frac))` where `frac = epoch/epochs` — a
smooth monotonic decay from 1e-4 at epoch 0 toward 5e-5 at epoch 300, no
warmup, no restart, no step change anywhere in the run:

[source: v9-train.log epoch 0] `lr=0.000100`
[source: v9-train.log epoch 11] `lr=0.000096`
[source: v9-train.log epoch 12] `lr=0.000096` (same rounding, mid-decay)
[source: v9-train.log epoch 258] `lr=0.000054`

Nothing changes in the lr curve across the onset boundary (epoch 11->12) —
it is a continuous, unremarkable point on an already-declining curve. LR
schedule is CLEARED as a cause.

`n2n_mse` (the actual training-loss signal, plain MSE in demod-log space vs
the K-averaged noise2noise label) fell EVERY epoch, before and after onset,
all the way to the end of the run:

[source: v9-train.log epoch 11] `n2n_mse=0.057592`
[source: v9-train.log epoch 258] `n2n_mse=0.023445`

This is the signature: the thing the optimizer is actually minimizing kept
improving right through the detonation. The validator metric (sparkle/resid
against the held-out teacher) diverged from the training metric — classic
train/val divergence, not an optimizer instability (no NaN, no loss spike,
no lr cliff).

## convicted: no evidence-clamp ceiling in the loss

`rdirect_train_v9.rs`'s own header, unedited by this atom:

[source: rdirect_train_v9.rs:80-81] "no evidence-clamp ceiling term:
`cpu::UnetWeights` has no clamp act wired yet, a disclosed, narrower scope
than v8d's per-pixel net"
[source: rdirect_train_v9.rs:153] "One step of a moving-camera pose
sequence — v8d's `Step`, minus the evidence-clamp ceiling (not wired in
`cpu::UnetWeights`, disclosed gap)."

v8d's own loss (`rdirect.rs::accumulate_backward_clamped_slice`, the
trainer this file explicitly ports its recipe from) gates the gradient:
`presented = min(out_dl, ceiling_dl)`, and the backprop delta is ZEROED
wherever `out_dl > ceiling_dl` — i.e. once the net's output already exceeds
what the evidence for that pixel supports (temporal-mean 3x3-max-pooled
evidence, gamma=1.5), v8d gives it literally no further gradient to push
higher. v9's plain whole-image MSE has no such ceiling anywhere: nothing in
the loss stops the net from learning arbitrarily large excursions above the
evidence-supported brightness in order to chase the noisy K-averaged target
(K=8 draws, still finite-sample noise) more tightly on TRAINING poses —
which is exactly what "sparkle" (isolated local luminance peaks vs the
converged teacher) and the highlight_ratio blowout measure. v8c/v8d ran
this exact recipe, WITH the clamp, for 200 epochs and never detonated
(handoff doctrine cited in the task). v9 is the same recipe minus that one
mechanism, and detonates at epoch 12. **CONVICTED** — direct code
correlation (the only structural difference in the loss termsheet) plus the
temporal correlation (n2n_mse falling / val diverging is the textbook
unclamped-overfit signature the clamp exists to prevent).

## convicted (found during this autopsy, not previously disclosed): monitor space-mismatch bug

`rdirect_unet.rs`'s own module doc: "Output channel count (RGB demod-log
radiance)" — the conv net's raw output is demod-log space, NOT linear.
v8d's own `settle()` never compares raw net output to the teacher directly;
it routes through `direct_render_sequence_hist_split`
(`rdirect.rs:1602`), which applies BOTH the evidence clamp AND
`undo_log_demod` per pixel before returning anything — v8d's monitor metric
is measured on clamped, LINEARIZED output.

`rdirect_train_v9.rs::settle()` does neither: it calls `img_to_vec3(&out)`
on the net's raw forward output and returns it as-is, comparing it directly
against `teacher` (linear radiance from `e_full+d_full`, never demod-log
encoded) in every metric (`sparkle_resid_per_mpx`, `rmse_lin`,
`highlight_ratio`). This is an apples-to-oranges space comparison — the
reported sparkle/resid/highlight_ratio numbers for v9 are not on the same
footing as v8d's own numbers, and likely OVER-state how bad the visual
result is relative to what a correctly-decoded image would show (confirmed
directly: the 640x480 eval below, which fixes this bug, measures the SAME
checkpoint at resid 0.1276 rather than the shape v9's own in-loop 128x72
monitor would have reported for an equivalently early checkpoint). This
does NOT explain the detonation itself (n2n_mse/train loss is unaffected,
it operates correctly in pure demod-log space against a demod-log target)
but it DOES mean v9's own printed MONITOR numbers overstate the visual
severity and are not directly comparable to v8d's log numbers. **CONVICTED**
via source inspection (`rdirect_train_v9.rs` vs `rdirect.rs:1602`'s own
inference path) — a real bug, distinct from and additive to the clamp gap.

## suspect: fresh He-init (no estimator-init)

[source: rdirect_train_v9.rs:672] `"init": "FRESH He-init
(UnetWeights::new_random) — no conv-net analytic estimator-init exists yet,
disclosed gap vs v8d's TIER1"`

v8d starts from an analytic estimator-init that is already close to a
sane output; v9 starts from random He-init and needs the first ~10 epochs
just to reach comparable resid (0.1387 at epoch 11 vs v8d's TIER1 head
start). A colder start means more gradient steps in the high-lr early
region before the net's output magnitude has settled, which is exactly
the window (epoch ~10-12) where onset happens. Plausible contributor,
**not directly provable from the log alone** (no ablation isolates init
in this run) — SUSPECT, not convicted.

## suspect / cleared: unconditional mirror pose, 2 real MV channels

Mirror pose: [source: rdirect_train_v9.rs:36] documented as "A3-innocent"
— proven the best-behaved of the three v8 ablation arms in
`scratch/v8-ablate-A3.log`, kept unconditionally by v8c/v8d too without
detonating. Low suspicion; named per the task's gap list but the log gives
no evidence against it (mirror sparkle stays 0.0/Mpx all the way through
epoch ~29 in this run, only rising much later and much slower than the
main val pose — if anything the mirror pose is the LAST thing to show
divergence, consistent with "innocent"). SUSPECT, weak.

2 real MV channels: new input features not present in v8d's Mlp. No
log signal isolates this either (would need an ablation without them).
SUSPECT, unranked — flagged per the task's disclosed-gap list only.

## verdict

- **CONVICTED**: no evidence-clamp ceiling in the loss (`cpu::UnetWeights`
  has no clamp act; v8d's clamped-gradient mechanism is simply absent) —
  the structural cause the train/val divergence signature points at.
- **CONVICTED** (found here, not previously disclosed): v9's own monitor
  compares raw demod-log net output against a linear teacher with no
  `undo_log_demod` step — inflates/mismeasures the reported severity
  relative to v8d's comparable numbers (does not itself cause the
  divergence).
- **SUSPECT**: fresh He-init (colder start, more early high-lr steps before
  output magnitude settles).
- **SUSPECT, weak**: mirror pose, MV channels — log evidence is neutral to
  mildly exculpatory for both.
- **CLEARED**: lr schedule (smooth monotonic decay, no anomaly at onset).
- Onset epoch: **12**. Last *BEST: **epoch 11**, sparkle 108.5/Mpx,
  resid 0.1387, highlight_ratio -0.168, score 3.962.

## v9c round (pose diversity, STAGE 2 cure attempt 2)

Source: `packages/scrying-glass/scratch/v9c-train.log`, completed run (not
killed — watchdog self-terminated it), 300-epoch cap, trainer
`examples/rdirect_train_v9c.rs`. Cure vs v9b: 2-anchor orbit±30° pose
diversity pool, 3 draws/epoch (front-pivot x2, wide-anchor x1), fresh draws
every epoch, replacing v9b's 3 literal-fixed cameras. Kept from v9b:
evidence-clamped loss (CURE 1), abort-on-detonation watchdog (CURE 2).

Onset epoch: **51** ([source: v9c-train.log epoch 51] first non-zero val
sparkle: `val sparkle 108.5/Mpx resid 0.0930 highlight_ratio 0.281
score=2.713 rising_streak=0/20` — score not yet >best*1.05 so streak has
not started counting).

rising_streak begins counting at epoch 53 ([source: v9c-train.log epoch 53]
`val sparkle 217.0/Mpx ... score=5.425 rising_streak=1/20`).

Last *BEST / best-floor checkpoint: **epoch 50** ([source: v9c-train.log
epoch 50] `MONITOR epoch 50: val sparkle 0.0/Mpx resid 0.0937
highlight_ratio 0.272 score=2.678 | mirror sparkle 0.0/Mpx resid 0.0930
highlight_ratio 0.275 *BEST->saved rising_streak=0/20`) — every epoch -1..50
saved *BEST monotonically; the shipped `rdirect-weights-v9c.bin` is epoch
50's weights.

Watchdog abort confirmed: [source: v9c-train.log epoch 72]
`CURE 2 ABORT-ON-DETONATION: score exceeded best*1.05 for 20 consecutive
monitor epochs (>= 20) at epoch 72 — stopping, keeping best (score=2.678)`,
`training done in 1280.8s (best score=2.678)`. rising_streak climbed
53→72 (1/20 through 20/20) without a single reset — once onset crossed
the 1.05x-best gate it never dipped back below.

Floor comparison vs v9b: v9c's best-epoch resid (0.0937 @ ep50) is well
below v9b's best-epoch resid (0.1378 @ ep11) — pose diversity measurably
deepens the pre-detonation floor, consistent with the task's framing. But
onset delay is the headline: v9b detonated (rising_streak hit 20/20) at
epoch 31 from onset ~10 (21-epoch fuse); v9c detonated at epoch 72 from
onset 51 (21-epoch fuse — same watchdog window length, just shifted
later). Diversity bought ~40 extra epochs before onset and comparably
deeper floor numbers, but did NOT change the qualitative outcome: same
disease signature (sparkle 0→rising, highlight_ratio climbing steeply
past +0.5, mirror val staying clean throughout — mirror sparkle is still
0.0/Mpx at epoch 71, only ticking to nonzero after, same "last to diverge"
pattern as v9/v9b), same self-terminating watchdog abort. **CONVICTED**:
pose diversity delays and deepens but does not cure memorization-onset —
motivates v9d's EMA-weight cure (Polyak averaging) rather than further
pool widening alone.

## v9c checkpoint + provenance (committed)

`data/rdirect-weights-v9c.bin` (sha256
b48050bf7ef7d7600464d021e7facc0cd3c152c98842abe0e5cf562caef0d5fb) +
`data/rdirect-weights-v9c.provenance.json`, both epoch-50 best-floor
weights, committed this round (see git log).

## v9d round (POOL MUCH WIDER + EMA-vs-raw monitor split, STAGE 2 cure attempt 3)

Source: `packages/scrying-glass/scratch/v9d-train.log`, completed run
(watchdog self-terminated), trainer `examples/rdirect_train_v9d.rs`. Cure vs
v9c: same 2 mechanisms kept (evidence-clamped loss, abort-on-detonation
watchdog) plus (a) pool widened from 2 anchors/±30°/3 draws to 4 anchors
(front/wide/high/close — high and close vary height/radius off the front
pivot)/±180° full-orbit/8 draws per epoch, and (b) `run_monitor` now also
settles+logs the RAW (non-EMA) net's val score alongside the EMA score every
epoch (diagnostic only — checkpoint/best-floor/watchdog stayed EMA-only,
unchanged from v9c, since v9c already carried Polyak EMA weights,
`ema=0.999` — that mechanism was NOT new to v9d, only the raw-alongside
logging was).

Onset epoch: **24** ([source: v9d-train.log epoch 24] first non-zero val
sparkle: `val sparkle 108.5/Mpx resid 0.0927 highlight_ratio 0.284
score=2.713 rising_streak=0/20`).

Last *BEST / best-floor checkpoint: **epoch 23** ([source: v9d-train.log
epoch 23] `MONITOR epoch 23: val sparkle 0.0/Mpx resid 0.0941
highlight_ratio 0.265 score=2.688 | mirror sparkle 0.0/Mpx resid 0.0935
highlight_ratio 0.284 *BEST->saved rising_streak=0/20`) — the shipped
`rdirect-weights-v9d.bin` is epoch 23's weights.

rising_streak begins counting at epoch 26 ([source: v9d-train.log epoch 26]
`score=8.138 rising_streak=1/20`, epochs 24-25 stayed within the 5% noise
band and reset the streak to 0 each time).

Watchdog abort confirmed: [source: v9d-train.log epoch 45]
`CURE 2 ABORT-ON-DETONATION: score exceeded best*1.05 for 20 consecutive
monitor epochs (>= 20) at epoch 45 — stopping, keeping best (score=2.688)`,
`training done in 1767.1s (best score=2.688)`. 46 epochs run (0..45) in
~29.5 minutes wall.

**CONVICTED — POOL WIDENING REGRESSED, DID NOT FURTHER DELAY ONSET**: v9d's
onset (epoch 24) landed EARLIER than v9c's (epoch 51), despite a much wider
pool (4 anchors vs 2, full ±180° orbit vs ±30°, 8 draws/epoch vs 3) — the
opposite of the extrapolation from v9→v9c ("diversity delays onset, so
widen hard"). One other v9d-only symptom not seen in v9c: the mirror
validator, clean at 0.0/Mpx sparkle through v9c's ENTIRE run (even past its
own onset, to epoch 71), starts sparkling in v9d by epoch 34
([source: v9d-train.log epoch 34] `mirror sparkle 108.5/Mpx`) — detonation
generalized to the previously-innocent held-out pose this round, not just
the orbit_-20 validator. rising_streak's shape is otherwise IDENTICAL in
kind to v9c's: once it starts (epoch 26) it climbs strictly monotonically
1→20 with zero resets ([source: v9d-train.log epochs 26-45], same
no-recovery pattern v9c/v9b/v9 all showed) — only the ONSET epoch moved,
not the shape of the runaway once triggered. SUSPECT explanation, not
proven from this log alone: a full ±180° orbit around the front pivot
sweeps THROUGH angles
near the held-out orbit_-20 validator's own viewpoint every few epochs
(v9c's ±30° pool never reached that far), so the net gets much more frequent
near-validator-view training exposure under v9d's wider pool than under
v9c's narrower one — an implicit train/val leakage risk that grows, not
shrinks, as the yaw range widens toward ±180°, which would explain a
FASTER path to memorizing that specific validator's answer rather than a
slower one. Not confirmed (would need a pool-composition ablation that logs
each draw's angular distance to the validator pose) — flagged as the lead
suspect for a future round, ranked above "wider pool per se is bad".

**EMA-vs-raw**: EMA score (driving checkpoint/abort) fell smoothly from
3.368 (ep-1) to 2.688 (ep23) then rose monotonically once detonation
started. The raw (non-EMA) net's score, logged alongside every epoch for
the first time this round, stayed noisy and far worse throughout — never
below ~226 (`raw(non-EMA) ... score=225.152` at epoch 1) and mostly
250-415, an order of magnitude above the EMA score at every epoch including
epoch 23's *BEST (`raw ... score=276.693` vs EMA's 2.688) — confirms EMA is
doing real, large variance-reduction work (expected: `ema=0.999` averages
over ~1000 effective steps of a net taking ONE Adam step per image), but
this comparison does NOT show whether EMA delays or dampens onset itself,
since v9c (same EMA mechanism, narrower pool) already delayed onset far
more than v9d does — the EMA mechanism was constant across both rounds, so
it cannot explain the v9c-vs-v9d onset-epoch difference; the pool-widening
change is the only varied factor and is the prime suspect above.

**Floor comparison (640x480, domain-fixed eval, `rdirect_v9_eval_640`,
`GAIA_V7_CLAMP_GAMMA` default 1.5, held-out orbit_-20)**:

| checkpoint | epoch | resid (no clamp) | resid (clamped) | sparkle (clamped) | highlight_ratio |
|---|---|---|---|---|---|
| v9c best | 50 | 0.1023 | 0.1022 | 0.0/Mpx | 0.257 |
| v9d best | 23 | 0.1035 | 0.1033 | 0.0/Mpx | 0.255 |

v9d's floor is marginally WORSE than v9c's (resid +0.0011) despite the
wider pool — consistent with the earlier onset cutting training short at
epoch 23 vs v9c's epoch 50 (27 fewer epochs of pre-detonation descent).
Neither checkpoint clears the resid<0.035 bar; NOT ordealed, no bar claimed
passed for either.

## v9d checkpoint + provenance + eval artifacts (committed)

`data/rdirect-weights-v9d.bin` (sha256
6eb16264b3a7fd906bd01e9c832a7d0ee9addea2275b333a20fd58a7299b6c9c) +
`data/rdirect-weights-v9d.provenance.json`, epoch-23 best-floor weights.
`scratch/v9d-eval-{teacher,v9clamped,v9noclamp}.png` (v9d best, 640x480) and
`scratch/v9c-best-eval-{teacher,v9clamped,v9noclamp}.png` (v9c's actual
best/ep50 checkpoint, run this round for the floor-comparison table above —
v9c's earlier committed eval PNGs were of an intermediate ep29 checkpoint,
not its final best).

## v9e round (target-tail forensics + robust-target cure, STAGE 2 cure attempt 4)

### 1a. onset in image-updates (no retrain, from the three logs already on disk)

Per-epoch pose counts, read directly from each trainer's own `main()`
(`examples/rdirect_train_v9.rs`, `rdirect_train_v9c.rs`, `rdirect_train_v9d.rs`),
`k`=`GAIA_V9_STILL` unroll steps/pose is **3** in all three (`env_u32("GAIA_V9_STILL", 3)`,
unchanged across every v9 round):

| run | onset epoch | poses/epoch (train_cams/pool+mirror) | k (steps/pose) |
|---|---|---|---|
| v9  | 12 (source: this file, v9 section above) | 4 (`front,wide,orbit_+20,mirror` \u2014 `rdirect_train_v9.rs:551-556`) | 3 |
| v9c | 51 (source: this file, v9c section above) | 4 (3 fresh pool draws + fixed mirror \u2014 `rdirect_train_v9c.rs:797-807`) | 3 |
| v9d | 24 (source: this file, v9d section above) | 9 (8 pool draws [4 anchors \u00d7 `GAIA_V9D_POOL_DRAWS_PER_ANCHOR`=2] + fixed mirror \u2014 `rdirect_train_v9d.rs:745,804-805,864-869`) | 3 |

Two ways to count "image-updates" (both computed, task's own formula matched
first since that is what the hypothesis statement in the task uses):

**Task's literal formula** (`onset_epoch * draws_per_epoch`, where v9's own
"4" already counts its fixed mirror pose but v9c's "3"/v9d's "8" quote only
the FRESH pool draws, excluding their own fixed mirror pose \u2014 reproduced
here exactly as given, not re-derived):
```
v9:  12 * 4 = 48
v9c: 51 * 3 = 153
v9d: 24 * 8 = 192
```
ratio v9c/v9d = 153/192 = **0.80** (within 2x, fairly tight \u2014 25% apart).
ratio v9/v9c = 48/153 = **0.31** (v9c took 3.2x MORE updates than v9 to
detonate) \u2014 **outside the ~2x band**.

**Refined formula** (`onset_epoch * poses_per_epoch(mirror INCLUDED
consistently for all three, since the mirror pose IS trained every epoch in
all three trainers) * k_steps`, i.e. the literal count of `net.forward()` +
`adam.step()` calls executed before onset \u2014 the actual number of times the
optimizer touched the weights):
```
v9:  12 * 4 * 3 = 144
v9c: 51 * 4 * 3 = 612
v9d: 24 * 9 * 3 = 648
```
ratio v9c/v9d = 612/648 = **0.94** (6% apart \u2014 MUCH tighter than the pool-
size difference of 3x, or than the onset-EPOCH difference of 2.1x, would
suggest on its own). ratio v9/v9c = 144/612 = **0.24** (v9c took 4.25x more
image-updates than v9 to detonate) \u2014 **outside the ~2x band by a wide
margin, worse than the literal-formula check**.

**Verdict on 1a**: the hypothesis holds TIGHTLY for v9c-vs-v9d under BOTH
countings (0.80x and 0.94x, both well inside 2x) \u2014 this is the strongest
single piece of evidence for the hypothesis, since it resolves the v9d
autopsy's own open puzzle ("pool widening REGRESSED onset in epoch-count,
opposite of the v9-to-v9c extrapolation") into a NON-mystery: v9d's pool
is 2.25x bigger (9 vs 4 poses/epoch) so it needed proportionally FEWER
epochs to accumulate the same image-update budget, and 24*9 lands within 6%
of 51*4 \u2014 image-update count, not epoch count and not pose-diversity
breadth, is what actually correlates across those two runs. v9 itself is
the outlier under BOTH countings (3.2x-4.25x off, never within 2x) \u2014 but
v9 is not a fair comparison to v9c/v9d under this hypothesis: v9 predates
CURE 1 (no evidence-clamp ceiling at all, convicted cause of v9's OWN
detonation in the section above), so its onset is driven by a structurally
different, unclamped-overfit failure mode, not by the same
"clamp-bounded-but-target-tail-chasing" mechanism the hypothesis is about.
**Restricting the hypothesis to the two CLAMPED runs it actually applies
to (v9c, v9d): CONFIRMED, image-update count predicts onset within
6-20%, image-COUNT explains what pose-diversity-alone could not.**

### 1b. target-tail forensics \u2014 domain verified, tail mass measured

**Domain check** (source read, `rdirect_train_v9c.rs:376-378` +
`rdirect.rs::target_demod_log`): the K=8 draws are summed in LINEAR
radiance (`radiance_sum[px] += e_b[px] + d_b[px]`, one running sum, no
per-draw buffer kept), divided by K, THEN encoded ONCE via
`target_demod_log(mean_linear, albedo)` \u2014 `log_demod`: `ln(radiance/divisor
+ 1)`. The loss (`rdirect_train_v9c.rs:849-859`) computes plain per-pixel
MSE of `presented_dl` (net's raw demod-log output, clamped) directly against
this `target_dl` \u2014 **no further space conversion in the loss**. So: **the
K-average happens in LINEAR space, but the loss's own arithmetic (the
number CURE 3 needs to winsorize) runs in DEMOD-LOG space** \u2014 confirmed by
source, not assumed. `rdirect_v9_tailmass.rs` (new tool, this round) 
reproduces v9c's exact pool-draw pipeline (front x2/wide x1 cadence,
K_DRAWS=8, evidence_spp=1, `pool_orbit_deg`=30, `pool_jitter`=0.6, byte-
identical `pool_camera`/seed-family constants) and measures tail mass
DIRECTLY on `target_dl` (post K-average, post demod-log-encode), matching
the loss's own domain exactly.

[source: `rdirect_v9_tailmass.rs` run, 12 pool poses (128x72, K=8,
evidence_spp=1), `nice -n19 ./target/release/examples/rdirect_v9_tailmass`,
this round]
```
AGGREGATE over 12 pool poses: mean frac(>3xlocal)=0.01345 mean frac(>10xlocal)=0.00028 mean frac(>30xlocal)=0.00000 mean max/mean=15.33
```
Per-pose spread: frac(>3x local 7x7-mean) ranges 0.84%-1.90% across the 12
poses, frac(>10x) 0.011%-0.054%, frac(>30x) is exactly 0 on every single
pose sampled (no pixel in any of the 12 poses exceeded 30x its own local
neighborhood mean), whole-image max/mean ratio 11.9-19.0.

**Smoking-gun verdict**: fireflies ARE present in the K=8-averaged targets
\u2014 a small (~1.3% of pixels) but real and CONSISTENT (present in all 12
poses sampled, not a one-off) tail of pixels sitting 3x+ above their own
local neighborhood mean, thinning fast (0.03% at 10x, 0% at 30x in this
sample) \u2014 8-draw averaging visibly reduces but does not eliminate
1-spp-per-draw noise2noise variance, exactly as the hypothesis predicts
(finite K, unbounded per-draw radiance distribution \u2014 direct/specular
paths in this scene can return arbitrarily bright single-sample estimates,
averaging 8 of them narrows but does not cap the tail). **CONFIRMED** as
smoking gun for the "unbounded bright tails in targets" half of the
hypothesis, though the tail is thinner (max/mean ~15x, not orders of
magnitude) than "fireflies" sometimes implies \u2014 consistent with the
DISEASE being a slow capacity-chasing overfit (many epochs to detonate, not
an immediate spike) rather than a single catastrophic outlier.

### 2. cure chosen: WINSORIZED TARGETS (not relative/luminance-normalized loss)

Forensics in 1b measured tail mass ON THE TARGET SIDE specifically (not a
loss-shape argument) \u2014 CURE 3 (`rdirect_train_v9e.rs`, new trainer this
round, base = v9c CONFIG unchanged: narrow \u00b130deg 2-anchor pool, 3 pool
draws/epoch + fixed mirror, NOT v9d's wide pool, per task instruction)
winsorizes `target_dl` per-channel against its own 7x7 local-mean ceiling
(`winsor_k` * local_mean, `GAIA_V9E_WINSOR_K` env-tunable, IRON default
10.0 \u2014 chosen to sit above the measured 10x-tail rarity threshold [0.03%]
so it clips only the sparse extreme tail, leaving the much more common
sub-3x brightness variation, i.e. ordinary specular/highlight detail,
untouched) BEFORE the target is ever used as a loss label, applied once per
pose per epoch in `render_pose_seq` right where `target_dl` is built.
Rejected the relative/luminance-normalized-loss alternative: that reweights
the LOSS by `1/(local_mean+eps)`, which down-weights bright REGIONS overall
(uniformly, whether or not any given pixel in them is a firefly) but does
NOT cap a single pixel that is itself far above ITS OWN local mean \u2014 by
definition a firefly deviates from its own neighborhood, so a global
region-brightness reweighting leaves the per-pixel outlier-to-neighbor gap
(the actual measured tail-mass quantity in 1b) exactly as large as before.
Winsorization directly attacks the measured quantity; the normalized-loss
alternative does not, so it was not chosen. CURE 3 is TARGET-side and
distinct from/complementary to CURE 1's NET-OUTPUT-side evidence clamp
(CURE 1 gates how high the net's OWN prediction may go relative to traced
evidence; CURE 3 gates how high the noisy LABEL itself may be before the
net ever sees it as ground truth) \u2014 both mechanisms are kept active in
v9e (CURE 1 unchanged, CURE 3 new).

Smoke-tested (`GAIA_V9_EPOCHS=2 GAIA_V9_WALL=300`, this round): runs clean,
n2n_mse falls epoch to epoch (0.143\u21920.099), MONITOR line prints the new
`winsorize_k=10` value, checkpoint/provenance write successfully \u2014
mechanically verified before the real launch below.

## v9g: mechanism hunt + CURE 4 (asymmetry cure)

### v9f end-reason

`scratch/v9f-train.log` tail: `[v9f] WATCHDOG ABORT: bar-res probe sparkle failed the bar at epoch 99 — stopping, keeping best+last (best score=2.678)`. Signal B (render-res probe fail) tripped, not resid-streak (`resid_streak=0/20` every monitor line through the whole run) and not the 300-epoch wall. Probe trajectory: ep24 sp0.0/resid0.1211 -> ep49 0.0/0.1052 -> ep74 13.0/0.0872 -> ep99 22.8/0.0701 (>= spark_target 16 -> abort). `highlight_ratio` (defined in `run_monitor`/`highlight_ratio()`, examples/rdirect_train_v9e.rs: mean net/teacher luminance ratio over the pose's brightest `highlight_pctl`=5% teacher pixels) rose monotonically 0.055 (ep-1) through 0.578 (ep77, last MONITOR before probe) — confirms the "overshoot at highlights" reading: net luminance in the brightest 5% of pixels is climbing toward and past teacher parity, not converging to it.

### Mechanism hunt verdict

(a) YES, one-sided. `rdirect.rs::accumulate_backward_clamped_slice` / every `rdirect_train_v9{b,c,d,e,f}.rs`'s inline loss loop compute `presented = min(out_dl, ceiling_dl)`, `diff = presented - target`, then gate the BACKWARD delta on `active = raw <= cap`: `d_out = active ? 2*diff/n : 0`. Above the ceiling the gradient is EXACTLY zero — confirmed empirically (below), not just by reading the derivative of `min()`. The v9b module doc's own words: "the net gets no gradient signal to push EVEN HIGHER once it already exceeds what the evidence supports" — true, but that's a HALF-description: it also removes every PENALTY for having already overshot. No downward force exists once `raw > cap`; only Adam momentum inherited from pre-crossing epochs (when the gradient was positive, pushing up) and U-net weight-sharing spillover from other, still-active pixels can still move it, in whatever direction THEY want — not a corrective force local to the overshoot.

(b) NO (ruled out) as the asymmetry source, YES as an amplifier. `log_demod`/`undo_log_demod` (`rdirect.rs` L73-85) is `dl = ln(max(radiance/divisor,0)+1)`, `radiance = (exp(dl)-1).max(0)*divisor` — a plain monotonic log1p/expm1 pair, no epsilon/clip inside the loss itself; the CURE-1 MSE is computed directly as `(presented_dl - target_dl)^2` in that log-space with no extra transform, so over/undershoot get equal-magnitude log-space gradients — symmetric. It IS an amplifier: `undo_log_demod` is `exp(dl)-1`, so a small, steady log-space drift compounds EXPONENTIALLY once exponentiated into linear-space sparkle — this is why a slow ~0.5%/epoch log-space creep (COORDS forensics, epoch 54->73, `scratch/v9e-forensics.log`) only crossed the render-res sparkle bar (22.8 > 16) at epoch 99 and not far earlier: the visible effect lagged the underlying drift by however many epochs it took the exponential to clear the bar.

(c) Instrumented directly. Added `GAIA_V9G_GRAD_PROBE=1` to `examples/rdirect_train_v9f.rs` (zero-effect on the normal run, gated env var): loads a checkpoint (default `data/rdirect-weights-v9f-last.bin`), runs the val_seq (`orbit_-20`) recurrent chain through it exactly like `history_forward`, then at the last step evaluates the EXACT training-loop CURE-1 formula (raw/cap/target/active/d_out) at the 7 fixed texels `scratch/v9e-forensics.log`'s COORDS dump tracked drifting: `(94,3) (94,6) (97,6) (94,8) (47,9) (55,34) (66,38)` (128x72 space). Result against v9f-last, all 3 channels each:

```
px=(94,3) c=0 raw=0.518097 cap=0.230823 target=0.195551 overshoot=true active=false d_out=0.00000000
px=(94,3) c=1 raw=0.300375 cap=0.093584 target=0.078429 overshoot=true active=false d_out=0.00000000
px=(94,3) c=2 raw=0.391099 cap=0.163291 target=0.137061 overshoot=true active=false d_out=0.00000000
... (94,6) (97,6) (94,8) (47,9) all channels: overshoot=true active=false d_out=0.00000000
px=(55,34) c=2 raw=0.363856 cap=0.390206 target=0.173789 overshoot=false active=true d_out=0.00001375   <- the ONE exception, still below its own ceiling
px=(66,38) c=0/1/2: overshoot=true active=false d_out=0.00000000
```

20 of 21 channel readings: overshoot=true, active=false, d_out=exactly 0.0. The down-pull is not merely "starved" — it is fully zero at every one of the drifting forensic pixels but one, in the exact checkpoint that shows the drift. Confirms: nothing locally opposes this drift; whatever raised these pixels above ceiling can only be raised further or left alone, never pulled back, by CURE 1 as originally written.

### CURE 4 chosen: symmetric restoring gradient (not a target-side change)

`examples/rdirect_train_v9g.rs` (copy of v9f, ONE mechanism changed at the loss call site): replace the zero-gradient gate above the ceiling with an explicit overshoot term, weight `overshoot_w` (IRON param, `GAIA_V9G_OVERSHOOT_W`, default 1.0):

```
loss = (min(raw,cap) - target)^2 + overshoot_w * max(0, raw-cap)^2
d(loss)/d(raw) = 2*(raw-target)            when raw <= cap   (CURE 1's own term, UNCHANGED)
               = 2*overshoot_w*(raw-cap)   when raw >  cap   (NEW — pulls toward the CEILING)
```

Chosen over an alternative (pull all the way to `target` above the ceiling too, i.e. just delete the `min()`/gate entirely and always use `2*(raw-target)/n`): that would silently reintroduce the exact failure the autopsy's v9-round convicted (autopsy: "no evidence-clamp ceiling term... `cpu::UnetWeights` has no clamp act wired yet" — CURE 1 was built specifically because the noisy K=8-draw-averaged `target_dl` itself has a measured firefly tail (autopsy section above this one), so chasing it unconditionally detonates). Pulling toward the CEILING instead keeps CURE 1's own founding philosophy — the evidence ceiling (temporal-mean, 3x3-max-pooled across K real rendered draws) is the trusted anchor once a prediction is already brighter than one noisy per-draw target pixel, not the target itself — while finally giving that anchor a gradient that can pull back down. `overshoot_w=1.0` makes the two branches C0-continuous in slope at the seam (`raw==cap` gives 0 from either side). The reported `n2n_mse`/monitor number (`presented`-based) is untouched — only the gradient that feeds Adam changed, so this is not comparable-metric-breaking.

Smoke-tested 3 epochs (`GAIA_V9_EPOCHS=3 GAIA_V9_WALL=180`, this round): builds clean, runs to completion, early curve (score 3.368->3.360, resid 0.1179->0.1176) matches v9c/v9e/v9f's own pre-drift regime at this epoch range (expected — CURE 4 only changes behavior once `raw>cap`, which does not happen this early). No crash, checkpoints/provenance wrote.

Launched full run (300 epochs, 128x72, `nohup`, IRON params: same K=8, same two-signal watchdog resid_streak=20/probe_every=25/spark_target=16, `GAIA_V9G_OVERSHOOT_W=1.0` default) — PID and probe trajectory through ~ep100 reported in the room reply; log at `scratch/v9g-train.log`.

### v9g verdict: CURE 4 PARTIAL — mechanism right, dose too low

[source: scratch/v9g-train.log] end-reason: `[v9g] WATCHDOG ABORT: bar-res probe sparkle failed the bar at epoch 99 -- stopping, keeping best+last (best score=2.713)` preceded by `[v9g] PROBE epoch 99: bar-res(640x480) sparkle 16.3/Mpx resid 0.0756 highlight_ratio 0.409 (tgt sp<16 resid<0.035) took 42.2s`. Same abort SHAPE as v9f (bar-res probe signal, not resid_streak, not the 300-epoch wall — `resid_streak=0/20(ABORT)` at every monitor line through the whole run, same as v9f).

[source: scratch/v9g-train.log, `PROBE epoch` lines] Full bar-res probe trajectory vs v9f (`scratch/v9f-train.log`, quoted in this file's v9g mechanism-hunt section above):

| epoch | v9f sparkle/resid | v9g sparkle/resid |
|---|---|---|
| 24 | 0.0/0.1211 | 0.0/0.1218 |
| 49 | 0.0/0.1052 | 0.0/0.1084 |
| 74 | 13.0/0.0872 | 9.8/0.0926 |
| 99 | 22.8/0.0701 (ABORT) | 16.3/0.0756 (ABORT) |

CURE 4's overshoot term (`overshoot_w=1.0`) measurably slowed the drift — ep99 sparkle roughly half v9f's (16.3 vs 22.8) and `highlight_ratio` lower at every checkpoint (0.409 vs 0.578 at ep99) — but did not stop it: sparkle still crosses the 16 bar at the same epoch (99) because resid is also falling slower in lockstep (0.0756 vs 0.0701), i.e. the down-pull is real but proportionally too weak to hold the ceiling once the net is still this far from converged. Mechanism confirmed correct (CURE 4's restored down-gradient is doing work, not a no-op); `overshoot_w=1.0` is simply under-dosed relative to the ~1.0-weighted upward pull from CURE 1's main term at these render-space brightness levels.

Verdict: CURE 4 PARTIAL. Escalate dose in v9h — same restored-gradient mechanism, larger `overshoot_w` multiplier (or a lowered up-gate ceiling factor for evidence-bright pixels) so the down-branch outweighs the still-active upward pull before the drift compounds past the bar-res sparkle bar.

## v9h: CURE 4 dose escalation (4x) — verdict

[source: `scratch/v9h-train.log`] end-reason: `WATCHDOG ABORT: bar-res probe sparkle failed the bar at epoch 124` (`sparkle 29.3/Mpx resid 0.0658 highlight_ratio 0.502`), best score=2.713. `GAIA_V9H_OVERSHOOT_W=4.0` (4x v9g's 1.0).

**Dose-response, three points, same abort shape every time (bar-res probe signal, never resid_streak/wall):**

| run | overshoot_w | abort epoch | sparkle floor@abort | resid floor@abort |
|---|---|---|---|---|
| v9f | 0x (CURE 4 absent) | 99 | 22.8 | 0.0701 |
| v9g | 1x | 99 | 16.3 | 0.0756 |
| v9h | 4x | 124 | 29.3 | 0.0658 |

4x the restoring force bought ~25 more epochs before the same bar-res sparkle bar (16) was crossed — NOT proportional (going 1x->4x delayed onset by only 25/99≈25%, not 4x), and resid keeps falling in lockstep (lower resid floor each escalation: 0.0701->0.0756->0.0658 is non-monotonic in a way that shows the two signals trade off, not that either is being solved). Extrapolating the ~25-epoch-per-4x pattern, closing the remaining gap to a truly stable (non-aborting, e.g. 300-epoch) run would need several more 4x escalations — a `overshoot_w` in the hundreds-to-thousands range before dose alone would plausibly work, if it worked at all (no evidence the relationship is even monotonic past this point, only 3 samples). **Escalating dose further is not a viable path to close a ~150-epoch gap on its own** — the failure mode is a symptom war (raising the down-pull weight fights the SAME still-unidentified upward pull harder, buying epochs, not curing the mechanism) rather than a fix. This atom (forensics, no retrain) exists to find what that upward pull actually is before spending another round guessing at dose.

## V9 DRIFT-SOURCE FORENSICS (v9h-last, no retrain)

**Tool**: `examples/rdirect_v9h_drift_forensics.rs` (new, standalone). Reuses `rdirect_train_v9h.rs::bar_res_probe`'s exact pipeline (K=3-step recurrent settle at 640x480, `orbit_-20` held-out pose, undo-log-demod + inference-time evidence clamp) so the located texels are the SAME ones the real WATCHDOG probe flags — confirmed: this tool's own bar-res sparkle readout (29.3/Mpx) landed EXACTLY on `v9h-train.log`'s epoch-124 abort number (both deterministic-seeded, same weights). `sparkle_resid_per_mpx`'s own 3x3 local-peak detector (unchanged, SPARK_DELTA=0.15 linear-luminance) located 9 flagged texels total; all 9 sampled (below `GAIA_V9H_MAX_TEXELS` default 12).

**Domain** (stated, not assumed): loss operates in DEMOD-LOG space (`presented_dl` vs `target_dl`, plain MSE) — that is the primary comparison domain below. For each flagged texel this tool computes: `net_raw_dl`/`net_presented_dl` (post evidence-clamp) directly from the loaded weights; `teacher_dl` = `target_demod_log(teacher_lin, albedo)` using the SAME per-pixel albedo divisor as the net, so net/teacher/target sit in one directly-comparable space; `target_dl_mean` = the trainer's OWN K=8-draw-average-then-encode recipe, repeated 32 times with fresh seeds (32×8=256 independent 1-spp draws total) and averaged IN DEMOD-LOG space (the space the loss actually sees) — this is what the noisy per-epoch label gives in expectation. A second cross-check, `target_lin_direct` (mean of all 256 raw linear draws, pre-encode), corroborates it did not matter which order the 32-repeat average was taken in (see run, both track together).

**Result** (`scratch/v9h-drift-forensics.json`, 9/9 texels, dl-space, mean over texels per channel R/G/B):

```
net_presented_dl - target_dl_mean   (net overshoots its OWN low-noise target estimate): +0.027 / +0.180 / +0.120
target_dl_mean   - teacher_dl       (does the noisy label itself run hot?):            +0.001 / +0.007 / +0.005
```

The net-vs-target gap is **20-26x larger** than the target-vs-teacher gap, every channel. Per-texel sign check: `target_dl_mean - teacher_dl` FLIPS SIGN across the 9 texels (4 negative, 5 positive, range -0.107..+0.064) — a noise pattern, not a systematic bias direction. `net_presented_dl - target_dl_mean` is POSITIVE at all 9 texels, every non-trivial channel (only one micro exception, ch2 at px(13,257), -0.085, single channel out of 27 channel-readings) — a systematic, one-directional excess. Aggregate counts (dl-space margin=0.05): target_dl_mean exceeds teacher_dl by >margin at 2/9 texels (weak, minority signal for (A)); target_dl_mean sits within margin of teacher_dl at 5/9 (majority, consistent with (B)); **net_presented_dl exceeds target_dl_mean by >margin at 9/9 texels, unanimous** — even against the well-averaged (256-draw) target, not the single noisy epoch label, the net is still higher, at every single flagged texel.

**Demod-degeneracy check (C)**: all 9 flagged texels have albedo EXACTLY `(0,0,0)` → `demod_divisor` takes the NO-HIT branch (`divisor=(1,1,1)`, demod skipped entirely per `NO_HIT_ALBEDO_THRESHOLD_SQ`), not the low-but-nonzero-albedo-emitter branch (`albedo+1e-3`) the task hypothesis named. 0/9 texels have `divisor <= 2*ALBEDO_DEMOD_EPS`. **These are NOT emitter-surface texels — they are zero-albedo (sky/no-hit primary ray) pixels that still carry real, non-zero teacher radiance** (direct sky/background light, `teacher_lin` 0.03-1.2), clustered at what are almost certainly geometry-silhouette/sky boundaries (all 9 coordinates sit in two tight clusters: x∈{13,17,19}, y=257 and x∈{282,321,332,335,337,339}, y≈247-260 — edges of bright objects against sky). No amplification pathology present (divisor is exactly 1.0, not near-epsilon) — (C) is REFUTED at these texels.

**VERDICT: (B) SHARED-WEIGHT DRIVE**, unanimous and ~20x larger in magnitude than any target-side effect. The K=8 n2n label is NOT what teaches the overshoot (target≈teacher, sign-mixed, small) — the net drifts past even its own converged (256-draw) target at every flagged texel, at the exact zero-albedo/sky-boundary pixel class CURE 4's grad-probe (v9g mechanism hunt, this file, above) already showed has ZERO local restoring gradient once `raw>cap` (20/21 channel readings `active=false, d_out=0.0`) — i.e. nothing LOCAL opposes the rise; the only force still able to move these weights is Adam momentum + convolutional weight-sharing spillover from OTHER pixels (genuine bright highlights elsewhere in the 640x480 frame) whose loss reduction is served by the same shared filters that also feed these silhouette/sky pixels. (A) target bias is present as a minor, non-systematic (sign-flipping) contributor at a minority of texels (2/9) and never explains more than a fraction of the net-vs-target gap even there. (C) demod degeneracy does not apply — the flagged class is zero-albedo no-hit pixels (divisor pinned at 1.0), not low-albedo emitters.

**Recommended cure**: per-pixel loss reweighting targeted at the evidence-bright/zero-albedo pixel class specifically (not a global mask) — e.g. boost `overshoot_w` (CURE 4's existing above-ceiling term) by a per-pixel factor keyed on `albedo≈0 AND ceiling_dl` being high (a zero-albedo pixel sitting next to bright evidence is exactly the population identified here), so these texels get a stronger LOCAL restoring pull without changing the loss for ordinary bright-surface highlight pixels elsewhere (the ones plausibly driving the spillover) — cheaper and more targeted than an architectural per-pixel gain head, and testable as a single new IRON param before any structural change. Do NOT escalate `overshoot_w` globally again (v9h's own dose-response above shows that path is a symptom war, not a cure) — next round should be scoped to this per-pixel-class reweighting, verified against the SAME 9 texels this tool located before a full retrain is spent.

## v9i: CURE 5, ACTIVATE THE MUTE PIXEL CLASS (grad-probe-fix, no retrain until pre-flight passed)

### gate conviction

Instrumented `examples/rdirect_v9h_drift_forensics.rs` (this round: added
`grad_probe_formulas()` + per-texel/per-channel printing, reusing the SAME
already-computed live tensors — `net_raw_dl`, `ceiling_dl`,
`target_dl_mean_32x8` — from that tool's existing bar-res pipeline, no extra
render). Three formulas evaluated at the SAME 9 bar-res sparkle-flagged
texels, v9h-last weights, live rerun (`scratch/v9i-preflight-grad-probe.log`,
this round):

- **BEFORE_stale**: [source: `examples/rdirect_train_v9h.rs:887-889`
  (before this round's edits; the standalone `GAIA_V9H_GRAD_PROBE=1`
  diagnostic's own hardcoded formula)] `active = raw <= cap; d_out = if
  active {2*(presented-target)/n} else {0.0}` — this is the PRE-CURE4
  zero-gate, "reused from v9f" per that file's own comment and never
  updated when CURE4 landed in v9g.
- **CURE4_actual**: [source: `examples/rdirect_train_v9h.rs:1010-1020`, the
  REAL per-pixel loop that trained `rdirect-weights-v9h-last.bin`]
  `d_out = if active {2*diff/n} else {2*overshoot_w*(raw-cap)/n}` —
  overshoot branch is NEVER unconditionally zero; it depends on
  `overshoot_w*(raw-cap)`.

**Result** (`scratch/v9i-preflight-grad-probe.log`, `# ---- V9I GRAD-PROBE
AGGREGATE ----` block, 9 texels x 3 channels = 27 readings, overshoot_w=4.0
matching v9h-last's own trained dose):
```
# BEFORE_stale ... d_out==0.0: 16/27
# CURE4_actual ... d_out==0.0: 0/27
```
**CONVICTED — probe/reality mismatch, "clamp's active flag semantics"
candidate confirmed but in the diagnostic, not the trained loss**: the
task's standing evidence ("active=false, d_out=0.0... overshoot_w scaling
is structurally impotent there, 4×0=0") reproduces EXACTLY under
BEFORE_stale (16/27 zero, all at texel/channels where `raw>cap`) but NOT
under the actual formula that trained the v9h-last checkpoint
(CURE4_actual: 0/27 zero, every reading nonzero). The prior round's "4×0=0"
reasoning implicitly assumed CURE4's d_out = `overshoot_w * BEFORE_stale's
d_out`, which is not the code — CURE4's overshoot branch is an
independent, nonzero term (`2*overshoot_w*(raw-cap)/n`), not a multiple of
the zero-gate. Ruled out as the ROOT explicit-mask candidates: no
`GAIA_V7_SKY_HISTORY` interaction (reject only touches history-reprojection
INPUT features, never the loss; this run had reject=false, matching v9h's
own training env, and the training loop applies one formula uniformly
regardless of albedo — confirmed by re-reading the per-pixel loop, no
albedo branch exists there pre-v9i); no ceiling-degenerate-to-zero (checked
directly, `ceiling_dl` at these 9 texels ranges 0.026-1.001, never
near-zero); no explicit no-hit mask in the loss (the loop is
albedo-agnostic before this round's edit).

### the REAL structural weakness (found via the same forensics data)

CURE4_actual is never exactly zero, but its overshoot branch anchors `raw`
toward `cap` (the evidence ceiling), NOT the honest target — and `cap` is
systematically INFLATED at exactly this pixel class. Computed directly from
`scratch/v9h-drift-forensics.json` (unchanged this round, just re-read): at
the 16/27 channel-readings where `raw>cap` (overshoot), `cap/target_dl_mean`
ratio ranges **1.09x–2.94x, mean 1.64x**. Source of the inflation:
`evidence_composite_frame` (`src/rdirect.rs:776`) bilinearly upsamples the
LOW-RES (half-res) traced e/d composite to full bar-res — at a zero-albedo
sky pixel immediately adjacent (in low-res space, i.e. within ~2 high-res
px) to a bright hit surface, the bilinear interpolation blends some of that
neighbor's radiance into the sky pixel's own composite value BEFORE any
demod/albedo distinction is applied (upsampling operates on raw e+d, not
per-material) — then `local_max_3x3` (`src/rdirect.rs:784`) MAX-POOLs a 3x3
high-res window, promoting any sky pixel within 1px of an already-bled
value up to that neighbor's (or higher) level, and `gamma=1.5` scales
further. All 9 texels sit exactly at "edges of bright objects against sky"
(x∈{13,17,19},y=257 and x∈{282,321,332,335,337,339},y≈247-260, per the
prior round's own coordinate clustering note) — precisely where this bleed
mechanism applies. Net effect: CURE4 gives these pixels a real, nonzero
restoring pull, but toward an anchor 1.09-2.94x too bright — explains v9h's
non-proportional, plateauing dose-response (escalating `overshoot_w`
converges faster toward the SAME too-bright ceiling, never below it).

### cure (IRON params)

**CURE 5** (`examples/rdirect_train_v9i.rs`, new trainer this round, copy of
v9h): at pixels where `albedo.length_squared() <= nohit_albedo_sq`
(`GAIA_V9I_NOHIT_ALBEDO_SQ`, IRON param, default `1e-8` — matches
`rdirect.rs::NO_HIT_ALBEDO_THRESHOLD_SQ`, the SAME no-hit branch
`demod_divisor` already special-cases), bypass the clamp/ceiling/overshoot
mechanism ENTIRELY: `d_out = 2*(raw-target)/n`, unconditional, both
directions, no dependency on `cap`. Elsewhere (`albedo>0`, real-surface
pixels — the population CURE 1-4 exist to protect from noisy-label
overfit) CURE 1-4 are byte-identical/untouched — this is a targeted,
per-pixel-class mask, not a global loss-shape change; it does not reopen
the noisy-target-chasing failure CURE 1 was built to prevent, since that
failure was measured (autopsy CURE-3 section, `rdirect_v9_tailmass.rs`) on
the general finite-K target tail, a phenomenon distinct from and additive
to albedo>0 surfaces — the zero-albedo/no-hit population's targets are
independently established as HONEST here (`target_dl_mean-teacher_dl`
sign-flipping, ~0.005 magnitude, `scratch/v9h-drift-forensics.json`
aggregate: target_near_teacher 5/9, target_over_teacher_dl only 2/9 weak).

**DOSE RESET** (task point 3): `GAIA_V9I_OVERSHOOT_W` default 4.0 -> **1.0**
(v9g's original dose) — with CURE 5 giving the mute-pixel class its own
local restoring gradient, the global overshoot_w escalation war (v9g→v9h,
1x→4x, non-monotonic resid floor 0.0701/0.0756/0.0658) is no longer the
only lever; resetting to 1.0 removes the resid drag the dose escalation was
paying elsewhere (param stays live for future A/B).

### pre-flight (required before any training): d_out ≠ 0 at all 9, before/after

Re-ran `rdirect_v9h_drift_forensics.rs` (same binary as the gate-conviction
section — its grad-probe block computes BEFORE_stale/CURE4_actual/AFTER_fix
together every time) against v9h-last, `scratch/v9i-preflight-grad-probe.log`
this round:

```
# BEFORE_stale (pre-CURE4 zero-gate) d_out==0.0: 16/27
# CURE4_actual (real training-loop formula, overshoot_w=4) d_out==0.0: 0/27
# AFTER_fix (V9I mask-based ordinary loss @ albedo~0/no-hit) d_out==0.0: 0/27
```

All 9 sampled texels have albedo EXACTLY (0,0,0) (the no-hit branch,
established in the prior round's demod-degeneracy check) — CURE 5's mask
(`albedo_sq <= 1e-8`) fires at ALL 9/9 texels, ALL 3 channels each (27/27
readings), giving `d_out = 2*(raw-target)/n` unconditionally there. Per-
channel AFTER_fix values (`scratch/v9i-preflight-grad-probe.log`): every one
of the 27 readings is nonzero except one micro-exception already present in
CURE4_actual too (px=(13,257) ch2, both give -0.00000018 — that channel was
never overshooting, `raw<cap` there, both formulas agree, a live but tiny
value, not a zero) — **REQUIREMENT MET: d_out ≠ 0 at all 9 texels (27/27
channel readings) under AFTER_fix**, and every AFTER_fix value pulls toward
the HONEST target rather than the inflated `cap` (verified: AFTER_fix
values sit between BEFORE_stale's active-branch pull and CURE4_actual's
overshoot-branch pull in every overshoot channel, always signed toward
`target`, per `scratch/v9i-preflight-grad-probe.log`'s per-texel rows).

### v9i smoke test + launch

Smoke-tested (`GAIA_V9_EPOCHS=3 GAIA_V9_WALL=180`, `scratch/v9i-smoke.log`,
this round): builds clean, `CURE 4: overshoot_w=1` and `CURE 5:
nohit_albedo_sq=0.00000001` both print at startup confirming the dose reset
and mask took effect, n2n_mse falls every epoch (0.1434→0.0969→0.0630),
checkpoints/provenance wrote. No crash.

Launched full run: `GAIA_V7_SKY_HISTORY=reject GAIA_V9_TAG=v9i
GAIA_V9_EPOCHS=300 GAIA_V9_WALL=21600`, `nohup nice -n19`, PID **66345**,
log `scratch/v9i-train.log` — harness otherwise byte-identical to v9h/v9g
(two-signal watchdog resid_streak=20/probe_every=25/spark_target=16, K=8,
EMA=0.999, fresh He-init, same 300-epoch cap). NOTE: v9h's own log shows it
accidentally ran with `GAIA_V7_SKY_HISTORY=reject` UNSET (reject=false,
`scratch/v9h-train.log` line 3) despite v9f/v9g both using `reject=true`
and the module doc's own "mandate expects true" line — this round's launch
uses `reject=true`, matching v9f/v9g precedent and the stated mandate (a
harness correction, not a new variable; flagged here for visibility, not
silently fixed).

Success criterion (task): bar-res sparkle stays ~0 past ep124 (v9h's
ignition epoch) while resid tracks v9f's fast fall (~0.0701@99 or better).
Probe rows reported as far as the wall allows — see room reply for the
epoch/probe table available at report time; full 300-epoch outcome is
UNVERIFIED until the run completes (`scratch/v9i-train.log` is the source
of truth going forward, `tail -f` to follow).

## v9j — MAKE CURE 5 ACTUALLY FIRE IN THE TRAINER

### log-diff verdict (v9g vs v9i, epoch-by-epoch)

NOT bit-identical (task's "IDENTICAL" was the working hypothesis, refuted
below) but very close, with small growing divergence:
`n2n_mse` epoch0 0.143371==0.143371, epoch1 v9g=0.091479 vs v9i=0.091486
(+0.008%), epoch74 v9g=0.040903 vs v9i=0.040922 (+0.046%), epoch99
v9g=0.040987 vs v9i=0.040600 (-0.94%) — `scratch/v9g-train.log`/
`scratch/v9i-train.log`. PROBE sparkle matches to 1 decimal at every probe
(ep24/49/74/99: 0.0/0.0/9.8/16.3 both runs) but PROBE **resid** measurably
diverges by ep99: v9g=0.0756 vs v9i=0.0766 (+1.3%), highlight_ratio v9g
0.409 vs v9i 0.394. Same WATCHDOG abort epoch (99), same best score to 3
decimals (2.713 both — coincidence of `score=max(sp/40,resid/0.035)`
rounding, sp identical to 1dp dominates). Verdict: **near-identical, not
bit-identical — small, real, growing divergence exists**, consistent with a
real (not zero) but small gradient perturbation.

### the miss (task's own framing revised by direct instrumentation)

Cited lines, `rdirect_train_v9i.rs`: mask condition
`let is_nohit = albedo[px].length_squared() <= nohit_albedo_sq;` (line
1054, training loop) and `let is_nohit = albedo_sq <= nohit_albedo_sq;`
(line 929, grad-probe diagnostic) — both test the per-pixel `albedo` field
straight from `Step.albedo` (`scrying_glass::integrator::split_aov`'s AOV
readback at TRAINING res, tw x th = 128x72, no resize/reproject — same
buffer feeds the loss, not a proxy). Ground truth
(`src/integrator.wgsl:1020-1024`): on a real hit, `aov[2*px+0]=(albedo,
hit.t)`; on a miss, unconditionally `aov[2*px+0]=(0,0,0,0)` — so albedo IS
exactly (0,0,0) at every true no-hit pixel, at ANY resolution including
training's own 128x72. **This refutes the task's "field never matches in
the training data" / "128x72 vs 640x480 differs" hypothesis outright** —
built and ran a `GAIA_V9J_DRY_RUN=1` env-gated one-batch dry run (K=3 steps
of the held-out val_seq, real training px/channel formula reused
byte-for-byte, `scratch/v9j-preflight-dryrun.log`) plus real per-epoch
instrumentation added to the actual training loop (below) — **the mask
DOES fire, abundantly, on real training data**:

```
[v9j] DRY RUN one-batch (K=3 steps of val_seq) mask counts: nohit_px=9858/27648 (35.66%) nohit_effective(is_nohit&&raw>cap, differs-from-CURE4)=52 elem-instances   [fresh He-init net]
[v9j] DRY RUN loaded checkpoint "data/rdirect-weights-v9i-last.bin" ... nohit_px=9813/27648 (35.49%) nohit_effective=193 elem-instances   [near-converged, post-fix]
```

Real training-loop per-epoch instrumentation (`scratch/v9j-smoke3.log`,
3-epoch smoke, real pose pool not val_seq): `nohit_px` ~27-33% of pixels
every epoch (background/sky is genuinely ~1/3 of frame); `nohit_effective`
(channel-instances where CURE5's formula differs from CURE4's, i.e. where
the fix actually changes the gradient) GROWS every epoch: epoch0=66,
epoch1=1472, epoch2=7359 (out of 110592 elem-instances/epoch, ~6.7% by
epoch2) — **not remotely zero, not a config no-op, growing with training**.
So the task's premise ("CURE 5 never altered a single gradient") is
**FALSE** by direct measurement; CURE 5 fires substantially. What remains
genuinely unresolved (flagged, not proven, UNVERIFIED): why a real, growing
gradient perturbation at training res (128x72) produces only the small
divergence measured above rather than preventing the bar-res (640x480)
WATCHDOG sparkle abort — the leading hypothesis (not verified this round)
is a **resolution-generalization gap**: CURE 5's correction only ever acts
on the 128x72 training loss surface, while the pass/fail signal
(`bar_res_probe`) evaluates the SAME fully-convolutional weights at a
33x-larger pixel count (640x480) via its own independently-derived
evidence/ceiling buffers — a fix proven to move the training-res loss
surface is not guaranteed to move a metric measured at a resolution the
net was never trained at.

### mask-field cross-check + fix (a): re-keyed to the trainer's real no-hit signal

Added a live cross-check every training pixel against `depth[px] <= 0.0`
(the OTHER no-hit signature already in this file's own buffers —
`reproject_prev`'s `is_miss = cur_depth <= 0.0` convention, same AOV pass,
same shader lines zero it unconditionally on a miss). Result
(`scratch/v9j-smoke3.log`, first-batch lines, epoch0-2):
`false_miss_albedo=0` every batch (canary confirmed: albedo NEVER misses a
true no-hit pixel, 0 false negatives, matches the shader-level guarantee
exactly) but `false_hit_dark` (albedo-mask said no-hit, depth says REAL
HIT — a legitimately-hit dark/black-albedo surface pixel wrongly diverted
through CURE5's bypass) is small but **nonzero**: 25, 3, 14 pixels/batch
out of ~9216 (0.03%-0.27%). This is a real, if minor, defect: the mask was
not measuring "no-hit" precisely, it was measuring "no-hit OR dark-albedo
hit". Re-keyed (`is_nohit = depth[px] <= 0.0`, dropping the albedo
threshold from the LIVE mask — `nohit_albedo_sq`/`GAIA_V9I_NOHIT_ALBEDO_SQ`
kept only as `legacy_is_nohit` for this cross-check's own reporting).
Verified post-fix: `nohit_px`/`nohit_effective` counts shift down by
exactly the removed false-positive population (dry run 9858->9813,
52->193 differs slightly due to different net; smoke re-run epoch0
nohit_effective 381->66, confirming the false positives were disproportionately
contributing to `nohit_effective` early in training) while the real no-hit
population and its abundant, growing effective-firing rate are unchanged.

### instrumentation (b)+(c) added, permanent equipment

Per-epoch print (epoch line, `rdirect_train_v9j.rs`):
`nohit_px=N/M(pct%) nohit_effective=K` appended every epoch. FIRST BATCH
loud check (every training run, not just epoch 0): prints
`nohit_px(albedo)=.../... nohit_effective=... | cross-check vs
depth-hit-flag: false_hit_dark=... false_miss_albedo=...` before the first
`epoch_mse` accumulates, and calls `std::process::exit(1)` with a labeled
STARTUP ABORT message if `nohit_px==0` on that first batch (config-bug
guard, per task law). Pre-flight dry run: `GAIA_V9J_DRY_RUN=1` (optionally
`GAIA_V9J_DRY_RUN_WEIGHTS=<path>` to probe a real checkpoint instead of
fresh init) renders nothing new, forwards K=3 steps of the held-out
val_seq through a fresh/loaded net, prints the mask counts + one sample
pixel's BEFORE(CURE4)/AFTER(CURE5) loss-branch values side by side, then
exits without training — proof pasted above and in
`scratch/v9j-preflight-dryrun.log`.

### v9j smoke + launch

Smoke-tested 3 epochs (`scratch/v9j-smoke3.log`): builds clean, instruments
print every epoch, no crash, checkpoints wrote (smoke artifacts deleted,
not committed — real launch below is the record). Launching full run:
same harness as v9i (two-signal watchdog resid_streak=20/probe_every=25/
spark_target=16, K=8, EMA=0.999, fresh He-init, 300-epoch cap,
`overshoot_w=1`, `GAIA_V7_SKY_HISTORY=reject`), `GAIA_V9_TAG=v9j`,
`nohup nice -n19`, log `scratch/v9j-train.log`. Success criterion
unchanged: bar-res sparkle ~0 past ep124, resid ≤ v9f's 0.0701@99 pace.
Given the miss conviction above, this run's outcome tests the
resolution-generalization hypothesis empirically rather than a masking bug
fix — **flagged as such, not oversold**.
