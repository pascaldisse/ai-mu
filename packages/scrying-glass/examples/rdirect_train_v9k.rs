//! R-DIRECT v9k-body — BAR-RES CROP TRAINING (kill the res gap).
//! AUTOPSY THIS ROUND ACTS ON (RES GAP CONFIRMED BY EXHAUSTION,
//! `scratch/v9j-train.log` autopsy verdict, commit d93bb595): v9g/v9i/v9j
//! are three independent loss-side cures (mute-pixel activation, CURE5 gate
//! conviction, CURE5 depth-hit-flag re-key) — all three fire at TRAINING res
//! (128x72, instrumented and confirmed non-no-op every epoch) yet the
//! bar-res (640x480) probe is BYTE-SIMILAR across all three runs: sparkle
//! 9.8/Mpx at epoch 74, 16.3/Mpx at epoch 99 (WATCHDOG ABORT), in v9g AND
//! v9i AND v9j (`scratch/v9g-train.log`/`v9i-train.log`/`v9j-train.log`
//! PROBE lines, diffed this round — identical to 1 decimal place at every
//! probe checkpoint: ep24 0.0/0.0, ep49 0.0/0.0, ep74 9.8/9.8/9.8, ep99
//! 16.3/16.3/16.3). Hypothesis: the 640x480 drift class (thin sky-
//! silhouette pixels the watchdog's own bar-res probe measures) is a
//! RESOLUTION-ABSENT statistic at 128x72 — each 128x72 pixel spans a ~25x
//! larger solid angle (640*480 / (128*72) ≈ 25), so a thin drift feature
//! that occupies a handful of 640x480 pixels is invisible or averaged away
//! in the 128x72 training signal entirely. No amount of LOSS surgery (what
//! v9g/v9i/v9j all tried) can teach the net to fix pixels its training
//! signal structurally cannot see. This file changes the TRAINING SAMPLE
//! UNIT instead: full 128x72 frames -> 128x72 CROPS of true 640x480
//! renders — the net now sees the SAME pixel-density/solid-angle the
//! bar-res judge measures, just windowed to a sub-rect each draw, at
//! UNCHANGED per-draw pixel cost (crop_w*crop_h, not 640*480).
//!
//! IMPLEMENTATION (asymmetric frustum, no WGSL change — see
//! `integrator.rs::IntegratorUniform::build_cropped` for the full algebraic
//! derivation, cited here only in summary): a crop window is a SUB-RECT of
//! a virtual `full_w x full_h` (640x480) frustum, specified by a full-frame
//! pixel offset `(crop_x, crop_y)` and extent `(crop_w, crop_h)` (default
//! 128x72, IRON params `GAIA_V9K_CROP_W`/`GAIA_V9K_CROP_H`). The tracer's
//! ray-generation basis (right/up/forward, already computed on the CPU
//! once per draw in `IntegratorUniform::build`) is pre-warped by an affine
//! remap of the per-pixel NDC coordinate (`center_x/y`, `scale_x/y` — a
//! translate+scale of the `[-1,1]` screen square) so the SAME unmodified
//! shader entry points (`integrate_split`/`integrate_aov`), dispatched at
//! ONLY `out_w x out_h` pixels (`out_w/out_h` may equal `crop_w/crop_h` for
//! a full-density pass, or half that for the low-res evidence pass — same
//! two-density split `low_w=tw/2` already used pre-crop), cast EXACTLY the
//! rays a genuine `full_w x full_h` render would have cast inside the
//! sub-rect — cost is `out_w*out_h`, the crop is never actually rendered at
//! 640x480 and cropped after the fact (that would be the naive/expensive
//! path this implementation avoids). New GPU entry points (additive,
//! existing `trace_headless_split`/`trace_headless_aov`/`IntegratorUniform::
//! build` byte-unchanged): `trace_headless_split_cropped`, `trace_headless_
//! aov_cropped`, `IntegratorUniform::build_cropped`.
//!
//! CPU-SIDE HISTORY/REPROJECTION: `reproject_prev`'s cross-step reprojection
//! (world point from step s-1's screen -> step s's screen, for the
//! recurrent history feature) needs a camera-pose type that applies the
//! SAME affine NDC remap. Rather than touch the shared `rdirect::CamPose`
//! (21 construction sites across the crate would need updating for two new
//! fields — out of scope, avoided), this file defines a LOCAL `SeqCam`
//! (full-frame `CamPose` + `center`/`scale`) with its own `ray_dir`/
//! `reproject`, algebraically the exact inverse of `build_cropped`'s
//! forward mapping (verified: `rz`/depth-ratio projection is only valid
//! against an ORTHONORMAL right/up/forward basis, so `reproject` computes
//! the ratio against the UNMODIFIED full-frame basis first, THEN applies
//! the inverse affine remap to land in crop-local NDC — it does NOT reuse
//! `CamPose::reproject`'s formula with a pre-warped, non-unit `forward`,
//! which would be invalid). `PoseSeq`/`build_input_img`/`history_forward`/
//! `reproject_prev`/`render_pose_seq` — every mirror/val/bar-res-probe path
//! — are UNTOUCHED, byte-identical to v9j; a NEW parallel set
//! (`CroppedPoseSeq`/`build_input_img_crop`/`history_forward_crop`/
//! `reproject_prev_crop`/`render_pose_seq_cropped`) exists ONLY for the
//! three per-epoch POOL draws (front-pivot x2, wide-anchor x1) — the
//! task's own "TRAINING SAMPLE UNIT", i.e. the pixels that actually feed
//! gradients. `TrainSeq` (new enum) lets the epoch loop iterate 3 cropped +
//! 1 uncropped (mirror) draw uniformly without touching the loss/backward
//! code at all (it is resolution-shape-only, camera-content-agnostic —
//! genuinely doesn't care whether a 128x72 buffer is a crop or a whole
//! frame). PADDING/EDGE HONESTY: a cropped step's reprojection into the
//! crop's OWN previous-frame window can legitimately miss (previous crop
//! content simply isn't there) — `SeqCam::reproject`'s existing out-of-
//! `[0,w)x[0,h)` rejection already handles this (returns `None` ->
//! `valid=0`, the same "no history this pixel" state `reproject_prev`
//! already produces at disocclusions) with ZERO special-casing needed;
//! `GAIA_V9K_PAD_MARGIN` (default 0) additionally keeps crop OFFSETS away
//! from the full frame's outer edge by a margin, so panning
//! (`GAIA_V9_PANSTEP`, unchanged) during the K=8 unroll doesn't walk the
//! crop window itself off the full frame.
//!
//! MIRROR/VAL POSES UNCHANGED (task instruction, verbatim): `mirror_seq`
//! (also used as the 4th per-epoch TRAINING draw, a FIXED anchor — cropping
//! it adds no diversity since it never moves) and `val_seq`/`mirror_val_seq`
//! (monitor-only, held-out) keep rendering full 128x72 frames via the
//! ORIGINAL `render_pose_seq`, exactly as v9j. `bar_res_probe` (the 640x480
//! judge) is BYTE-IDENTICAL to v9j — it already rendered true 640x480, the
//! thing this file changes is what the net TRAINS on, never what judges it.
//!
//! EQUIPMENT KEPT (task mandate, ALL from v9j, unchanged): two-signal
//! watchdog (resid-streak abort + bar-res-probe abort, score-streak
//! log-only), bar-res probe every 25 epochs, dual checkpoints (best-score +
//! unconditional -last), masked-px counters + first-batch abort guard
//! (CURE5 depth-hit-flag re-key, `nohit_px`/`nohit_effective`/
//! `false_hit_dark`/`false_miss_albedo`), K=8 n2n draws, EMA, `GAIA_V7_SKY_
//! HISTORY=reject` mandate default, `overshoot_w=1.0` (CURE4 dose reset).
//! The v9i/v9j opt-in diagnostics (`GAIA_V9I_GRAD_PROBE`, `GAIA_V9J_DRY_
//! RUN`) are ALSO kept verbatim below (dead by default, harmless) — this
//! file adds a NEW opt-in `GAIA_V9K_CROP_DRY_RUN=1` pre-flight (no-silent-
//! cure law): one K=8 batch at TWO different random crop offsets, printing
//! both offsets, the nohit-masked-px count ON CROPS (must be >0 — a crop
//! that never lands on a no-hit/sky pixel would silently make CURE5's
//! mask a no-op), and a checksum (sum of the K-draws' `target_dl` buffer)
//! for each offset that must DIFFER between the two draws (proof the crop
//! offset genuinely changes what gets rendered, not silently ignored/
//! clamped to a fixed window).
//!
//! R-DIRECT v9j-body — MAKE CURE 5 ACTUALLY FIRE IN THE TRAINER.
//! v9i's CURE 5 (mask on albedo≈0/no-hit) reproduced v9g's trajectory
//! BIT-FOR-BIT (probes ep74 sparkle 9.8, ep99 16.3, abort ep99, best=2.713
//! — identical to v9g, see scratch/v9-autopsy.md v9j section): the mask's
//! condition (`albedo[px].length_squared() <= nohit_albedo_sq`) is a real
//! no-hit signature at TRAINING res (128x72) too (the AOV shader zeroes
//! albedo on a geometric miss unconditionally, see
//! `integrator.wgsl:1023-1024`) BUT CURE 5's formula is IDENTICAL to the
//! `active` branch's formula whenever `raw <= cap` — it only differs from
//! CURE4 when `is_nohit && raw > cap` (mask actually overrides the
//! overshoot branch). v9j instruments both counts every epoch
//! (`nohit_px`/`nohit_effective`) plus a first-batch loud abort if
//! `nohit_px==0`, and an env-gated one-batch dry run
//! (`GAIA_V9J_DRY_RUN=1`) printing the counts + one sample pixel's
//! before/after loss branch. Base (unless noted): `rdirect_train_v9i.rs`
//! byte-identical.
//!
//! (`scratch/v9-autopsy.md` v9h grad-probe-fix section). Base:
//! `rdirect_train_v9h.rs` UNCHANGED (two-signal watchdog, dual checkpoints,
//! periodic bar-res probe, K=8, EMA, fresh init, CURE 1-4 mechanisms —
//! every mechanism not listed here is byte-identical, see that file's own
//! header below) except:
//!   (1) CURE 5 (NEW): at albedo≈0/no-hit pixels (`GAIA_V9I_NOHIT_ALBEDO_SQ`,
//!       default 1e-8, matches `rdirect.rs::NO_HIT_ALBEDO_THRESHOLD_SQ`),
//!       the clamp/ceiling/overshoot mechanism is bypassed entirely —
//!       ordinary, unconditionally-active two-sided MSE against the honest
//!       n2n target (`2*(raw-target)/n`). Elsewhere (albedo>0, real-surface
//!       pixels) CURE 1-4 are byte-identical/untouched.
//!   (2) DOSE RESET: `GAIA_V9I_OVERSHOOT_W` default 4.0 -> 1.0 (v9h's 4x
//!       escalation was a symptom war against a mechanism CURE 5 now fixes
//!       locally; param stays for A/B, see autopsy "DOSE RESET" section).
//!
//! GATE CONVICTION (autopsy, this round, full derivation + quoted lines
//! there): the task's standing evidence ("active=false, d_out=0.0 in the
//! grad probe" at the 9 bar-res sparkle-flagged texels) reproduces EXACTLY
//! under the standalone `GAIA_V9H_GRAD_PROBE=1` diagnostic's own hardcoded
//! formula (pre-CURE4 zero-gate, `rdirect_train_v9h.rs:887-889` — STALE,
//! never updated when CURE4 landed in v9g) — 16/27 channel readings zero
//! there. Rerunning the SAME 9 texels/v9h-last weights through the REAL,
//! currently-training CURE4 formula (`rdirect_train_v9h.rs:1010-1020`)
//! gives ZERO exactly-zero readings (0/27) — CURE4's overshoot branch was
//! never actually mute, it was measured with dead diagnostic code. The REAL
//! defect: CURE4's restoring pull anchors to `ceiling_dl` (`cap`), not the
//! honest target — and `cap` is systematically INFLATED at this exact
//! pixel class (mean 1.64x, up to 2.94x the honest 256-draw target at the
//! 16 overshoot-channel readings), because `evidence_composite_frame`
//! bilinearly upsamples LOW-RES e/d traces to full res (bleeding a
//! neighboring bright-surface texel's radiance into the adjacent sky/no-hit
//! texel right at the silhouette edge) and `local_max_3x3` widens that
//! bleed further — so CURE4 converges these pixels toward a too-bright
//! anchor, never down to the honest target, regardless of `overshoot_w`
//! dose (explains v9h's non-proportional, plateauing dose-response). CURE 5
//! removes the (wrong, for this class) ceiling anchor entirely at exactly
//! the population it is unreliable for, using the (honest, |target-teacher|
//! ~0.005) n2n target directly instead.
//!
//! V9H VERDICT THIS ROUND ACTS ON (`scratch/v9g-train.log`,
//! `scratch/v9-autopsy.md` v9g verdict): v9g (CURE 4, overshoot_w=1.0) ran
//! to the SAME abort shape as v9f — `WATCHDOG ABORT: bar-res probe sparkle
//! failed the bar at epoch 99` — with the drift measurably slowed but not
//! stopped: bar-res probe sparkle/resid 0.0/0.1218 (ep24) -> 0.0/0.1084
//! (ep49) -> 9.8/0.0926 (ep74) -> 16.3/0.0756 (ep99, ABORT), vs v9f's
//! 0.0/0.1211 -> 0.0/0.1052 -> 13.0/0.0872 -> 22.8/0.0701 (ABORT).
//! `highlight_ratio` at ep99 also lower (0.409 vs 0.578). The restoring
//! gradient (CURE 4's mechanism) is doing real work — not a no-op — but
//! proportionally too weak: sparkle still crosses the tgt<16 bar at the
//! SAME epoch (99) because resid falls in lockstep at a similar rate, so
//! the down-pull never outweighs CURE 1's upward pull enough to hold the
//! ceiling. This file escalates DOSE ONLY (`GAIA_V9H_OVERSHOOT_W` default
//! 4.0, upper end of the evidence-indicated 3-4x range) — same mechanism,
//! same seam (C0-continuous at raw==cap only when overshoot_w=1; at 4.0 the
//! above-ceiling branch's slope is 4x steeper, still 0 at the seam itself),
//! no other change, to keep this an interpretable single-variable step.
//!
//! MECHANISM HUNT VERDICT (v9f ran to WATCHDOG ABORT ep99, `scratch/v9f-
//! train.log`; texel forensics `scratch/v9e-forensics.log` COORDS): CURE 1
//! (`rdirect_train_v9b.rs` doc, `rdirect.rs::accumulate_backward_clamped_
//! slice`/`evidence_ceiling_demod_log`) computes `presented = min(out_dl,
//! ceiling_dl)` and sets the backprop delta to EXACTLY ZERO wherever
//! `out_dl > ceiling_dl` (v9b doc, verbatim: "the backprop delta is 0 — the
//! net gets no gradient signal to push EVEN HIGHER once it already exceeds
//! what the evidence supports"). That description is only HALF true: zero
//! gradient does stop REWARDING further rise, but it ALSO removes every
//! PENALTY for having already overshot — a one-sided gate, not a
//! restoring force. `GAIA_V9H_GRAD_PROBE=1` (added this round, reused from
//! v9f) instruments this directly: loaded `data/rdirect-weights-v9f-last.
//! bin`, ran the val_seq (orbit_-20) recurrent chain through it, and
//! evaluated the EXACT training-loop CURE-1 formula at the 7 fixed texels
//! `scratch/v9e-forensics.log` COORDS tracked drifting (~0.5%/ep, epoch 54-
//! 73): 20/21 channel readings had `raw > cap` (overshoot=true) AND
//! `active=false, d_out=0.00000000` exactly — the downward gradient was
//! COMPLETELY STARVED at every one of the drifting pixels, at every
//! channel but one, in the very checkpoint that shows the drift. This
//! confirms the autopsy hypothesis: monotonic, unopposed drift because
//! nothing pulls back once a pixel crosses the ceiling — Adam momentum
//! carried from the pre-crossing epochs (when the gradient WAS positive and
//! pushing up) plus U-net weight-sharing spillover from OTHER pixels'
//! active gradients are the only things still moving these pixels, and
//! both push in whatever direction those other signals want, not down.
//! (Ruled out (b): `log_demod`/`undo_log_demod` (`rdirect.rs` L73-85) is a
//! plain monotonic log1p/expm1 pair; the CURE-1 MSE is computed directly in
//! that log-space `(presented_dl-target_dl)^2` with NO extra clamp/epsilon
//! inside the loss itself, so the log domain does not by itself introduce
//! gradient asymmetry — over/undershoot get equal-magnitude log-space
//! gradients. It IS an AMPLIFIER, not the cause: `undo_log_demod` is `exp(dl)
//! -1`, so a fixed small log-space drift compounds into exponentially
//! larger LINEAR sparkle once exponentiated — why a slow, steady ~0.5%/ep
//! log-space creep only crossed the render-res sparkle bar (22.8 > 16) at
//! epoch 99, not earlier.)
//!
//! CURE 4 (introduced v9g, this file's IRON param `GAIA_V9H_OVERSHOOT_W`,
//! DEFAULT ESCALATED to 4.0 this round — see V9H VERDICT above):
//! replace the single `min(out_dl,ceiling_dl)` clamp (whose subgradient is
//! ONE-SIDED: 1 below the ceiling, 0 above) with a loss that ALSO penalizes
//! the overshoot itself, so the clamp becomes symmetric — a restoring force
//! exists on BOTH sides of the ceiling, and overshoot ALWAYS hurts:
//!   loss = (min(out_dl,ceiling_dl) - target_dl)^2
//!        + OVERSHOOT_W * max(0, out_dl-ceiling_dl)^2
//! d(loss)/d(out_dl) = 2*(out_dl-target_dl)      when out_dl <= ceiling_dl  (UNCHANGED, CURE 1's own term)
//!                     2*OVERSHOOT_W*(out_dl-ceiling_dl)  when out_dl > ceiling_dl  (NEW — pulls toward the CEILING, not the possibly-noisy K=8 target; keeps CURE 1's original philosophy that the evidence ceiling, not one noisy draw-averaged target pixel, is the trusted anchor above the bar)
//! At OVERSHOOT_W=1.0 the two branches are C0-continuous in slope at the
//! seam (both equal 0 exactly at out_dl==ceiling_dl); this is the minimal
//! change that keeps CURE 1's reported MSE value (`presented`-based, used
//! for `n2n_mse`/monitor logging) byte-identical while fixing ONLY the
//! gradient that feeds Adam.
//!
//! Original v9f header, still true for every OTHER mechanism in this file:
//!
//! R-DIRECT v9f-body — TRAIN THROUGH THE "DETONATION", JUDGE AT BAR RES.
//! Base: `rdirect_train_v9e.rs` (== `rdirect_train_v9c.rs` CONFIG UNCHANGED +
//! CURE 3 winsorize, now DEFAULT OFF — see below). Every mechanism not
//! listed here is byte-identical to `rdirect_train_v9e.rs`/`rdirect_train_v9c.rs`
//! — not re-duplicated, see those files' own headers.
//!
//! EVIDENCE THIS ROUND ACTS ON (`scratch/v9e-train.log`, `scratch/v9-autopsy.md`):
//!   (a) v9e (winsorized targets) reproduced v9c EXACTLY — onset ep51->52,
//!       floor resid 0.0938, best score 2.680 vs v9c's 2.678. Target
//!       treatment changed nothing measurable -> target-tail-chasing
//!       WEAKENED as the score-rise cause (fireflies ARE present in targets
//!       per `rdirect_v9_tailmass`, but clipping them didn't move onset or
//!       floor).
//!   (b) Through the "detonation" (ep51->72) resid KEPT FALLING
//!       (0.0938->0.0790) while the LOW-RES (128x72) monitor's sparkle rose
//!       in exact discrete steps: 1,2,4,5,6,7 peak pixels (108.5, 217.0,
//!       434.0, 542.5, 651.0, 759.5 /Mpx * 9216px/1e6 == that integer count,
//!       confirmed by direct computation) — a handful of DISCRETE coarse
//!       pixels, not a spreading fire.
//!   (c) Every 640x480 eval ever run (v9/v9b/v9c/v9d, `scratch/v9*-eval-*`)
//!       measured sparkle 0.0, including post-onset-era checkpoints.
//! HYPOTHESIS UNDER TEST: the 128x72 probe-res monitor's sparkle spike is a
//! COARSE-RESOLUTION EVALUATION ARTIFACT — highlight energy concentrating
//! into a few 128x72 pixels while the net is still LEARNING highlights
//! (highlight_ratio climbing 0.27->0.50 toward teacher parity through the
//! same window, not runaway) — NOT a real quality collapse at render
//! resolution. The v9c/v9d/v9e watchdog (score-streak alone) has been
//! killing runs that were still improving. v9f trains straight through the
//! score rise and judges quality at the BAR'S OWN RESOLUTION instead.
//!
//! CHANGES vs v9e (three, all additive/config — no mechanism removed):
//!
//! 1. TWO-SIGNAL WATCHDOG. The OLD sole trigger (score > best*1.05 sustained
//!    `abort_streak` monitor calls) is now LOG-ONLY (`rising_streak` still
//!    printed every monitor line, unchanged meaning) — it no longer aborts
//!    anything by itself, since evidence (b) shows the low-res score can
//!    detonate on a handful of coarse pixels while everything else (resid,
//!    highlight_ratio, render-res sparkle) keeps improving. Abort now fires
//!    on EITHER of two independent signals:
//!      A. RESID-STREAK — `resid_rising_streak` mirrors the OLD score-streak
//!         logic exactly, just on `rs` (val resid) instead of `score`:
//!         resets on improvement-or-within-5%-of-best-ever, increments
//!         otherwise; abort when it reaches `GAIA_V9F_RESID_STREAK`
//!         (default 20). Catches a REAL quality collapse (resid sustained-
//!         rising), the failure mode v9's own unclamped detonation actually
//!         showed and CURE 1/2 were built against.
//!      B. RENDER-RES PROBE FAIL — signal 3 below: if the periodic bar-res
//!         probe's sparkle >= `spark_target` (16, same bar the low-res
//!         monitor targets) at ANY probe checkpoint, abort immediately (no
//!         streak — a probe is already expensive/infrequent, one hit is
//!         enough since it directly measures the thing the bar cares
//!         about).
//!
//! 2. DUAL CHECKPOINTS. `data/rdirect-weights-v9f.bin` (+ .provenance.json)
//!    keeps the OLD best-score convention (score = max(sp/40, resid/0.035)
//!    at LOW res, unchanged formula — still useful as a fast in-loop
//!    ranking signal even though we no longer trust it alone to gate
//!    training). NEW: `data/rdirect-weights-v9f-last.bin` (+
//!    .provenance.json) is the EMA net as of the most recent monitor call,
//!    unconditionally overwritten every monitor epoch regardless of score —
//!    since training now runs THROUGH the low-res detonation on purpose,
//!    the run's true end state (or the state at any abort) needs its own
//!    artifact distinct from whichever low-res epoch happened to have the
//!    best low-res score.
//!
//! 3. PERIODIC BAR-RES PROBE. Every `GAIA_V9F_PROBE_EVERY` epochs (default
//!    25), `bar_res_probe` below re-derives `rdirect_v9_eval_640.rs`'s own
//!    measurement (same held-out `orbit_-20` pose, same
//!    undo-log-demod + v7 structural evidence-clamp-at-inference fix that
//!    file introduced, same `GAIA_V9_EVAL_W/H/K/REF` env vars) on the
//!    CURRENT ema weights (not reloaded from disk — the live in-memory net)
//!    and logs `sparkle/resid/highlight_ratio` AT RENDER RES into the
//!    training log every time — the bar's own resolution becomes a monitor
//!    of record, not just a post-hoc eval script run once at the end.
//!
//! Run: cargo run --release -j2 --example rdirect_train_v9f
//!   GAIA_V7_SKY_HISTORY=reject GAIA_V9_K=8 (mandate defaults)

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, trace_headless_aov, trace_headless_aov_cropped,
    trace_headless_split, trace_headless_split_cropped,
};
use scrying_glass::rdirect::{
    CamPose, HIST_FEATURES_SPLIT, INPUT_FEATURES_SPLIT, OUTPUT_CHANNELS, evidence_ceiling_demod_log,
    evidence_clamp_gamma, evidence_composite_frame, hist_features_split, local_max_3x3,
    pixel_features_split, sky_history_reject, target_demod_log,
};
use scrying_glass::rdirect_unet::cpu::{Img, UnetAdam, UnetWeights, deserialize_weights, serialize_weights, weights_sha256};
use scrying_glass::rdirect_unet::{MOTION_VECTOR_CHANNELS, UnetConfig};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};
use scrying_glass::denoiser_dataset::{camera_at, orbit_camera};

const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;
const DRAW_A_SEED_BASE: u32 = 0x7abc; // v8d parity — net's own input evidence
const DRAW_B_SEED_BASE: u32 = 0xB222; // v8d parity — noise2noise label draws
// DOMAIN FIX (autopsy conviction #2, scratch/v9-autopsy.md): the net's raw
// output is demod-log radiance, not linear — undo it before comparing
// against the linear teacher, exactly like v8d's own inference path.
const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;
fn demod_divisor(albedo: GVec3) -> GVec3 {
    if albedo.length_squared() > NO_HIT_ALBEDO_THRESHOLD_SQ {
        albedo + GVec3::splat(ALBEDO_DEMOD_EPS)
    } else {
        GVec3::ONE
    }
}
fn undo_log_demod(dl: GVec3, divisor: GVec3) -> GVec3 {
    let expm1 = GVec3::new(dl.x.exp() - 1.0, dl.y.exp() - 1.0, dl.z.exp() - 1.0);
    GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * divisor
}
// V9Q forward transform (`rdirect.rs::log_demod`, private there — local
// copy, same convention as `demod_divisor`/`undo_log_demod` above): needed
// to convert the despeckle cap (derived in LINEAR radiance, the sparkle
// criterion's own domain) back into the loss's demod-log domain, the exact
// inverse of `undo_log_demod` above.
fn log_demod(radiance: GVec3, divisor: GVec3) -> GVec3 {
    let d = radiance / divisor;
    GVec3::new((d.x.max(0.0) + 1.0).ln(), (d.y.max(0.0) + 1.0).ln(), (d.z.max(0.0) + 1.0).ln())
}

// ── V9W DISPLAY-DOMAIN LOSS (task mandate) ─────────────────────────────────
// v9v's autopsy left "loss-domain" as the last named suspect: the loss is
// computed in demod-log space (`(presented_dl-target_dl)^2`), a domain the
// eye never sees directly — error there is NOT proportional to perceived
// error once it passes through `undo_log_demod` (exp) and the display
// transform (sRGB gamma) every actual viewing (write_png/eval PNGs) applies.
// `GAIA_V9W_LOSS_DOMAIN=display` moves the loss's OWN diff into that
// presented-pixel domain: forward-transform demod-log -> linear
// (`undo_log_demod`'s own domain, ALREADY defined above) -> display
// (`linear_to_srgb`, exposure=1.0 — verbatim from `rdirect_reference.rs`/
// `rdirect_v9_eval_640.rs::write_png`, the ONLY exposure any write_png call
// in this crate ever uses), keep the CURE4/CURE5 active-MSE/overshoot shape
// UNCHANGED (same branches, same weights), just evaluated on transformed
// values, then chain-rule the gradient back through both stages to the
// demod-log `raw` value backward() actually needs. Default '' (unset) takes
// NONE of this — the old branches below run byte-identical (IRON LAW).
fn v9w_loss_domain() -> String {
    std::env::var("GAIA_V9W_LOSS_DOMAIN").unwrap_or_default()
}
// ── V9X MIXED-DOMAIN LOSS (task mandate) ───────────────────────────────────
// V9W's display-domain grad (module doc above) kills the sparkle drive but
// its plateau (resid gap 0.0473 vs 0.035, v9w2 postmortem) suggests the
// classic demod-log MSE's own gradient carries a "linear reason to render
// the glints" the pure display-domain shape damps out (sRGB's own slope
// flattens hard near white). `GAIA_V9X_MIX` blends the two EXISTING d_out
// shapes as a weighted sum, no new gradient math invented: total =
// mix*display_grad + (1-mix)*classic_grad. Default -1.0 (unset/negative) is
// the feature OFF switch — old behavior byte-identical (IRON LAW), the
// original `display_domain`-gated branch runs untouched. Any value >=0.0
// (clamped to [0,1] as a safety net; task range is 0.0..1.0) turns mixing on
// and OVERRIDES/IGNORES `GAIA_V9W_LOSS_DOMAIN` entirely — the blend, not the
// single-domain switch, decides every px's grad once mix is set (mix=0.0 ==
// pure classic, mix=1.0 == pure display, regardless of GAIA_V9W_LOSS_DOMAIN).
fn v9x_mix() -> f32 {
    env_f32("GAIA_V9X_MIX", -1.0)
}
/// Verbatim from `rdirect_reference.rs`/`rdirect_v9_eval_640.rs::write_png`
/// (separate binaries, no shared crate fn to import — local copy, kept
/// byte-identical on purpose so this loss measures the SAME curve a PNG
/// viewer would).
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}
/// d(linear_to_srgb)/d(c) at the ORIGINAL (pre-clamp) c: the clamp's own
/// subgradient is exactly 0 outside [0,1] (a saturated display pixel — over
/// or under exposed — carries no gradient, same convention `undo_log_demod`'s
/// `.max(0.0)` already uses elsewhere in this file); inside, the analytic
/// derivative of the piecewise sRGB curve.
fn dsrgb_dlinear(c: f32) -> f32 {
    if c <= 0.0 || c >= 1.0 {
        0.0
    } else if c <= 0.003_130_8 {
        12.92
    } else {
        (1.055 / 2.4) * c.powf(1.0 / 2.4 - 1.0)
    }
}
/// Forward chain: demod-log `dl` -> linear (albedo-demodulated composited
/// radiance, exactly `undo_log_demod`'s scalar-channel form) -> display
/// [0,1] (`linear_to_srgb`, exposure=1.0). `divisor_c` is one channel of
/// `demod_divisor(albedo[px])` (a per-px constant, not a function of `dl`).
fn dl_to_display(dl: f32, divisor_c: f32) -> f32 {
    let expm1 = dl.exp() - 1.0;
    let linear = expm1.max(0.0) * divisor_c;
    linear_to_srgb(linear)
}
/// d(display)/d(dl), chain-ruled through both stages of `dl_to_display`:
/// `max(expm1,0)` zeroes the gradient when `dl<=0` (same subgradient
/// `undo_log_demod` already accepts); `dsrgb_dlinear` zeroes it again if the
/// resulting linear value saturates the display clamp.
fn ddisplay_ddl(dl: f32, divisor_c: f32) -> f32 {
    let expm1 = dl.exp() - 1.0;
    let linear = expm1.max(0.0) * divisor_c;
    let dlinear_ddl = if expm1 > 0.0 { dl.exp() * divisor_c } else { 0.0 };
    dsrgb_dlinear(linear) * dlinear_ddl
}

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(n: &str, d: f32) -> f32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// ── V9V INPUT FIREFLY CLAMP (task mandate) ─────────────────────────────────
// v9u falsified the noisy-teacher hypothesis (scratch/v9u-train.log: clean
// 512spp targets did NOT kill sparkle, curve identical to v9t) — the drive
// is INPUT-side, not target-side. This is the FIRST cure applied where
// fireflies actually ENTER the net: `pixel_features_split`'s own 2x2
// demod-log evidence taps (E block then D block, 12 features each). Per
// channel per 2x2 block: ceiling = median-of-the-4-taps + delta; any tap
// above it is pulled down. Teacher-free, cheap, live-path replicable
// (mirrors exactly what a runtime denoiser could do to its own taps).
// `GAIA_V9V_INPUT_CLAMP` default 0.0 disables (IRON LAW: byte-identical to
// v9k..v9u). Applied IDENTICALLY at all 3 feature-builder call sites
// (build_input_img/build_input_img_crop/bar_res_probe) — each calls this
// right after its own `pixel_features_split`.
fn v9v_input_clamp_delta() -> f32 {
    env_f32("GAIA_V9V_INPUT_CLAMP", 0.0)
}
static INPUT_CLAMP_TAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn input_clamp_taps_snapshot() -> u64 {
    INPUT_CLAMP_TAPS.load(std::sync::atomic::Ordering::Relaxed)
}
// V9W2 NUMERIC-CHECK HARDENING (postmortem of the v9w ep24 false abort):
// the check's own verdict must never flip on a single noisy round — track
// CONSECUTIVE failing rounds here (reset to 0 on any PASS) and abort only
// at 2 in a row. See the check site (V9W NUMERIC CHECK block) for the
// full diagnosis (f32 catastrophic cancellation, not a chain-rule bug).
static V9W_NUMERIC_CHECK_CONSEC_FAILS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// `base` layout (`pixel_features_split`): [0..12)=E's 2x2x3 demod-log taps,
/// [12..24)=D's 2x2x3 demod-log taps, tap stride 3 (one f32 per channel).
/// `delta<=0.0` is a no-op (old behavior, byte-identical). Bumps the shared
/// no-silent-cure counter once per channel-instance actually pulled down.
fn apply_input_clamp(base: &mut [f32; INPUT_FEATURES_SPLIT], delta: f32) {
    if delta <= 0.0 {
        return;
    }
    for block_start in [0usize, 12usize] {
        for c in 0..3usize {
            let vals = [
                base[block_start + c],
                base[block_start + 3 + c],
                base[block_start + 6 + c],
                base[block_start + 9 + c],
            ];
            let mut sorted = vals;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = 0.5 * (sorted[1] + sorted[2]);
            let ceiling = median + delta;
            for tap in 0..4usize {
                let idx = block_start + tap * 3 + c;
                if base[idx] > ceiling {
                    base[idx] = ceiling;
                    INPUT_CLAMP_TAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
}

// ── V9V REAL MOTION SEQUENCES (task mandate) ───────────────────────────────
// Per-epoch pool draws' virtual camera now genuinely MOVES between the K
// history steps (translation + yaw), so the recurrent mv/history-reprojection
// channels (fed ~0 before — GAIA_V9_PANSTEP's tiny 0.004rad yaw-only turn
// produced sub-pixel motion) carry a REAL signal, resembling actual play
// speeds. `GAIA_V9V_MOTION` default 0 disables (IRON LAW: byte-identical
// old yaw-only `pan_step` stepping in `render_pose_seq_cropped`).
fn v9v_motion_on() -> bool {
    env_u32("GAIA_V9V_MOTION", 0) != 0
}
fn v9v_cam_step() -> f32 {
    env_f32("GAIA_V9V_CAM_STEP", 0.15)
}
fn v9v_yaw_step() -> f32 {
    env_f32("GAIA_V9V_YAW_STEP", 0.02)
}

/// V9U CONVERGED-TEACHER TRAINING (module doc / task mandate): all prior
/// training used a K=8 (`GAIA_V9_K`) noise2noise average of 1spp draws as
/// the LOSS target (`target_dl`) — noisy by construction, the convicted
/// root cause of persistent bar-res speckle (`examples/rdirect_reference.rs`
/// commit 920b3e63 proved a 512spp accumulated render is clean AND only
/// 1.7s/frame at 640x480 on GPU). `GAIA_V9_TEACHER_SPP` (default 0) opts
/// in: >0 replaces the K-draw n2n target with a TRUE CONVERGED render at
/// that many total samples, via the SAME chunked-accumulation machinery
/// `rdirect_reference.rs::render_converged` uses (equal-chunk weighted
/// mean — the result is the exact overall mean regardless of chunking).
/// Default 0 is BYTE-IDENTICAL to v9k..v9t (IRON LAW): the old `k_draws`
/// branch is left in place, untouched, never removed. Inputs (the net's
/// own 1spp live-evidence taps, `low_e`/`low_d`/albedo/normal/depth) are
/// NOT touched by this env — only the target the loss compares against.
fn env_teacher_spp() -> u32 {
    env_u32("GAIA_V9_TEACHER_SPP", 0)
}
/// Samples per progress/accumulation chunk for the converged-teacher
/// render above — mirrors `rdirect_reference.rs`'s own `GAIA_REF_CHUNK`
/// (default 100), same rationale (equal-chunk running mean, no precision
/// or timeout concern at any chunk size).
fn env_teacher_chunk() -> u32 {
    env_u32("GAIA_V9_TEACHER_CHUNK", 100)
}

/// V9N HIT-GATED BODY (module doc / task mandate): architectural cure —
/// six rounds (v9g..v9m) proved no LOSS-SIDE term stops the no-hit/sky
/// sparkle drift class, so this GATES THE OUTPUT ITSELF: composed =
/// m*net_out + (1-m)*evidence, m = per-px hit flag (1.0=hit, 0.0=no-hit,
/// same `depth[px] <= 0.0` signature CURE 5 already uses), evidence = the
/// accumulated E+D composite the net's own input taps are built from
/// (`evidence_composite_frame`, exactly what `pixel_features_split` samples)
/// converted into the loss's demod-log domain via `target_demod_log` — so a
/// no-hit pixel's composed output is EXACTLY its evidence, by construction,
/// never approximately via a loss weight. `GAIA_V9N_HITGATE=1` opts in;
/// default 0 is BYTE-IDENTICAL to v9k/v9m (IRON LAW).
fn hitgate_on() -> bool {
    matches!(std::env::var("GAIA_V9N_HITGATE").as_deref(), Ok("1"))
}

/// V9O DEGENERATE-DEMOD GATE (module doc / task mandate): v9n forensics
/// (scratch/v9n-barres-forensics.log/json, commit 13349d3d) proved the
/// bar-res sparkle class is 11/11 HIT px (depth>0 — v9n's hit-gate
/// correctly leaves them on the net-output path) whose albedo is EXACTLY
/// 0.0 — the demod divisor degenerates to 1.0 exactly like a true no-hit/
/// sky px (`demod_divisor` above), so the class is architecturally
/// identical to no-hit even though it is a real geometric hit (this is the
/// trainer's own `false_hit_dark` population, re-keyed on a luminance
/// threshold instead of `false_hit_dark`'s squared-length one). This widens
/// V9N's gate mask from `no-hit` to `no-hit OR (hit AND albedo luminance <=
/// GAIA_V9O_DARK_ALB)` — gated px still get output:=evidence, grad:=0,
/// exactly as v9n. Default 1e-4; `<= 0.0` disables the extension entirely
/// (an explicit guard below, not just a threshold that happens to never
/// match) = v9n behavior byte-identical (IRON LAW parameter).
fn v9o_dark_alb() -> f32 {
    env_f32("GAIA_V9O_DARK_ALB", 1.0e-4)
}
/// Shared predicate used by settle()/bar_res_probe()/the training loop so
/// the 3 sites can never drift apart: true on a real no-hit px, OR on a hit
/// px whose albedo luminance falls at/under the dark-albedo threshold
/// (disabled when the threshold is <= 0.0).
#[allow(dead_code)] // superseded by inline is_nohit/is_dark_hit splits (v9p needs to treat them differently); kept for history/reference.
fn v9o_gate_px(depth_val: f32, albedo: GVec3) -> bool {
    depth_val <= 0.0 || (v9o_dark_alb() > 0.0 && lum(albedo) <= v9o_dark_alb())
}

/// V9P ROBUST FIREFLY CAP (module doc / task mandate): v9o's dark-hit
/// widening (above) replaced net output with the RAW evidence composite at
/// dark-albedo hit px — but that evidence is itself firefly-ridden at
/// exactly this specular/degenerate-demod population (measured: instant
/// ignition 81.4@24, worse than v9n's drift 35.8@124). The cure keeps v9n's
/// no-hit gate untouched (composed:=evidence, grad:=0) and, at dark-hit px
/// ONLY, keeps the NET's own output but caps it with a firefly-immune
/// statistic: the per-channel MEDIAN of `evidence_dl` over the px's 3x3
/// neighborhood (median is insensitive to a single-pixel firefly, unlike
/// the sparkle detector's own local-max/evidence-ceiling machinery which a
/// firefly trivially wins) plus `GAIA_V9P_DARK_CAP_DELTA` (default 0.10,
/// deliberately BELOW the sparkle detector's own 0.15 local-peak delta in
/// `sparkle_resid_per_mpx` — so a capped px is structurally incapable of
/// registering as a sparkle peak). `delta<=0.0` disables the v9p cap
/// entirely: dark-hit px then fall through to the ordinary CURE4/CURE5
/// branches exactly as if `is_dark_hit` were false (v9n behavior byte-
/// identical, IRON LAW parameter — v9o's own evidence-passthrough dark
/// branch is fully replaced, never reachable again).
fn v9p_dark_cap_delta() -> f32 {
    env_f32("GAIA_V9P_DARK_CAP_DELTA", 0.10)
}

/// Per-channel 3x3-neighborhood MEDIAN of a demod-log evidence buffer
/// (`Step::evidence_dl` / the bar-res probe's own per-px demod-log'd
/// composite) — shared by settle()/bar_res_probe()/the training loop so the
/// 3 sites can never drift apart (task mandate). In-bounds neighbors only
/// (edges/corners median over fewer than 9 samples, matching
/// `winsorize_targets`'s own bounds-check convention elsewhere in this
/// file) — median, not mean, so a single-pixel firefly in the evidence
/// buffer itself cannot inflate the cap.
fn median3x3_dl(evidence_dl: &[[f32; OUTPUT_CHANNELS]], w: u32, h: u32) -> Vec<[f32; OUTPUT_CHANNELS]> {
    let (wi, hi) = (w as i32, h as i32);
    let mut out = vec![[0.0f32; OUTPUT_CHANNELS]; evidence_dl.len()];
    let mut vals: Vec<f32> = Vec::with_capacity(9);
    for y in 0..hi {
        for x in 0..wi {
            let px = (y * wi + x) as usize;
            for c in 0..OUTPUT_CHANNELS {
                vals.clear();
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (sx, sy) = (x + dx, y + dy);
                        if sx >= 0 && sx < wi && sy >= 0 && sy < hi {
                            vals.push(evidence_dl[(sy * wi + sx) as usize][c]);
                        }
                    }
                }
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[px][c] = vals[vals.len() / 2];
            }
        }
    }
    out
}

/// V9Q AIRTIGHT DESPECKLE CAP (module doc / task mandate): v9p forensics
/// (`scratch/v9p-barres-forensics.log`, commit 4e45211e) proved the v9p cap
/// ABOVE engages at every remaining bar-res sparkle texel but its own value
/// still exceeds the sparkle criterion: v9p compares in DEMOD-LOG space
/// against the EVIDENCE's 3x3 median, while `sparkle_resid_per_mpx`'s own
/// criterion (`SPARK_DELTA=0.15`) is a LINEAR local-peak against the
/// OUTPUT's own 3x3 neighborhood, evaluated AFTER the whole frame composes
/// (net-output px + no-hit evidence-passthrough px side by side) — a fixed
/// demod-log delta at bright radiance blows up past 0.15 linear once
/// exponentiated, and the evidence median itself sits high vs the net's
/// actual dark-neighbor renders. This cap instead operates in the
/// criterion's OWN domain: `GAIA_V9Q_DESPECKLE_DELTA` (default 0.10,
/// deliberately BELOW `SPARK_DELTA`=0.15, same headroom logic as v9p's own
/// delta) bounds a dark-class px's PRESENTED LINEAR radiance to at most the
/// MAX of its 3x3 neighbors in the composed frame (excluding self, that
/// same composed frame — no-hit evidence-passthrough + other px's own
/// presented value) plus the delta. `delta<=0.0` disables: dark-hit px then
/// fall through to the OLD v9p median+delta cap (if `GAIA_V9P_DARK_CAP_
/// DELTA>0`) or, if that is also disabled, to the ordinary evidence-ceiling
/// branch exactly as v9n (byte-identical, IRON LAW parameter) — v9q takes
/// PRIORITY over v9p at a dark-hit px whenever both deltas are positive
/// (the launch config disables v9p's own delta, but the parameter stays
/// live/reachable for A/B, never hardcoded away).
fn v9q_despeckle_delta() -> f32 {
    env_f32("GAIA_V9Q_DESPECKLE_DELTA", 0.10)
}

/// Shared despeckle-cap primitive (settle()/bar_res_probe()/the training
/// loop all call this so the 3 sites can never drift apart, task mandate):
/// given the FULL FRAME's pre-cap composed LINEAR radiance (`precap_lin`,
/// frozen — "no iteration", module doc) and one dark-class pixel's index,
/// returns `neigh_max_lin + delta` — the linear-domain cap value at that
/// pixel, taken over its up-to-8 in-bounds 3x3 neighbors, self excluded.
fn despeckle_cap_lin(precap_lin: &[GVec3], w: u32, h: u32, px: usize, delta: f32) -> GVec3 {
    let (wi, hi) = (w as i32, h as i32);
    let x = (px as i32) % wi;
    let y = (px as i32) / wi;
    let mut m = GVec3::ZERO;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (sx, sy) = (x + dx, y + dy);
            if sx >= 0 && sx < wi && sy >= 0 && sy < hi {
                m = m.max(precap_lin[(sy * wi + sx) as usize]);
            }
        }
    }
    m + GVec3::splat(delta)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

// ── V9R DEGENERATE-DEMOD FLAG CHANNEL (module doc / task mandate) ─────────
// v9o/v9p/v9q (above) all cure the dark-hit class by SURGERY on the loss or
// the composed output — the net's OWN input never learns the class exists
// (its 2x2 demod-log taps are RAW radiance at these px, scale-incompatible
// with every demodulated neighbor, but nothing in the feature vector marks
// that fact — v9n/v9o/v9p/v9q forensics + the sealed alignment audit ruled
// every teacher-anchored/loss-side cap ILLEGAL for the live path, which has
// no teacher). REPRESENTATIONAL cure instead: append ONE extra per-px input
// channel, `flag = 1.0 if (no-hit OR dark-albedo hit) else 0.0` — same class
// `is_dark_hit` already detects at all 3 monitor sites, just also handed to
// the net's own eyes so shared-weight highlight drive can learn to treat
// the population differently instead of inflating it. Env-gated
// `GAIA_V9R_DEGEN_FLAG=1`; default 0 keeps `v9r_input_channels()` ==
// `HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS` (the v9k/v9j/.. layout,
// byte-identical weight shapes) — an explicit disabled-by-default guard,
// not a threshold that merely never fires (IRON LAW).
fn v9r_degen_flag_on() -> bool {
    matches!(std::env::var("GAIA_V9R_DEGEN_FLAG").as_deref(), Ok("1"))
}
/// V9R optional loss weight at dark-hit px: multiplies whatever n2n grad
/// already reached `d_out` at those px (active-branch MSE term or the
/// overshoot/cap-substituted term — no new gradient shape, task mandate)
/// — they already carry honest K-draw n2n targets, this only reweights how
/// hard the net is pushed to match them. Default 1.0 = no-op (byte-
/// identical); independent of `GAIA_V9R_DEGEN_FLAG` (a pure loss-side dial,
/// like `overshoot_w`/`nohit_w` above), but only ever touches px where
/// `is_dark_hit` is true — itself gated off entirely when `GAIA_V9O_DARK_
/// ALB<=0.0` (v9n byte-identical class detection).
fn v9r_dark_w() -> f32 {
    env_f32("GAIA_V9R_DARK_W", 1.0)
}
/// Extra input channels this file's own feature builders append beyond the
/// shared `HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS` layout when V9R's
/// flag is on — an IRON channel-count constant of the FEATURE CONTRACT (one
/// flag scalar), never a hardcoded total dimension: every `Img::zeros`/
/// `UnetConfig{in_channels}` call site below reads `v9r_input_channels()`,
/// not a literal.
const V9R_FLAG_CHANNELS: usize = 1;
fn v9r_extra_channels() -> usize {
    if v9r_degen_flag_on() { V9R_FLAG_CHANNELS } else { 0 }
}
/// This file's own net input channel count — the shared v7/v9 layout plus
/// V9R's optional flag channel. Feeds BOTH the `Img::zeros` feature buffers
/// (build_input_img/build_input_img_crop/bar_res_probe) AND every
/// `UnetConfig.in_channels` constructed below, so the weight shapes and the
/// features fed to them can never drift apart (`UnetWeights::forward`'s own
/// `assert_eq!` on `(input.h, input.w, input.c)` vs `config` catches it
/// loudly if they ever did).
fn v9r_input_channels() -> usize {
    HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS + v9r_extra_channels()
}
/// Shared degenerate-demod flag value for one px — same class `is_dark_hit`
/// detects at the 3 monitor sites (settle/bar_res_probe/train loop), reused
/// here so the flag channel and the loss-side class detection can never
/// drift apart. `is_nohit`/`is_dark_hit` are the caller's own per-px booleans
/// (callers already compute them via `depth[px]<=0.0` / `v9o_dark_alb`).
fn v9r_flag_value(is_nohit: bool, is_dark_hit: bool) -> f32 {
    if is_nohit || is_dark_hit { 1.0 } else { 0.0 }
}

/// GAIA_V9_WIDTHS (CAPACITY ROUND, IRON): comma-separated per-scale conv
/// widths overriding `UnetConfig::default().widths` ([24,40,64]) — same
/// override pattern as `v9r_input_channels()` above (a plain fn feeding the
/// `UnetConfig{..}` struct-update literal, not a literal baked into the
/// graph builder). Unset ⇒ the parsed default equals `UnetConfig::default()`
/// exactly ⇒ byte-identical to every prior v9k..v9z run (IRON LAW).
/// `n_scales` is DERIVED from this list's length at every call site below —
/// never set independently, so the two fields the builder's own
/// `widths.len()==n_scales` assert checks (`rdirect_unet.rs` build()/
/// new_random()) can never drift apart from this fn's output.
fn v9_widths() -> Vec<usize> {
    match std::env::var("GAIA_V9_WIDTHS") {
        Ok(s) => {
            let widths: Vec<usize> = s
                .split(',')
                .map(|tok| {
                    tok.trim()
                        .parse::<usize>()
                        .unwrap_or_else(|e| panic!("GAIA_V9_WIDTHS: bad token {tok:?} in {s:?}: {e}"))
                })
                .collect();
            assert!(!widths.is_empty(), "GAIA_V9_WIDTHS: parsed to zero widths from {s:?}");
            widths
        }
        Err(_) => vec![24, 40, 64],
    }
}

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h as f32 }
}

fn rand_uniform(rng: &mut Rng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next() >> 11) as f64 / (1u64 << 53) as f64) as f32; // [0,1)
    lo + (hi - lo) * u
}

/// POSE DIVERSITY (v9c/v9e parity, unchanged) — one fresh draw from a
/// continuous pool around `anchor_eye`/`anchor_pivot`.
fn pool_camera(anchor_eye: [f32; 3], anchor_pivot: [f32; 3], fov_deg: f32, orbit_deg: f32, jitter: f32, rng: &mut Rng) -> (Camera, f32) {
    let yaw = rand_uniform(rng, -orbit_deg, orbit_deg);
    let oriented = orbit_camera(anchor_eye, anchor_pivot, yaw, fov_deg);
    let j = GVec3::new(
        rand_uniform(rng, -jitter, jitter),
        rand_uniform(rng, -jitter, jitter) * 0.4,
        rand_uniform(rng, -jitter, jitter),
    );
    let eye2 = (oriented.eye + j).to_array();
    (camera_at(eye2, anchor_pivot, fov_deg), yaw)
}

/// One step of a moving-camera pose sequence (v9c/v9e parity, unchanged).
#[derive(Clone)]
struct Step {
    low_e: Vec<GVec3>,
    low_d: Vec<GVec3>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    teacher: Vec<GVec3>,       // VALIDATOR ONLY (never in the loss)
    target_dl: Vec<[f32; OUTPUT_CHANNELS]>, // K-averaged noise2noise label
    ceiling_dl: Vec<[f32; OUTPUT_CHANNELS]>, // CURE 1: per-pixel evidence-clamp ceiling, demod-log space
    // V9N: this step's OWN E+D evidence composite (same taps `pixel_features_split`
    // feeds the net as input), demod-log space — the hit-gated compose's
    // (1-m) term, filled below alongside ceiling_dl (both derive from the
    // same `evidence_composite_frame`/`evidence_mean` machinery, unchanged).
    evidence_dl: Vec<[f32; OUTPUT_CHANNELS]>,
}

#[derive(Clone)]
struct PoseSeq {
    steps: Vec<Step>,
    cams: Vec<CamPose>,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
}

/// CURE 3, v9e's mechanism, KEPT but DEFAULT OFF this round (`winsor_k<=0.0`
/// disables — module doc: v9e proved it target-neutral, see evidence (a)).
/// `GAIA_V9F_WINSOR_K` re-enables it for A/B curiosity, not this round's bet.
fn winsorize_targets(target_dl: &mut [[f32; OUTPUT_CHANNELS]], w: u32, h: u32, winsor_k: f32) {
    if winsor_k <= 0.0 {
        return;
    }
    let (wi, hi) = (w as i32, h as i32);
    let r = 3i32; // 7x7
    let n = (w * h) as usize;
    for c in 0..OUTPUT_CHANNELS {
        let chan: Vec<f32> = (0..n).map(|px| target_dl[px][c]).collect();
        for y in 0..hi {
            for x in 0..wi {
                let mut sum = 0.0f64;
                let mut cnt = 0u32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (sx, sy) = (x + dx, y + dy);
                        if sx >= 0 && sx < wi && sy >= 0 && sy < hi {
                            sum += chan[(sy * wi + sx) as usize] as f64;
                            cnt += 1;
                        }
                    }
                }
                let local_mean = (sum / cnt.max(1) as f64) as f32;
                let ceiling = winsor_k * local_mean.max(1e-6);
                let px = (y * wi + x) as usize;
                if target_dl[px][c] > ceiling {
                    target_dl[px][c] = ceiling;
                }
            }
        }
    }
}

/// V9U CONVERGED-TEACHER accumulation (module doc / task mandate) — the
/// FULL-FRAME counterpart, byte-for-byte the same algorithm as
/// `rdirect_reference.rs::render_converged` (equal-size chunks of
/// `trace_headless_split` calls at `spp:1`, summed in LINEAR radiance
/// space, weighted by chunk size so the result is the exact overall mean
/// regardless of chunking — the last chunk may be short). Shared by
/// `render_pose_seq`'s target_dl/teacher (mirror/val/mirror_val, computed
/// ONCE before the epoch loop) and `bar_res_probe`'s teacher (computed
/// every `probe_every` epochs, not per-epoch) — never called per-epoch at
/// full-frame resolution, so its cost never rides the training loop.
#[allow(clippy::too_many_arguments)]
fn render_converged_lin(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    cam: &Camera,
    sun: &scrying_glass::scene::SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    w: u32,
    h: u32,
    total_spp: u32,
    chunk: u32,
) -> Vec<GVec3> {
    let n = (w * h) as usize;
    let mut sum = vec![GVec3::ZERO; n];
    let mut done = 0u32;
    while done < total_spp {
        let this_chunk = chunk.min(total_spp - done);
        let params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
        let (e, d) = trace_headless_split(device, queue, bvh, cam, sun, sky_top, sky_horizon, w, h, this_chunk, &params);
        for i in 0..n {
            sum[i] += (e[i] + d[i]) * (this_chunk as f32);
        }
        done += this_chunk;
    }
    let inv = 1.0 / (total_spp.max(1) as f32);
    sum.into_iter().map(|s| s * inv).collect()
}

/// `render_converged_lin`'s CROP counterpart (module doc / task mandate) —
/// same chunked-accumulation algorithm, `trace_headless_split_cropped` in
/// place of `trace_headless_split`. THIS is the one that rides the
/// per-epoch pool draws (the task's own "training sample unit") — called
/// at CROP resolution (`crop_w x crop_h`, same as `tw x th`, e.g. 128x72),
/// not full-frame, so its per-call cost is `crop_w*crop_h*total_spp`
/// integrator samples, unrelated to `full_w x full_h`.
#[allow(clippy::too_many_arguments)]
fn render_converged_lin_cropped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    cam: &Camera,
    sun: &scrying_glass::scene::SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    full_w: u32,
    full_h: u32,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
    total_spp: u32,
    chunk: u32,
) -> Vec<GVec3> {
    let n = (crop_w * crop_h) as usize;
    let mut sum = vec![GVec3::ZERO; n];
    let mut done = 0u32;
    while done < total_spp {
        let this_chunk = chunk.min(total_spp - done);
        let params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
        let (e, d) = trace_headless_split_cropped(
            device, queue, bvh, cam, sun, sky_top, sky_horizon,
            full_w, full_h, crop_x, crop_y, crop_w, crop_h, crop_w, crop_h, this_chunk, &params,
        );
        for i in 0..n {
            sum[i] += (e[i] + d[i]) * (this_chunk as f32);
        }
        done += this_chunk;
    }
    let inv = 1.0 / (total_spp.max(1) as f32);
    sum.into_iter().map(|s| s * inv).collect()
}

#[allow(clippy::too_many_arguments)]
fn render_pose_seq(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    base_cam: &Camera,
    k: u32,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
    ref_frames: u32,
    pan_step: f32,
    evidence_spp: u32,
    k_draws: u32,
    winsor_k: f32,
) -> PoseSeq {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    let mut steps = Vec::with_capacity(k as usize);
    let mut cams = Vec::with_capacity(k as usize);
    let n = (tw * th) as usize;
    for step in 0..k {
        let mut cam = *base_cam;
        cam.yaw += pan_step * step as f32;
        cams.push(cam_pose(&cam, tw, th));

        let np_a = IntegratorParams { spp: evidence_spp, seed: DRAW_A_SEED_BASE + step * 131 + 5, ..IntegratorParams::default() };
        let (low_e, low_d) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np_a,
        );
        let (albedo, normal, depth) = scrying_glass::integrator::split_aov(&trace_headless_aov(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
        ));

        // V9U (module doc / task mandate): GAIA_V9_TEACHER_SPP>0 replaces
        // BOTH the K-draw n2n target_dl (the loss's own target) AND this
        // seq's teacher field (validator-only — render_pose_seq feeds
        // mirror_seq/val_seq/mirror_val_seq, all rendered ONCE before the
        // epoch loop, so upgrading teacher here costs nothing per-epoch)
        // with a converged accumulation. Default 0 is the OLD path,
        // byte-identical (IRON LAW).
        let teacher_spp = env_teacher_spp();
        let mut target_dl: Vec<[f32; OUTPUT_CHANNELS]> = if teacher_spp > 0 {
            let converged = render_converged_lin(
                device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, teacher_spp, env_teacher_chunk(),
            );
            (0..n).map(|px| target_demod_log(converged[px], albedo[px])).collect()
        } else {
            let mut radiance_sum = vec![GVec3::ZERO; n];
            for kd in 0..k_draws {
                let np_b = IntegratorParams { spp: evidence_spp, seed: DRAW_B_SEED_BASE + step * 257 + 11 + kd * 9973, ..IntegratorParams::default() };
                let (e_b, d_b) = trace_headless_split(
                    device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, 1, &np_b,
                );
                for px in 0..n {
                    radiance_sum[px] += e_b[px] + d_b[px];
                }
            }
            let inv_k = 1.0 / (k_draws.max(1) as f32);
            (0..n).map(|px| target_demod_log(radiance_sum[px] * inv_k, albedo[px])).collect()
        };
        winsorize_targets(&mut target_dl, tw, th, winsor_k);

        let teacher: Vec<GVec3> = if teacher_spp > 0 {
            render_converged_lin(
                device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, teacher_spp, env_teacher_chunk(),
            )
        } else {
            let (e_full, d_full) = trace_headless_split(
                device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
                &IntegratorParams::default(),
            );
            (0..n).map(|i| e_full[i] + d_full[i]).collect()
        };

        steps.push(Step { low_e, low_d, albedo, normal, depth, teacher, target_dl, ceiling_dl: Vec::new(), evidence_dl: Vec::new() });
    }

    let gamma = evidence_clamp_gamma();
    let mut evidence_sum = vec![GVec3::ZERO; n];
    for s in steps.iter_mut() {
        let composite = evidence_composite_frame(&s.low_e, &s.low_d, low_w, low_h, tw, th);
        // V9N: this step's OWN composite, demod-log'd against this step's OWN
        // albedo — the exact per-px value the hit-gated compose substitutes
        // for net_out on no-hit pixels (same source/domain as the net's own
        // input taps and the loss's target_dl).
        s.evidence_dl = (0..n).map(|px| target_demod_log(composite[px], s.albedo[px])).collect();
        for (acc, c) in evidence_sum.iter_mut().zip(composite.iter()) {
            *acc += *c;
        }
    }
    let inv_steps = 1.0 / (steps.len().max(1) as f32);
    let evidence_mean: Vec<GVec3> = evidence_sum.iter().map(|&s| s * inv_steps).collect();
    let evidence_ceiling_lin = local_max_3x3(&evidence_mean, tw, th);
    for s in steps.iter_mut() {
        s.ceiling_dl = (0..n).map(|px| evidence_ceiling_demod_log(evidence_ceiling_lin[px], gamma, s.albedo[px])).collect();
    }

    PoseSeq { steps, cams, low_w, low_h, tw, th }
}

/// Reproject step `s-1`'s screen into step `s` (v8d/v9c/v9e parity, unchanged).
#[allow(clippy::too_many_arguments)]
fn reproject_prev(
    cur_cam: &CamPose,
    cur_depth: f32,
    cur_normal: GVec3,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    prev_cam: &CamPose,
    prev_out_dl: &[GVec3],
    prev_depth: &[f32],
    prev_normal: &[GVec3],
    pw: u32,
    ph: u32,
    sky_reject: bool,
) -> ([f32; 3], f32, [f32; 2]) {
    let is_miss = cur_depth <= 0.0;
    let dir = cur_cam.ray_dir(tx, ty, tw, th);
    let dist = if is_miss { 1.0e5 } else { cur_depth };
    let world = cur_cam.eye + dir * dist;
    match prev_cam.reproject(world, pw, ph) {
        None => ([0.0; 3], 0.0, [0.0; 2]),
        Some((fx, fy)) => {
            let mv = [fx - tx as f32, fy - ty as f32];
            let ipx = fx.round().clamp(0.0, (pw - 1) as f32) as usize;
            let ipy = fy.round().clamp(0.0, (ph - 1) as f32) as usize;
            let pj = ipy * pw as usize + ipx;
            let prev_d = prev_depth[pj];
            let prev_miss = prev_d <= 0.0;
            let ok = if is_miss {
                prev_miss && !sky_reject
            } else if prev_miss {
                false
            } else {
                let dist_prev = (world - prev_cam.eye).length();
                let depth_ok = (dist_prev - prev_d).abs() <= DEPTH_TOL * dist_prev.max(1e-4);
                let normal_ok = cur_normal.dot(prev_normal[pj]) >= NORMAL_THRESH;
                depth_ok && normal_ok
            };
            if ok {
                let s = scrying_glass::rdirect::bilinear_vec3(prev_out_dl, fx, fy, pw, ph);
                ([s.x, s.y, s.z], 1.0, mv)
            } else {
                ([0.0; 3], 0.0, [0.0; 2])
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_input_img(
    seq: &PoseSeq,
    step_idx: usize,
    prev_out: Option<&[GVec3]>,
    sky_reject: bool,
) -> Img {
    let (tw, th) = (seq.tw, seq.th);
    let step = &seq.steps[step_idx];
    let flag_on = v9r_degen_flag_on();
    let dark_alb = v9o_dark_alb();
    let input_clamp = v9v_input_clamp_delta();
    let mut img = Img::zeros(th as usize, tw as usize, v9r_input_channels());
    for ty in 0..th {
        for tx in 0..tw {
            let px = (ty * tw + tx) as usize;
            let mut base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                &step.low_e, &step.low_d, seq.low_w, seq.low_h, tw, th, tx, ty,
                step.albedo[px], step.normal[px], step.depth[px], Vec2::ZERO,
            );
            apply_input_clamp(&mut base, input_clamp);
            let (prev_dl, valid, mv) = match (step_idx, prev_out) {
                (0, _) | (_, None) => ([0.0f32; 3], 0.0f32, [0.0f32; 2]),
                (_, Some(prev)) => {
                    let prev_step = &seq.steps[step_idx - 1];
                    reproject_prev(
                        &seq.cams[step_idx], step.depth[px], step.normal[px], tx, ty, tw, th,
                        &seq.cams[step_idx - 1], prev, &prev_step.depth, &prev_step.normal, tw, th,
                        sky_reject,
                    )
                }
            };
            let feat = hist_features_split(&base, prev_dl, valid);
            for c in 0..HIST_FEATURES_SPLIT {
                img.set(ty as usize, tx as usize, c, feat[c]);
            }
            img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT, mv[0]);
            img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT + 1, mv[1]);
            if flag_on {
                let is_nohit = step.depth[px] <= 0.0;
                let is_dark_hit = !is_nohit && dark_alb > 0.0 && lum(step.albedo[px]) <= dark_alb;
                img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS, v9r_flag_value(is_nohit, is_dark_hit));
            }
        }
    }
    img
}

fn history_forward(ema: &UnetWeights, seq: &PoseSeq, sky_reject: bool) -> Vec<Vec<GVec3>> {
    let (tw, th) = (seq.tw as usize, seq.th as usize);
    let mut chain: Vec<Vec<GVec3>> = Vec::with_capacity(seq.steps.len());
    for s in 0..seq.steps.len() {
        let prev: Option<&[GVec3]> = if s == 0 { None } else { Some(&chain[s - 1]) };
        let input = build_input_img(seq, s, prev, sky_reject);
        let (out, _cache) = ema.forward(&input);
        let mut v = vec![GVec3::ZERO; tw * th];
        for y in 0..th {
            for x in 0..tw {
                v[y * tw + x] = GVec3::new(out.at(y, x, 0), out.at(y, x, 1), out.at(y, x, 2));
            }
        }
        chain.push(v);
    }
    chain
}

// ═══════════════════════════ V9K CROP MACHINERY ═══════════════════════════
// Everything above this line (Step/PoseSeq/render_pose_seq/reproject_prev/
// build_input_img/history_forward) is BYTE-IDENTICAL to v9j and stays the
// path for mirror_seq/val_seq/mirror_val_seq (module doc: "MIRROR/VAL POSES
// UNCHANGED"). Everything below is NEW, additive, used ONLY by the three
// per-epoch pool draws (module doc: "IMPLEMENTATION"/"CPU-SIDE HISTORY").

/// A crop's own camera pose for CPU-side ray-gen/reprojection — mirrors
/// `rdirect::CamPose::ray_dir`/`reproject` exactly, but folds in the SAME
/// affine NDC remap `IntegratorUniform::build_cropped` applies on the GPU
/// (module doc "CPU-SIDE HISTORY/REPROJECTION" for why this can't just be a
/// pre-warped `CamPose`). `full` stays the UNMODIFIED full-frame pose (its
/// right/up/forward are a genuine orthonormal basis — required for
/// `reproject`'s `rz`/ratio projection to be valid at all); `center`/`scale`
/// are the crop's affine remap of the `[-1,1]` NDC square, computed exactly
/// like `build_cropped`'s `center_x/y`/`scale_x/y` (same formulas, cited
/// there in full).
#[derive(Clone, Copy)]
struct SeqCam {
    full: CamPose,
    center: Vec2,
    scale: Vec2,
}

impl SeqCam {
    fn crop(full: CamPose, full_w: u32, full_h: u32, crop_x: u32, crop_y: u32, crop_w: u32, crop_h: u32) -> Self {
        let scale_x = crop_w as f32 / (full_w.max(1)) as f32;
        let scale_y = crop_h as f32 / (full_h.max(1)) as f32;
        let center_x = 2.0 * (crop_x as f32 + crop_w as f32 * 0.5) / (full_w.max(1)) as f32 - 1.0;
        let center_y = 1.0 - 2.0 * (crop_y as f32 + crop_h as f32 * 0.5) / (full_h.max(1)) as f32;
        SeqCam { full, center: Vec2::new(center_x, center_y), scale: Vec2::new(scale_x, scale_y) }
    }

    /// World-space ray direction through crop-local target pixel (tx,ty),
    /// crop dispatched at `w x h`. Same construction as `CamPose::ray_dir`,
    /// with `cx_crop`/`cy_crop` remapped through `center`/`scale` into the
    /// full frame's own NDC before applying the full frame's basis/half_tan/
    /// aspect — the exact inverse-of-inverse of `build_cropped`'s forward map.
    fn ray_dir(&self, tx: u32, ty: u32, w: u32, h: u32) -> GVec3 {
        let cx_crop = (2.0 * (tx as f32 + 0.5) / w as f32) - 1.0;
        let cy_crop = 1.0 - (2.0 * (ty as f32 + 0.5) / h as f32);
        let cx_full = self.center.x + self.scale.x * cx_crop;
        let cy_full = self.center.y + self.scale.y * cy_crop;
        (self.full.forward
            + self.full.right * cx_full * self.full.half_tan * self.full.aspect
            + self.full.up * cy_full * self.full.half_tan)
            .normalize_or_zero()
    }

    /// Reproject a world point into THIS (previous) crop's fractional screen
    /// pixel, `w x h` the crop's own dispatch size. `rz`/`cx_full`/`cy_full`
    /// are computed against the UNMODIFIED full-frame orthonormal basis
    /// (`self.full`, matching `CamPose::reproject` exactly — valid only
    /// there), THEN inverse-affine-mapped into crop-local NDC via
    /// `center`/`scale` before the final pixel conversion. `None` on a
    /// disocclusion OR when the reprojected point falls OUTSIDE this crop's
    /// own window (module doc "PADDING/EDGE HONESTY" — this is the honest,
    /// no-special-casing miss path, identical in spirit to `CamPose::
    /// reproject`'s own off-screen rejection).
    fn reproject(&self, world: GVec3, w: u32, h: u32) -> Option<(f32, f32)> {
        let rel = world - self.full.eye;
        let rz = rel.dot(self.full.forward);
        if rz <= 1e-4 {
            return None;
        }
        let cx_full = rel.dot(self.full.right) / (rz * self.full.half_tan * self.full.aspect);
        let cy_full = rel.dot(self.full.up) / (rz * self.full.half_tan);
        let cx_crop = (cx_full - self.center.x) / self.scale.x;
        let cy_crop = (cy_full - self.center.y) / self.scale.y;
        let mut fpx = (cx_crop + 1.0) * 0.5 * w as f32 - 0.5;
        let mut fpy = (1.0 - cy_crop) * 0.5 * h as f32 - 0.5;
        // Same SNAP_EPS pixel-boundary tie-break as `CamPose::reproject`
        // (rdirect.rs) — duplicated here identically (that module's SNAP_EPS
        // is private, not exported).
        const SNAP_EPS: f32 = 1.0e-3;
        let snap_x = (fpx + 0.5).floor();
        if (fpx - snap_x).abs() < SNAP_EPS {
            fpx = snap_x;
        }
        let snap_y = (fpy + 0.5).floor();
        if (fpy - snap_y).abs() < SNAP_EPS {
            fpy = snap_y;
        }
        if fpx < 0.0 || fpy < 0.0 || fpx > (w - 1) as f32 || fpy > (h - 1) as f32 {
            return None;
        }
        Some((fpx, fpy))
    }
}

/// `PoseSeq`'s crop counterpart — SAME `Step` type (resolution-shape-only,
/// camera-content-agnostic), `cams: Vec<SeqCam>` instead of `Vec<CamPose>`.
struct CroppedPoseSeq {
    steps: Vec<Step>,
    cams: Vec<SeqCam>,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
}

/// `render_pose_seq`'s crop counterpart: renders each of the `k` unroll
/// steps as a `crop_w x crop_h` (default 128x72) sub-rect of a virtual
/// `full_w x full_h` (default 640x480) frame at the SAME crop offset
/// `(crop_x, crop_y)` for every step (module doc — "K=8 n2n targets +
/// evidence + history all on the same crop"); only the camera itself yaws
/// per-step (`pan_step`, unchanged from v9j). Evidence/target/ceiling maths
/// (`evidence_composite_frame`/`local_max_3x3`/`evidence_ceiling_demod_log`/
/// `target_demod_log`) are UNCHANGED calls — they operate purely on
/// resolution-shaped arrays, agnostic to whether those pixels are a crop or
/// a whole frame.
#[allow(clippy::too_many_arguments)]
fn render_pose_seq_cropped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    base_cam: &Camera,
    k: u32,
    full_w: u32,
    full_h: u32,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
    low_w: u32,
    low_h: u32,
    ref_frames: u32,
    pan_step: f32,
    evidence_spp: u32,
    k_draws: u32,
    winsor_k: f32,
) -> CroppedPoseSeq {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    let mut steps = Vec::with_capacity(k as usize);
    let mut cams = Vec::with_capacity(k as usize);
    let n = (crop_w * crop_h) as usize;
    // V9V REAL MOTION SEQUENCES (task mandate, see helper doc above): with
    // GAIA_V9V_MOTION=1 the pool camera genuinely MOVES between the K unroll
    // steps (translation along its own turning look direction, at real-play
    // magnitudes) instead of yawing-in-place at v9j's tiny GAIA_V9_PANSTEP
    // (0.004rad) — this is what feeds the mv/history-reprojection channels a
    // real (non-~0) signal. Default 0 is byte-identical (IRON LAW): same
    // `cam.yaw += pan_step * step`, zero translation.
    let motion_on = v9v_motion_on();
    let cam_step = v9v_cam_step();
    let yaw_step = v9v_yaw_step();
    // V9V TARGET-AT-FINAL-POSE (task mandate): "target = converged render
    // (GAIA_V9_TEACHER_SPP path) at FINAL pose". With motion on, the net's
    // job is "predict what this real camera move CONVERGES to", not "predict
    // this step's own noisy frame" — every step's teacher_spp>0 target_dl
    // (below) renders at this fixed FINAL-step pose instead of its own
    // moving one. The teacher_spp==0 K-draws n2n fallback is UNTOUCHED (the
    // task scoped this to "the GAIA_V9_TEACHER_SPP path" only). Motion off
    // ⇒ `final_cam` == `base_cam`'s own per-step math collapses to `cam`
    // itself, so the `if motion_on` gate below picks `cam` anyway (IRON LAW:
    // byte-identical either way when GAIA_V9V_MOTION=0).
    let final_cam = if motion_on {
        let mut c = *base_cam;
        let last = k.saturating_sub(1) as f32;
        c.yaw += yaw_step * last;
        c.eye += c.direction() * (cam_step * last);
        c
    } else {
        *base_cam
    };
    for step in 0..k {
        let mut cam = *base_cam;
        if motion_on {
            cam.yaw += yaw_step * step as f32;
            cam.eye += cam.direction() * (cam_step * step as f32);
        } else {
            cam.yaw += pan_step * step as f32;
        }
        let full_pose = cam_pose(&cam, full_w, full_h);
        cams.push(SeqCam::crop(full_pose, full_w, full_h, crop_x, crop_y, crop_w, crop_h));

        let np_a = IntegratorParams { spp: evidence_spp, seed: DRAW_A_SEED_BASE + step * 131 + 5, ..IntegratorParams::default() };
        let (low_e, low_d) = trace_headless_split_cropped(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon,
            full_w, full_h, crop_x, crop_y, crop_w, crop_h, low_w, low_h, 1, &np_a,
        );
        let (albedo, normal, depth) = scrying_glass::integrator::split_aov(&trace_headless_aov_cropped(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon,
            full_w, full_h, crop_x, crop_y, crop_w, crop_h, crop_w, crop_h,
        ));

        // V9U (module doc / task mandate): GAIA_V9_TEACHER_SPP>0 replaces
        // the K-draw n2n target_dl — THE per-epoch pool draw's own loss
        // target, the task's own "training sample unit" — with a converged
        // CROP accumulation (`render_converged_lin_cropped`, same chunked
        // algorithm as `rdirect_reference.rs`). The `teacher` field below
        // is DELIBERATELY left on the OLD `ref_frames` path unconditionally
        // — pool draws' own `teacher` is validator-only and NEVER read by
        // the training loop (`TrainSeq::steps()` only reads target_dl/
        // ceiling_dl/evidence_dl/albedo/depth), so upgrading it here would
        // burn GPU time every epoch for a value nothing consumes (mind
        // epoch time, task mandate). Default 0 is the OLD path, byte-
        // identical (IRON LAW).
        let teacher_spp = env_teacher_spp();
        // V9V: see `final_cam` doc above — motion on picks the FIXED final
        // pose for the converged target render; motion off (or the K-draws
        // fallback just below, always) keeps this step's own `cam`.
        let target_render_cam: &Camera = if motion_on { &final_cam } else { &cam };
        let mut target_dl: Vec<[f32; OUTPUT_CHANNELS]> = if teacher_spp > 0 {
            let converged = render_converged_lin_cropped(
                device, queue, &bvh, target_render_cam, &scene.sun, scene.sky_top, scene.sky_horizon,
                full_w, full_h, crop_x, crop_y, crop_w, crop_h, teacher_spp, env_teacher_chunk(),
            );
            (0..n).map(|px| target_demod_log(converged[px], albedo[px])).collect()
        } else {
            let mut radiance_sum = vec![GVec3::ZERO; n];
            for kd in 0..k_draws {
                let np_b = IntegratorParams { spp: evidence_spp, seed: DRAW_B_SEED_BASE + step * 257 + 11 + kd * 9973, ..IntegratorParams::default() };
                let (e_b, d_b) = trace_headless_split_cropped(
                    device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon,
                    full_w, full_h, crop_x, crop_y, crop_w, crop_h, crop_w, crop_h, 1, &np_b,
                );
                for px in 0..n {
                    radiance_sum[px] += e_b[px] + d_b[px];
                }
            }
            let inv_k = 1.0 / (k_draws.max(1) as f32);
            (0..n).map(|px| target_demod_log(radiance_sum[px] * inv_k, albedo[px])).collect()
        };
        winsorize_targets(&mut target_dl, crop_w, crop_h, winsor_k);

        let (e_full, d_full) = trace_headless_split_cropped(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon,
            full_w, full_h, crop_x, crop_y, crop_w, crop_h, crop_w, crop_h, ref_frames,
            &IntegratorParams::default(),
        );
        let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();

        steps.push(Step { low_e, low_d, albedo, normal, depth, teacher, target_dl, ceiling_dl: Vec::new(), evidence_dl: Vec::new() });
    }

    let gamma = evidence_clamp_gamma();
    let mut evidence_sum = vec![GVec3::ZERO; n];
    for s in steps.iter_mut() {
        let composite = evidence_composite_frame(&s.low_e, &s.low_d, low_w, low_h, crop_w, crop_h);
        // V9N: see full-frame `render_pose_seq`'s identical comment.
        s.evidence_dl = (0..n).map(|px| target_demod_log(composite[px], s.albedo[px])).collect();
        for (acc, c) in evidence_sum.iter_mut().zip(composite.iter()) {
            *acc += *c;
        }
    }
    let inv_steps = 1.0 / (steps.len().max(1) as f32);
    let evidence_mean: Vec<GVec3> = evidence_sum.iter().map(|&s| s * inv_steps).collect();
    let evidence_ceiling_lin = local_max_3x3(&evidence_mean, crop_w, crop_h);
    for s in steps.iter_mut() {
        s.ceiling_dl = (0..n).map(|px| evidence_ceiling_demod_log(evidence_ceiling_lin[px], gamma, s.albedo[px])).collect();
    }

    CroppedPoseSeq { steps, cams, low_w, low_h, tw: crop_w, th: crop_h }
}

/// `reproject_prev`'s crop counterpart — identical control flow, `SeqCam`
/// instead of `CamPose`.
#[allow(clippy::too_many_arguments)]
fn reproject_prev_crop(
    cur_cam: &SeqCam,
    cur_depth: f32,
    cur_normal: GVec3,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    prev_cam: &SeqCam,
    prev_out_dl: &[GVec3],
    prev_depth: &[f32],
    prev_normal: &[GVec3],
    pw: u32,
    ph: u32,
    sky_reject: bool,
) -> ([f32; 3], f32, [f32; 2]) {
    let is_miss = cur_depth <= 0.0;
    let dir = cur_cam.ray_dir(tx, ty, tw, th);
    let dist = if is_miss { 1.0e5 } else { cur_depth };
    let world = cur_cam.full.eye + dir * dist;
    match prev_cam.reproject(world, pw, ph) {
        None => ([0.0; 3], 0.0, [0.0; 2]),
        Some((fx, fy)) => {
            let mv = [fx - tx as f32, fy - ty as f32];
            let ipx = fx.round().clamp(0.0, (pw - 1) as f32) as usize;
            let ipy = fy.round().clamp(0.0, (ph - 1) as f32) as usize;
            let pj = ipy * pw as usize + ipx;
            let prev_d = prev_depth[pj];
            let prev_miss = prev_d <= 0.0;
            let ok = if is_miss {
                prev_miss && !sky_reject
            } else if prev_miss {
                false
            } else {
                let dist_prev = (world - prev_cam.full.eye).length();
                let depth_ok = (dist_prev - prev_d).abs() <= DEPTH_TOL * dist_prev.max(1e-4);
                let normal_ok = cur_normal.dot(prev_normal[pj]) >= NORMAL_THRESH;
                depth_ok && normal_ok
            };
            if ok {
                let s = scrying_glass::rdirect::bilinear_vec3(prev_out_dl, fx, fy, pw, ph);
                ([s.x, s.y, s.z], 1.0, mv)
            } else {
                ([0.0; 3], 0.0, [0.0; 2])
            }
        }
    }
}

/// `build_input_img`'s crop counterpart.
#[allow(clippy::too_many_arguments)]
fn build_input_img_crop(
    seq: &CroppedPoseSeq,
    step_idx: usize,
    prev_out: Option<&[GVec3]>,
    sky_reject: bool,
) -> Img {
    let (tw, th) = (seq.tw, seq.th);
    let step = &seq.steps[step_idx];
    let flag_on = v9r_degen_flag_on();
    let dark_alb = v9o_dark_alb();
    let input_clamp = v9v_input_clamp_delta();
    let mut img = Img::zeros(th as usize, tw as usize, v9r_input_channels());
    for ty in 0..th {
        for tx in 0..tw {
            let px = (ty * tw + tx) as usize;
            let mut base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                &step.low_e, &step.low_d, seq.low_w, seq.low_h, tw, th, tx, ty,
                step.albedo[px], step.normal[px], step.depth[px], Vec2::ZERO,
            );
            apply_input_clamp(&mut base, input_clamp);
            let (prev_dl, valid, mv) = match (step_idx, prev_out) {
                (0, _) | (_, None) => ([0.0f32; 3], 0.0f32, [0.0f32; 2]),
                (_, Some(prev)) => {
                    let prev_step = &seq.steps[step_idx - 1];
                    reproject_prev_crop(
                        &seq.cams[step_idx], step.depth[px], step.normal[px], tx, ty, tw, th,
                        &seq.cams[step_idx - 1], prev, &prev_step.depth, &prev_step.normal, tw, th,
                        sky_reject,
                    )
                }
            };
            let feat = hist_features_split(&base, prev_dl, valid);
            for c in 0..HIST_FEATURES_SPLIT {
                img.set(ty as usize, tx as usize, c, feat[c]);
            }
            img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT, mv[0]);
            img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT + 1, mv[1]);
            if flag_on {
                let is_nohit = step.depth[px] <= 0.0;
                let is_dark_hit = !is_nohit && dark_alb > 0.0 && lum(step.albedo[px]) <= dark_alb;
                img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS, v9r_flag_value(is_nohit, is_dark_hit));
            }
        }
    }
    img
}

/// `history_forward`'s crop counterpart.
fn history_forward_crop(ema: &UnetWeights, seq: &CroppedPoseSeq, sky_reject: bool) -> Vec<Vec<GVec3>> {
    let (tw, th) = (seq.tw as usize, seq.th as usize);
    let mut chain: Vec<Vec<GVec3>> = Vec::with_capacity(seq.steps.len());
    for s in 0..seq.steps.len() {
        let prev: Option<&[GVec3]> = if s == 0 { None } else { Some(&chain[s - 1]) };
        let input = build_input_img_crop(seq, s, prev, sky_reject);
        let (out, _cache) = ema.forward(&input);
        let mut v = vec![GVec3::ZERO; tw * th];
        for y in 0..th {
            for x in 0..tw {
                v[y * tw + x] = GVec3::new(out.at(y, x, 0), out.at(y, x, 1), out.at(y, x, 2));
            }
        }
        chain.push(v);
    }
    chain
}

/// Unifies the 3 cropped pool draws + 1 uncropped mirror draw into ONE
/// per-epoch training batch without touching the loss/backward loop (which
/// only ever reads `.steps()[s]` — the shared `Step` type — and calls
/// `build_input`/`history_forward`; genuinely resolution-shape-only, not
/// camera-aware).
enum TrainSeq {
    Full(PoseSeq),
    Crop(CroppedPoseSeq),
}

impl TrainSeq {
    fn steps(&self) -> &[Step] {
        match self {
            TrainSeq::Full(p) => &p.steps,
            TrainSeq::Crop(p) => &p.steps,
        }
    }
    fn build_input(&self, step_idx: usize, prev: Option<&[GVec3]>, sky_reject: bool) -> Img {
        match self {
            TrainSeq::Full(p) => build_input_img(p, step_idx, prev, sky_reject),
            TrainSeq::Crop(p) => build_input_img_crop(p, step_idx, prev, sky_reject),
        }
    }
    fn history_forward(&self, ema: &UnetWeights, sky_reject: bool) -> Vec<Vec<GVec3>> {
        match self {
            TrainSeq::Full(p) => history_forward(ema, p, sky_reject),
            TrainSeq::Crop(p) => history_forward_crop(ema, p, sky_reject),
        }
    }
}

/// Crop offset picker (module doc IRON params): uniform over the valid
/// `[pad, full-crop-pad]` range on each axis, with an OPTIONAL bias toward
/// the upper third of the frame (`sky_bias_prob`, default 0.0 = OFF = plain
/// uniform) — a documented HEURISTIC (naruko's sky is generically toward
/// the top of frame; this is NOT a measured horizon line, just a coarse
/// prior), never applied unless explicitly enabled.
fn pick_crop_offset(rng: &mut Rng, full_w: u32, full_h: u32, crop_w: u32, crop_h: u32, pad: u32, sky_bias_prob: f32) -> (u32, u32) {
    let max_x = full_w.saturating_sub(crop_w + 2 * pad).max(1);
    let max_y_full = full_h.saturating_sub(crop_h + 2 * pad).max(1);
    let x = pad + rand_uniform(rng, 0.0, max_x as f32).clamp(0.0, max_x as f32) as u32;
    let biased = sky_bias_prob > 0.0 && rand_uniform(rng, 0.0, 1.0) < sky_bias_prob;
    let max_y = if biased { (max_y_full / 3).max(1) } else { max_y_full };
    let y = pad + rand_uniform(rng, 0.0, max_y as f32).clamp(0.0, max_y as f32) as u32;
    (x, y)
}

// ═════════════════════════ END V9K CROP MACHINERY ══════════════════════════

fn img_to_vec3(img: &Img) -> Vec<GVec3> {
    let mut v = vec![GVec3::ZERO; img.h * img.w];
    for y in 0..img.h {
        for x in 0..img.w {
            v[y * img.w + x] = GVec3::new(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
        }
    }
    v
}

fn target_to_img(target: &[[f32; OUTPUT_CHANNELS]], h: usize, w: usize) -> Img {
    let mut img = Img::zeros(h, w, OUTPUT_CHANNELS);
    for y in 0..h {
        for x in 0..w {
            let px = y * w + x;
            for c in 0..OUTPUT_CHANNELS {
                img.set(y, x, c, target[px][c]);
            }
        }
    }
    img
}

fn rmse_lin(net: &[GVec3], teacher: &[GVec3]) -> f64 {
    let mut s = 0.0f64;
    for (a, b) in net.iter().zip(teacher) {
        let d = *a - *b;
        s += (d.x * d.x + d.y * d.y + d.z * d.z) as f64;
    }
    (s / (net.len() as f64 * 3.0)).sqrt()
}
fn sparkle_resid_per_mpx(net: &[GVec3], teacher: &[GVec3], w: u32, h: u32) -> f64 {
    const SPARK_DELTA: f32 = 0.15;
    let idx = |x: i32, y: i32| (y as usize) * w as usize + x as usize;
    let err = |x: i32, y: i32| lum(net[idx(x, y)]) - lum(teacher[idx(x, y)]);
    let mut count = 0u64;
    for y in 1..h as i32 - 1 {
        for x in 1..w as i32 - 1 {
            let e = err(x, y);
            if e <= SPARK_DELTA {
                continue;
            }
            let mut is_peak = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx != 0 || dy != 0) && err(x + dx, y + dy) >= e {
                        is_peak = false;
                    }
                }
            }
            if is_peak {
                count += 1;
            }
        }
    }
    (count as f64) * 1.0e6 / (w as f64 * h as f64)
}
fn highlight_ratio(net: &[GVec3], teacher: &[GVec3], pctl: f32) -> f64 {
    let mut order: Vec<usize> = (0..teacher.len()).collect();
    order.sort_by(|&a, &b| lum(teacher[b]).partial_cmp(&lum(teacher[a])).unwrap());
    let n = ((teacher.len() as f32 * pctl).ceil() as usize).max(1).min(teacher.len());
    let mut net_sum = 0.0f64;
    let mut teacher_sum = 0.0f64;
    for &i in &order[..n] {
        net_sum += lum(net[i]) as f64;
        teacher_sum += lum(teacher[i]) as f64;
    }
    if teacher_sum > 1e-9 { net_sum / teacher_sum } else { 1.0 }
}

fn settle(net: &UnetWeights, seq: &PoseSeq, sky_reject: bool) -> (Vec<GVec3>, Vec<GVec3>) {
    let mut prev: Option<Vec<GVec3>> = None;
    let mut last_dl = Vec::new();
    for s in 0..seq.steps.len() {
        let input = build_input_img(seq, s, prev.as_deref(), sky_reject);
        let (out, _cache) = net.forward(&input);
        last_dl = img_to_vec3(&out);
        prev = Some(last_dl.clone());
    }
    let last_step = seq.steps.last().unwrap();
    // V9N no-hit gate (unchanged): monitors MUST evaluate the SAME composed
    // output the training loss uses (task mandate) — gate applies here at
    // the demod-log stage, before `undo_log_demod`, exactly like the
    // training loop composes in `target_dl` domain. gate_on false leaves
    // `composed[px]` == `last_dl[px]` exactly (byte-identical old path).
    // V9Q (replaces v9p's demod-log median cap): at dark-hit px (real hit,
    // degenerate demod, see `v9o_dark_alb`) the net's own output is KEPT
    // uncapped in the PRE-cap compose below — the cap itself is applied
    // POST-COMPOSE, in the sparkle criterion's own LINEAR domain, against
    // the composed frame's own 3x3 neighborhood (module doc `despeckle_cap_
    // lin`). Falls back to the OLD v9p median+delta cap when v9q's own
    // delta is disabled, then to the ordinary evidence ceiling (v9n).
    let gate_on = hitgate_on();
    let dark_alb = v9o_dark_alb();
    let dark_cap_delta = v9p_dark_cap_delta();
    let despeckle_delta = v9q_despeckle_delta();
    let median_dl = median3x3_dl(&last_step.evidence_dl, seq.tw, seq.th);
    let n = last_dl.len();
    let mut precap_dl: Vec<GVec3> = Vec::with_capacity(n);
    let mut dark_v9q = vec![false; n];
    for (px, dl) in last_dl.iter().enumerate() {
        let is_nohit = last_step.depth[px] <= 0.0;
        if gate_on && is_nohit {
            let e = last_step.evidence_dl[px];
            precap_dl.push(GVec3::new(e[0], e[1], e[2]));
            continue;
        }
        let is_dark_hit = !is_nohit && dark_alb > 0.0 && lum(last_step.albedo[px]) <= dark_alb;
        let use_v9q = gate_on && is_dark_hit && despeckle_delta > 0.0;
        if use_v9q {
            dark_v9q[px] = true;
            precap_dl.push(*dl);
            continue;
        }
        if gate_on && is_dark_hit && dark_cap_delta > 0.0 {
            let m = median_dl[px];
            precap_dl.push(GVec3::new(
                dl.x.min(m[0] + dark_cap_delta),
                dl.y.min(m[1] + dark_cap_delta),
                dl.z.min(m[2] + dark_cap_delta),
            ));
            continue;
        }
        precap_dl.push(*dl);
    }
    let precap_lin: Vec<GVec3> = precap_dl
        .iter()
        .zip(last_step.albedo.iter())
        .map(|(dl, albedo)| undo_log_demod(*dl, demod_divisor(*albedo)))
        .collect();
    let net_lin: Vec<GVec3> = (0..n)
        .map(|px| {
            if dark_v9q[px] {
                let cap_lin = despeckle_cap_lin(&precap_lin, seq.tw, seq.th, px, despeckle_delta);
                precap_lin[px].min(cap_lin)
            } else {
                precap_lin[px]
            }
        })
        .collect();
    (net_lin, last_step.teacher.clone())
}

/// TWO-SIGNAL WATCHDOG, signal A (module doc change 1): mirrors the OLD
/// score-streak logic exactly, on `metric` (resid, not score) — reset on
/// improvement-or-within-5%-of-best-ever, increment otherwise; caller
/// decides what to do with the returned streak. Shared by both the
/// score-log-only counter and the resid-abort counter (same shape, two
/// independent trackers).
fn update_streak(metric: f64, best_ever: &mut f64, streak: &mut u32) {
    if metric < *best_ever {
        *best_ever = metric;
        *streak = 0;
    } else if metric > *best_ever * 1.05 {
        *streak += 1;
    } else {
        *streak = 0;
    }
}

#[allow(clippy::too_many_arguments)]
/// v9e's `run_monitor`, PLUS module doc change 1: `score_streak` is now
/// LOG-ONLY (no abort effect — the field is still tracked/printed exactly
/// as before so log format stays legible against v9c/v9d/v9e); a NEW
/// independent `resid_best`/`resid_streak` pair (same `update_streak` shape)
/// is the ONLY in-monitor abort signal now, returned as the 4th field.
fn run_monitor(
    run_tag: &str,
    tag: &str,
    ema: &UnetWeights,
    val_seq: &PoseSeq,
    mirror_seq: &PoseSeq,
    sky_reject: bool,
    highlight_pctl: f32,
    spark_target: f32,
    resid_gate: f32,
    best_score: &mut f64,
    best_bytes: &mut Vec<u8>,
    wpath_best: &Path,
    score_streak: &mut u32,
    score_streak_param: u32,
    resid_best: &mut f64,
    resid_streak: &mut u32,
    resid_streak_param: u32,
    // V9Z: gates the *BEST->saved file write (and its log marker) below.
    // true (score metric, default) is byte-identical to v9k..v9y — every
    // caller passes true unless GAIA_V9_BEST_METRIC=bar, which passes
    // false so this scorer's OWN 128x72 pick never overwrites the file the
    // bar-res-probe selection (main loop) is maintaining; best_score/
    // score_streak bookkeeping still runs either way (LOG-ONLY).
    save_to_disk: bool,
) -> (f64, f64, f64, bool) {
    let (net, teacher) = settle(ema, val_seq, sky_reject);
    let sp = sparkle_resid_per_mpx(&net, &teacher, val_seq.tw, val_seq.th);
    let rs = rmse_lin(&net, &teacher);
    let hl = highlight_ratio(&net, &teacher, highlight_pctl);
    let (mnet, mteacher) = settle(ema, mirror_seq, sky_reject);
    let msp = sparkle_resid_per_mpx(&mnet, &mteacher, mirror_seq.tw, mirror_seq.th);
    let mrs = rmse_lin(&mnet, &mteacher);
    let mhl = highlight_ratio(&mnet, &mteacher, highlight_pctl);
    let passes = sp < spark_target as f64 && rs < resid_gate as f64;
    let score = (sp / 40.0).max(rs / 0.035);

    let better = score < *best_score;
    if better {
        *best_score = score;
        if save_to_disk {
            *best_bytes = serialize_weights(ema);
            std::fs::write(wpath_best, &*best_bytes).unwrap();
        }
    }
    update_streak(score, best_score, score_streak);
    // score_streak resets to 0 on `better` too via update_streak's own
    // `metric < best_ever` branch — consistent with the OLD sole-signal
    // semantics, just no longer wired to abort.

    update_streak(rs, resid_best, resid_streak);
    let resid_should_abort = *resid_streak >= resid_streak_param;

    eprintln!(
        "[{run_tag}] MONITOR {tag}: val sparkle {sp:.1}/Mpx resid {rs:.4} highlight_ratio {hl:.3} score={score:.3}{} | mirror sparkle {msp:.1}/Mpx resid {mrs:.4} highlight_ratio {mhl:.3} (tgt sp<{spark_target} resid<{resid_gate}){} score_streak={score_streak}/{score_streak_param}(LOG-ONLY) resid_streak={resid_streak}/{resid_streak_param}(ABORT)",
        if passes { " PASS" } else { "" }, if better && save_to_disk { " *BEST->saved" } else { "" },
    );
    std::io::stderr().flush().ok();
    (sp, rs, hl, resid_should_abort)
}

/// MODULE DOC CHANGE 3 — periodic bar-res probe: re-derives
/// `rdirect_v9_eval_640.rs`'s own measurement (held-out `orbit_-20` pose,
/// undo-log-demod + v7 structural evidence-clamp-at-inference) on the
/// CURRENT `ema` weights at `tw`x`th` (native "bar" resolution, default
/// 640x480 via `GAIA_V9_EVAL_W/H`) — same K-step recurrent settle +
/// converged `ref_frames`-sample teacher, same inference-only clamp maths
/// (`evidence_composite_frame`/`local_max_3x3`/`evidence_ceiling_demod_log`).
/// Config override trick matches `rdirect_v9_eval_640.rs`: the net is
/// fully-convolutional, only `forward`'s own dim asserts care about
/// `config.render_w/h`/`output_w/h`, so a CLONE of `ema` gets its config
/// bumped to the probe resolution — the trainer's own `ema` (and its
/// low-res config) is never touched.
#[allow(clippy::too_many_arguments)]
fn bar_res_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    base_cam: &Camera,
    ema: &UnetWeights,
    sky_reject: bool,
    tw: u32,
    th: u32,
    k: u32,
    ref_frames: u32,
    highlight_pctl: f32,
) -> (f64, f64, f64) {
    let low_w = tw / 2;
    let low_h = th / 2;
    let n = (tw * th) as usize;
    let gamma = evidence_clamp_gamma();
    let bvh = Bvh::build(base_tris, &BvhParams::default());

    let mut steps_low_e = Vec::with_capacity(k as usize);
    let mut steps_low_d = Vec::with_capacity(k as usize);
    let mut steps_albedo = Vec::with_capacity(k as usize);
    let mut steps_normal = Vec::with_capacity(k as usize);
    let mut steps_depth = Vec::with_capacity(k as usize);
    let mut cams = Vec::with_capacity(k as usize);
    for step in 0..k {
        let cam = *base_cam;
        cams.push(cam_pose(&cam, tw, th));
        let np_a = IntegratorParams { spp: 1, seed: DRAW_A_SEED_BASE + step * 131 + 5, ..IntegratorParams::default() };
        let (low_e, low_d) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np_a,
        );
        let (albedo, normal, depth) = scrying_glass::integrator::split_aov(&trace_headless_aov(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
        ));
        steps_low_e.push(low_e);
        steps_low_d.push(low_d);
        steps_albedo.push(albedo);
        steps_normal.push(normal);
        steps_depth.push(depth);
    }
    // V9U (module doc / task mandate): GAIA_V9_TEACHER_SPP>0 makes the
    // bar-res judge's OWN teacher a true converged render too ("the
    // sparkle criterion finally measures vs truth") — same chunked
    // accumulation as render_pose_seq's teacher above. Runs only every
    // `probe_every` epochs (default 25), never per-epoch, so the extra
    // cost is amortized. Default 0 is the OLD `ref_frames` path, byte-
    // identical (IRON LAW).
    let teacher_spp = env_teacher_spp();
    let teacher: Vec<GVec3> = if teacher_spp > 0 {
        render_converged_lin(
            device, queue, &bvh, base_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, teacher_spp, env_teacher_chunk(),
        )
    } else {
        let (e_full, d_full) = trace_headless_split(
            device, queue, &bvh, base_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
            &IntegratorParams::default(),
        );
        (0..n).map(|i| e_full[i] + d_full[i]).collect()
    };

    let mut probe_net = ema.clone();
    probe_net.config.render_w = tw as usize;
    probe_net.config.render_h = th as usize;
    probe_net.config.output_w = tw as usize;
    probe_net.config.output_h = th as usize;

    let mut chain_dl: Vec<Vec<GVec3>> = Vec::with_capacity(k as usize);
    let mut out_imgs: Vec<Img> = Vec::with_capacity(k as usize);
    let flag_on = v9r_degen_flag_on();
    let dark_alb_probe = v9o_dark_alb();
    let input_clamp = v9v_input_clamp_delta();
    for s in 0..k as usize {
        let mut img = Img::zeros(th as usize, tw as usize, v9r_input_channels());
        for ty in 0..th {
            for tx in 0..tw {
                let px = (ty * tw + tx) as usize;
                let mut base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                    &steps_low_e[s], &steps_low_d[s], low_w, low_h, tw, th, tx, ty,
                    steps_albedo[s][px], steps_normal[s][px], steps_depth[s][px], Vec2::ZERO,
                );
                apply_input_clamp(&mut base, input_clamp);
                let (prev_dl, valid, mv) = if s == 0 {
                    ([0.0f32; 3], 0.0f32, [0.0f32; 2])
                } else {
                    reproject_prev(
                        &cams[s], steps_depth[s][px], steps_normal[s][px], tx, ty, tw, th,
                        &cams[s - 1], &chain_dl[s - 1], &steps_depth[s - 1], &steps_normal[s - 1], tw, th, sky_reject,
                    )
                };
                let feat = hist_features_split(&base, prev_dl, valid);
                for c in 0..HIST_FEATURES_SPLIT {
                    img.set(ty as usize, tx as usize, c, feat[c]);
                }
                img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT, mv[0]);
                img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT + 1, mv[1]);
                if flag_on {
                    let is_nohit = steps_depth[s][px] <= 0.0;
                    let is_dark_hit = !is_nohit && dark_alb_probe > 0.0 && lum(steps_albedo[s][px]) <= dark_alb_probe;
                    img.set(ty as usize, tx as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS, v9r_flag_value(is_nohit, is_dark_hit));
                }
            }
        }
        let (out, _cache) = probe_net.forward(&img);
        chain_dl.push(img_to_vec3(&out));
        out_imgs.push(out);
    }

    let mut evidence_sum = vec![GVec3::ZERO; n];
    for s in 0..k as usize {
        let composite = evidence_composite_frame(&steps_low_e[s], &steps_low_d[s], low_w, low_h, tw, th);
        for (acc, c) in evidence_sum.iter_mut().zip(composite.iter()) {
            *acc += *c;
        }
    }
    let inv_k = 1.0 / (k.max(1) as f32);
    let evidence_mean: Vec<GVec3> = evidence_sum.iter().map(|&s| s * inv_k).collect();
    let evidence_ceiling = local_max_3x3(&evidence_mean, tw, th);

    let last_dl = &out_imgs[k as usize - 1];
    let last_albedo = &steps_albedo[k as usize - 1];
    let last_depth = &steps_depth[k as usize - 1];
    // V9N no-hit gate (unchanged): the probe must evaluate the SAME
    // composed output the training loss/settle() use (task mandate) —
    // this step's OWN evidence composite (not the temporal `evidence_mean`
    // the ceiling uses), demod-log'd against this step's own albedo, same
    // as the trainer's `Step::evidence_dl`. gate_on false leaves every px
    // on the existing clamp-only path (byte-identical old probe). V9Q
    // (replaces v9p's demod-log median cap): at dark-hit px the net's own
    // output is KEPT uncapped in the PRE-cap compose, then POST-COMPOSE
    // capped in LINEAR space against the composed frame's own 3x3
    // neighborhood (`despeckle_cap_lin`, same as settle()/the training
    // loop) — falls back to the OLD v9p cap, then the ordinary ceiling.
    let gate_on = hitgate_on();
    let dark_alb = v9o_dark_alb();
    let dark_cap_delta = v9p_dark_cap_delta();
    let despeckle_delta = v9q_despeckle_delta();
    let last_composite = evidence_composite_frame(&steps_low_e[k as usize - 1], &steps_low_d[k as usize - 1], low_w, low_h, tw, th);
    let last_evidence_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n)
        .map(|px| target_demod_log(last_composite[px], last_albedo[px]))
        .collect();
    let median_dl = median3x3_dl(&last_evidence_dl, tw, th);
    let mut precap_lin = vec![GVec3::ZERO; n];
    let mut dark_v9q = vec![false; n];
    for px in 0..n {
        let dl = GVec3::new(last_dl.data[px * 3], last_dl.data[px * 3 + 1], last_dl.data[px * 3 + 2]);
        let divisor = demod_divisor(last_albedo[px]);
        let is_nohit = last_depth[px] <= 0.0;
        if gate_on && is_nohit {
            let e = last_evidence_dl[px];
            precap_lin[px] = undo_log_demod(GVec3::new(e[0], e[1], e[2]), divisor);
            continue;
        }
        let is_dark_hit = !is_nohit && dark_alb > 0.0 && lum(last_albedo[px]) <= dark_alb;
        let use_v9q = gate_on && is_dark_hit && despeckle_delta > 0.0;
        if use_v9q {
            dark_v9q[px] = true;
            precap_lin[px] = undo_log_demod(dl, divisor);
            continue;
        }
        if gate_on && is_dark_hit && dark_cap_delta > 0.0 {
            let m = median_dl[px];
            let capped = GVec3::new(dl.x.min(m[0] + dark_cap_delta), dl.y.min(m[1] + dark_cap_delta), dl.z.min(m[2] + dark_cap_delta));
            precap_lin[px] = undo_log_demod(capped, divisor);
            continue;
        }
        let ceiling_dl: [f32; OUTPUT_CHANNELS] = evidence_ceiling_demod_log(evidence_ceiling[px], gamma, last_albedo[px]);
        let presented_dl = GVec3::new(dl.x.min(ceiling_dl[0]), dl.y.min(ceiling_dl[1]), dl.z.min(ceiling_dl[2]));
        precap_lin[px] = undo_log_demod(presented_dl, divisor);
    }
    let mut net_clamped = vec![GVec3::ZERO; n];
    for px in 0..n {
        if dark_v9q[px] {
            let cap_lin = despeckle_cap_lin(&precap_lin, tw, th, px, despeckle_delta);
            net_clamped[px] = precap_lin[px].min(cap_lin);
        } else {
            net_clamped[px] = precap_lin[px];
        }
    }

    let sp = sparkle_resid_per_mpx(&net_clamped, &teacher, tw, th);
    let rs = rmse_lin(&net_clamped, &teacher);
    let hl = highlight_ratio(&net_clamped, &teacher, highlight_pctl);
    (sp, rs, hl)
}

fn main() {
    let run_tag = std::env::var("GAIA_V9_TAG").unwrap_or_else(|_| "v9k".to_string());
    // V9K BAR-RES CROP TRAINING (module doc): the pool draws' virtual full
    // frame + crop window. Crop size IS the training resolution `tw x th`
    // below (`GAIA_V9_W`/`GAIA_V9_H`, default 128x72 — the task's own IRON
    // default, satisfied for free since that's already this file's net
    // input/output resolution — a crop MUST equal it, the UnetConfig is
    // fixed-size). `full_w`/`full_h` default 640x480 (the bar-res judge's
    // own resolution — the whole point) are the only genuinely NEW knobs.
    let crop_full_w = env_u32("GAIA_V9K_FULL_W", 640);
    let crop_full_h = env_u32("GAIA_V9K_FULL_H", 480);
    let crop_pad = env_u32("GAIA_V9K_PAD_MARGIN", 0);
    let crop_sky_bias_prob = env_f32("GAIA_V9K_SKY_BIAS_PROB", 0.0); // default OFF — heuristic, see pick_crop_offset doc
    let score_streak_param = env_u32("GAIA_V9F_SCORE_STREAK", 20); // LOG-ONLY now (module doc change 1)
    let resid_streak_param = env_u32("GAIA_V9F_RESID_STREAK", 20); // the abort signal A
    let probe_every = env_u32("GAIA_V9F_PROBE_EVERY", 25); // module doc change 3, 0 disables
    let pool_orbit_deg = env_f32("GAIA_V9F_POOL_ORBIT_DEG", 30.0); // v9c/v9e parity
    let pool_jitter = env_f32("GAIA_V9F_POOL_JITTER", 0.6); // v9c/v9e parity
    let winsor_k = env_f32("GAIA_V9F_WINSOR_K", 0.0); // CURE 3 default OFF this round (evidence (a))
    // V9I DOSE RESET (autopsy v9h grad-probe-fix section): CURE4's overshoot
    // term is no longer the war-winning knob now that CURE 5 (below) gives
    // the mute pixel class its own local restoring gradient — default back
    // to 1.0 (v9g's original dose), param stays for A/B.
    let overshoot_w = env_f32("GAIA_V9I_OVERSHOOT_W", 1.0);
    // CURE 5 (V9I, this round): at albedo≈0/no-hit pixels (the class
    // convicted in the autopsy's grad-probe-fix section — evidence ceiling
    // there is systematically INFLATED, mean 1.64x/up to 2.94x the honest
    // 256-draw target, via bilinear-upsample + 3x3-maxpool bleed from
    // neighboring bright surface pixels at sky/silhouette boundaries),
    // bypass the clamp+overshoot mechanism ENTIRELY and use ordinary,
    // unconditionally-active two-sided MSE against the honest n2n target.
    // Mask threshold matches `rdirect.rs::NO_HIT_ALBEDO_THRESHOLD_SQ` (the
    // same no-hit branch demod_divisor already special-cases) by default.
    let nohit_albedo_sq = env_f32("GAIA_V9I_NOHIT_ALBEDO_SQ", 1.0e-8);
    // V9M PROMOTION (task mandate): CURE 5's honest-target term was a full-
    // formula bypass but its weight vs. the primary n2n loss was an
    // unexposed hardcoded 1.0 (IRON LAW violation — parameter, never
    // hardcode). `GAIA_V9M_NOHIT_W` makes that weight an explicit dial:
    // the no-hit class (15-33% of px, `epoch_nohit_px` below) is supervised
    // at `nohit_w` against its honest K=8 n2n target, `2*nohit_w*(raw-
    // target)/n` — default 1.0 keeps v9k/v9j's existing full-weight
    // behavior byte-identical (no domain change needed: the target already
    // lives in the demod-log `target_dl` domain CURE 5 uses, and per
    // forensics the demod divisor is 1.0 exactly at zero-albedo, i.e. the
    // demod-log and plain domains coincide there — degenerate but honest).
    // `nohit_w<=0.0` reaches the OLD tiny-side-term behavior (pre-v9i): the
    // no-hit class falls through to the ordinary CURE4 active/overshoot
    // branches instead of CURE 5's bypass (param 0 = CURE 5 off).
    let nohit_w = env_f32("GAIA_V9M_NOHIT_W", 1.0);
    let sky_reject = sky_history_reject();
    eprintln!("[{run_tag}] GAIA_V7_SKY_HISTORY reject={sky_reject} — mandate expects true (set GAIA_V7_SKY_HISTORY=reject)");

    let Some((device, queue)) = headless_device() else {
        panic!("[{run_tag}] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let tw = env_u32("GAIA_V9_W", 128);
    let th = env_u32("GAIA_V9_H", 72);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V9_STILL", 3);
    let ref_frames = env_u32("GAIA_V9_REF", 64);
    let epochs = env_u32("GAIA_V9_EPOCHS", 300);
    let lr0 = env_f32("GAIA_V9_LR", 1.0e-4);
    let ema_decay = env_f32("GAIA_V9_EMA", 0.999);
    let pan_step = env_f32("GAIA_V9_PANSTEP", 0.004);
    let mirror_spp = env_u32("GAIA_V9_MIRROR_SPP", 4);
    let highlight_pctl = env_f32("GAIA_V9_HIGHLIGHT_PCTL", 0.05);
    let spark_target = env_f32("GAIA_V9_SPARK_TGT", 16.0);
    // v9t: GAIA_V9_SPARKLE_ABORT — default 1 keeps the bar-res probe's
    // sparkle-breach ABORT byte-identical to v9k..v9r/s (IRON LAW); 0 makes
    // the breach OBSERVATIONAL ONLY (logs loudly, training continues) so the
    // full 300-ep curve can be watched post-ignition. resid_streak abort and
    // every other watchdog signal stay ACTIVE regardless — this dial touches
    // ONLY the bar-res probe sparkle>=spark_target abort branch below.
    let sparkle_abort_on = !matches!(std::env::var("GAIA_V9_SPARKLE_ABORT").as_deref(), Ok("0"));
    let resid_gate = env_f32("GAIA_V9_RESID_GATE", 0.035);
    let monitor_every = env_u32("GAIA_V9_MONITOR", 1);
    let wall_budget = env_f32("GAIA_V9_WALL", 10_800.0);
    let k_draws = env_u32("GAIA_V9_K", 8);
    let seed = env_u32("GAIA_V9_SEED", 0x5eed_c0de) as u64;
    // module doc change 3 — bar-res probe params, GAIA_V9_EVAL_* per task
    // instruction (reuses rdirect_v9_eval_640.rs's own env var names).
    let eval_w = env_u32("GAIA_V9_EVAL_W", 640);
    let eval_h = env_u32("GAIA_V9_EVAL_H", 480);
    let eval_k = env_u32("GAIA_V9_EVAL_K", 3);
    let eval_ref = env_u32("GAIA_V9_EVAL_REF", 64);
    // V9Z CHECKPOINT-CADENCE FIX (task mandate, sweet-zone recovery): v9y's
    // ep~300 sweet-zone body (bar-res sparkle 13.0, resid 0.0465, drive
    // 0.810) was LOST because (a) the *BEST->saved scorer below uses the
    // small-res val monitor, which flared on a 325/Mpx artifact and stopped
    // saving at ep60, and (b) no periodic checkpoints existed independent of
    // that scorer. Both knobs default to the OLD behavior byte-identical
    // (IRON LAW): ckpt_every=0 writes nothing extra; best_metric='score' is
    // the untouched v9k..v9y run_monitor scorer.
    let ckpt_every = env_u32("GAIA_V9_CKPT_EVERY", 0); // 0 (default) disables periodic checkpoints
    let best_metric_raw = std::env::var("GAIA_V9_BEST_METRIC").unwrap_or_else(|_| "score".to_string());
    let best_metric = if best_metric_raw == "score" || best_metric_raw == "bar" {
        best_metric_raw
    } else {
        eprintln!("[{run_tag}] GAIA_V9_BEST_METRIC='{best_metric_raw}' unrecognized (want 'score' or 'bar') — falling back to 'score' (no-silent-cure law: this is loud, not a quiet default)");
        "score".to_string()
    };
    let best_metric_is_bar = best_metric == "bar";
    // GAIA_V9_START_EPOCH (RESUME, IRON default 0 = old behavior byte-
    // identical to v9k..v9z: loop starts at epoch 0 against a fresh
    // He-init net). >0 continues a prior run's schedule position instead
    // of restarting it: the epoch loop below iterates `start_epoch..epochs`
    // (not `0..epochs`), so `frac=epoch/epochs` (lr anneal), ckpt cadence
    // (`(epoch+1)%ckpt_every`), probe cadence (`(epoch+1)%probe_every`) and
    // the epoch/ep_n numbers printed in every log line all fall out of that
    // ONE range change with zero further special-casing — this is the same
    // numbering scheme the dead run already used, just resumed mid-range.
    // GAIA_V9_RESUME_WEIGHTS (paired env, required when start_epoch>0):
    // path to a checkpoint (any of this trainer's own .bin outputs, e.g. a
    // `-last.bin` EMA snapshot) loaded as the STARTING POINT for both `net`
    // (the raw Adam-updated weights) and `ema` (the reported/saved model) —
    // both seeded equal, exactly like a fresh run seeds `ema = net.clone()`
    // off a fresh He-init `net`. Shape-validated against this run's own
    // UnetConfig (GAIA_V9_WIDTHS etc.), same loud panic-on-mismatch as the
    // existing GAIA_V9J_DRY_RUN_WEIGHTS path. DISCLOSED GAP: Adam's m/v
    // momentum state is NOT serialized anywhere in this codebase and is NOT
    // resumed here — `adam` below is always freshly constructed, so the
    // resumed run's first few steps ride cold (zero-initialized, bias-
    // corrected) momentum at whatever lr the schedule has already annealed
    // to, not the true momentum the dead run had accumulated. No mechanism
    // exists yet to fix this; stated honestly rather than silently assumed
    // equivalent (no-silent-cure law).
    let start_epoch = env_u32("GAIA_V9_START_EPOCH", 0);
    let resume_weights_env = std::env::var("GAIA_V9_RESUME_WEIGHTS").ok();
    eprintln!("[{run_tag}] GAIA_V9_START_EPOCH={start_epoch} (default 0=old behavior, fresh He-init net at epoch 0) GAIA_V9_RESUME_WEIGHTS={resume_weights_env:?} — start_epoch>0 continues the lr anneal (frac=epoch/epochs) and ckpt/probe cadence at that epoch instead of resetting to 0, loading net+ema from the resume checkpoint; Adam m/v momentum is NOT persisted/resumed anywhere in this codebase (disclosed gap, starts cold either way).");
    if start_epoch > 0 && resume_weights_env.is_none() {
        eprintln!("[{run_tag}] WARNING: GAIA_V9_START_EPOCH={start_epoch}>0 but GAIA_V9_RESUME_WEIGHTS unset — training will resume the SCHEDULE (lr/ckpt/probe numbering) against a FRESH He-init net. This is almost certainly not what you want; set GAIA_V9_RESUME_WEIGHTS to the checkpoint you're continuing from.");
    }
    if start_epoch >= epochs {
        eprintln!("[{run_tag}] WARNING: GAIA_V9_START_EPOCH={start_epoch} >= GAIA_V9_EPOCHS={epochs} — the training loop below will not execute a single epoch.");
    }
    eprintln!("[{run_tag}] res {tw}x{th} K={k} steps, GAIA_V9_K={k_draws} noise2noise draws (variance/{k_draws}, teacher validator-only)");
    eprintln!("[{run_tag}] WATCHDOG two-signal: score_streak={score_streak_param} LOG-ONLY, resid_streak={resid_streak_param} ABORT, bar-res probe every {probe_every} epochs at {eval_w}x{eval_h} (K={eval_k} ref={eval_ref}) ABORT on sparkle>={spark_target}");
    eprintln!("[{run_tag}] GAIA_V9_SPARKLE_ABORT={sparkle_abort_on} — 1 (default) keeps the bar-res probe sparkle>={spark_target} ABORT byte-identical to v9k..v9s; 0 makes that breach OBSERVATIONAL ONLY (loud log, training continues); resid_streak abort and every other watchdog signal stay ACTIVE either way.");
    eprintln!("[{run_tag}] GAIA_V9_CKPT_EVERY: ckpt_every={ckpt_every} — 0 (default) disables; >0 writes an UNCONDITIONAL EMA snapshot data/rdirect-weights-{run_tag}-ep<N>.bin (+ .provenance.json, N=epoch+1) every N epochs, independent of the best-checkpoint scorer below — insurance against a val-monitor flare freezing *BEST while the render-res body keeps improving (v9y postmortem, this banner's own module doc).");
    eprintln!("[{run_tag}] GAIA_V9_BEST_METRIC: best_metric={best_metric} — 'score' (default, IRON byte-identical to v9k..v9y) selects wpath_best by the lowest 128x72 demod-log score=max(sp/40,resid/0.035) seen across all monitor calls (run_monitor's own *BEST->saved); 'bar' selects wpath_best ONLY from the periodic bar-res PROBE pass (every {probe_every} epochs): legal (sparkle<{spark_target}) beats illegal, lowest resid wins among legal candidates, falls back to lowest sparkle when no legal candidate has been seen yet — run_monitor's 128x72 *BEST->saved file-write is DISABLED in this mode (its best_score/score_streak bookkeeping keeps logging, unchanged, LOG-ONLY).");
    eprintln!("[{run_tag}] CURE 4: overshoot_w={overshoot_w} — above-ceiling gradient is 2*overshoot_w*(raw-cap)/n (pulls toward the ceiling), no longer zero (see module doc mechanism-hunt verdict)");
    eprintln!("[{run_tag}] CURE 5: nohit_albedo_sq={nohit_albedo_sq} — albedo~0/no-hit pixels bypass the clamp/ceiling entirely, ordinary 2*(raw-target)/n always active (autopsy grad-probe-fix section)");
    eprintln!("[{run_tag}] CURE 5 PROMOTION (v9m): nohit_w={nohit_w} — no-hit class supervised at weight nohit_w against its honest n2n target (2*nohit_w*(raw-target)/n), nohit_w<=0.0 falls back to the OLD tiny-side-term/CURE5-off behavior (no-hit pixels run through ordinary CURE4 active/overshoot branches)");
    let v9n_hitgate = hitgate_on();
    eprintln!("[{run_tag}] V9N HIT-GATED BODY: GAIA_V9N_HITGATE={v9n_hitgate} — architectural cure, monad-specified: no-hit px (depth<=0) composed output := evidence EXACTLY (its own E+D composite, demod-log domain), backward grad to net_out is EXACT ZERO on those px (not a loss-side weight); CURE4/CURE5 branches above are dead for no-hit px when this is on. Default OFF = byte-identical to v9k/v9m (IRON LAW). Monitors (settle/run_monitor) and the bar-res probe evaluate the SAME composed output — see settle()/bar_res_probe.");
    let dark_alb = v9o_dark_alb();
    eprintln!("[{run_tag}] V9O DEGENERATE-DEMOD GATE (class detector): GAIA_V9O_DARK_ALB={dark_alb} — identifies the dark-hit class (settle/bar_res_probe/train loop, all 3 sites, `is_dark_hit`) as no-hit OR (hit AND albedo luminance<=dark_alb); v9n forensics (scratch/v9n-barres-forensics.log/json, commit 13349d3d) measured the bar-res sparkle class = 11/11 HIT px with albedo EXACTLY 0.0 (demod divisor degenerates to 1.0 like sky) — the trainer's own false_hit_dark population. dark_alb<=0.0 disables the dark-hit class detection entirely (v9n byte-identical). v9o's OWN dark-branch behavior (evidence passthrough at these px) proved firefly-ridden (81.4@24) and is fully REPLACED by V9P below.");
    let dark_cap_delta = v9p_dark_cap_delta();
    eprintln!("[{run_tag}] V9P ROBUST FIREFLY CAP (legacy, superseded by V9Q below whenever GAIA_V9Q_DESPECKLE_DELTA>0): GAIA_V9P_DARK_CAP_DELTA={dark_cap_delta} — at dark-hit px (v9o's class, above) NOT taking the v9q path, the net's own output is KEPT but capped at this px's 3x3 evidence-median (demod-log domain, `median3x3_dl`, all 3 sites: settle/bar_res_probe/train loop) + delta. dark_cap_delta<=0.0 disables: dark-hit px fall through to the ordinary evidence ceiling exactly as v9n (byte-identical, IRON LAW parameter). Per-epoch dark_capped_px/dark_clamped counted (no-silent-cure law).");
    let despeckle_delta = v9q_despeckle_delta();
    eprintln!("[{run_tag}] V9Q AIRTIGHT DESPECKLE CAP: GAIA_V9Q_DESPECKLE_DELTA={despeckle_delta} — v9p forensics (scratch/v9p-barres-forensics.log, commit 4e45211e) proved v9p's cap engages at every remaining bar-res sparkle texel but the cap VALUE itself still exceeds the sparkle criterion (demod-log vs evidence-median ≠ the criterion's own linear local-peak vs the OUTPUT's own neighborhood). At dark-hit px (v9o's class, above), TAKES PRIORITY over V9P: the net's own output is KEPT uncapped through the PRE-cap compose, then POST-COMPOSE bounded to at most its 3x3 neighbors' (excluding self, the frozen PRE-cap frame) max LINEAR radiance + delta (default 0.10, deliberately BELOW `sparkle_resid_per_mpx`'s own 0.15 local-peak delta — a capped px cannot register as a sparkle peak), converted back to the loss's demod-log domain (`log_demod`, the exact inverse of `undo_log_demod`) before feeding the SAME CURE4-style active/overshoot machinery (cap value substituted, no new gradient shape). despeckle_delta<=0.0 disables: dark-hit px fall through to V9P (above), then to the ordinary evidence ceiling (v9n byte-identical, IRON LAW parameter). Per-epoch despeckle_px/despeckle_clamped counted (no-silent-cure law).");
    let v9r_flag_on = v9r_degen_flag_on();
    let v9r_dw = v9r_dark_w();
    eprintln!("[{run_tag}] V9R DEGENERATE-DEMOD FLAG CHANNEL (representational cure, class made visible to the body): GAIA_V9R_DEGEN_FLAG={v9r_flag_on} — appends ONE extra per-px input channel (1.0 if no-hit OR dark-hit else 0.0, same `is_dark_hit` class as V9O above) beyond the shared HIST_FEATURES_SPLIT+MOTION_VECTOR_CHANNELS={} layout (disabled baseline, byte-identical to v9k/../v9q's layout, IRON LAW); flag on → resolved in_channels={} (this run). GAIA_V9R_DARK_W={v9r_dw} — multiplies whatever n2n grad already reaches dark-hit px (default 1.0 = no-op); independent of the flag (a pure loss-side reweight, like overshoot_w/nohit_w), only ever touches px where V9O's is_dark_hit fires.", HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS, v9r_input_channels());
    let loss_domain = v9w_loss_domain();
    let display_domain = loss_domain == "display";
    eprintln!("[{run_tag}] V9W LOSS DOMAIN: GAIA_V9W_LOSS_DOMAIN='{loss_domain}' — '' (default) is byte-identical demod-log MSE (v9k..v9v, IRON LAW); 'display' forward-transforms out_dl/target_dl/ceiling_dl through undo_log_demod->linear_to_srgb (exposure=1.0, the write_png/eval presentation chain) before applying the UNCHANGED CURE4/CURE5 active-MSE/overshoot shape, then chain-rules d(display)/d(dl) back for backward() — error now measured where the eye looks, not in raw demod-log space.");
    let v9x_mix_raw = v9x_mix();
    let mix_on = v9x_mix_raw >= 0.0;
    let mix = if mix_on { v9x_mix_raw.clamp(0.0, 1.0) } else { -1.0 };
    eprintln!("[{run_tag}] V9X MIXED-DOMAIN LOSS: GAIA_V9X_MIX={v9x_mix_raw} resolved_mix={mix} mix_on={mix_on} — mix<0.0 (default) is feature OFF, byte-identical to the plain V9W display_domain={display_domain}/classic branch above (IRON LAW); mix>=0.0 (clamped [0,1]) blends d_out=mix*display_grad+(1-mix)*classic_grad per px (both EXISTING gradient shapes, weighted sum, no new math) and OVERRIDES/IGNORES GAIA_V9W_LOSS_DOMAIN for the purpose of picking the grad (mix=0.0==pure classic, mix=1.0==pure display).");
    let teacher_spp = env_teacher_spp();
    let teacher_chunk = env_teacher_chunk();
    if teacher_spp > 0 {
        eprintln!("[{run_tag}] V9U CONVERGED-TEACHER TRAINING: GAIA_V9_TEACHER_SPP={teacher_spp} (chunk={teacher_chunk}) — pool draws' target_dl (the loss target) is now a TRUE CONVERGED render (rdirect_reference.rs-style chunked accumulation) at this crop resolution, replacing the K={k_draws}-draw n2n average entirely; mirror/val/mirror_val teachers AND the periodic bar-res probe's own teacher are ALSO converged at this spp (monitors measure vs truth, not a 64spp yardstick). Inputs (1spp live evidence taps) are UNCHANGED. Mind epoch time: this rides EVERY pool draw, EVERY epoch.");
    } else {
        eprintln!("[{run_tag}] V9U CONVERGED-TEACHER TRAINING: GAIA_V9_TEACHER_SPP=0 (default) — byte-identical to v9k..v9t, K={k_draws}-draw n2n target_dl, ref_frames={ref_frames} teacher (validator-only, never in the loss).");
    }

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let mirror_cam = scrying_glass::denoiser_dataset::mirror_camera();
    let fov_deg = params.fov_y_degrees;
    let front_eye = params.camera_position;
    let front_pivot = [0.0f32, 2.0, 0.0];
    let wide_eye = [-4.5f32, 8.5, 33.0];
    let wide_pivot = [-5.5f32, 2.0, 15.5];
    let val_cam = find("orbit_-20");
    eprintln!(
        "[{run_tag}] POSE DIVERSITY POOL (v9c/v9e parity): orbit±{pool_orbit_deg}deg jitter±{pool_jitter} around 2 anchors (front-pivot family x2 draws/epoch, wide-anchor family x1 draw/epoch) + mirror unconditional (fixed) — fresh draws every epoch, CURE 3 winsorize_k={winsor_k} (OFF if 0)"
    );

    let t_render = Instant::now();
    let mirror_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &mirror_cam, k, low_w, low_h, tw, th, ref_frames, pan_step, mirror_spp, k_draws, winsor_k);
    let val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &val_cam, k, low_w, low_h, tw, th, ref_frames, 0.0, 1, k_draws, winsor_k);
    let mirror_val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &mirror_cam, k, low_w, low_h, tw, th, ref_frames, 0.0, mirror_spp, k_draws, winsor_k);
    eprintln!(
        "[{run_tag}] rendered mirror(train) + 2 VALIDATOR (orbit_-20, mirror — FIXED, held-out) pose sequences ({tw}x{th}, teacher {} VALIDATOR ONLY, pan_step={pan_step}, mirror_spp={mirror_spp}) in {:.1}s",
        if teacher_spp > 0 { format!("{teacher_spp}spp CONVERGED") } else { format!("{ref_frames}spp") },
        t_render.elapsed().as_secs_f64()
    );
    eprintln!(
        "[{run_tag}] V9K CROP TRAINING: pool draws (front-pivot x2, wide-anchor x1) now render {tw}x{th} CROPS of a virtual {crop_full_w}x{crop_full_h} frame (asymmetric frustum, cost={tw}x{th} not {crop_full_w}x{crop_full_h}), pad_margin={crop_pad} sky_bias_prob={crop_sky_bias_prob} (0.0=uniform) — mirror/val stay full-frame, bar-res probe (below) stays the judge, unchanged"
    );
    std::io::stderr().flush().ok();

    // V9K CROP DRY RUN (PRE-FLIGHT, no-silent-cure law): opt-in,
    // GAIA_V9K_CROP_DRY_RUN=1. Renders ONE K=8 crop draw at TWO different
    // random offsets (front-pivot anchor, same anchor the real pool draws
    // use), prints both offsets, the depth-hit-flag nohit-masked-px count
    // ON EACH CROP (must be >0 — a crop that structurally never lands on a
    // no-hit/sky pixel would silently make CURE5's mask a no-op on this
    // training path), and a checksum (f64 sum of every step's target_dl
    // buffer) for each offset that must DIFFER (proof the crop offset
    // genuinely changes what gets rendered — the plumbing is real, not
    // ignored/clamped to a fixed window). Exits without training.
    if std::env::var("GAIA_V9K_CROP_DRY_RUN").ok().as_deref() == Some("1") {
        let mut dry_rng = Rng(0xc20d_ba5e_d2a0_c0de ^ seed);
        let mut checksums = Vec::new();
        for trial in 0..2 {
            let (draw, _yaw) = pool_camera(front_eye, front_pivot, fov_deg, pool_orbit_deg, pool_jitter, &mut dry_rng);
            let (cx, cy) = pick_crop_offset(&mut dry_rng, crop_full_w, crop_full_h, tw, th, crop_pad, crop_sky_bias_prob);
            let seq = render_pose_seq_cropped(&device, &queue, &base_tris, &scene, &draw, k, crop_full_w, crop_full_h, cx, cy, tw, th, low_w, low_h, ref_frames, pan_step, 1, k_draws, winsor_k);
            let mut nohit_px: u64 = 0;
            let mut total_px: u64 = 0;
            let mut checksum: f64 = 0.0;
            for s in &seq.steps {
                for px in 0..s.depth.len() {
                    total_px += 1;
                    if s.depth[px] <= 0.0 {
                        nohit_px += 1;
                    }
                }
                for row in &s.target_dl {
                    for &v in row {
                        checksum += v as f64;
                    }
                }
            }
            eprintln!(
                "[{run_tag}] CROP DRY RUN trial {trial}: offset=({cx},{cy}) full={crop_full_w}x{crop_full_h} crop={tw}x{th} nohit_px={nohit_px}/{total_px} ({:.2}%) target_dl_checksum={checksum:.6}",
                100.0 * nohit_px as f64 / total_px.max(1) as f64
            );
            std::io::stderr().flush().ok();
            if nohit_px == 0 {
                eprintln!("[{run_tag}] CROP DRY RUN ABORT: trial {trial} crop matched ZERO no-hit/sky pixels — CURE5's mask would be silently inert on this crop. Not necessarily a bug (a crop can legitimately land entirely on geometry), but flagged loudly per the no-silent-cure law; re-run or widen the crop pool if this persists across trials.");
                std::io::stderr().flush().ok();
            }
            checksums.push(checksum);
        }
        let differ = (checksums[0] - checksums[1]).abs() > 1e-6;
        eprintln!("[{run_tag}] CROP DRY RUN checksum check: trial0={:.6} trial1={:.6} DIFFER={differ}", checksums[0], checksums[1]);
        std::io::stderr().flush().ok();
        if !differ {
            eprintln!("[{run_tag}] CROP DRY RUN ABORT: two different crop offsets produced IDENTICAL evidence — the crop offset is being ignored (configuration bug), not a silent no-op. Fix before launching training.");
            std::io::stderr().flush().ok();
            std::process::exit(1);
        }
        eprintln!("[{run_tag}] CROP DRY RUN done — exiting (no training run)");
        std::io::stderr().flush().ok();
        return;
    }

    // V9J DRY RUN (FIX+INSTRUMENT LAW (c)): opt-in, GAIA_V9J_DRY_RUN=1.
    // One batch (val_seq's fully-populated K steps, fresh He-init net, NO
    // weight update) through the REAL training px/channel loop below
    // (kept byte-identical here on purpose — this is the actual formula,
    // not a reimplementation), printing the mask counts
    // (nohit_px/nohit_effective, defined below at the per-epoch
    // instrumentation) plus one sample pixel's before/after loss branch,
    // then exits without training.
    if std::env::var("GAIA_V9J_DRY_RUN").ok().as_deref() == Some("1") {
        let v9_w = v9_widths();
        let config = UnetConfig { render_w: tw as usize, render_h: th as usize, output_w: tw as usize, output_h: th as usize, in_channels: v9r_input_channels(), n_scales: v9_w.len(), widths: v9_w, ..UnetConfig::default() };
        // GAIA_V9J_DRY_RUN_WEIGHTS (optional): load a checkpoint instead of
        // fresh He-init, to check whether the no-hit-overshoot population
        // shrinks as training converges (fresh init is NOT representative
        // of the trained net's raw/cap relationship near the failure).
        let net = if let Ok(wpath) = std::env::var("GAIA_V9J_DRY_RUN_WEIGHTS") {
            let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("read {wpath:?}: {e}"));
            let mut w = deserialize_weights(&bytes).unwrap_or_else(|| panic!("deserialize {wpath:?}"));
            // CAPACITY ROUND (in_channels pattern, extended honestly): a
            // checkpoint carries its OWN widths/n_scales/in_channels in its
            // header (round-trips generically, see rdirect_unet.rs
            // serialize_weights/deserialize_weights) — overwriting `.config`
            // wholesale with THIS run's config (e.g. a different
            // GAIA_V9_WIDTHS) would silently pair new shape metadata with
            // old-shaped conv tensors, index-panicking obscurely deep in
            // `forward` instead of here with a clear reason. Loud assert,
            // same spirit as `forward`'s own `assert_eq!` on in_channels.
            if let Err(e) = w.validate_shapes_against(&config) {
                panic!("[{run_tag}] GAIA_V9J_DRY_RUN_WEIGHTS {wpath:?}: checkpoint shape mismatch vs this run's UnetConfig (widths={:?} in_channels={}) — {e}. Checkpoints trained with different widths/in_channels cannot be silently reused.", config.widths, config.in_channels);
            }
            w.config = config;
            eprintln!("[{run_tag}] DRY RUN loaded checkpoint {wpath:?} ({} bytes) instead of fresh init", bytes.len());
            w
        } else {
            UnetWeights::new_random(config, seed)
        };
        let mut dry_nohit_px: u64 = 0;
        let mut dry_nohit_effective: u64 = 0;
        let mut dry_total_px: u64 = 0;
        let mut sample_printed = false;
        for s in 0..k as usize {
            let prev: Option<&[GVec3]> = None; // dry run: no history chain needed for the count
            let input = build_input_img(&val_seq, s, prev, sky_reject);
            let (out, _cache) = net.forward(&input);
            let target = target_to_img(&val_seq.steps[s].target_dl, th as usize, tw as usize);
            let ceiling = &val_seq.steps[s].ceiling_dl;
            let depth = &val_seq.steps[s].depth;
            let n_elems = out.data.len() as f32;
            for px in 0..(out.h * out.w) {
                dry_total_px += 1;
                // FIX (a): re-keyed on the trainer's REAL no-hit signature
                // (`depth[px] <= 0.0`, the same hit-flag `reproject_prev`'s
                // `is_miss` already uses — zeroed unconditionally on a true
                // miss by `integrator.wgsl:1023-1024`). The old albedo≈0
                // threshold was PROVEN (per-epoch cross-check instrumentation
                // below/in the training loop) to have zero false negatives
                // but a small nonzero false-POSITIVE rate: legitimately-hit
                // dark/black-albedo surface pixels wrongly diverted through
                // CURE5's bypass (a few/epoch out of thousands — see
                // scratch/v9-autopsy.md v9j section).
                let is_nohit = depth[px] <= 0.0;
                if is_nohit {
                    dry_nohit_px += 1;
                }
                for c in 0..out.c {
                    let i = px * out.c + c;
                    let cap = ceiling[px][c];
                    let raw = out.data[i];
                    let active = raw <= cap;
                    if is_nohit && !active {
                        dry_nohit_effective += 1;
                    }
                    if is_nohit && !active && !sample_printed {
                        sample_printed = true;
                        let diff = raw.min(cap) - target.data[i];
                        let d_cure4 = 2.0 * overshoot_w * (raw - cap) / n_elems;
                        let d_cure5 = 2.0 * (raw - target.data[i]) / n_elems;
                        eprintln!(
                            "[{run_tag}] DRY RUN sample px={px} step={s} c={c} raw={raw:.6} cap={cap:.6} target={:.6} depth={:.6} BEFORE(CURE4 formula, overshoot anchored to cap)={d_cure4:.8} AFTER(CURE5 formula, anchored to target)={d_cure5:.8}",
                            target.data[i], depth[px]
                        );
                        let _ = diff;
                    }
                }
            }
        }
        std::io::stderr().flush().ok();
        eprintln!(
            "[{run_tag}] DRY RUN one-batch (K={k} steps of val_seq) mask counts: nohit_px={dry_nohit_px}/{dry_total_px} ({:.2}%) nohit_effective(is_nohit&&raw>cap, differs-from-CURE4)={dry_nohit_effective} elem-instances",
            100.0 * dry_nohit_px as f64 / dry_total_px.max(1) as f64
        );
        if dry_nohit_px == 0 {
            eprintln!("[{run_tag}] DRY RUN ABORT: nohit_px==0 on a full K-step batch — CURE 5's mask NEVER matched a single training pixel. This is a configuration bug (wrong field/threshold/resolution), not a silent no-op. Fix before launching training.");
            std::io::stderr().flush().ok();
            std::process::exit(1);
        }
        if !sample_printed {
            eprintln!("[{run_tag}] DRY RUN NOTE: nohit_px>0 but nohit_effective==0 in this one-batch sample — CURE 5 matched no-hit pixels but none of them exceeded their ceiling (raw>cap) in this batch, so CURE 5's formula was identical to CURE4's/active-branch's formula everywhere it fired. Real effect may still be ~0 across training; per-epoch instrumentation below is the authoritative signal.");
            std::io::stderr().flush().ok();
        }
        eprintln!("[{run_tag}] DRY RUN done — exiting (no training run)");
        std::io::stderr().flush().ok();
        return;
    }

    // V9I GRAD-PROBE (c), updated this round: opt-in, GAIA_V9I_GRAD_PROBE=1.
    // Loads a checkpoint (GAIA_V9I_PROBE_WEIGHTS, default v9h-last), runs
    // the val_seq's (orbit_-20) recurrent chain through it, then at the
    // LAST step prints, per pixel/channel, THREE formulas side by side:
    // BEFORE_stale (pre-CURE4 zero-gate — the OLD diagnostic's own dead
    // formula, kept for historical comparison only), CURE4_actual (the
    // real pre-v9i training formula), AFTER_fix (v9i's CURE 5: ordinary
    // MSE, unconditional, at albedo≈0/no-hit). Zero effect on the normal
    // run (env unset by default).
    if std::env::var("GAIA_V9I_GRAD_PROBE").ok().as_deref() == Some("1") {
        let data_dir0 = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let wpath = std::env::var("GAIA_V9I_PROBE_WEIGHTS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| data_dir0.join("rdirect-weights-v9h-last.bin"));
        let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("read {wpath:?}: {e}"));
        let probe = deserialize_weights(&bytes).unwrap_or_else(|| panic!("deserialize {wpath:?}"));
        eprintln!("[v9i-grad-probe] loaded {wpath:?} ({} bytes)", bytes.len());
        let chain = history_forward(&probe, &val_seq, sky_reject);
        let last_s = val_seq.steps.len() - 1;
        let prev: Option<&[GVec3]> = if last_s == 0 { None } else { Some(&chain[last_s - 1]) };
        let input = build_input_img(&val_seq, last_s, prev, sky_reject);
        let (out, _cache) = probe.forward(&input);
        let step = &val_seq.steps[last_s];
        // (x,y) list from scratch/v9e-forensics.log COORDS epoch 73 (128x72 space).
        let texels: &[(u32, u32)] = &[(94, 3), (94, 6), (97, 6), (94, 8), (47, 9), (55, 34), (66, 38)];
        eprintln!("[v9i-grad-probe] val_seq {}x{}, last step {last_s}, texel raw/cap/target/active/d_out_before/d_out_cure4/d_out_after per channel (overshoot_w={overshoot_w} nohit_albedo_sq={nohit_albedo_sq}):", val_seq.tw, val_seq.th);
        for &(x, y) in texels {
            let px = (y * val_seq.tw + x) as usize;
            let albedo_sq = step.albedo[px].length_squared();
            let is_nohit = albedo_sq <= nohit_albedo_sq;
            for c in 0..OUTPUT_CHANNELS {
                let i = px * OUTPUT_CHANNELS + c;
                let cap = step.ceiling_dl[px][c];
                let raw = out.data[i];
                let target = step.target_dl[px][c];
                let presented = raw.min(cap);
                let active = raw <= cap;
                let d_before = if active { 2.0 * (presented - target) / (out.data.len() as f32) } else { 0.0 };
                let d_cure4 = if active { 2.0 * (presented - target) / (out.data.len() as f32) } else { 2.0 * overshoot_w * (raw - cap) / (out.data.len() as f32) };
                let d_after = if is_nohit { 2.0 * (raw - target) / (out.data.len() as f32) } else { d_cure4 };
                eprintln!(
                    "[v9i-grad-probe]   px=({x},{y}) c={c} raw={raw:.6} cap={cap:.6} target={target:.6} overshoot={} active={active} is_nohit={is_nohit} d_out_before={d_before:.8} d_out_cure4={d_cure4:.8} d_out_after={d_after:.8}",
                    raw > cap,
                );
            }
        }
        eprintln!("[v9i-grad-probe] done — exiting (no training run)");
        std::io::stderr().flush().ok();
        return;
    }

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    // IRON paths (module doc change 2): best-score convention unchanged,
    // NEW "-last" sibling for the unconditional most-recent-epoch snapshot.
    let wpath_best = data_dir.join(format!("rdirect-weights-{run_tag}.bin"));
    let wpath_last = data_dir.join(format!("rdirect-weights-{run_tag}-last.bin"));
    let v9_w = v9_widths();
    let config = UnetConfig { render_w: tw as usize, render_h: th as usize, output_w: tw as usize, output_h: th as usize, in_channels: v9r_input_channels(), n_scales: v9_w.len(), widths: v9_w, ..UnetConfig::default() };
    eprintln!("[{run_tag}] UnetConfig widths={:?} n_scales={} params≈{}", config.widths, config.n_scales, config.approx_param_count());
    eprintln!("[{run_tag}] RESOLVED in_channels={} (v9r_degen_flag_on={v9r_flag_on}, base HIST_FEATURES_SPLIT+MOTION_VECTOR_CHANNELS={}, +{} if flag on) — this is the value baked into config.in_channels/UnetWeights/provenance below, not doc text", config.in_channels, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS, V9R_FLAG_CHANNELS);

    let mut net = if let Some(wpath) = &resume_weights_env {
        let bytes = std::fs::read(wpath).unwrap_or_else(|e| panic!("[{run_tag}] GAIA_V9_RESUME_WEIGHTS {wpath:?}: read failed: {e}"));
        let mut w = deserialize_weights(&bytes).unwrap_or_else(|| panic!("[{run_tag}] GAIA_V9_RESUME_WEIGHTS {wpath:?}: deserialize failed"));
        // Same shape-mismatch law as GAIA_V9J_DRY_RUN_WEIGHTS above: a
        // checkpoint carries its OWN widths/n_scales/in_channels in its
        // header; overwriting `.config` wholesale with THIS run's config
        // (e.g. a different GAIA_V9_WIDTHS) would silently pair new shape
        // metadata with old-shaped conv tensors, panicking obscurely deep
        // in `forward` instead of here with a clear reason.
        if let Err(e) = w.validate_shapes_against(&config) {
            panic!("[{run_tag}] GAIA_V9_RESUME_WEIGHTS {wpath:?}: checkpoint shape mismatch vs this run's UnetConfig (widths={:?} in_channels={}) — {e}. Checkpoints trained with different widths/in_channels cannot be silently reused.", config.widths, config.in_channels);
        }
        w.config = config.clone();
        eprintln!("[{run_tag}] RESUME: loaded {wpath:?} ({} bytes, sha256={}) as starting net+ema at GAIA_V9_START_EPOCH={start_epoch}", bytes.len(), weights_sha256(&w));
        w
    } else {
        UnetWeights::new_random(config.clone(), seed)
    };
    let mut ema = net.clone();
    let mut adam = UnetAdam::new(&net, lr0 as f32, 0.9, 0.999, 1e-8);

    let mut best_score: f64 = if wpath_best.exists() {
        if let Some(bytes) = std::fs::read(&wpath_best).ok() {
            if let Some(prior) = deserialize_weights(&bytes) {
                let (n, t) = settle(&prior, &val_seq, sky_reject);
                let sp = sparkle_resid_per_mpx(&n, &t, val_seq.tw, val_seq.th);
                let rs = rmse_lin(&n, &t);
                let s = (sp / 40.0).max(rs / 0.035);
                eprintln!("[{run_tag}] cross-run floor: existing checkpoint score={s:.3} (sparkle {sp:.1} resid {rs:.4})");
                s
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    };
    let mut best_bytes: Vec<u8> = serialize_weights(&ema);
    let mut score_streak: u32 = 0;
    let mut resid_best: f64 = f64::INFINITY;
    let mut resid_streak: u32 = 0;

    // V9Z 'bar' metric bookkeeping (dead/unused when best_metric_is_bar is
    // false — IRON LAW, no effect on the default 'score' path). Same
    // cross-run-floor idea as best_score above, but measured at bar-res
    // (rdirect_v9_eval_640.rs-equivalent, bar_res_probe) instead of the
    // 128x72 val monitor: seeds the selection from an existing wpath_best
    // checkpoint so a restart in bar mode can't regress below it.
    let mut bar_best_legal = false;
    let mut bar_best_resid: f64 = f64::INFINITY;
    let mut bar_best_sparkle: f64 = f64::INFINITY;
    if best_metric_is_bar && wpath_best.exists() {
        if let Some(prior) = std::fs::read(&wpath_best).ok().and_then(|b| deserialize_weights(&b)) {
            let (psp0, prs0, _) = bar_res_probe(&device, &queue, &base_tris, &scene, &val_cam, &prior, sky_reject, eval_w, eval_h, eval_k, eval_ref, highlight_pctl);
            bar_best_legal = psp0 < spark_target as f64;
            bar_best_resid = prs0;
            bar_best_sparkle = psp0;
            eprintln!("[{run_tag}] cross-run floor (bar): existing checkpoint bar-res sparkle={psp0:.1}/Mpx resid={prs0:.4} legal={bar_best_legal}");
            std::io::stderr().flush().ok();
        }
    }

    let monitor_label0 = if resume_weights_env.is_some() { format!("epoch {}-1 (resumed net)", start_epoch) } else { "epoch -1 (fresh He-init)".to_string() };
    run_monitor(&run_tag, &monitor_label0, &ema, &val_seq, &mirror_val_seq, sky_reject, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath_best, &mut score_streak, score_streak_param, &mut resid_best, &mut resid_streak, resid_streak_param, !best_metric_is_bar);

    let mut rng = Rng(0xd15e_ed00_08f0_0dc0 ^ seed);
    let t_train = Instant::now();
    let mut stop_reason = String::from("wall/epoch budget exhausted");

    'train: for epoch in start_epoch..epochs {
        if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
            eprintln!("[{run_tag}] WALL budget {wall_budget}s reached at epoch {epoch} — stopping, keeping best+last");
            break;
        }
        let frac = epoch as f32 / epochs as f32;
        adam.set_lr(lr0 / (1.0 + 1.0 * frac));

        let t_pool = Instant::now();
        let (draw_a1, yaw_a1) = pool_camera(front_eye, front_pivot, fov_deg, pool_orbit_deg, pool_jitter, &mut rng);
        let (draw_a2, yaw_a2) = pool_camera(front_eye, front_pivot, fov_deg, pool_orbit_deg, pool_jitter, &mut rng);
        let (draw_b, yaw_b) = pool_camera(wide_eye, wide_pivot, fov_deg, pool_orbit_deg, pool_jitter, &mut rng);
        // V9K CROP TRAINING SAMPLE UNIT (module doc): each of the 3 pool
        // draws gets its OWN independent random crop offset into the
        // virtual full_w x full_h frame (task: "random crop offsets per
        // draw"); the mirror draw stays the ORIGINAL uncropped render
        // (module doc "MIRROR/VAL POSES UNCHANGED" — it's a fixed anchor,
        // cropping it adds no diversity).
        let (cx_a1, cy_a1) = pick_crop_offset(&mut rng, crop_full_w, crop_full_h, tw, th, crop_pad, crop_sky_bias_prob);
        let (cx_a2, cy_a2) = pick_crop_offset(&mut rng, crop_full_w, crop_full_h, tw, th, crop_pad, crop_sky_bias_prob);
        let (cx_b, cy_b) = pick_crop_offset(&mut rng, crop_full_w, crop_full_h, tw, th, crop_pad, crop_sky_bias_prob);
        let poses: Vec<TrainSeq> = vec![
            TrainSeq::Crop(render_pose_seq_cropped(&device, &queue, &base_tris, &scene, &draw_a1, k, crop_full_w, crop_full_h, cx_a1, cy_a1, tw, th, low_w, low_h, ref_frames, pan_step, 1, k_draws, winsor_k)),
            TrainSeq::Crop(render_pose_seq_cropped(&device, &queue, &base_tris, &scene, &draw_a2, k, crop_full_w, crop_full_h, cx_a2, cy_a2, tw, th, low_w, low_h, ref_frames, pan_step, 1, k_draws, winsor_k)),
            TrainSeq::Crop(render_pose_seq_cropped(&device, &queue, &base_tris, &scene, &draw_b, k, crop_full_w, crop_full_h, cx_b, cy_b, tw, th, low_w, low_h, ref_frames, pan_step, 1, k_draws, winsor_k)),
            TrainSeq::Full(mirror_seq.clone()),
        ];
        let pool_ms = t_pool.elapsed().as_secs_f64() * 1000.0;

        let t_hist = Instant::now();
        let history: Vec<Vec<Vec<GVec3>>> = poses.iter().map(|seq| seq.history_forward(&ema, sky_reject)).collect();
        let hist_ms = t_hist.elapsed().as_secs_f64() * 1000.0;

        let mut order: Vec<(usize, usize)> = Vec::with_capacity(poses.len() * k as usize);
        for pi in 0..poses.len() {
            for s in 0..k as usize {
                order.push((pi, s));
            }
        }
        for i in (1..order.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut epoch_mse = 0.0f64;
        let mut n_px_total = 0u64;
        // FIX+INSTRUMENT LAW (b): per-epoch CURE5-masked-pixel equipment.
        // nohit_px = pixel-instances (px, counted once/pixel not/channel)
        // where the mask condition matched. nohit_effective = channel-
        // instances where the mask ACTUALLY changed the gradient vs CURE4
        // (is_nohit && raw>cap — elsewhere CURE5's formula is identical to
        // the plain `active` branch, so it is a silent no-op there by
        // construction, not a bug).
        let mut epoch_nohit_px: u64 = 0;
        let mut epoch_nohit_effective: u64 = 0;
        // V9O: degenerate-demod (albedo~0) HIT px, counted separately from
        // `epoch_nohit_px` (no-silent-cure law) — see `v9o_gate_px`.
        let mut epoch_dark_alb_px: u64 = 0;
        // V9P: dark-hit px actually given the cap treatment (gate_on &&
        // is_dark_hit && dark_cap_delta>0), counted separately from
        // `epoch_dark_alb_px` (which counts is_dark_hit regardless of
        // gate/delta) — no-silent-cure law.
        let mut epoch_dark_capped_px: u64 = 0;
        // V9P: channel-instances where the dark cap ACTUALLY fired
        // (raw>cap) — "how many were actually clamped", task-mandated.
        let mut epoch_dark_clamped: u64 = 0;
        // V9Q: dark-hit px actually given the POST-COMPOSE despeckle
        // treatment (gate_on && is_dark_hit && despeckle_delta>0, takes
        // PRIORITY over the v9p branch above whenever both deltas are
        // positive) — counted separately, no-silent-cure law.
        let mut epoch_despeckle_px: u64 = 0;
        // V9Q: channel-instances where the despeckle cap ACTUALLY fired
        // (raw_lin > neigh_max_lin+delta, i.e. raw_dl > cap_dl once both
        // sides are compared in the loss's own dl domain).
        let mut epoch_despeckle_clamped: u64 = 0;
        let mut epoch_total_px: u64 = 0;
        // V9V no-silent-cure counters: input-clamp channel-instances
        // actually pulled down this epoch (diffed against the shared
        // atomic's snapshot at epoch start — bar_res_probe/settle also
        // bump the same atomic on their own periodic/monitor calls, which
        // legitimately inflates this a little on probe/monitor epochs) and
        // the mean |mv| (reprojection motion-vector magnitude, pixels) fed
        // to the net across every batch this epoch.
        let epoch_clamp_taps_start = input_clamp_taps_snapshot();
        let mut epoch_mv_abs_sum: f64 = 0.0;
        let mut epoch_mv_px: u64 = 0;
        let mut first_batch = true;
        let t_batch = Instant::now();
        for &(pi, s) in &order {
            if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
                eprintln!("[{run_tag}] WALL budget {wall_budget}s reached MID-epoch {epoch} — stopping, keeping best+last");
                std::io::stderr().flush().ok();
                break 'train;
            }
            let seq = &poses[pi];
            let prev: Option<&[GVec3]> = if s == 0 { None } else { Some(&history[pi][s - 1]) };
            let input = seq.build_input(s, prev, sky_reject);
            for py in 0..input.h {
                for pxi in 0..input.w {
                    let mvx = input.at(py, pxi, HIST_FEATURES_SPLIT);
                    let mvy = input.at(py, pxi, HIST_FEATURES_SPLIT + 1);
                    epoch_mv_abs_sum += (mvx * mvx + mvy * mvy).sqrt() as f64;
                    epoch_mv_px += 1;
                }
            }
            let (out, cache) = net.forward(&input);
            let step = &seq.steps()[s];
            let target = target_to_img(&step.target_dl, th as usize, tw as usize);
            let ceiling = &step.ceiling_dl;
            let albedo = &step.albedo;
            let depth = &step.depth;
            // V9N hit-gated compose (module doc): `evidence_dl` is this
            // step's own E+D composite already in the loss's demod-log
            // domain (filled at render time, same source as the net's own
            // input taps). `hitgate_on` false ⇒ every gate branch below is
            // dead and this run is byte-identical to v9k/v9m (IRON LAW).
            let evidence = &step.evidence_dl;
            let gate_on = hitgate_on();
            let n_elems = out.data.len() as f32;
            let mut d_out = Img::zeros(out.h, out.w, out.c);
            let mut mse = 0.0f64;
            let mut batch_nohit_px: u64 = 0;
            let mut batch_nohit_effective: u64 = 0;
            let mut batch_dark_alb_px: u64 = 0;
            let mut batch_dark_capped_px: u64 = 0;
            let mut batch_dark_clamped: u64 = 0;
            let mut batch_despeckle_px: u64 = 0;
            let mut batch_despeckle_clamped: u64 = 0;
            let dark_alb = v9o_dark_alb();
            let dark_cap_delta = v9p_dark_cap_delta();
            let despeckle_delta = v9q_despeckle_delta();
            let dark_w = v9r_dark_w();
            let median_dl = median3x3_dl(evidence, out.w as u32, out.h as u32);
            let n_px = (out.h * out.w) as usize;
            // MASK-FIELD CROSS-CHECK (FIX+INSTRUMENT LAW (a)): depth<=0.0
            // is the OTHER no-hit signature already in the trainer's own
            // Step buffers (`reproject_prev`'s `is_miss = cur_depth <= 0.0`
            // convention, same AOV pass, integrator.wgsl:1023-1024 zeros
            // BOTH albedo and depth(=hit.t) on a true miss unconditionally)
            // — false_hit_dark = albedo-mask says nohit but depth says REAL
            // HIT (a legitimately dark/black-albedo surface wrongly diverted
            // through CURE5's bypass); false_miss_albedo = depth says miss
            // but albedo-mask missed it (should be structurally impossible
            // per the shader, kept as a canary).
            let mut batch_false_hit_dark: u64 = 0;
            let mut batch_false_miss_albedo: u64 = 0;
            // V9Q PASS 1 (module doc — "one pass, neighbors read from the
            // PRE-cap composed frame, no iteration"): classify every px AND
            // build the frame's PRE-cap composed dl value — nohit-gate
            // px:=evidence; dark-hit px whose despeckle path wins (priority
            // over the OLD v9p median cap, see `v9q_despeckle_delta` doc)
            // KEPT RAW/uncapped here (the despeckle cap itself is applied in
            // PASS 2, in the criterion's own LINEAR domain, against this
            // frozen frame); dark-hit px falling through to the legacy v9p
            // branch get the OLD median+delta cap exactly as before; every
            // other px gets the ordinary evidence-ceiling clamp, unchanged.
            let mut is_nohit_all = vec![false; n_px];
            let mut is_dark_hit_all = vec![false; n_px];
            let mut dark_v9q_all = vec![false; n_px];
            let mut dark_capped_all = vec![false; n_px];
            let mut precap_dl: Vec<[f32; OUTPUT_CHANNELS]> = vec![[0.0; OUTPUT_CHANNELS]; n_px];
            for px in 0..n_px {
                // CURE 5 (V9I, module doc): albedo≈0/no-hit pixels
                // (sky/silhouette-boundary class the autopsy's grad-probe-
                // fix section convicted — evidence ceiling there is
                // systematically INFLATED by bilinear-upsample+3x3-maxpool
                // bleed from neighboring bright surfaces, mean 1.64x the
                // honest target) skip the clamp/ceiling mechanism entirely:
                // ordinary MSE against the honest n2n target, unconditional.
                // FIX (a): re-keyed on the trainer's REAL no-hit signature
                // — `depth[px] <= 0.0` (the true hit-flag; `depth`/`albedo`
                // are BOTH zeroed unconditionally on a real miss by the same
                // AOV pass, `integrator.wgsl:1023-1024`, and this is the
                // exact test `reproject_prev`'s `is_miss` already uses
                // elsewhere in this file). The OLD albedo≈`nohit_albedo_sq`
                // threshold is kept below ONLY as `legacy_is_nohit`, for the
                // false-positive cross-check this instrumentation reports
                // (a legitimately-hit dark/black-albedo surface pixel was
                // being wrongly diverted through CURE5's bypass — measured
                // nonzero but small, see scratch/v9-autopsy.md v9j section).
                let is_nohit = depth[px] <= 0.0;
                let legacy_is_nohit = albedo[px].length_squared() <= nohit_albedo_sq;
                if is_nohit {
                    epoch_nohit_px += 1;
                    batch_nohit_px += 1;
                }
                if legacy_is_nohit && !is_nohit {
                    batch_false_hit_dark += 1;
                }
                if is_nohit && !legacy_is_nohit {
                    batch_false_miss_albedo += 1;
                }
                // V9O DEGENERATE-DEMOD GATE (module doc, forensics-measured
                // class): a REAL hit (is_nohit==false) whose albedo
                // luminance is at/under `dark_alb` degenerates to the SAME
                // demod_divisor==1.0 as a true no-hit px — v9n's hit-gate
                // structurally missed this class (depth>0). This is exactly
                // the `false_hit_dark` population above, re-keyed on the
                // v9o luminance threshold (not the squared-length one) and
                // counted separately from `epoch_nohit_px`/`nohit_px` per
                // the no-silent-cure law. `dark_alb<=0.0` disables this
                // branch entirely (explicit guard, not a threshold that
                // merely never matches) = v9n byte-identical.
                let is_dark_hit = !is_nohit && dark_alb > 0.0 && lum(albedo[px]) <= dark_alb;
                if is_dark_hit {
                    epoch_dark_alb_px += 1;
                    batch_dark_alb_px += 1;
                }
                epoch_total_px += 1;
                is_nohit_all[px] = is_nohit;
                is_dark_hit_all[px] = is_dark_hit;
                // V9N: gate applies to REAL no-hit px ONLY, when the gate
                // param is on. When gated, the composed/backward-facing
                // output for EVERY channel of this px is hard-replaced by
                // `evidence` — CURE4/CURE5's branches below become dead code
                // for this px, per the task's "leave code, param-gated"
                // mandate.
                let gated_nohit = gate_on && is_nohit;
                // V9Q (replaces v9p's demod-log median cap, module doc):
                // dark-hit px whose despeckle delta is enabled win priority
                // — their PRE-cap value is the RAW net output, uncapped; the
                // despeckle cap itself is a POST-COMPOSE, linear-domain
                // operation applied in PASS 2 below.
                let use_v9q = gate_on && is_dark_hit && despeckle_delta > 0.0;
                dark_v9q_all[px] = use_v9q;
                if use_v9q {
                    epoch_despeckle_px += 1;
                    batch_despeckle_px += 1;
                }
                // V9P (legacy, kept param-gated): a dark-hit px NOT taking
                // the v9q path keeps the NET's own output but its ceiling
                // becomes the ROBUST median+delta cap — `dark_cap_delta<=0.0`
                // disables: dark-hit px fall through to the ordinary
                // evidence ceiling exactly as if `is_dark_hit` were false
                // (v9n behavior byte-identical).
                let dark_capped = gate_on && is_dark_hit && !use_v9q && dark_cap_delta > 0.0;
                dark_capped_all[px] = dark_capped;
                if dark_capped {
                    epoch_dark_capped_px += 1;
                    batch_dark_capped_px += 1;
                }
                for c in 0..out.c {
                    let i = px * out.c + c;
                    let raw = out.data[i];
                    precap_dl[px][c] = if gated_nohit {
                        evidence[px][c]
                    } else if use_v9q {
                        raw
                    } else if dark_capped {
                        raw.min(median_dl[px][c] + dark_cap_delta)
                    } else {
                        raw.min(ceiling[px][c])
                    };
                }
            }
            // V9Q PASS 2: convert the frozen PRE-cap composed frame to LINEAR
            // radiance (the sparkle criterion's own domain, `undo_log_demod`
            // — the exact conversion `sparkle_resid_per_mpx`/the forensics
            // tool use), then for every dark-hit px taking the v9q path,
            // bound its presented linear radiance to at most its 3x3
            // neighbors' (excluding self, THIS frozen frame) max + delta
            // (`despeckle_cap_lin`) — convert that cap back to the loss's
            // demod-log domain (`log_demod`, the exact inverse) and feed it
            // into the SAME CURE4-style active/overshoot machinery every
            // other branch already uses below (cap value substituted, no
            // new gradient shape invented — task mandate).
            let precap_lin: Vec<GVec3> = (0..n_px)
                .map(|px| undo_log_demod(GVec3::new(precap_dl[px][0], precap_dl[px][1], precap_dl[px][2]), demod_divisor(albedo[px])))
                .collect();
            for px in 0..n_px {
                let is_nohit = is_nohit_all[px];
                let dark_capped = dark_capped_all[px];
                let use_v9q = dark_v9q_all[px];
                let gated_nohit = gate_on && is_nohit;
                let cap_arr: [f32; OUTPUT_CHANNELS] = if use_v9q {
                    let cap_lin = despeckle_cap_lin(&precap_lin, out.w as u32, out.h as u32, px, despeckle_delta);
                    let cap_dl = log_demod(cap_lin, demod_divisor(albedo[px]));
                    [cap_dl.x, cap_dl.y, cap_dl.z]
                } else if dark_capped {
                    let m = median_dl[px];
                    [m[0] + dark_cap_delta, m[1] + dark_cap_delta, m[2] + dark_cap_delta]
                } else {
                    ceiling[px]
                };
                // V9W (module doc): per-channel divisor for this px, shared
                // by every dl->display transform below (a constant w.r.t.
                // `raw`/`out_dl` — albedo is an INPUT, never backprop'd
                // through here). Only ever read when `display_domain`.
                let divisor_px = demod_divisor(albedo[px]);
                let divisor_arr = [divisor_px.x, divisor_px.y, divisor_px.z];
                for c in 0..out.c {
                    let i = px * out.c + c;
                    let cap = cap_arr[c];
                    let raw = out.data[i];
                    let presented = if gated_nohit { evidence[px][c] } else { raw.min(cap) };
                    let diff = presented - target.data[i];
                    mse += (diff as f64) * (diff as f64);
                    if is_nohit && raw > cap {
                        epoch_nohit_effective += 1;
                        batch_nohit_effective += 1;
                    }
                    if dark_capped && raw > cap {
                        epoch_dark_clamped += 1;
                        batch_dark_clamped += 1;
                    }
                    if use_v9q && raw > cap {
                        epoch_despeckle_clamped += 1;
                        batch_despeckle_clamped += 1;
                    }
                    // CURE 4 (v9g, module doc): the OLD gate zeroed the
                    // gradient outright above the ceiling (one-sided —
                    // stopped reward for rising further but removed every
                    // penalty for having already overshot). Now BOTH sides
                    // of the ceiling carry a restoring gradient: below,
                    // CURE 1's own (raw-target) term unchanged; above, an
                    // explicit overshoot term pulls raw back toward the
                    // CEILING (not the possibly-noisy per-draw target),
                    // weight `overshoot_w` (IRON param, default 1.0) — C0
                    // continuous at the seam (both branches are 0 exactly
                    // at raw==cap). `presented`/`mse` above (the REPORTED
                    // quality number, n2n_mse/monitor) are untouched.
                    let active = raw <= cap;
                    // V9X (module doc): classic demod-log grad, exact SAME
                    // shape as v9k..v9v (CURE5 bypass / CURE4 active/
                    // overshoot) — pulled into a variable so it can feed
                    // either the plain (mix off) branch below or the mix
                    // blend, unchanged in formula either way.
                    let classic_grad = if is_nohit && nohit_w > 0.0 {
                        // CURE 5 PROMOTED (v9m): bypass clamp/ceiling —
                        // ordinary two-sided MSE against the honest target,
                        // always active, at explicit weight `nohit_w` (IRON
                        // param, default 1.0 = v9k/v9j's prior full-weight
                        // behavior byte-identical). Dead when gate_on (v9n).
                        2.0 * nohit_w * (raw - target.data[i]) / n_elems
                    } else if active {
                        2.0 * diff / n_elems
                    } else {
                        2.0 * overshoot_w * (raw - cap) / n_elems
                    };
                    // V9X: display_grad is needed either under the plain
                    // GAIA_V9W_LOSS_DOMAIN=display switch, or under mixing
                    // whenever mix>0.0 (mix==0.0 is pure classic_grad, skip
                    // the extra work). mix_on OVERRIDES/IGNORES
                    // display_domain for the FINAL value below — this bool
                    // only decides whether display_grad needs computing.
                    let need_display_grad = display_domain || (mix_on && mix > 0.0);
                    d_out.data[i] = if gated_nohit {
                        // V9N HIT-GATE (architectural cure, module doc): no
                        // grad reaches net_out on a gated no-hit px — EXACT
                        // zero (this px's composed output does not depend on
                        // raw at all), not an epsilon/small-weight side term.
                        0.0
                    } else if need_display_grad {
                        // V9W (module doc): SAME branch shape as below
                        // (CURE5 bypass / CURE4 active / CURE4 overshoot),
                        // just evaluated on dl->display-transformed values,
                        // chain-ruled back to d(loss)/d(raw_dl) via
                        // `ddisplay_ddl`. `active`/`is_nohit` (computed in
                        // the ORIGINAL dl domain, above) stay the correct
                        // gate: `dl_to_display` is monotonic nondecreasing,
                        // so it preserves both `<=` comparisons and `min`
                        // (f(min(a,b))==min(f(a),f(b))) — no separate
                        // recomputation needed.
                        let divisor_c = divisor_arr[c];
                        let raw_disp = dl_to_display(raw, divisor_c);
                        let target_disp = dl_to_display(target.data[i], divisor_c);
                        let ddl = ddisplay_ddl(raw, divisor_c);
                        let display_grad = if is_nohit && nohit_w > 0.0 {
                            2.0 * nohit_w * (raw_disp - target_disp) * ddl / n_elems
                        } else if active {
                            2.0 * (raw_disp - target_disp) * ddl / n_elems
                        } else {
                            let cap_disp = dl_to_display(cap, divisor_c);
                            2.0 * overshoot_w * (raw_disp - cap_disp) * ddl / n_elems
                        };
                        // V9X (task mandate): mix_on is the override — once
                        // set, the blend ALONE decides the value (ignores
                        // display_domain even if also set: mix=0.0 here
                        // still resolves to classic_grad via the weighted
                        // sum). mix_on==false takes the ORIGINAL v9w path
                        // byte-identical (display_grad alone).
                        if mix_on {
                            mix * display_grad + (1.0 - mix) * classic_grad
                        } else {
                            display_grad
                        }
                    } else {
                        classic_grad
                    };
                    // V9R DARK-HIT LOSS WEIGHT (module doc): reweights
                    // whatever grad shape was just picked above — no new
                    // formula, default 1.0 is a byte-identical no-op. Never
                    // touches the gated-nohit branch's exact-zero (dark-hit
                    // px are never `is_nohit`, `is_dark_hit` requires
                    // `!is_nohit` by construction — see V9O above).
                    if is_dark_hit_all[px] && dark_w != 1.0 {
                        d_out.data[i] *= dark_w;
                    }
                }
            }
            epoch_mse += mse;
            n_px_total += out.data.len() as u64;
            if first_batch {
                first_batch = false;
                eprintln!(
                    "[{run_tag}] FIRST BATCH mask check (epoch {epoch}, pi={pi} s={s}): nohit_px(albedo)={batch_nohit_px}/{} nohit_effective(differs-from-CURE4)={batch_nohit_effective} dark_alb_px(v9o degenerate-demod HIT px, dark_alb={dark_alb})={batch_dark_alb_px} dark_capped_px(v9p cap applied, delta={dark_cap_delta})={batch_dark_capped_px} dark_clamped(v9p cap actually fired)={batch_dark_clamped} despeckle_px(v9q post-compose cap applied, delta={despeckle_delta})={batch_despeckle_px} despeckle_clamped(v9q cap actually fired)={batch_despeckle_clamped} | cross-check vs depth-hit-flag: false_hit_dark(albedo-mask says nohit, depth says REAL HIT)={batch_false_hit_dark} false_miss_albedo(depth says miss, albedo-mask missed it — should be 0)={batch_false_miss_albedo}",
                    out.h * out.w
                );
                std::io::stderr().flush().ok();
                // V9K CROP AMENDMENT: with cropped pool draws, a SINGLE
                // frame legitimately can land entirely on geometry (no sky/
                // no-hit pixel at all — the crop dry-run itself measured
                // this, trial 1: nohit_px=0 for one draw while trial 0 and
                // the aggregate both fire) — that is real per-crop variance,
                // NOT a configuration bug, so a single-batch zero no longer
                // hard-aborts (v9j's own per-batch abort here assumed every
                // frame was a full 128x72 view, near-guaranteed some sky —
                // false for a small crop of a bigger frame). The real
                // no-silent-cure gate moves to the FIRST EPOCH's AGGREGATE
                // count (12 frames: 3 pool draws + mirror, x K=3 steps),
                // checked below after the epoch completes — a mask that
                // fires zero times across an ENTIRE epoch remains a loud,
                // exit(1) configuration bug.
                if batch_nohit_px == 0 {
                    eprintln!("[{run_tag}] NOTE: this single first batch had zero no-hit pixels (crop landed entirely on geometry) — not aborting, waiting for the first EPOCH's aggregate count below.");
                    std::io::stderr().flush().ok();
                }
                // V9W NO-SILENT-CURE NUMERIC CHECK (task mandate): 10
                // (px, channel) samples off THIS real batch's own net output
                // + albedo (not synthetic values) — finite-difference
                // `dl_to_display` against the analytic `ddisplay_ddl` used
                // for backprop above. Runs when GAIA_V9W_LOSS_DOMAIN=display
                // OR (V9X) mix_on with mix>0.0 — both paths touch
                // `ddisplay_ddl`/`dl_to_display`, old plain-classic path
                // (display_domain false, mix off or mix==0.0) never does,
                // nothing to check there. Aborts (exit 1) if max rel error
                // > 1e-3 — a wrong chain rule must never train silently.
                if display_domain || (mix_on && mix > 0.0) {
                    let mut max_rel_err = 0.0f32;
                    let mut worst_px = 0usize;
                    let mut worst_c = 0usize;
                    let mut worst_analytic = 0.0f32;
                    let mut worst_fd = 0.0f32;
                    let mut worst_abs_err = 0.0f32;
                    let mut worst_h = 0.0f32;
                    let mut check_rng = Rng(0xd15a_10c1_c0de_dddd ^ (epoch as u64));
                    let mut n_checked = 0u32;
                    let mut n_skipped_seam = 0u32;
                    let mut n_skipped_tiny = 0u32;
                    let mut attempts = 0u32;
                    let mut any_sample_fail = false;
                    // V9W2 POSTMORTEM (v9w aborted ep24 on ONE round: rel_err
                    // 0.001041 at analytic=0.135311 fd=0.135452 — an O(1)
                    // slope, not a tiny-derivative or huge-dl case). Offline
                    // reproduction (2M random (dl in [-2,12], divisor_c in
                    // [0.001,5]) draws, seams excluded by the OLD fixed
                    // margin): the OLD check (h=1e-4 fixed) false-fails
                    // 2748/2M (0.137%) draws from f32 CATASTROPHIC
                    // CANCELLATION alone — h=1e-4 is too small relative to
                    // f32 epsilon once f_plus/f_minus are O(0.1-1) display
                    // values, so `(f_plus-f_minus)` loses precision before
                    // the /(2h) amplifies it back into a derivative; worst
                    // observed 0.35% rel err, same order as the ep24 sample
                    // — matches to well within noise, NOT a chain-rule
                    // branch miss (24 prior epochs PASSed at 3e-4..6e-4,
                    // consistent with clean-gradient noise, not a
                    // systematic bug). Fix (ill-conditioning, not the
                    // gradient): (1) h now SCALES to the sample itself,
                    // h=max(1e-3*|dl|, 1e-4) — large enough at O(1) dl to
                    // keep f32 rounding negligible, still tiny near dl=0
                    // where truncation error would otherwise dominate;
                    // seam margin (8h) scales with it so seam exclusion
                    // stays proportionate; (2) samples with |analytic|<1e-6
                    // are skipped (a near-flat slope makes rel_err's own
                    // denominator meaningless); (3) a sample only counts as
                    // a FAILURE if BOTH rel_err>1e-3 AND abs_err>1e-5 (abs
                    // floor filters the exact rounding-noise band measured
                    // above, ~1e-6..1e-5, while a real chain-rule bug's
                    // abs_err is orders of magnitude larger); (4) the abort
                    // itself now requires 2 CONSECUTIVE failing ROUNDS
                    // (V9W_NUMERIC_CHECK_CONSEC_FAILS, reset on any PASS) —
                    // a lone noisy round logs FAIL and continues instead of
                    // killing a 24-epoch-clean run. Reverified offline with
                    // this exact scheme: 0/2M false positives (same sweep
                    // that broke the old check). `dl_to_display` still has
                    // its three intentional value-continuous/slope-
                    // discontinuous seams (dl==0 expm1 clamp, sRGB
                    // breakpoint, display clamp at linear==1.0) — excluded
                    // by the (now scaled) margin exactly as before.
                    while n_checked < 10 && attempts < 2000 {
                        attempts += 1;
                        let cpx = (check_rng.next() as usize) % n_px;
                        let cc = (check_rng.next() as usize) % out.c;
                        let raw_v = out.data[cpx * out.c + cc];
                        let div3 = demod_divisor(albedo[cpx]);
                        let divisor_c = match cc { 0 => div3.x, 1 => div3.y, _ => div3.z };
                        let h = (1.0e-3f32 * raw_v.abs()).max(1.0e-4);
                        let seam_dl = [
                            0.0f32,
                            (1.0 + 0.003_130_8 / divisor_c).ln(),
                            (1.0 + 1.0 / divisor_c).ln(),
                        ];
                        let margin = 8.0 * h;
                        if seam_dl.iter().any(|s| (raw_v - s).abs() < margin) {
                            n_skipped_seam += 1;
                            continue;
                        }
                        let analytic = ddisplay_ddl(raw_v, divisor_c);
                        if analytic.abs() < 1.0e-6 {
                            n_skipped_tiny += 1;
                            continue;
                        }
                        n_checked += 1;
                        let f_plus = dl_to_display(raw_v + h, divisor_c);
                        let f_minus = dl_to_display(raw_v - h, divisor_c);
                        let fd = (f_plus - f_minus) / (2.0 * h);
                        let abs_err = (analytic - fd).abs();
                        let denom = analytic.abs().max(fd.abs()).max(1.0e-8);
                        let rel_err = abs_err / denom;
                        if rel_err > 1.0e-3 && abs_err > 1.0e-5 {
                            any_sample_fail = true;
                        }
                        if rel_err > max_rel_err {
                            max_rel_err = rel_err;
                            worst_px = cpx;
                            worst_c = cc;
                            worst_analytic = analytic;
                            worst_fd = fd;
                            worst_abs_err = abs_err;
                            worst_h = h;
                        }
                    }
                    let round_failed = n_checked >= 10 && any_sample_fail;
                    let verdict = if n_checked < 10 { "INCONCLUSIVE(too few non-seam/non-tiny samples)" } else if round_failed { "FAIL" } else { "PASS" };
                    eprintln!(
                        "[{run_tag}] V9W NUMERIC CHECK (finite-diff vs analytic ddisplay_ddl, {n_checked}/10 (px,channel) samples off this real batch, {n_skipped_seam} skipped as seam-adjacent, {n_skipped_tiny} skipped as |analytic|<1e-6, h scaled per-sample max(1e-3*|dl|,1e-4)): max_rel_err={max_rel_err:.6} worst_px={worst_px} worst_c={worst_c} analytic={worst_analytic:.6} fd={worst_fd:.6} abs_err={worst_abs_err:.6} h={worst_h:.6} — {verdict}"
                    );
                    std::io::stderr().flush().ok();
                    if n_checked >= 10 {
                        if round_failed {
                            let now_consec = V9W_NUMERIC_CHECK_CONSEC_FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            eprintln!("[{run_tag}] V9W NUMERIC CHECK: round FAILED (consecutive={now_consec}/2, mixed tolerance rel_err>1e-3 AND abs_err>1e-5) — aborting only on 2 in a row (no-silent-cure law kept, single-round noise no longer fatal).");
                            std::io::stderr().flush().ok();
                            if now_consec >= 2 {
                                eprintln!("[{run_tag}] ABORT: V9W display-domain gradient finite-difference check FAILED 2 consecutive rounds — chain rule is wrong, refusing to train on a broken gradient (no-silent-cure law).");
                                std::io::stderr().flush().ok();
                                std::process::exit(1);
                            }
                        } else {
                            V9W_NUMERIC_CHECK_CONSEC_FAILS.store(0, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
            let mut grads = net.zeros_like();
            net.backward(&cache, &d_out, &mut grads);
            adam.step(&mut net, &grads);
            ema.ema_update(&net, ema_decay);
        }
        let batch_ms = t_batch.elapsed().as_secs_f64() * 1000.0;

        let mse_mean = epoch_mse / (n_px_total.max(1) as f64);
        let nohit_pct = 100.0 * epoch_nohit_px as f64 / epoch_total_px.max(1) as f64;
        // V9N no-silent-cure print: gated-px count reuses `epoch_nohit_px`
        // (gate_on ⇒ every no-hit px this epoch was gated by construction;
        // gate off ⇒ 0, matching the byte-identical old path).
        let epoch_gated_px = if hitgate_on() { epoch_nohit_px } else { 0 };
        // V9P no-silent-cure print: `epoch_dark_capped_px` (px given the
        // cap treatment) and `epoch_dark_clamped` (channel-instances where
        // the cap actually fired, raw>cap) are real per-batch tallies now,
        // not a reuse of `epoch_dark_alb_px` (v9o's old dark_gated_px did
        // that, since v9o's dark branch had no notion of "actually fired" —
        // it hard-replaced every channel unconditionally). 0 whenever
        // gate_on is off or `GAIA_V9P_DARK_CAP_DELTA<=0.0` (v9n behavior).
        // V9V no-silent-cure print: input_clamp_taps (channel-instances
        // actually pulled down this epoch, 0 whenever GAIA_V9V_INPUT_CLAMP
        // <=0.0) and mean_abs_mv (pixels, 0 whenever GAIA_V9V_MOTION=0 —
        // proves the motion lever is genuinely feeding the net non-~0
        // reprojected motion, not silently still zero).
        let epoch_clamp_taps = input_clamp_taps_snapshot() - epoch_clamp_taps_start;
        let epoch_mean_abs_mv = if epoch_mv_px > 0 { epoch_mv_abs_sum / epoch_mv_px as f64 } else { 0.0 };
        println!(
            "[{run_tag}] epoch {}/{} n2n_mse={:.6} lr={:.6} pool_yaws=[{yaw_a1:.1},{yaw_a2:.1},{yaw_b:.1}] pool_ms={pool_ms:.0} hist_ms={hist_ms:.0} batch_ms={batch_ms:.0} nohit_px={epoch_nohit_px}/{epoch_total_px}({nohit_pct:.2}%) nohit_effective={epoch_nohit_effective} hitgate={} gated_px={epoch_gated_px} dark_alb_px={epoch_dark_alb_px} dark_capped_px={epoch_dark_capped_px} dark_clamped={epoch_dark_clamped} despeckle_px={epoch_despeckle_px} despeckle_clamped={epoch_despeckle_clamped} input_clamp_taps={epoch_clamp_taps} mean_abs_mv={epoch_mean_abs_mv:.4} ({:.1}s)",
            epoch, epochs, mse_mean, adam.lr(), hitgate_on(), t_train.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().ok();
        // FIRST-EPOCH AGGREGATE STARTUP ABORT (v9k crop amendment, see the
        // per-batch NOTE above): the real no-silent-cure gate. A single crop
        // draw can legitimately miss every no-hit pixel; a WHOLE epoch
        // (every pool draw + mirror, every unroll step) missing them all
        // cannot be crop variance — it's a configuration bug (wrong field/
        // threshold/resolution), exactly like v9j's original gate intended.
        if epoch == 0 && epoch_nohit_px == 0 {
            eprintln!("[{run_tag}] STARTUP ABORT: CURE 5's mask matched ZERO pixels across the ENTIRE first epoch ({epoch_total_px} pixel-instances, every pool draw + mirror, every unroll step) — a cure that fires zero times is a configuration bug, never a silent no-op. Check nohit_albedo_sq/GAIA_V9I_NOHIT_ALBEDO_SQ and the albedo AOV field.");
            std::io::stderr().flush().ok();
            std::process::exit(1);
        }

        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            let (_, _, _, resid_abort) = run_monitor(&run_tag, &format!("epoch {epoch}"), &ema, &val_seq, &mirror_val_seq, sky_reject, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath_best, &mut score_streak, score_streak_param, &mut resid_best, &mut resid_streak, resid_streak_param, !best_metric_is_bar);

            // module doc change 2 — unconditional last-epoch checkpoint.
            let last_bytes = serialize_weights(&ema);
            std::fs::write(&wpath_last, &last_bytes).unwrap();

            let mut probe_abort = false;
            if probe_every > 0 && (epoch + 1) % probe_every == 0 {
                let t_probe = Instant::now();
                let (psp, prs, phl) = bar_res_probe(&device, &queue, &base_tris, &scene, &val_cam, &ema, sky_reject, eval_w, eval_h, eval_k, eval_ref, highlight_pctl);
                let probe_pass = psp < spark_target as f64 && prs < resid_gate as f64;
                eprintln!(
                    "[{run_tag}] PROBE epoch {epoch}: bar-res({eval_w}x{eval_h}) sparkle {psp:.1}/Mpx resid {prs:.4} highlight_ratio {phl:.3} (tgt sp<{spark_target} resid<{resid_gate}){} took {:.1}s",
                    if probe_pass { " PASS" } else { "" }, t_probe.elapsed().as_secs_f64(),
                );
                std::io::stderr().flush().ok();

                // V9Z GAIA_V9_BEST_METRIC=bar selection: best re-evaluated
                // ONLY here (a PROBE epoch), never by the 128x72 run_monitor
                // call above. Lexicographic: legal (sparkle<spark_target)
                // always beats illegal; among legal, lowest resid wins;
                // among illegal (no legal candidate seen yet), lowest
                // sparkle wins. No-op (byte-identical) when best_metric is
                // 'score' (default).
                if best_metric_is_bar {
                    let probe_legal = psp < spark_target as f64;
                    let is_new_bar_best = if probe_legal && !bar_best_legal {
                        true
                    } else if probe_legal {
                        prs < bar_best_resid
                    } else if !bar_best_legal {
                        psp < bar_best_sparkle
                    } else {
                        false
                    };
                    if is_new_bar_best {
                        bar_best_legal = probe_legal;
                        bar_best_resid = prs;
                        bar_best_sparkle = psp;
                        best_bytes = serialize_weights(&ema);
                        std::fs::write(&wpath_best, &best_bytes).unwrap();
                        eprintln!("[{run_tag}] PROBE epoch {epoch}: *BAR-BEST->saved legal={bar_best_legal} sparkle={psp:.1}/Mpx resid={prs:.4} -> {}", wpath_best.display());
                        std::io::stderr().flush().ok();
                    }
                }

                if psp >= spark_target as f64 {
                    if sparkle_abort_on {
                        probe_abort = true;
                    } else {
                        eprintln!("[{run_tag}] SPARKLE BREACH (observational, no abort): psp={psp:.1} >= spark_target={spark_target} at epoch {epoch} — GAIA_V9_SPARKLE_ABORT=0, training continues");
                        std::io::stderr().flush().ok();
                    }
                }
            }

            if resid_abort || probe_abort {
                stop_reason = if resid_abort && probe_abort {
                    format!("resid_streak {resid_streak}/{resid_streak_param} AND bar-res probe sparkle failed")
                } else if resid_abort {
                    format!("resid_streak {resid_streak}/{resid_streak_param} (sustained resid rise)")
                } else {
                    "bar-res probe sparkle failed the bar".to_string()
                };
                eprintln!("[{run_tag}] WATCHDOG ABORT: {stop_reason} at epoch {epoch} — stopping, keeping best+last (best score={best_score:.3})");
                std::io::stderr().flush().ok();
                break 'train;
            }
        }

        // V9Z GAIA_V9_CKPT_EVERY: unconditional periodic snapshot, decoupled
        // from monitor_every so the cadence is exact regardless of how
        // often the val monitor runs. 0 (default) never fires — IRON LAW,
        // byte-identical to v9k..v9y (no extra files, no extra I/O).
        if ckpt_every > 0 && (epoch + 1) % ckpt_every == 0 {
            let ep_n = epoch + 1;
            let ckpt_bytes = serialize_weights(&ema);
            let wpath_ckpt = data_dir.join(format!("rdirect-weights-{run_tag}-ep{ep_n}.bin"));
            std::fs::write(&wpath_ckpt, &ckpt_bytes).unwrap();
            let ckpt_net = deserialize_weights(&ckpt_bytes).expect("reload periodic ckpt");
            let wsha_ckpt = weights_sha256(&ckpt_net);
            let (cn, ct) = settle(&ema, &val_seq, sky_reject);
            let csp = sparkle_resid_per_mpx(&cn, &ct, val_seq.tw, val_seq.th);
            let crs = rmse_lin(&cn, &ct);
            let chl = highlight_ratio(&cn, &ct, highlight_pctl);
            let prov_ckpt = serde_json::json!({
                "artifact": format!("rdirect-weights-{run_tag}-ep{ep_n}.bin"), "ablation_tag": format!("{run_tag}-ep{ep_n}"),
                "weights_sha256": wsha_ckpt,
                "checkpoint_kind": format!("PERIODIC — unconditional EMA snapshot at epoch {ep_n} (GAIA_V9_CKPT_EVERY={ckpt_every}), independent of the best-checkpoint scorer (best_metric={best_metric}) — insurance against a *BEST scorer losing track of a good body mid-training (v9y sweet-zone-loss postmortem)"),
                "val_128x72_at_snapshot": { "sparkle_per_mpx": csp, "resid": crs, "highlight_ratio": chl },
                "architecture": { "kind": "v9-body shape-parametric multi-scale conv U-Net (rdirect_unet.rs), CPU-trainable twin (rdirect_unet.rs::cpu)",
                    "widths": config.widths, "n_scales": config.n_scales, "kernel": config.kernel,
                    "in_channels": config.in_channels, "out_channels": config.out_channels, "params_approx": config.approx_param_count() },
            });
            std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}-ep{ep_n}.provenance.json")), serde_json::to_string_pretty(&prov_ckpt).unwrap()).unwrap();
            eprintln!("[{run_tag}] CHECKPOINT ep{ep_n}: wrote {} sha256={wsha_ckpt} val128 sparkle={csp:.1}/Mpx resid={crs:.4} highlight_ratio={chl:.3} (ckpt_every={ckpt_every})", wpath_ckpt.display());
            std::io::stderr().flush().ok();
        }
    }
    if stop_reason == "wall/epoch budget exhausted" {
        eprintln!("[{run_tag}] training completed all {epochs} epochs without watchdog abort (best score={best_score:.3})");
    }
    eprintln!("[{run_tag}] training done in {:.1}s (best score={best_score:.3})", t_train.elapsed().as_secs_f64());
    std::io::stderr().flush().ok();

    // Final unconditional last-epoch snapshot (covers a WALL-budget/natural-
    // epoch-limit exit that didn't already hit a monitor call this instant).
    let last_bytes = serialize_weights(&ema);
    std::fs::write(&wpath_last, &last_bytes).unwrap();

    std::fs::write(&wpath_best, &best_bytes).unwrap();
    let best_net = deserialize_weights(&best_bytes).expect("reload best");
    let last_net = deserialize_weights(&last_bytes).expect("reload last");
    let wsha_best = weights_sha256(&best_net);
    let wsha_last = weights_sha256(&last_net);
    println!("[{run_tag}] wrote {} sha256={wsha_best}", wpath_best.display());
    println!("[{run_tag}] wrote {} sha256={wsha_last}", wpath_last.display());
    std::io::stdout().flush().ok();

    let common_training = serde_json::json!({
        "resolution_train": [tw, th], "epochs_requested": epochs, "unroll_steps": k, "lr0": lr0,
        "ref_frames_validator_only": ref_frames,
        "v9u_converged_teacher": { "teacher_spp": teacher_spp, "chunk": teacher_chunk, "env": "GAIA_V9_TEACHER_SPP", "note": "0 (default) = byte-identical K-draw n2n target_dl + ref_frames-spp validator teacher (v9k..v9t); >0 = target_dl AND val/mirror/bar-res-probe teachers become TRUE CONVERGED renders (rdirect_reference.rs-style chunked accumulation) at this spp — pool draws' own per-epoch teacher field stays ref_frames (unused by the training loop, left alone to protect epoch time)" },
        "init": "FRESH He-init (UnetWeights::new_random) — no conv-net analytic estimator-init exists yet, disclosed gap vs v8d's TIER1",
        "loss": format!("whole-image MSE(net(step features), mean of K={k_draws} independent draw radiances) — teacher NEVER in the loss, validator only (v8d TIER2 doctrine, ported); CURE 4 overshoot_w={overshoot_w} (above-ceiling gradient 2*overshoot_w*(raw-cap)/n for albedo>0 pixels); CURE 5 (v9i, NEW) nohit_albedo_sq={nohit_albedo_sq}: albedo~0/no-hit pixels bypass clamp/ceiling entirely, ordinary 2*(raw-target)/n always active (autopsy grad-probe-fix section)"),
        "k_draws": k_draws,
        "sky_history_reject_active": sky_reject,
        "watchdog": {
            "signal_a_resid_streak": { "param": resid_streak_param, "note": "abort if val resid stays > best_resid_ever*1.05 for this many CONSECUTIVE monitor calls — mirrors v9c/v9d/v9e's OLD score-streak shape, now on resid" },
            "signal_b_bar_res_probe": { "probe_every_epochs": probe_every, "eval_res": [eval_w, eval_h], "eval_k": eval_k, "eval_ref_frames": eval_ref, "abort_if_sparkle_at_least": spark_target, "note": "periodic rdirect_v9_eval_640.rs-equivalent measurement on the LIVE ema net, one failing probe aborts immediately" },
            "score_streak_LOG_ONLY": { "param": score_streak_param, "note": "v9c/v9d/v9e's OLD sole abort trigger (score=max(sp/40,resid/0.035) at 128x72 > best*1.05 sustained) — kept for log continuity, NO LONGER aborts training by itself; evidence this round (scratch/v9e-train.log ep51-72) showed it can detonate on a handful of coarse low-res pixels while resid/highlight_ratio/render-res sparkle all kept improving" },
        },
        "pose_diversity": { "pool_orbit_deg": pool_orbit_deg, "pool_jitter": pool_jitter, "draws_per_epoch": 3, "anchors": ["front-pivot (x2 draws/epoch)", "wide-anchor (x1 draw/epoch)"] },
        "winsorized_targets": { "winsor_k": winsor_k, "note": "CURE 3 (v9e), DEFAULT OFF this round — v9e forensics (scratch/v9e-train.log) showed it target-neutral (reproduced v9c exactly), see module doc" },
        "domain_fix": "settle()/run_monitor undo_log_demod the final step's output before every metric — v9c/v9e parity",
        "v9r_degenerate_demod_flag_channel": { "enabled": v9r_flag_on, "env": "GAIA_V9R_DEGEN_FLAG", "extra_input_channels": v9r_extra_channels(), "note": "representational cure (v9o/v9p/v9q were all loss/output-side surgery): one extra per-px input channel, 1.0 if no-hit OR dark-hit (V9O's class) else 0.0, so the net's own input marks the population whose demod divisor degenerated to identity (raw-radiance-scale taps) instead of leaving it invisible in feature space. Disabled (default) keeps in_channels byte-identical to v9k..v9q." },
        "v9r_dark_hit_loss_weight": { "dark_w": v9r_dw, "env": "GAIA_V9R_DARK_W", "note": "multiplies whatever n2n grad already reached V9O's is_dark_hit px (default 1.0 = no-op, byte-identical); independent of the flag channel above, a pure loss-side reweight like overshoot_w/nohit_w — these px already carry honest K-draw n2n targets." },
        "v9w_loss_domain": { "domain": loss_domain, "env": "GAIA_V9W_LOSS_DOMAIN", "note": "'' (default) = byte-identical demod-log MSE (v9k..v9v, IRON LAW); 'display' = CURE4/CURE5 active-MSE/overshoot shape UNCHANGED in form but evaluated on out_dl/target_dl/ceiling_dl forward-transformed through undo_log_demod->linear_to_srgb (exposure=1.0, the write_png/eval presentation chain), gradient chain-ruled back to demod-log via ddisplay_ddl; verified every epoch's first batch by a 10-sample finite-difference check (abort on >1e-3 rel error). OVERRIDDEN/IGNORED when v9x_mixed_domain_loss.mix_on is true (below)." },
        "v9x_mixed_domain_loss": { "mix_raw": v9x_mix_raw, "mix_resolved": mix, "mix_on": mix_on, "env": "GAIA_V9X_MIX", "note": "mix<0.0 (default) = feature OFF, byte-identical to the plain v9w_loss_domain branch above (IRON LAW); mix in [0.0,1.0] (clamped) blends d_out=mix*display_grad+(1-mix)*classic_grad per px, both EXISTING V9W/classic gradient shapes weighted-summed (no new math), OVERRIDING/IGNORING GAIA_V9W_LOSS_DOMAIN for the purpose of picking the grad; numeric finite-diff guard on the display component stays active whenever mix>0.0, same 1e-3 rel-err abort law as v9w_loss_domain." },
        "stop_reason": stop_reason,
        "v9z_best_metric": { "metric": best_metric, "env": "GAIA_V9_BEST_METRIC", "ckpt_every": ckpt_every, "ckpt_every_env": "GAIA_V9_CKPT_EVERY", "note": "'score' (default, IRON) = run_monitor's own 128x72 *BEST->saved, byte-identical to v9k..v9y; 'bar' = bar-res PROBE-selected (legal beats illegal, lowest resid among legal, else lowest sparkle), re-evaluated only at PROBE epochs. ckpt_every>0 also wrote independent periodic rdirect-weights-{run_tag}-ep<N>.bin snapshots (N=epoch+1) regardless of metric." },
    });

    let checkpoint_kind_best = if best_metric_is_bar {
        format!("BEST-BAR — bar-res PROBE-selected (every {probe_every} epochs, GAIA_V9_BEST_METRIC=bar): legal (sparkle<{spark_target}) beats illegal, lowest resid wins among legal candidates, falls back to lowest sparkle when no legal candidate has been seen yet; final pick sparkle={bar_best_sparkle:.1}/Mpx resid={bar_best_resid:.4} legal={bar_best_legal} (v9z checkpoint-cadence fix)")
    } else {
        "BEST-SCORE — lowest 128x72 score=max(sp/40,resid/0.035) seen across all monitor calls, same convention v9..v9j used for their sole/best checkpoint".to_string()
    };
    let prov_best = serde_json::json!({
        "artifact": format!("rdirect-weights-{run_tag}.bin"), "ablation_tag": format!("{run_tag}-best"),
        "weights_sha256": wsha_best,
        "checkpoint_kind": checkpoint_kind_best,
        "supersedes": "rdirect-weights-v9j.bin (v9k = v9j mechanisms UNCHANGED [two-signal watchdog, bar-res probe every 25, dual checkpoints, masked-px counters+first-batch guard, K=8, EMA, reject=true, overshoot_w=1, CURE5 depth-keyed] + pool draws now render CROPS of a virtual 640x480 frame instead of full 128x72 frames, see module doc)",
        "architecture": { "kind": "v9-body shape-parametric multi-scale conv U-Net (rdirect_unet.rs), CPU-trainable twin (rdirect_unet.rs::cpu)",
            "widths": config.widths, "n_scales": config.n_scales, "kernel": config.kernel,
            "in_channels": config.in_channels, "out_channels": config.out_channels, "params_approx": config.approx_param_count() },
        "training": common_training,
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th], "crop_full": [crop_full_w, crop_full_h],
            "train": ["front-pivot-pool x2/epoch (CROP of virtual 640x480)", "wide-anchor-pool x1/epoch (CROP of virtual 640x480)", "mirror (fixed, unconditional, FULL-FRAME unchanged)"], "val": ["orbit_-20 (still, FIXED, held-out, FULL-FRAME)", "mirror (still, FIXED, held-out, FULL-FRAME)"] },
        "gate": "NOT ordealed — v9k output only, no bar claimed passed",
    });
    let prov_last = serde_json::json!({
        "artifact": format!("rdirect-weights-{run_tag}-last.bin"), "ablation_tag": format!("{run_tag}-last"),
        "weights_sha256": wsha_last,
        "checkpoint_kind": "LAST-EPOCH — the ema net as of the FINAL monitor call before training stopped (natural epoch limit / wall budget / watchdog abort) — v9f-introduced dual-checkpoint convention, unchanged since",
        "supersedes": "rdirect-weights-v9j-last.bin (v9k crop-training change, see prov_best.supersedes)",
        "architecture": { "kind": "v9-body shape-parametric multi-scale conv U-Net (rdirect_unet.rs), CPU-trainable twin (rdirect_unet.rs::cpu)",
            "widths": config.widths, "n_scales": config.n_scales, "kernel": config.kernel,
            "in_channels": config.in_channels, "out_channels": config.out_channels, "params_approx": config.approx_param_count() },
        "training": common_training,
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th], "crop_full": [crop_full_w, crop_full_h],
            "train": ["front-pivot-pool x2/epoch (CROP of virtual 640x480)", "wide-anchor-pool x1/epoch (CROP of virtual 640x480)", "mirror (fixed, unconditional, FULL-FRAME unchanged)"], "val": ["orbit_-20 (still, FIXED, held-out, FULL-FRAME)", "mirror (still, FIXED, held-out, FULL-FRAME)"] },
        "gate": "NOT ordealed — v9k output only, no bar claimed passed",
    });
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")), serde_json::to_string_pretty(&prov_best).unwrap()).unwrap();
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}-last.provenance.json")), serde_json::to_string_pretty(&prov_last).unwrap()).unwrap();
    println!("[{run_tag}] wrote provenance. tag={run_tag} best={} last={}", wpath_best.display(), wpath_last.display());
    std::io::stdout().flush().ok();
}
