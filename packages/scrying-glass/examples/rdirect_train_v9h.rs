//! R-DIRECT v9h-body — CURE 4 DOSE ESCALATION (mechanism hunt,
//! `scratch/v9-autopsy.md` v9g verdict section). Base: `rdirect_train_v9g.rs`
//! UNCHANGED (two-signal watchdog, dual checkpoints, periodic bar-res probe,
//! K=8, EMA, fresh init — every mechanism not listed here is byte-identical,
//! see that file's own header) — the ONLY change is the `overshoot_w`
//! DEFAULT at the loss call site below (1.0 -> 4.0).
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
    IntegratorParams, headless_device, trace_headless_aov, trace_headless_split,
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

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(n: &str, d: f32) -> f32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
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
        let mut target_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n)
            .map(|px| target_demod_log(radiance_sum[px] * inv_k, albedo[px]))
            .collect();
        winsorize_targets(&mut target_dl, tw, th, winsor_k);

        let (e_full, d_full) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
            &IntegratorParams::default(),
        );
        let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();

        steps.push(Step { low_e, low_d, albedo, normal, depth, teacher, target_dl, ceiling_dl: Vec::new() });
    }

    let gamma = evidence_clamp_gamma();
    let mut evidence_sum = vec![GVec3::ZERO; n];
    for s in &steps {
        let composite = evidence_composite_frame(&s.low_e, &s.low_d, low_w, low_h, tw, th);
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
    let mut img = Img::zeros(th as usize, tw as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS);
    for ty in 0..th {
        for tx in 0..tw {
            let px = (ty * tw + tx) as usize;
            let base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                &step.low_e, &step.low_d, seq.low_w, seq.low_h, tw, th, tx, ty,
                step.albedo[px], step.normal[px], step.depth[px], Vec2::ZERO,
            );
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
    let net_lin: Vec<GVec3> = last_dl
        .iter()
        .zip(last_step.albedo.iter())
        .map(|(dl, albedo)| undo_log_demod(*dl, demod_divisor(*albedo)))
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
        *best_bytes = serialize_weights(ema);
        std::fs::write(wpath_best, &*best_bytes).unwrap();
    }
    update_streak(score, best_score, score_streak);
    // score_streak resets to 0 on `better` too via update_streak's own
    // `metric < best_ever` branch — consistent with the OLD sole-signal
    // semantics, just no longer wired to abort.

    update_streak(rs, resid_best, resid_streak);
    let resid_should_abort = *resid_streak >= resid_streak_param;

    eprintln!(
        "[{run_tag}] MONITOR {tag}: val sparkle {sp:.1}/Mpx resid {rs:.4} highlight_ratio {hl:.3} score={score:.3}{} | mirror sparkle {msp:.1}/Mpx resid {mrs:.4} highlight_ratio {mhl:.3} (tgt sp<{spark_target} resid<{resid_gate}){} score_streak={score_streak}/{score_streak_param}(LOG-ONLY) resid_streak={resid_streak}/{resid_streak_param}(ABORT)",
        if passes { " PASS" } else { "" }, if better { " *BEST->saved" } else { "" },
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
    let (e_full, d_full) = trace_headless_split(
        device, queue, &bvh, base_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
        &IntegratorParams::default(),
    );
    let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();

    let mut probe_net = ema.clone();
    probe_net.config.render_w = tw as usize;
    probe_net.config.render_h = th as usize;
    probe_net.config.output_w = tw as usize;
    probe_net.config.output_h = th as usize;

    let mut chain_dl: Vec<Vec<GVec3>> = Vec::with_capacity(k as usize);
    let mut out_imgs: Vec<Img> = Vec::with_capacity(k as usize);
    for s in 0..k as usize {
        let mut img = Img::zeros(th as usize, tw as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS);
        for ty in 0..th {
            for tx in 0..tw {
                let px = (ty * tw + tx) as usize;
                let base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                    &steps_low_e[s], &steps_low_d[s], low_w, low_h, tw, th, tx, ty,
                    steps_albedo[s][px], steps_normal[s][px], steps_depth[s][px], Vec2::ZERO,
                );
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
    let mut net_clamped = vec![GVec3::ZERO; n];
    for px in 0..n {
        let dl = GVec3::new(last_dl.data[px * 3], last_dl.data[px * 3 + 1], last_dl.data[px * 3 + 2]);
        let divisor = demod_divisor(last_albedo[px]);
        let ceiling_dl: [f32; OUTPUT_CHANNELS] = evidence_ceiling_demod_log(evidence_ceiling[px], gamma, last_albedo[px]);
        let presented_dl = GVec3::new(dl.x.min(ceiling_dl[0]), dl.y.min(ceiling_dl[1]), dl.z.min(ceiling_dl[2]));
        net_clamped[px] = undo_log_demod(presented_dl, divisor);
    }

    let sp = sparkle_resid_per_mpx(&net_clamped, &teacher, tw, th);
    let rs = rmse_lin(&net_clamped, &teacher);
    let hl = highlight_ratio(&net_clamped, &teacher, highlight_pctl);
    (sp, rs, hl)
}

fn main() {
    let run_tag = std::env::var("GAIA_V9_TAG").unwrap_or_else(|_| "v9h".to_string());
    let score_streak_param = env_u32("GAIA_V9F_SCORE_STREAK", 20); // LOG-ONLY now (module doc change 1)
    let resid_streak_param = env_u32("GAIA_V9F_RESID_STREAK", 20); // the abort signal A
    let probe_every = env_u32("GAIA_V9F_PROBE_EVERY", 25); // module doc change 3, 0 disables
    let pool_orbit_deg = env_f32("GAIA_V9F_POOL_ORBIT_DEG", 30.0); // v9c/v9e parity
    let pool_jitter = env_f32("GAIA_V9F_POOL_JITTER", 0.6); // v9c/v9e parity
    let winsor_k = env_f32("GAIA_V9F_WINSOR_K", 0.0); // CURE 3 default OFF this round (evidence (a))
    let overshoot_w = env_f32("GAIA_V9H_OVERSHOOT_W", 4.0); // CURE 4 DOSE ESCALATION (v9h, module doc): v9g's weight=1.0 only halved the drift (bar-res sparkle 16.3 vs v9f's 22.8, still crossed the tgt<16 bar at ep99) -- 4x stronger restoring force this round
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
    eprintln!("[{run_tag}] res {tw}x{th} K={k} steps, GAIA_V9_K={k_draws} noise2noise draws (variance/{k_draws}, teacher validator-only)");
    eprintln!("[{run_tag}] WATCHDOG two-signal: score_streak={score_streak_param} LOG-ONLY, resid_streak={resid_streak_param} ABORT, bar-res probe every {probe_every} epochs at {eval_w}x{eval_h} (K={eval_k} ref={eval_ref}) ABORT on sparkle>={spark_target}");
    eprintln!("[{run_tag}] CURE 4: overshoot_w={overshoot_w} — above-ceiling gradient is 2*overshoot_w*(raw-cap)/n (pulls toward the ceiling), no longer zero (see module doc mechanism-hunt verdict)");

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
        "[{run_tag}] rendered mirror(train) + 2 VALIDATOR (orbit_-20, mirror — FIXED, held-out) pose sequences ({tw}x{th}, teacher {ref_frames} VALIDATOR ONLY, pan_step={pan_step}, mirror_spp={mirror_spp}) in {:.1}s",
        t_render.elapsed().as_secs_f64()
    );
    std::io::stderr().flush().ok();

    // V9G MECHANISM-HUNT PROBE (c): opt-in, GAIA_V9H_GRAD_PROBE=1. Loads a
    // checkpoint (GAIA_V9H_PROBE_WEIGHTS, default the v9f-last artifact),
    // runs the val_seq's (orbit_-20) recurrent chain through it exactly like
    // `history_forward`/`settle`, then at the LAST step recomputes the
    // EXACT training-loop CURE-1 formula per pixel/channel for the fixed
    // forensic texel list (`scratch/v9e-forensics.log` COORDS, same 128x72
    // space) and prints raw/cap/target/active/d_out — answers whether the
    // downward gradient is zero/starved at pixels already past the ceiling.
    // Zero effect on the normal v9f run (env unset by default).
    if std::env::var("GAIA_V9H_GRAD_PROBE").ok().as_deref() == Some("1") {
        let data_dir0 = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let wpath = std::env::var("GAIA_V9H_PROBE_WEIGHTS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| data_dir0.join("rdirect-weights-v9f-last.bin"));
        let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("read {wpath:?}: {e}"));
        let probe = deserialize_weights(&bytes).unwrap_or_else(|| panic!("deserialize {wpath:?}"));
        eprintln!("[v9h-grad-probe] loaded {wpath:?} ({} bytes)", bytes.len());
        let chain = history_forward(&probe, &val_seq, sky_reject);
        let last_s = val_seq.steps.len() - 1;
        let prev: Option<&[GVec3]> = if last_s == 0 { None } else { Some(&chain[last_s - 1]) };
        let input = build_input_img(&val_seq, last_s, prev, sky_reject);
        let (out, _cache) = probe.forward(&input);
        let step = &val_seq.steps[last_s];
        // (x,y) list from scratch/v9e-forensics.log COORDS epoch 73 (128x72 space).
        let texels: &[(u32, u32)] = &[(94, 3), (94, 6), (97, 6), (94, 8), (47, 9), (55, 34), (66, 38)];
        eprintln!("[v9h-grad-probe] val_seq {}x{}, last step {last_s}, texel raw/cap/target/active/d_out per channel:", val_seq.tw, val_seq.th);
        for &(x, y) in texels {
            let px = (y * val_seq.tw + x) as usize;
            for c in 0..OUTPUT_CHANNELS {
                let i = px * OUTPUT_CHANNELS + c;
                let cap = step.ceiling_dl[px][c];
                let raw = out.data[i];
                let target = step.target_dl[px][c];
                let presented = raw.min(cap);
                let diff = presented - target;
                let active = raw <= cap;
                let d_out = if active { 2.0 * diff / (out.data.len() as f32) } else { 0.0 };
                eprintln!(
                    "[v9h-grad-probe]   px=({x},{y}) c={c} raw={raw:.6} cap={cap:.6} target={target:.6} overshoot={} active={active} d_out={d_out:.8}",
                    raw > cap,
                );
            }
        }
        eprintln!("[v9h-grad-probe] done — exiting (no training run)");
        std::io::stderr().flush().ok();
        return;
    }

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    // IRON paths (module doc change 2): best-score convention unchanged,
    // NEW "-last" sibling for the unconditional most-recent-epoch snapshot.
    let wpath_best = data_dir.join(format!("rdirect-weights-{run_tag}.bin"));
    let wpath_last = data_dir.join(format!("rdirect-weights-{run_tag}-last.bin"));
    let config = UnetConfig { render_w: tw as usize, render_h: th as usize, output_w: tw as usize, output_h: th as usize, ..UnetConfig::default() };
    eprintln!("[{run_tag}] UnetConfig widths={:?} n_scales={} params≈{}", config.widths, config.n_scales, config.approx_param_count());

    let mut net = UnetWeights::new_random(config.clone(), seed);
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

    run_monitor(&run_tag, "epoch -1 (fresh He-init)", &ema, &val_seq, &mirror_val_seq, sky_reject, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath_best, &mut score_streak, score_streak_param, &mut resid_best, &mut resid_streak, resid_streak_param);

    let mut rng = Rng(0xd15e_ed00_08f0_0dc0 ^ seed);
    let t_train = Instant::now();
    let mut stop_reason = String::from("wall/epoch budget exhausted");

    'train: for epoch in 0..epochs {
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
        let poses: Vec<PoseSeq> = vec![
            render_pose_seq(&device, &queue, &base_tris, &scene, &draw_a1, k, low_w, low_h, tw, th, ref_frames, pan_step, 1, k_draws, winsor_k),
            render_pose_seq(&device, &queue, &base_tris, &scene, &draw_a2, k, low_w, low_h, tw, th, ref_frames, pan_step, 1, k_draws, winsor_k),
            render_pose_seq(&device, &queue, &base_tris, &scene, &draw_b, k, low_w, low_h, tw, th, ref_frames, pan_step, 1, k_draws, winsor_k),
            mirror_seq.clone(),
        ];
        let pool_ms = t_pool.elapsed().as_secs_f64() * 1000.0;

        let t_hist = Instant::now();
        let history: Vec<Vec<Vec<GVec3>>> = poses.iter().map(|seq| history_forward(&ema, seq, sky_reject)).collect();
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
        let t_batch = Instant::now();
        for &(pi, s) in &order {
            if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
                eprintln!("[{run_tag}] WALL budget {wall_budget}s reached MID-epoch {epoch} — stopping, keeping best+last");
                std::io::stderr().flush().ok();
                break 'train;
            }
            let seq = &poses[pi];
            let prev: Option<&[GVec3]> = if s == 0 { None } else { Some(&history[pi][s - 1]) };
            let input = build_input_img(seq, s, prev, sky_reject);
            let (out, cache) = net.forward(&input);
            let target = target_to_img(&seq.steps[s].target_dl, th as usize, tw as usize);
            let ceiling = &seq.steps[s].ceiling_dl;
            let n_elems = out.data.len() as f32;
            let mut d_out = Img::zeros(out.h, out.w, out.c);
            let mut mse = 0.0f64;
            for px in 0..(out.h * out.w) {
                for c in 0..out.c {
                    let i = px * out.c + c;
                    let cap = ceiling[px][c];
                    let raw = out.data[i];
                    let presented = raw.min(cap);
                    let diff = presented - target.data[i];
                    mse += (diff as f64) * (diff as f64);
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
                    d_out.data[i] = if active {
                        2.0 * diff / n_elems
                    } else {
                        2.0 * overshoot_w * (raw - cap) / n_elems
                    };
                }
            }
            epoch_mse += mse;
            n_px_total += out.data.len() as u64;
            let mut grads = net.zeros_like();
            net.backward(&cache, &d_out, &mut grads);
            adam.step(&mut net, &grads);
            ema.ema_update(&net, ema_decay);
        }
        let batch_ms = t_batch.elapsed().as_secs_f64() * 1000.0;

        let mse_mean = epoch_mse / (n_px_total.max(1) as f64);
        println!(
            "[{run_tag}] epoch {}/{} n2n_mse={:.6} lr={:.6} pool_yaws=[{yaw_a1:.1},{yaw_a2:.1},{yaw_b:.1}] pool_ms={pool_ms:.0} hist_ms={hist_ms:.0} batch_ms={batch_ms:.0} ({:.1}s)",
            epoch, epochs, mse_mean, adam.lr(), t_train.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().ok();

        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            let (_, _, _, resid_abort) = run_monitor(&run_tag, &format!("epoch {epoch}"), &ema, &val_seq, &mirror_val_seq, sky_reject, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath_best, &mut score_streak, score_streak_param, &mut resid_best, &mut resid_streak, resid_streak_param);

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
                if psp >= spark_target as f64 {
                    probe_abort = true;
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
        "init": "FRESH He-init (UnetWeights::new_random) — no conv-net analytic estimator-init exists yet, disclosed gap vs v8d's TIER1",
        "loss": format!("whole-image MSE(net(step features), mean of K={k_draws} independent draw radiances) — teacher NEVER in the loss, validator only (v8d TIER2 doctrine, ported); CURE 4 overshoot_w={overshoot_w} (module doc): above-ceiling gradient is 2*overshoot_w*(raw-cap)/n, no longer zero"),
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
        "stop_reason": stop_reason,
    });

    let prov_best = serde_json::json!({
        "artifact": format!("rdirect-weights-{run_tag}.bin"), "ablation_tag": format!("{run_tag}-best"),
        "weights_sha256": wsha_best,
        "checkpoint_kind": "BEST-SCORE — lowest 128x72 score=max(sp/40,resid/0.035) seen across all monitor calls, same convention v9/v9b/v9c/v9d/v9e used for their sole checkpoint",
        "supersedes": "rdirect-weights-v9e.bin (v9f = v9c/v9e mechanisms UNCHANGED + two-signal watchdog + dual checkpoints + periodic bar-res probe, see module doc)",
        "architecture": { "kind": "v9-body shape-parametric multi-scale conv U-Net (rdirect_unet.rs), CPU-trainable twin (rdirect_unet.rs::cpu)",
            "widths": config.widths, "n_scales": config.n_scales, "kernel": config.kernel,
            "in_channels": config.in_channels, "out_channels": config.out_channels, "params_approx": config.approx_param_count() },
        "training": common_training,
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": ["front-pivot-pool x2/epoch", "wide-anchor-pool x1/epoch", "mirror (fixed, unconditional)"], "val": ["orbit_-20 (still, FIXED, held-out)", "mirror (still, FIXED, held-out)"] },
        "gate": "NOT ordealed — v9f output only, no bar claimed passed",
    });
    let prov_last = serde_json::json!({
        "artifact": format!("rdirect-weights-{run_tag}-last.bin"), "ablation_tag": format!("{run_tag}-last"),
        "weights_sha256": wsha_last,
        "checkpoint_kind": "LAST-EPOCH — the ema net as of the FINAL monitor call before training stopped (natural epoch limit / wall budget / watchdog abort) — NEW this round (module doc change 2): since v9f trains through the low-res score detonation on purpose, the run's true end state needs its own artifact distinct from whichever low-res epoch happened to have the best low-res score",
        "supersedes": "n/a — new artifact kind, no prior -last checkpoint existed",
        "architecture": { "kind": "v9-body shape-parametric multi-scale conv U-Net (rdirect_unet.rs), CPU-trainable twin (rdirect_unet.rs::cpu)",
            "widths": config.widths, "n_scales": config.n_scales, "kernel": config.kernel,
            "in_channels": config.in_channels, "out_channels": config.out_channels, "params_approx": config.approx_param_count() },
        "training": common_training,
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": ["front-pivot-pool x2/epoch", "wide-anchor-pool x1/epoch", "mirror (fixed, unconditional)"], "val": ["orbit_-20 (still, FIXED, held-out)", "mirror (still, FIXED, held-out)"] },
        "gate": "NOT ordealed — v9f output only, no bar claimed passed",
    });
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")), serde_json::to_string_pretty(&prov_best).unwrap()).unwrap();
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}-last.provenance.json")), serde_json::to_string_pretty(&prov_last).unwrap()).unwrap();
    println!("[{run_tag}] wrote provenance. tag={run_tag} best={} last={}", wpath_best.display(), wpath_last.display());
    std::io::stdout().flush().ok();
}
