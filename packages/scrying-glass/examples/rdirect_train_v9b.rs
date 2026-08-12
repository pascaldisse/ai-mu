//! R-DIRECT v9b-body — CURED ROUND (FINISH ATOM step 3, `scratch/v9-autopsy.md`):
//! v9 (STAGE 2, `rdirect_train_v9.rs`) DETONATED — healthy through epoch
//! ~11 (resid 0.1387 falling, *BEST) then val sparkle ran away monotonically
//! to 51323/Mpx by epoch 258 while the train loss (n2n_mse) kept FALLING
//! (0.023) — a classic unclamped-output train/val divergence, convicted by
//! the autopsy against the trainer's own disclosed gap: "no evidence-clamp
//! ceiling term ... `cpu::UnetWeights` has no clamp act wired yet". This
//! file is v9 UNCHANGED (same mechanisms 1-4, same recipe, FRESH init —
//! never fine-tuned from v9's best, 5/5 law) plus TWO cures, both additive
//! at the TRAINING-LOOP call site (`rdirect_unet.rs::cpu` itself is NOT
//! touched — no clamp act inside the net, the clamp lives in how the loss
//! gradient is computed, exactly where v8d's own
//! `accumulate_backward_clamped_slice` puts it):
//!
//! CURE 1 — INFERENCE-CONSISTENT EVIDENCE CLAMP INSIDE THE LOSS. Per pose,
//! a temporal-mean evidence ceiling (`evidence_composite_frame` +
//! `local_max_3x3`, `rdirect.rs`, v7's OWN structural anti-sparkle
//! mechanism) is precomputed ONCE over that pose's K rendered steps'
//! low_e/low_d (the same evidence source the net's own input features
//! already read — no new render pass), converted to demod-log
//! (`evidence_ceiling_demod_log`, gamma=1.5 `evidence_clamp_gamma()`
//! default) and stored per `Step`. During training, each pixel's gradient
//! is GATED exactly like v8d's `accumulate_backward_clamped_slice`:
//! `presented = min(out_dl, ceiling_dl)`, loss = (presented-target)^2, and
//! the backprop delta is ZERO wherever `out_dl > ceiling_dl` (the net gets
//! no gradient signal to push EVEN HIGHER once it already exceeds what the
//! evidence supports — the exact mechanism the autopsy found absent).
//! Disclosed simplification vs v8d: v8d's ceiling accumulates
//! INCREMENTALLY frame-by-frame during a strictly-sequential recurrent
//! settle; v9's image-level batching shuffles (pose,step) pairs across a
//! whole epoch, so there is no single "we are mid-sequence" running state
//! to accumulate into — this file instead accumulates each pose's FULL
//! K-step temporal mean once (before training starts) and reuses that same
//! ceiling for every step of that pose, every epoch. Approximates the same
//! contract (temporal-mean, spatially 3x3-pooled evidence ceiling) without
//! requiring sequential batch order.
//!
//! CURE 2 — ABORT-ON-DETONATION WATCHDOG. `run_monitor` now tracks a
//! rising-streak counter: any epoch whose score exceeds the best-ever score
//! by >5% extends the streak, anything else (an improvement OR a flat/noisy
//! epoch within 5% of best) resets it to 0. `GAIA_V9B_ABORT_STREAK`
//! (default 20) consecutive worsening epochs stops training immediately
//! (keeping the best checkpoint already on disk) instead of burning the
//! rest of the wall budget inside a runaway — the autopsy's own v9 log
//! shows score rising for 250+ consecutive epochs past the last *BEST
//! without ever recovering, so 20 is a conservative early trip relative to
//! that observed runaway length.
//!
//! Everything below is otherwise `rdirect_train_v9.rs` (the v7/v8 lineage's
//! OWN training recipe; every mechanism has a v7/v8 file it is
//! copied/adapted from, cited inline) — this file is NOT a new trainer
//! design, it is v9 plus the two cures above.
//!
//! WHY A NEW BODY (recap, HANDOFF.md v8d CAPACITY VERDICT, sealed 07-21):
//! the per-pixel `rdirect::Mlp` has no spatial receptive field and cannot
//! reach resid bar 0.035 (v8d plateaued flat at 0.0416 — a body-capacity
//! ceiling, not a training-recipe problem). `rdirect_unet.rs` STAGE 1
//! (392a7ff7) built the conv shape + a perf spike
//! (`docs/perf/2026-07-21-v9-spike.md`: C-med widths=[24,40,64] GPU-only
//! 4.29ms, well under the 9.07ms budget — chosen as `UnetConfig::default()`).
//! This file is STAGE 2: wire that body into the v7/v8 TRAINING RECIPE
//! (moving-camera history with REAL reprojection, curved-mirror/low-
//! roughness pose, K-averaged noise2noise targets, the −38.7% highlight fix)
//! using the CPU-trainable twin (`rdirect_unet.rs::cpu`, this lane's other
//! salvaged commit) since the live MPSGraph graph has no gradient path (the
//! same house split `rdirect.rs::Mlp` / `rdirect_live.rs` already uses).
//!
//! ── mechanisms, each an EXPLICIT port from v8d (`rdirect_train_v8d.rs`) ──
//!
//! 1. MOVING-CAMERA HISTORY, REAL REPROJECTION (not identity feedback).
//!    `render_pose_seq`/`reproject_prev` below are the v8d functions
//!    UNCHANGED in their math (byte-identical `CamPose::reproject` call,
//!    same depth/normal guard, same `sky_history_reject()`), only the
//!    CONSUMER changed: v8d's `history_forward` ran `Mlp::forward` per
//!    pixel; this file's `history_forward` runs `UnetWeights::forward` ONCE
//!    per whole image (the receptive field the conv body exists for) and
//!    slices the resulting `Img` back into a `Vec<GVec3>` chain for the next
//!    step's per-pixel reprojection sampling — same recurrent contract
//!    (EMA-sourced chain, kept from v8/cf8bd7b), different granularity.
//!
//! 2. CURVED-MIRROR / LOW-ROUGHNESS POSE. `denoiser_dataset::mirror_camera`
//!    at `mirror_spp`=4 evidence — the SAME pose v8/v8c/v8d train on
//!    unconditionally (ablation A3: proven innocent, best-behaved of the
//!    three v8-ablation arms, `scratch/v8-ablate-A3.log`). No new pose
//!    invented here.
//!
//! 3. K-AVERAGED NOISE2NOISE TARGETS (v8d TIER 2, K=8 default, teacher
//!    DEMOTED TO VALIDATOR ONLY). `render_pose_seq` traces `k_draws`
//!    independent native-res evidence draws per step (seed family
//!    `DRAW_B_SEED_BASE`, disjoint from the net's own input draw
//!    `DRAW_A_SEED_BASE`) and averages their raw per-pixel radiance into
//!    `target_dl` — unbiased (E[mean of K iid draws]=truth), variance/K.
//!    The converged teacher (`ref_frames` frames) is rendered ONLY for
//!    `run_monitor`'s sparkle/resid/highlight_ratio reporting and the
//!    score-floor gate; `accumulate_backward` (this file's hand-rolled
//!    conv loss, see below) NEVER reads it — copied verbatim from v8d's
//!    TIER 2 doctrine (NEURAL.md §TRAINING DOCTRINE, 07-20 enforcement).
//!
//! 4. HIGHLIGHT-RECOVERY (the gamma-sweep's −38.7% highlight under-render,
//!    `scratch/v7-live-lane.md:1645`, HANDOFF.md:794/801/848/916). THIS IS
//!    NOT A SEPARATE LOSS TERM — v8c's own header is explicit that TIER 2's
//!    K-averaged noise2noise target IS the doctrine-compliant fix for this
//!    gap ("directly attacks the gamma-sweep's -38.7% highlight
//!    under-render... by giving the net an UNBIASED highlight signal on
//!    every pixel every epoch, instead of a spatially-biased oversampling
//!    hack" — v8's OWN `GAIA_V8_HIGHLIGHT_FRAC` sampling-side hack was
//!    A2-convicted as a runaway detonator in `scratch/v8-ablate-A2.log` and
//!    dropped entirely in v8c/v8d). This file inherits that verdict:
//!    mechanism 3 above IS mechanism 4; `run_monitor`'s `highlight_ratio`
//!    reports the gap as a measured number every epoch (validator-only,
//!    never fed back into sampling or the loss) so the claim stays honest,
//!    not asserted.
//!
//! WHAT IS DIFFERENT FROM v8d (the conv-specific additions):
//!   - Per-pixel `hist_features_split` (39-wide) is evaluated at EVERY
//!     pixel of the WHOLE training-res image to build one `Img` per step
//!     (image channels, not a flat per-pixel vector — see
//!     `rdirect_unet.rs` module docs) plus `MOTION_VECTOR_CHANNELS` (2) of
//!     REAL reprojected screen-space motion (v7's own `hi_motion` slot
//!     inside the 35-feature base was always fed `Vec2::ZERO` — never
//!     computed; this file computes the real thing for the 2 NEW v9
//!     channels only, leaving the legacy 35-feature slot's own motion pair
//!     at zero for exact byte-parity with v7/v8's feature contract).
//!   - Loss/optimizer: `cpu::UnetWeights::backward` + `cpu::UnetAdam`
//!     (whole-image MSE vs the K-averaged target `Img`, plain MSE — same
//!     loss SHAPE as v8d's `accumulate_backward_clamped_slice`, no
//!     evidence-clamp ceiling term: `cpu::UnetWeights` has no clamp act
//!     wired yet, a disclosed, narrower scope than v8d's per-pixel net;
//!     flagged, not hidden).
//!   - Batch granularity: v8d subsamples 5000px/pose/epoch, batch=64px.
//!     A whole-image conv forward+backward already touches every pixel at
//!     once, so the natural "batch" here is IMAGES: one Adam step per
//!     (pose, K-step) image, all K*n_poses images visited once per epoch
//!     (shuffled) — the image-level analogue of v8d's pixel minibatching.
//!   - Training resolution defaults SMALL (`GAIA_V9_W/H`, default 128x72)
//!     versus v8d's 384x288 — `rdirect_unet.rs`'s own module doc: "conv
//!     weights are resolution-independent... training resolution is just
//!     `UnetConfig::render_w/h` set smaller for CPU wall budget". This
//!     file's CPU forward/backward has no GPU acceleration (that is what
//!     STAGE 1's `imp::UnetLive` MPSGraph path is for, at inference time,
//!     not here); the resolution choice is a wall-clock necessity, not an
//!     architecture change — a checkpoint trained here forwards correctly
//!     at any `render_w/h` later (fully-convolutional, IRON: no literal
//!     dimension in the graph builder).
//!
//! Run: cargo run --release -j2 --example rdirect_train_v9
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

const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;
const DRAW_A_SEED_BASE: u32 = 0x7abc; // v8d parity — net's own input evidence
const DRAW_B_SEED_BASE: u32 = 0xB222; // v8d parity — noise2noise label draws
// DOMAIN FIX (autopsy conviction #2, scratch/v9-autopsy.md): the net's raw
// output is demod-log radiance, not linear — `settle()` must undo_log_demod
// the FINAL step's output (using that pixel's albedo divisor) before ever
// comparing it against the linear teacher, exactly like v8d's own inference
// path (`rdirect.rs::direct_render_sequence_hist_split`). Duplicated here
// (both fns are private in `rdirect.rs`), same as `rdirect_v9_eval_640.rs`.
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

/// One step of a moving-camera pose sequence — v8d's `Step`, PLUS CURE 1's
/// evidence-clamp ceiling (`ceiling_dl`, precomputed once per pose — see
/// module doc CURE 1 for the per-pose-temporal-mean simplification vs
/// v8d's per-step-incremental accumulator).
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

struct PoseSeq {
    steps: Vec<Step>,
    cams: Vec<CamPose>,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
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

        // v8d TIER 2 — K-averaged noise2noise label (mechanism 3+4, see
        // module doc): K independent native-res draws, disjoint seed family,
        // running-sum only (no per-draw buffers retained).
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
        let target_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n)
            .map(|px| target_demod_log(radiance_sum[px] * inv_k, albedo[px]))
            .collect();

        let (e_full, d_full) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
            &IntegratorParams::default(),
        );
        let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();

        // CURE 1 placeholder — filled in the second pass below once every
        // step's evidence composite is known (temporal mean over the WHOLE
        // pose, not incremental — see module doc CURE 1).
        steps.push(Step { low_e, low_d, albedo, normal, depth, teacher, target_dl, ceiling_dl: Vec::new() });
    }

    // CURE 1: per-pose temporal-mean evidence ceiling (v7's structural
    // anti-sparkle mechanism, `evidence_composite_frame` + `local_max_3x3`,
    // `rdirect.rs`) — folded across ALL of this pose's K steps' own input
    // evidence (`low_e`/`low_d`, the SAME draw the net's own features read),
    // then converted to demod-log per step (each step keeps its own
    // albedo-dependent divisor via `evidence_ceiling_demod_log`).
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

/// Reproject step `s-1`'s screen into step `s` — v8d's `reproject_prev`,
/// unchanged math, returns BOTH the sampled `(prev_dl, valid)` history pair
/// (byte-identical contract to v7's `hist_features_split`) AND the raw
/// screen-space displacement `(fx-tx, fy-ty)` in pixels for the v9 motion-
/// vector channels (0 when invalid/miss — same validity gate).
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

/// Build one step's full 41-channel input `Img` (39 = v7 `hist_features_split`
/// + 2 v9 motion-vector channels) from the step's raw buffers + the previous
/// step's REPROJECTED net output (mechanism 1 — real reprojection, not
/// identity feedback; `prev` is `None` for step 0 of a sequence).
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

/// EMA-sourced whole-sequence forward chain (mechanism 1) — mirrors v8d's
/// `history_forward` (same "EMA weights, once per epoch" contract,
/// cf8bd7b), but each step is now ONE whole-image conv forward instead of a
/// per-pixel MLP forward loop.
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
/// (diagnostic, VALIDATOR ONLY — mechanism 4, see module doc) mean net/
/// teacher luminance ratio over the pose's brightest `pctl` teacher pixels.
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

/// Settle a whole pose sequence through `net` (fresh forward chain, NOT the
/// EMA history chain — this is the eval/monitor act, same as v8d's
/// `settle`/`direct_render_sequence_hist_split`).
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

#[allow(clippy::too_many_arguments)]
/// CURE 2 — return value's 3rd field is the abort-on-detonation verdict
/// (module doc CURE 2): `rising_streak` counts consecutive monitor calls
/// whose score exceeds the best-EVER score by >5% (an improvement OR a
/// flat/noisy epoch within 5% of best resets it to 0); once the streak
/// reaches `abort_streak` the caller stops training and keeps the best
/// checkpoint already on disk — v9's own detonation ran the score up for
/// 250+ consecutive epochs past its last *BEST without ever recovering, so
/// `abort_streak`=20 (default) trips FAR earlier than that observed length.
#[allow(clippy::too_many_arguments)]
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
    wpath: &Path,
    rising_streak: &mut u32,
    abort_streak: u32,
) -> (f64, f64, bool) {
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
        std::fs::write(wpath, &*best_bytes).unwrap();
        *rising_streak = 0;
    } else if score > *best_score * 1.05 {
        *rising_streak += 1;
    } else {
        *rising_streak = 0;
    }
    let should_abort = *rising_streak >= abort_streak;
    eprintln!(
        "[{run_tag}] MONITOR {tag}: val sparkle {sp:.1}/Mpx resid {rs:.4} highlight_ratio {hl:.3} score={score:.3}{} | mirror sparkle {msp:.1}/Mpx resid {mrs:.4} highlight_ratio {mhl:.3} (tgt sp<{spark_target} resid<{resid_gate}){} rising_streak={rising_streak}/{abort_streak}",
        if passes { " PASS" } else { "" }, if better { " *BEST->saved" } else { "" },
    );
    std::io::stderr().flush().ok();
    (sp, rs, should_abort)
}

fn main() {
    let run_tag = std::env::var("GAIA_V9_TAG").unwrap_or_else(|_| "v9b".to_string());
    let abort_streak = env_u32("GAIA_V9B_ABORT_STREAK", 20); // CURE 2
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

    // Training resolution SMALL for CPU wall budget — architecture-agnostic
    // (fully-convolutional; forwards at any render_w/h later). See module doc.
    let tw = env_u32("GAIA_V9_W", 128);
    let th = env_u32("GAIA_V9_H", 72);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V9_STILL", 3);
    let ref_frames = env_u32("GAIA_V9_REF", 64);
    let epochs = env_u32("GAIA_V9_EPOCHS", 300);
    let lr0 = env_f32("GAIA_V9_LR", 1.0e-4); // TRAINING LAW: corner-crawl lr 1e-4
    let ema_decay = env_f32("GAIA_V9_EMA", 0.999);
    let pan_step = env_f32("GAIA_V9_PANSTEP", 0.004); // v8d parity
    let mirror_spp = env_u32("GAIA_V9_MIRROR_SPP", 4); // v8d parity
    let highlight_pctl = env_f32("GAIA_V9_HIGHLIGHT_PCTL", 0.05);
    let spark_target = env_f32("GAIA_V9_SPARK_TGT", 16.0);
    let resid_gate = env_f32("GAIA_V9_RESID_GATE", 0.035); // the IRON bar
    let monitor_every = env_u32("GAIA_V9_MONITOR", 1); // TRAINING LAW: every epoch
    let wall_budget = env_f32("GAIA_V9_WALL", 10_800.0);
    let k_draws = env_u32("GAIA_V9_K", 8); // v8d parity — mechanisms 3+4
    let seed = env_u32("GAIA_V9_SEED", 0x5eed_c0de) as u64;
    eprintln!("[{run_tag}] res {tw}x{th} K={k} steps, GAIA_V9_K={k_draws} noise2noise draws (variance/{k_draws}, teacher validator-only)");

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let mirror_cam = scrying_glass::denoiser_dataset::mirror_camera();
    let train_cams: Vec<(&str, Camera, u32)> = vec![
        ("front", find("front"), 1),
        ("wide", find("wide"), 1),
        ("orbit_+20", find("orbit_+20"), 1),
        ("mirror", mirror_cam, mirror_spp), // curved-mirror/low-roughness pose (mechanism 2)
    ];
    eprintln!("[{run_tag}] training poses (mirror always on, A3-innocent): {:?}", train_cams.iter().map(|(n, ..)| *n).collect::<Vec<_>>());

    let t_render = Instant::now();
    let poses: Vec<PoseSeq> = train_cams
        .iter()
        .map(|(_, c, spp)| render_pose_seq(&device, &queue, &base_tris, &scene, c, k, low_w, low_h, tw, th, ref_frames, pan_step, *spp, k_draws))
        .collect();
    let val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &find("orbit_-20"), k, low_w, low_h, tw, th, ref_frames, 0.0, 1, k_draws);
    let mirror_val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &mirror_cam, k, low_w, low_h, tw, th, ref_frames, 0.0, mirror_spp, k_draws);
    eprintln!(
        "[{run_tag}] rendered {}+2 MOVING/STILL pose sequences ({tw}x{th}, teacher {ref_frames} VALIDATOR ONLY, pan_step={pan_step}, mirror_spp={mirror_spp}) in {:.1}s",
        poses.len(), t_render.elapsed().as_secs_f64()
    );
    std::io::stderr().flush().ok();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join(format!("rdirect-weights-{run_tag}.bin"));
    let config = UnetConfig { render_w: tw as usize, render_h: th as usize, output_w: tw as usize, output_h: th as usize, ..UnetConfig::default() };
    eprintln!("[{run_tag}] UnetConfig widths={:?} n_scales={} params≈{}", config.widths, config.n_scales, config.approx_param_count());

    // FRESH init only (training law) — random conv init (no TIER1 analytic
    // estimator-init exists for a conv net yet; disclosed, narrower than
    // v8d's per-pixel body — He-init via `UnetWeights::new_random`).
    let mut net = UnetWeights::new_random(config.clone(), seed);
    let mut ema = net.clone();
    let mut adam = UnetAdam::new(&net, lr0 as f32, 0.9, 0.999, 1e-8);

    let mut best_score: f64 = if wpath.exists() {
        if let Some(bytes) = std::fs::read(&wpath).ok() {
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
    let mut rising_streak: u32 = 0;

    run_monitor(&run_tag, "epoch -1 (fresh He-init)", &ema, &val_seq, &mirror_val_seq, sky_reject, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath, &mut rising_streak, abort_streak);

    let mut rng = Rng(0xd15e_ed00_08f0_0dc0 ^ seed);
    let t_train = Instant::now();

    'train: for epoch in 0..epochs {
        if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
            eprintln!("[{run_tag}] WALL budget {wall_budget}s reached at epoch {epoch} — stopping, keeping best");
            break;
        }
        let frac = epoch as f32 / epochs as f32;
        adam.set_lr(lr0 / (1.0 + 1.0 * frac));

        // Mechanism 1: EMA-sourced whole-sequence history, once per epoch.
        let t_hist = Instant::now();
        let history: Vec<Vec<Vec<GVec3>>> = poses.iter().map(|seq| history_forward(&ema, seq, sky_reject)).collect();
        let hist_ms = t_hist.elapsed().as_secs_f64() * 1000.0;

        // Image-level batch: every (pose, step) image once per epoch, shuffled.
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
                eprintln!("[{run_tag}] WALL budget {wall_budget}s reached MID-epoch {epoch} — stopping, keeping best");
                std::io::stderr().flush().ok();
                break 'train;
            }
            let seq = &poses[pi];
            let prev: Option<&[GVec3]> = if s == 0 { None } else { Some(&history[pi][s - 1]) };
            let input = build_input_img(seq, s, prev, sky_reject);
            let (out, cache) = net.forward(&input);
            let target = target_to_img(&seq.steps[s].target_dl, th as usize, tw as usize);
            // CURE 1: evidence-CLAMPED whole-image loss, v8d's
            // `accumulate_backward_clamped_slice` shape (module doc CURE 1):
            // presented = min(out, ceiling); loss on presented vs the
            // K-averaged noise2noise label (mechanisms 3+4, teacher NEVER
            // appears here); gradient is ZEROED wherever out already exceeds
            // the evidence ceiling — no signal to push higher once there.
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
                    let active = raw <= cap;
                    d_out.data[i] = if active { 2.0 * diff / n_elems } else { 0.0 };
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
            "[{run_tag}] epoch {}/{} n2n_mse={:.6} lr={:.6} hist_ms={hist_ms:.0} batch_ms={batch_ms:.0} ({:.1}s)",
            epoch, epochs, mse_mean, adam.lr(), t_train.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().ok();

        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            let (_, _, should_abort) = run_monitor(&run_tag, &format!("epoch {epoch}"), &ema, &val_seq, &mirror_val_seq, sky_reject, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath, &mut rising_streak, abort_streak);
            if should_abort {
                eprintln!("[{run_tag}] CURE 2 ABORT-ON-DETONATION: score exceeded best*1.05 for {rising_streak} consecutive monitor epochs (>= {abort_streak}) at epoch {epoch} — stopping, keeping best (score={best_score:.3})");
                std::io::stderr().flush().ok();
                break 'train;
            }
        }
    }
    eprintln!("[{run_tag}] training done in {:.1}s (best score={best_score:.3})", t_train.elapsed().as_secs_f64());
    std::io::stderr().flush().ok();

    std::fs::write(&wpath, &best_bytes).unwrap();
    let best_net = deserialize_weights(&best_bytes).expect("reload best");
    let wsha = weights_sha256(&best_net);
    println!("[{run_tag}] wrote {} sha256={wsha}", wpath.display());
    std::io::stdout().flush().ok();
    let prov = serde_json::json!({
        "artifact": format!("rdirect-weights-{run_tag}.bin"),
        "ablation_tag": run_tag,
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v8d.bin (different body — U-Net conv, NOT a drop-in for the per-pixel MLP's live path)",
        "doctrine_concordance": "STAGE 2 of the v9-body lane: v8d's TIER1(estimator-init: NOT ported, no analytic estimator-init exists for a conv net yet — FRESH He-init instead, disclosed)+TIER2(K-averaged noise2noise, K={k_draws}, teacher=validator-only)+TIER3(structure: moving-camera history w/ REAL reprojection + curved-mirror/low-roughness pose + EMA history-source, kept as structural mechanisms, NOT the training signal) — same doctrine, whole-image conv body",
        "architecture": {
            "kind": "v9-body shape-parametric multi-scale conv U-Net (rdirect_unet.rs), CPU-trainable twin (rdirect_unet.rs::cpu)",
            "widths": config.widths, "n_scales": config.n_scales, "kernel": config.kernel,
            "in_channels": config.in_channels, "out_channels": config.out_channels,
            "params_approx": config.approx_param_count(),
        },
        "training": {
            "resolution_train": [tw, th], "epochs": epochs, "unroll_steps": k, "lr0": lr0,
            "ref_frames_validator_only": ref_frames,
            "init": "FRESH He-init (UnetWeights::new_random) — no conv-net analytic estimator-init exists yet, disclosed gap vs v8d's TIER1",
            "loss": format!("whole-image MSE(net(step features), mean of K={k_draws} independent draw radiances) — teacher NEVER in the loss, validator only (v8d TIER2 doctrine, ported)"),
            "k_draws": k_draws,
            "sky_history_reject_active": sky_reject,
            "moving_camera_settle": { "pan_step_rad_per_step": pan_step, "note": "REAL reprojection via CamPose::reproject + depth/normal guard, NOT identity feedback — v8d parity" },
            "mirror_pose": { "camera": "denoiser_dataset::mirror_camera", "evidence_spp": mirror_spp, "note": "curved-mirror/low-roughness pose, kept unconditionally (v8/A3-innocent parity)" },
            "highlight_recovery": "same mechanism as K-averaged noise2noise (v8c/v8d doctrine) — NOT a separate loss term; highlight_ratio reported validator-only every epoch",
            "history_precompute": "once per epoch, sourced from EMA net (kept, cf8bd7b), whole-image conv forward per step",
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": train_cams.iter().map(|(n, ..)| *n).collect::<Vec<_>>(), "val": ["orbit_-20 (still)"] },
        "gate": "NOT ordealed — STAGE 2/3 output only, no bar claimed passed",
    });
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")), serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[{run_tag}] wrote provenance. tag={run_tag} weights={}", wpath.display());
    std::io::stdout().flush().ok();
}
