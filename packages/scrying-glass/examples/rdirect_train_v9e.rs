//! R-DIRECT v9e-body — TARGET-TAIL FORENSICS + WINSORIZED-TARGET CURE
//! (v9e task, `scratch/v9-autopsy.md` v9e section). Base: `rdirect_train_v9c.rs`
//! CONFIG UNCHANGED (narrow ±30deg 2-anchor pool, 3 pool draws/epoch + fixed
//! mirror, EMA eval, evidence-clamped loss CURE 1, abort-on-detonation
//! watchdog CURE 2, K=8, real reprojection, mirror pose, corner-crawl lr,
//! fresh He-init — NOT v9d's wide pool, per task instruction) — every
//! mechanism below this header is byte-identical to `rdirect_train_v9c.rs`
//! except ONE new addition:
//!
//! CURE 3 — WINSORIZED NOISE2NOISE TARGETS. Forensics
//! (`scratch/v9-autopsy.md` v9e section, `rdirect_v9_tailmass.rs`)
//! instrumented v9c's own K=8-averaged `target_dl` at v9c's pool poses, IN
//! THE EXACT DOMAIN THE LOSS COMPUTES IN (verified by source read: the
//! K-average happens in LINEAR radiance, `radiance_sum * inv_k`, THEN is
//! demod-log-encoded ONCE via `target_demod_log` — so the loss's plain MSE
//! runs entirely in demod-log space, and that is the space both the
//! forensics tool and this cure operate in) — found a real, if modest,
//! firefly tail even after 8-draw averaging: ~1.3% of pixels exceed 3x
//! their local 7x7-mean luminance, ~0.03% exceed 10x, whole-image max/mean
//! ratio ~15x. CURE: before computing the loss, each `target_dl` pixel's
//! luminance is winsorized against a per-channel local-mean ceiling —
//! `target_dl[px][c] = min(target_dl[px][c], winsor_k * local_mean_7x7(target_dl)[px][c])`,
//! `winsor_k` = `GAIA_V9E_WINSOR_K` (env-tunable, IRON — default 10.0,
//! chosen to sit above the forensics' 10x-tail rarity threshold so it clips
//! only the sparse extreme fireflies, not ordinary bright/specular detail
//! which forensics showed sits mostly below 3x). This is a TARGET-side fix,
//! complementary to and distinct from CURE 1's NET-OUTPUT-side evidence
//! clamp (CURE 1 gates how high the net's own prediction may go relative to
//! traced evidence; CURE 3 gates how high the noisy LABEL itself may be
//! before the net ever sees it as ground truth) — see `winsorize_targets`
//! below, applied once per pose per epoch right after `target_dl` is built,
//! same place `ceiling_dl` is derived.
//!
//! MONAD HYPOTHESIS UNDER TEST (see `scratch/v9-autopsy.md` v9e section for
//! the full forensics writeup): detonation onset tracks TRAINING PROGRESS
//! (cumulative image-gradient-updates = onset_epoch * poses/epoch *
//! unroll_steps), not pose diversity/exposure — v9c and v9d's onsets land
//! within ~6% of each other in image-update count despite a 3x pool-size
//! difference, while v9 (no clamp, different failure mode) is a >4x
//! outlier, consistent with unbounded target-tail-chasing capacity once
//! `n2n_mse` is low, gated by whichever mechanism (clamp presence) is
//! active. v9e tests whether removing the target-side tail mass (this
//! cure) delays or removes onset entirely at the SAME image-update budget
//! v9c/v9d both detonated within.
//!
//! Everything else (mechanisms 1-4, CURE 1, CURE 2, pool_camera, domain
//! fix) is `rdirect_train_v9c.rs` UNCHANGED — see that file's own header
//! for the full mechanism-by-mechanism doc; not re-duplicated here to keep
//! this file's diff against v9c legible.
//!
//! Run: cargo run --release -j2 --example rdirect_train_v9e
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

fn rand_uniform(rng: &mut Rng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next() >> 11) as f64 / (1u64 << 53) as f64) as f32; // [0,1)
    lo + (hi - lo) * u
}

/// POSE DIVERSITY (v9c task, see module doc): one fresh draw from a
/// continuous pool around `anchor_eye`/`anchor_pivot` — `orbit_camera`
/// (`denoiser_dataset.rs`, unchanged math) gives a random yaw in
/// `[-orbit_deg, +orbit_deg]` about the anchor's own pivot (azimuth
/// diversity — the SAME rotation `law_poses`' derived orbit views use, just
/// continuously resampled instead of frozen at fixed degrees), then a small
/// eye-position jitter box (`±jitter` world units, vertical jitter scaled
/// ×0.4 to avoid clipping through the floor or into the sky) is added and
/// the camera is re-aimed at the SAME pivot (`camera_at`) — off-orbit
/// viewpoint diversity on top of the azimuth sweep. Returns `(camera, yaw)`
/// so the caller can log what was actually drawn.
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

/// One step of a moving-camera pose sequence — v8d's `Step`, PLUS CURE 1's
/// evidence-clamp ceiling (`ceiling_dl`, precomputed once per pose — see
/// module doc CURE 1 for the per-pose-temporal-mean simplification vs
/// v8d's per-step-incremental accumulator). `Clone` so the fixed mirror
/// pose sequence (rendered once, module doc POSE DIVERSITY) can be cloned
/// into each epoch's fresh pose list alongside the 3 newly-drawn ones.
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

/// CURE 3 (module doc): winsorize a K-averaged noise2noise `target_dl`
/// buffer against ITS OWN per-channel 7x7 local-mean ceiling — caps the
/// residual firefly tail forensics found survives K=8 averaging
/// (`scratch/v9-autopsy.md` v9e section, `rdirect_v9_tailmass.rs`), in the
/// SAME demod-log domain the loss computes in (no space conversion). Box
/// filter, not `local_max_3x3` (that fn is a MAX, a different statistic
/// used elsewhere for the evidence ceiling, not a mean).
fn winsorize_targets(target_dl: &mut [[f32; OUTPUT_CHANNELS]], w: u32, h: u32, winsor_k: f32) {
    if winsor_k <= 0.0 {
        return; // 0 or negative disables the cure (diagnostic escape hatch)
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
        let mut target_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n)
            .map(|px| target_demod_log(radiance_sum[px] * inv_k, albedo[px]))
            .collect();
        // CURE 3 (module doc): winsorize the noise2noise label itself
        // against its own per-channel local-mean ceiling, BEFORE it is ever
        // used as a loss target — removes the residual firefly tail forensics
        // found survives K=8 averaging (scratch/v9-autopsy.md v9e section).
        winsorize_targets(&mut target_dl, tw, th, winsor_k);

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
/// FORENSICS (v9f-round pixel-coordinate instrumentation, opt-in via
/// `GAIA_V9E_DUMP_COORDS=1`, default OFF — zero effect on the byte-identical
/// v9e run otherwise) — same peak-detector as `sparkle_resid_per_mpx` but
/// returns the actual `(x,y,err)` of every local-maximum pixel instead of
/// just the count, so recurrence across epochs can be checked directly.
fn sparkle_peak_coords(net: &[GVec3], teacher: &[GVec3], w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    const SPARK_DELTA: f32 = 0.15;
    let idx = |x: i32, y: i32| (y as usize) * w as usize + x as usize;
    let err = |x: i32, y: i32| lum(net[idx(x, y)]) - lum(teacher[idx(x, y)]);
    let mut out = Vec::new();
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
                out.push((x as u32, y as u32, e));
            }
        }
    }
    out
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
    if std::env::var("GAIA_V9E_DUMP_COORDS").ok().as_deref() == Some("1") {
        let peaks = sparkle_peak_coords(&net, &teacher, val_seq.tw, val_seq.th);
        eprintln!("[{run_tag}] COORDS {tag}: {} peak px: {:?}", peaks.len(), peaks);
    }
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
    let run_tag = std::env::var("GAIA_V9_TAG").unwrap_or_else(|_| "v9e".to_string());
    let abort_streak = env_u32("GAIA_V9E_ABORT_STREAK", 20); // CURE 2, v9c parity
    let pool_orbit_deg = env_f32("GAIA_V9E_POOL_ORBIT_DEG", 30.0); // POSE DIVERSITY, v9c parity (narrow pool, base config unchanged)
    let pool_jitter = env_f32("GAIA_V9E_POOL_JITTER", 0.6); // POSE DIVERSITY, v9c parity, world units
    let winsor_k = env_f32("GAIA_V9E_WINSOR_K", 10.0); // CURE 3 — target winsorize ceiling factor, IRON default (see module doc)
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
    // POSE DIVERSITY POOL ANCHORS (module doc) — the SAME two anchors
    // `denoiser_dataset::law_poses` already authors: front's own
    // camera_position/pivot, and "wide"'s eye/look-at (duplicated constants,
    // that fn doesn't expose them separately — cited at each use).
    let fov_deg = params.fov_y_degrees;
    let front_eye = params.camera_position;
    let front_pivot = [0.0f32, 2.0, 0.0]; // == law_poses' front_pivot
    let wide_eye = [-4.5f32, 8.5, 33.0]; // == law_poses' wide_camera eye
    let wide_pivot = [-5.5f32, 2.0, 15.5]; // == law_poses' wide_camera look_at
    eprintln!(
        "[{run_tag}] POSE DIVERSITY POOL (v9c parity): orbit±{pool_orbit_deg}deg jitter±{pool_jitter} around 2 anchors (front-pivot family x2 draws/epoch, wide-anchor family x1 draw/epoch) + mirror unconditional (fixed) — fresh draws every epoch, PLUS CURE 3 target winsorize_k={winsor_k}, see module doc"
    );

    let t_render = Instant::now();
    let mirror_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &mirror_cam, k, low_w, low_h, tw, th, ref_frames, pan_step, mirror_spp, k_draws, winsor_k);
    let val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &find("orbit_-20"), k, low_w, low_h, tw, th, ref_frames, 0.0, 1, k_draws, winsor_k);
    let mirror_val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &mirror_cam, k, low_w, low_h, tw, th, ref_frames, 0.0, mirror_spp, k_draws, winsor_k);
    eprintln!(
        "[{run_tag}] rendered mirror(train) + 2 VALIDATOR (orbit_-20, mirror — FIXED, held-out) pose sequences ({tw}x{th}, teacher {ref_frames} VALIDATOR ONLY, pan_step={pan_step}, mirror_spp={mirror_spp}) in {:.1}s",
        t_render.elapsed().as_secs_f64()
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

        // POSE DIVERSITY (module doc): 3 fresh draws this epoch (2 from the
        // front-pivot family, 1 from the wide-anchor family) + the FIXED
        // mirror sequence cloned in — replaces v9b's 4 literal-fixed cameras.
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
            "[{run_tag}] epoch {}/{} n2n_mse={:.6} lr={:.6} pool_yaws=[{yaw_a1:.1},{yaw_a2:.1},{yaw_b:.1}] pool_ms={pool_ms:.0} hist_ms={hist_ms:.0} batch_ms={batch_ms:.0} ({:.1}s)",
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
        "supersedes": "rdirect-weights-v9c.bin (v9e = v9c CONFIG UNCHANGED + CURE 3 winsorized noise2noise targets, robust-target cure per v9-autopsy forensics, see module doc)",
        "doctrine_concordance": "STAGE 2 of the v9-body lane: v8d's TIER1(estimator-init: NOT ported, no analytic estimator-init exists for a conv net yet — FRESH He-init instead, disclosed)+TIER2(K-averaged noise2noise, K={k_draws}, teacher=validator-only)+TIER3(structure: moving-camera history w/ REAL reprojection + curved-mirror/low-roughness pose + EMA history-source, kept as structural mechanisms, NOT the training signal) — same doctrine, whole-image conv body, v9c's domain-fixed monitor + narrow pose-diversity pool (unchanged), plus v9e's CURE 3 target-side winsorization (this round's own cure for the forensically-confirmed residual firefly tail in the K=8 noise2noise labels)",
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
            "mirror_pose": { "camera": "denoiser_dataset::mirror_camera", "evidence_spp": mirror_spp, "note": "curved-mirror/low-roughness pose, kept unconditionally FIXED (v8/A3-innocent parity) — NOT part of the diversity pool" },
            "highlight_recovery": "same mechanism as K-averaged noise2noise (v8c/v8d doctrine) — NOT a separate loss term; highlight_ratio reported validator-only every epoch",
            "history_precompute": "once per epoch, sourced from EMA net (kept, cf8bd7b), whole-image conv forward per step",
            "pose_diversity": { "pool_orbit_deg": pool_orbit_deg, "pool_jitter": pool_jitter, "draws_per_epoch": 3, "anchors": ["front-pivot (x2 draws/epoch)", "wide-anchor (x1 draw/epoch)"], "note": "v9c CONFIG UNCHANGED (narrow ±30deg pool, NOT v9d's wide pool) — orbit yaw in [-pool_orbit_deg,+pool_orbit_deg] about each anchor's pivot + small eye-position jitter box (±pool_jitter world units, vertical ×0.4), fresh draw every epoch" },
            "winsorized_targets": { "winsor_k": winsor_k, "note": "CURE 3 (module doc) — K=8-averaged noise2noise target_dl winsorized per-channel against its own 7x7 local-mean * winsor_k ceiling BEFORE the loss ever sees it, distinct from CURE 1's net-output-side evidence clamp; forensics (scratch/v9-autopsy.md v9e section, rdirect_v9_tailmass.rs) found ~1.3% of v9c pool-pose target pixels >3x local mean, ~0.03% >10x, whole-image max/mean~15x — winsor_k=10 clips only the sparse extreme tail" },
            "domain_fix": "settle()/run_monitor now undo_log_demod the final step's output (per-pixel albedo divisor) before every metric — same fix as rdirect_v9_eval_640.rs, ported back into the trainer's own in-loop monitor (scratch/v9-autopsy.md conviction #2)",
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": ["front-pivot-pool x2/epoch", "wide-anchor-pool x1/epoch", "mirror (fixed, unconditional)"], "val": ["orbit_-20 (still, FIXED, held-out)", "mirror (still, FIXED, held-out)"] },
        "gate": "NOT ordealed — v9e output only, no bar claimed passed",
    });
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")), serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[{run_tag}] wrote provenance. tag={run_tag} weights={}", wpath.display());
    std::io::stdout().flush().ok();
}
