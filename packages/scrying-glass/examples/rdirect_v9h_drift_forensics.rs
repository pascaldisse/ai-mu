//! V9 DRIFT-SOURCE FORENSICS (v9h-last weights, no retrain).
//!
//! Task: at the render-res (640x480 "bar-res") sparkle-flagged texels, what
//! is pulling the net up — TARGET BIAS (the K=8 n2n label itself sits above
//! teacher there), SHARED-WEIGHT DRIVE (label ≈ teacher but net overshoots
//! anyway, i.e. global weight-sharing profit elsewhere), or DEMOD DEGENERACY
//! (the albedo divisor is pathologically small at these texels, amplifying
//! noise through the log-demod encode)?
//!
//! Pipeline reused byte-for-byte from `rdirect_train_v9h.rs::bar_res_probe`
//! (K-step recurrent settle at bar-res, undo-log-demod + inference-time
//! evidence clamp) so the located pixels are the SAME ones the real
//! WATCHDOG probe flagged. `sparkle_resid_per_mpx`'s own 3x3 local-peak
//! detector (SPARK_DELTA=0.15, linear luminance err) is reused UNCHANGED to
//! locate texels — this file only adds a second measurement (the target
//! side) at exactly those locations, nothing about detection changes.
//!
//! Domain: the loss operates in DEMOD-LOG space (`presented_dl` vs
//! `target_dl`, plain MSE, see `rdirect_train_v9h.rs`'s per-pixel loop) —
//! that is the PRIMARY reporting domain here. LINEAR radiance (via
//! `undo_log_demod`) is reported alongside for intuition/sparkle-metric
//! cross-check (sparkle itself is a linear-luminance metric). Every number
//! below states which domain it is in; conversions use the exact
//! `target_demod_log`/`undo_log_demod`/`demod_divisor` functions the
//! trainer uses, applied with the SAME per-pixel albedo divisor for net,
//! teacher and target so the three are directly comparable.
//!
//! TARGET MEAN: the trainer's per-epoch label is `target_dl = 
//! target_demod_log(mean_linear_over_K=8_draws, albedo)` — one noisy
//! realization per epoch. To estimate what that noisy procedure gives IN
//! EXPECTATION at the flagged texels (not just one epoch's roll), this tool
//! repeats the exact 8-draw-average-then-encode recipe 32 times (fresh
//! seeds each repeat, same integrator/spp/camera) and reports:
//!   - `target_dl_mean`  = mean over 32 repeats of `target_dl` (demod-log
//!     space average — the space the loss actually sees).
//!   - `target_lin_of_dl_mean` = undo_log_demod(target_dl_mean) — linear
//!     radiance implied by that demod-log average (for teacher comparison).
//!   - `target_lin_direct` = mean over all 32*8=256 raw linear draws taken
//!     directly (order-of-operations cross-check: averaging 256 unbiased
//!     linear draws directly is the lowest-variance unbiased MC estimate of
//!     the TRUE scene radiance available here, independent of the log-demod
//!     nonlinearity — compares directly against `teacher_lin`).
use std::io::Write;
use std::path::Path;

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
use scrying_glass::rdirect_unet::cpu::{Img, UnetWeights, deserialize_weights};
use scrying_glass::rdirect_unet::MOTION_VECTOR_CHANNELS;
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;
const DRAW_A_SEED_BASE: u32 = 0x7abc;
const DRAW_B_SEED_BASE: u32 = 0xB222;
const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

/// V9I GRAD-PROBE (added this round): reproduces, from data already computed
/// in this tool's own pipeline (no extra render needed — `net_raw_dl`,
/// `ceiling_dl`, `target_dl_mean` ARE the live forward-pass/ceiling/target
/// tensors, not resimulated), three per-channel gradient formulas at every
/// flagged texel:
///   BEFORE_STALE  — the standalone `GAIA_V9H_GRAD_PROBE=1` diagnostic's own
///     hardcoded formula (`examples/rdirect_train_v9h.rs:887-889`), pre-CURE4,
///     zero above the ceiling regardless of `overshoot_w`.
///   CURE4_ACTUAL  — the REAL, currently-training formula (same file,
///     :1010-1020): active branch unchanged, overshoot branch
///     `2*overshoot_w*(raw-cap)/n` (nonzero, but anchored to `cap`, not
///     `target`).
///   AFTER_FIX     — V9I's cure: at albedo≈0/no-hit pixels (mask threshold
///     `GAIA_V9I_NOHIT_ALBEDO_SQ`, default matches `NO_HIT_ALBEDO_THRESHOLD_SQ`),
///     bypass the clamp/ceiling entirely — ordinary two-sided MSE against the
///     honest target, unconditionally active.
fn grad_probe_formulas(raw: f32, cap: f32, target: f32, albedo_sq: f32, overshoot_w: f32, nohit_albedo_sq: f32, n_elems: f32) -> (bool, f32, f32, f32) {
    let active = raw <= cap;
    let d_before_stale = if active { 2.0 * (raw - target) / n_elems } else { 0.0 };
    let d_cure4_actual = if active { 2.0 * (raw - target) / n_elems } else { 2.0 * overshoot_w * (raw - cap) / n_elems };
    let is_nohit = albedo_sq <= nohit_albedo_sq;
    let d_after_fix = if is_nohit { 2.0 * (raw - target) / n_elems } else { d_cure4_actual };
    (active, d_before_stale, d_cure4_actual, d_after_fix)
}

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

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}
fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h as f32 }
}

/// `sparkle_resid_per_mpx`'s OWN detection, unchanged (SPARK_DELTA=0.15,
/// linear-luminance err, 3x3 local peak) — reused, but returning the
/// flagged (x,y,err) list instead of just a count.
const SPARK_DELTA: f32 = 0.15;
fn sparkle_flagged_pixels(net: &[GVec3], teacher: &[GVec3], w: u32, h: u32) -> Vec<(u32, u32, f32)> {
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
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    out
}

fn sparkle_resid_per_mpx(net: &[GVec3], teacher: &[GVec3], w: u32, h: u32) -> f64 {
    let count = sparkle_flagged_pixels(net, teacher, w, h).len() as f64;
    count * 1.0e6 / (w as f64 * h as f64)
}

#[allow(clippy::too_many_arguments)]
fn reproject_prev(
    cur_cam: &CamPose, cur_depth: f32, cur_normal: GVec3, tx: u32, ty: u32, tw: u32, th: u32,
    prev_cam: &CamPose, prev_out_dl: &[GVec3], prev_depth: &[f32], prev_normal: &[GVec3],
    pw: u32, ph: u32, sky_reject: bool,
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

fn img_to_vec3(img: &Img) -> Vec<GVec3> {
    let mut v = vec![GVec3::ZERO; img.h * img.w];
    for y in 0..img.h {
        for x in 0..img.w {
            v[y * img.w + x] = GVec3::new(img.at(y, x, 0), img.at(y, x, 1), img.at(y, x, 2));
        }
    }
    v
}

fn main() {
    let tag = "v9h-drift-forensics";
    let sky_reject = sky_history_reject();
    eprintln!("[{tag}] GAIA_V7_SKY_HISTORY reject={sky_reject}");

    let Some((device, queue)) = headless_device() else {
        panic!("[{tag}] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();
    let bvh = Bvh::build(&base_tris, &BvhParams::default());

    let tw = env_u32("GAIA_V9_EVAL_W", 640);
    let th = env_u32("GAIA_V9_EVAL_H", 480);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V9_EVAL_K", 3);
    let ref_frames = env_u32("GAIA_V9_EVAL_REF", 64);
    let k_draws = env_u32("GAIA_V9_K", 8); // K=8, matches training's noise2noise draw count
    let n_repeats = env_u32("GAIA_V9H_TARGET_REPEATS", 32); // "many target draws"
    let gamma = evidence_clamp_gamma();

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let val_cam = all.iter().find(|(pn, _)| *pn == "orbit_-20").unwrap().1.clone();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = std::env::var("GAIA_V9H_DRIFT_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("rdirect-weights-v9h-last.bin"));
    let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("read {wpath:?}: {e}"));
    let ema = deserialize_weights(&bytes).unwrap_or_else(|| panic!("deserialize {wpath:?}"));
    eprintln!("[{tag}] loaded {wpath:?} ({} bytes)", bytes.len());
    eprintln!("[{tag}] bar-res {tw}x{th} K={k} steps, ref_frames={ref_frames}, K_DRAWS={k_draws}, target n_repeats={n_repeats}");

    // ---- bar_res_probe pipeline, verbatim (rdirect_train_v9h.rs) ----
    let mut steps_low_e = Vec::with_capacity(k as usize);
    let mut steps_low_d = Vec::with_capacity(k as usize);
    let mut steps_albedo = Vec::with_capacity(k as usize);
    let mut steps_normal = Vec::with_capacity(k as usize);
    let mut steps_depth = Vec::with_capacity(k as usize);
    let mut cams = Vec::with_capacity(k as usize);
    for step in 0..k {
        let cam = val_cam;
        cams.push(cam_pose(&cam, tw, th));
        let np_a = IntegratorParams { spp: 1, seed: DRAW_A_SEED_BASE + step * 131 + 5, ..IntegratorParams::default() };
        let (low_e, low_d) = trace_headless_split(
            &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np_a,
        );
        let (albedo, normal, depth) = scrying_glass::integrator::split_aov(&trace_headless_aov(
            &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
        ));
        steps_low_e.push(low_e);
        steps_low_d.push(low_d);
        steps_albedo.push(albedo);
        steps_normal.push(normal);
        steps_depth.push(depth);
    }
    let n = (tw * th) as usize;
    let (e_full, d_full) = trace_headless_split(
        &device, &queue, &bvh, &val_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
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

    let last = k as usize - 1;
    let last_dl = &out_imgs[last];
    let last_albedo = &steps_albedo[last];
    let mut net_clamped = vec![GVec3::ZERO; n];
    let mut ceiling_dl_all: Vec<[f32; OUTPUT_CHANNELS]> = vec![[0.0; OUTPUT_CHANNELS]; n];
    for px in 0..n {
        let dl = GVec3::new(last_dl.data[px * 3], last_dl.data[px * 3 + 1], last_dl.data[px * 3 + 2]);
        let divisor = demod_divisor(last_albedo[px]);
        let ceiling_dl: [f32; OUTPUT_CHANNELS] = evidence_ceiling_demod_log(evidence_ceiling[px], gamma, last_albedo[px]);
        ceiling_dl_all[px] = ceiling_dl;
        let presented_dl = GVec3::new(dl.x.min(ceiling_dl[0]), dl.y.min(ceiling_dl[1]), dl.z.min(ceiling_dl[2]));
        net_clamped[px] = undo_log_demod(presented_dl, divisor);
    }

    let sp = sparkle_resid_per_mpx(&net_clamped, &teacher, tw, th);
    eprintln!("[{tag}] bar-res sparkle {sp:.1}/Mpx (reference: v9h-last PROBE epoch 124 reported 29.3/Mpx at this same res/pose — this run's own trace RNG differs from training's, small drift expected)");

    let flagged = sparkle_flagged_pixels(&net_clamped, &teacher, tw, th);
    eprintln!("[{tag}] {} sparkle-flagged texels located", flagged.len());

    // Cap how many texels get the expensive 32x8-draw target-mean treatment.
    let max_texels = env_u32("GAIA_V9H_MAX_TEXELS", 12) as usize;
    let sample: Vec<(u32, u32, f32)> = flagged.iter().take(max_texels).cloned().collect();

    // V9I GRAD-PROBE params (IRON, env-tunable).
    let overshoot_w = env_f32("GAIA_V9H_OVERSHOOT_W", 4.0); // matches v9h-last's own trained dose
    let nohit_albedo_sq = env_f32("GAIA_V9I_NOHIT_ALBEDO_SQ", NO_HIT_ALBEDO_THRESHOLD_SQ);
    let n_elems_grad = (tw * th * OUTPUT_CHANNELS as u32) as f32;
    println!("# V9I GRAD-PROBE this run: overshoot_w={overshoot_w} nohit_albedo_sq={nohit_albedo_sq} n_elems={n_elems_grad}");

    println!("# V9H DRIFT-SOURCE FORENSICS — bar-res {tw}x{th}, val_cam=orbit_-20, weights={wpath:?}");
    println!("# domain: dl = demod-log (loss space); lin = linear radiance (undo_log_demod). Luminance = 0.2126R+0.7152G+0.0722B.");
    println!("# columns: px err_lin(sparkle) | net_raw_dl net_presented_dl net_lin(clamped) | teacher_lin teacher_dl(same divisor) | target_dl_mean target_lin_of_dl_mean target_lin_direct(256-draw MC) | ceiling_dl | albedo divisor");

    let mut rows: Vec<serde_json::Value> = Vec::new();

    for &(x, y, e) in &sample {
        let px = (y * tw + x) as usize;
        let albedo = last_albedo[px];
        let divisor = demod_divisor(albedo);
        let cap = ceiling_dl_all[px];

        // ---- 32-repeat, 8-draw-each target reconstruction (this pose, last step) ----
        let mut target_dl_sum = GVec3::ZERO;
        let mut linear_grand_sum = GVec3::ZERO;
        let mut n_draws_total = 0u32;
        for r in 0..n_repeats {
            let mut radiance_sum = GVec3::ZERO;
            for kd in 0..k_draws {
                let seed = DRAW_B_SEED_BASE + (last as u32) * 257 + 11 + kd * 9973 + r * 1_000_003;
                let np_b = IntegratorParams { spp: 1, seed, ..IntegratorParams::default() };
                // Full-frame trace is expensive; restrict to a tight window
                // around the flagged pixel would require a windowed
                // integrator entry point which doesn't exist headless, so
                // this measures the FULL frame per draw (matches training's
                // own render_pose_seq cost exactly) and reads back just the
                // flagged pixel.
                let (e_b, d_b) = trace_headless_split(
                    &device, &queue, &bvh, &val_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, 1, &np_b,
                );
                let r_px = e_b[px] + d_b[px];
                radiance_sum += r_px;
                linear_grand_sum += r_px;
                n_draws_total += 1;
            }
            let mean_linear = radiance_sum / (k_draws.max(1) as f32);
            let dl: [f32; OUTPUT_CHANNELS] = target_demod_log(mean_linear, albedo);
            target_dl_sum += GVec3::new(dl[0], dl[1], dl[2]);
        }
        let target_dl_mean = target_dl_sum / (n_repeats.max(1) as f32);
        let target_lin_of_dl_mean = undo_log_demod(target_dl_mean, divisor);
        let target_lin_direct = linear_grand_sum / (n_draws_total.max(1) as f32);

        let raw_dl = GVec3::new(last_dl.data[px * 3], last_dl.data[px * 3 + 1], last_dl.data[px * 3 + 2]);
        let presented_dl = GVec3::new(raw_dl.x.min(cap[0]), raw_dl.y.min(cap[1]), raw_dl.z.min(cap[2]));
        let net_lin = net_clamped[px];
        let teacher_lin = teacher[px];
        let teacher_dl_arr: [f32; OUTPUT_CHANNELS] = target_demod_log(teacher_lin, albedo);
        let teacher_dl = GVec3::new(teacher_dl_arr[0], teacher_dl_arr[1], teacher_dl_arr[2]);

        println!(
            "px=({x:>3},{y:>3}) err_lin={e:.4} | net_raw_dl=({:.4},{:.4},{:.4}) net_presented_dl=({:.4},{:.4},{:.4}) net_lin=({:.4},{:.4},{:.4}) | teacher_lin=({:.4},{:.4},{:.4}) teacher_dl=({:.4},{:.4},{:.4}) | target_dl_mean=({:.4},{:.4},{:.4}) target_lin_of_dl_mean=({:.4},{:.4},{:.4}) target_lin_direct256=({:.4},{:.4},{:.4}) | ceiling_dl=({:.4},{:.4},{:.4}) | albedo=({:.4},{:.4},{:.4}) divisor=({:.4},{:.4},{:.4})",
            raw_dl.x, raw_dl.y, raw_dl.z,
            presented_dl.x, presented_dl.y, presented_dl.z,
            net_lin.x, net_lin.y, net_lin.z,
            teacher_lin.x, teacher_lin.y, teacher_lin.z,
            teacher_dl.x, teacher_dl.y, teacher_dl.z,
            target_dl_mean.x, target_dl_mean.y, target_dl_mean.z,
            target_lin_of_dl_mean.x, target_lin_of_dl_mean.y, target_lin_of_dl_mean.z,
            target_lin_direct.x, target_lin_direct.y, target_lin_direct.z,
            cap[0], cap[1], cap[2],
            albedo.x, albedo.y, albedo.z,
            divisor.x, divisor.y, divisor.z,
        );
        std::io::stdout().flush().ok();

        // ---- V9I GRAD-PROBE: BEFORE_STALE / CURE4_ACTUAL / AFTER_FIX d_out per channel ----
        let albedo_sq = albedo.length_squared();
        let mut gp_active = [false; OUTPUT_CHANNELS];
        let mut gp_before = [0.0f32; OUTPUT_CHANNELS];
        let mut gp_cure4 = [0.0f32; OUTPUT_CHANNELS];
        let mut gp_after = [0.0f32; OUTPUT_CHANNELS];
        for c in 0..OUTPUT_CHANNELS {
            let raw_c = raw_dl[c];
            let cap_c = cap[c];
            let target_c = target_dl_mean[c];
            let (active, d_before, d_cure4, d_after) = grad_probe_formulas(raw_c, cap_c, target_c, albedo_sq, overshoot_w, nohit_albedo_sq, n_elems_grad);
            gp_active[c] = active;
            gp_before[c] = d_before;
            gp_cure4[c] = d_cure4;
            gp_after[c] = d_after;
        }
        println!(
            "[grad-probe] px=({x:>3},{y:>3}) active={:?} d_out_BEFORE_stale=({:.8},{:.8},{:.8}) d_out_CURE4_actual=({:.8},{:.8},{:.8}) d_out_AFTER_fix=({:.8},{:.8},{:.8})",
            gp_active, gp_before[0], gp_before[1], gp_before[2], gp_cure4[0], gp_cure4[1], gp_cure4[2], gp_after[0], gp_after[1], gp_after[2],
        );
        std::io::stdout().flush().ok();

        rows.push(serde_json::json!({
            "px": [x, y], "err_lin_lum": e,
            "net_raw_dl": [raw_dl.x, raw_dl.y, raw_dl.z],
            "net_presented_dl": [presented_dl.x, presented_dl.y, presented_dl.z],
            "net_lin": [net_lin.x, net_lin.y, net_lin.z],
            "teacher_lin": [teacher_lin.x, teacher_lin.y, teacher_lin.z],
            "teacher_dl": [teacher_dl.x, teacher_dl.y, teacher_dl.z],
            "target_dl_mean_32x8": [target_dl_mean.x, target_dl_mean.y, target_dl_mean.z],
            "target_lin_of_dl_mean": [target_lin_of_dl_mean.x, target_lin_of_dl_mean.y, target_lin_of_dl_mean.z],
            "target_lin_direct_256draw_mc": [target_lin_direct.x, target_lin_direct.y, target_lin_direct.z],
            "ceiling_dl": cap,
            "albedo": [albedo.x, albedo.y, albedo.z],
            "divisor": [divisor.x, divisor.y, divisor.z],
            "grad_probe_active": gp_active,
            "grad_probe_d_out_before_stale": gp_before,
            "grad_probe_d_out_cure4_actual": gp_cure4,
            "grad_probe_d_out_after_fix": gp_after,
        }));
    }

    // ---- verdict aggregation ----
    let mut target_over_teacher_dl = 0usize; // (A) target_dl_mean > teacher_dl by margin
    let mut net_over_target_dl = 0usize;     // net_presented overshoots even target_dl_mean
    let mut target_near_teacher = 0usize;    // (B) target ≈ teacher (net alone drifts)
    let mut divisor_near_floor = 0usize;     // (C) divisor near ALBEDO_DEMOD_EPS => low albedo
    let margin = 0.05f32; // dl-space margin, ~5% log-radiance
    for row in &rows {
        let target_dl_mean: Vec<f64> = row["target_dl_mean_32x8"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let teacher_dl: Vec<f64> = row["teacher_dl"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let net_presented: Vec<f64> = row["net_presented_dl"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let divisor: Vec<f64> = row["divisor"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let over = (0..3).any(|c| target_dl_mean[c] > teacher_dl[c] + margin as f64);
        let near = (0..3).all(|c| (target_dl_mean[c] - teacher_dl[c]).abs() <= margin as f64);
        let net_over_target = (0..3).any(|c| net_presented[c] > target_dl_mean[c] + margin as f64);
        let low_divisor = divisor.iter().any(|&d| d <= (ALBEDO_DEMOD_EPS as f64) * 2.0);
        if over { target_over_teacher_dl += 1; }
        if near { target_near_teacher += 1; }
        if net_over_target { net_over_target_dl += 1; }
        if low_divisor { divisor_near_floor += 1; }
    }

    println!("\n# ---- VERDICT AGGREGATION over {} sampled flagged texels (dl-space margin={margin}) ----", rows.len());
    println!("# target_dl_mean > teacher_dl + margin (TARGET BIAS signal): {target_over_teacher_dl}/{}", rows.len());
    println!("# target_dl_mean within margin of teacher_dl (TARGET~TEACHER, SHARED-WEIGHT signal): {target_near_teacher}/{}", rows.len());
    println!("# net_presented_dl > target_dl_mean + margin (net overshoots EVEN the noisy label): {net_over_target_dl}/{}", rows.len());
    println!("# divisor <= 2*ALBEDO_DEMOD_EPS ({:.4}) (DEMOD DEGENERACY signal, near-zero albedo): {divisor_near_floor}/{}", 2.0 * ALBEDO_DEMOD_EPS, rows.len());

    // ---- V9I GRAD-PROBE aggregate (channel-reading counts, 3 per texel) ----
    let mut n_ch = 0usize;
    let mut n_before_zero = 0usize;
    let mut n_cure4_zero = 0usize;
    let mut n_after_zero = 0usize;
    for row in &rows {
        let before: Vec<f64> = row["grad_probe_d_out_before_stale"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let cure4: Vec<f64> = row["grad_probe_d_out_cure4_actual"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let after: Vec<f64> = row["grad_probe_d_out_after_fix"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        for c in 0..3 {
            n_ch += 1;
            if before[c] == 0.0 { n_before_zero += 1; }
            if cure4[c] == 0.0 { n_cure4_zero += 1; }
            if after[c] == 0.0 { n_after_zero += 1; }
        }
    }
    println!("\n# ---- V9I GRAD-PROBE AGGREGATE over {} texels x 3 channels = {n_ch} readings ----", rows.len());
    println!("# BEFORE_stale (pre-CURE4 zero-gate, matches GAIA_V9H_GRAD_PROBE=1's hardcoded formula) d_out==0.0: {n_before_zero}/{n_ch}");
    println!("# CURE4_actual (real training-loop formula, overshoot_w={overshoot_w}) d_out==0.0: {n_cure4_zero}/{n_ch}");
    println!("# AFTER_fix (V9I mask-based ordinary loss @ albedo~0/no-hit) d_out==0.0: {n_after_zero}/{n_ch}");

    let out_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scratch/v9h-drift-forensics.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&serde_json::json!({
        "tag": tag, "weights": wpath.to_string_lossy(), "bar_res": [tw, th], "k_draws": k_draws, "n_repeats": n_repeats,
        "sparkle_per_mpx_this_run": sp, "n_flagged": flagged.len(), "n_sampled": rows.len(),
        "texels": rows,
        "aggregate": {
            "target_over_teacher_dl": target_over_teacher_dl,
            "target_near_teacher": target_near_teacher,
            "net_over_target_dl": net_over_target_dl,
            "divisor_near_floor": divisor_near_floor,
        },
        "grad_probe_aggregate": {
            "overshoot_w": overshoot_w,
            "nohit_albedo_sq": nohit_albedo_sq,
            "n_channel_readings": n_ch,
            "before_stale_zero": n_before_zero,
            "cure4_actual_zero": n_cure4_zero,
            "after_fix_zero": n_after_zero,
        }
    })).unwrap()).unwrap();
    eprintln!("[{tag}] wrote {out_path:?}");
    std::io::stderr().flush().ok();
}
