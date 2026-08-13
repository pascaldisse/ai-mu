//! V9N BAR-RES SPARKLE CLASS FORENSICS (v9n-last weights, no retrain).
//!
//! Task: v9n's hit-gate (`GAIA_V9N_HITGATE`) architecturally forces no-hit
//! px (depth<=0) to `compose := evidence` with EXACT ZERO backward grad —
//! by construction, that class cannot sparkle. Yet the live watchdog probe
//! still ignited at bar-res (128x72 sparkle stayed 0.0/Mpx while the
//! 640x480 probe hit 6.5@74, 13.0@99, 35.8@124 — WATCHDOG ABORT). So the
//! bar-res sparkle class must be the OTHER side of the gate: HIT pixels
//! (depth>0), still running the CURE4/CURE5 clamp-and-overshoot machinery.
//! This tool locates every bar-res sparkle-flagged texel on v9n-last and
//! classifies it (hit vs no-hit, edge adjacency, albedo/depth) to confirm
//! or refute that verdict directly, at the SAME 640x480 the probe actually
//! failed at (prior forensics, `rdirect_v9h_drift_forensics.rs`, only ran
//! at 128x72 — the wrong resolution for this question).
//!
//! Pipeline reused BYTE-FOR-BYTE from `examples/rdirect_train_v9k.rs::bar_res_probe`
//! (K-step recurrent settle at bar-res, undo-log-demod + inference-time
//! evidence clamp, AND the v9n hit-gate compose branch — instrument-audit
//! law: same formula as the live probe, not a reinvention) so the located
//! pixels are the SAME ones the real WATCHDOG probe flagged. Detection is
//! `sparkle_resid_per_mpx`'s own 3x3 local-peak criterion (SPARK_DELTA=0.15,
//! linear luminance err), copied unchanged from the same file.
//!
//! Domain: the loss operates in DEMOD-LOG space (`presented_dl`/`target_dl`,
//! plain MSE) — PRIMARY reporting domain. LINEAR radiance (via
//! `undo_log_demod`) reported alongside (sparkle itself is linear-luminance).
//! Every number states which domain it is in.
//!
//! n2n-TARGET-EQUIVALENT: reconstructs, at each flagged texel, the honest
//! K=8-draw noise2noise label the trainer would have used at THIS step —
//! repeated `GAIA_V9N_TARGET_REPEATS` times (fresh seeds) and averaged in
//! demod-log space (the loss's own space), exactly the `rdirect_v9h_drift_
//! forensics.rs` TARGET MEAN recipe. Full-frame renders are expensive at
//! 640x480 (no windowed headless integrator entry point exists), so this is
//! genuinely "if computable" — capped by `GAIA_V9N_MAX_TEXELS` (a full-frame
//! render per draw, `n_repeats * k_draws` draws per texel).
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
use scrying_glass::rdirect_unet::cpu::{Img, deserialize_weights};
use scrying_glass::rdirect_unet::MOTION_VECTOR_CHANNELS;
use scrying_glass::scene::{Camera, RenderScene};

const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;
const DRAW_A_SEED_BASE: u32 = 0x7abc;
const DRAW_B_SEED_BASE: u32 = 0xB222;
const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;
const SPARK_DELTA: f32 = 0.15;

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

/// `bar_res_probe`'s own gate switch (`examples/rdirect_train_v9k.rs::hitgate_on`),
/// COPIED — except the default here is TRUE, not false: v9n-last was
/// trained (and its watchdog probes were evaluated) with
/// `GAIA_V9N_HITGATE=1` for the entire run (verified: `scratch/v9n-train.log`
/// line 8, "V9N HIT-GATED BODY: GAIA_V9N_HITGATE=true"). Replicating the
/// LIVE probe means replicating that value, not the trainer file's own
/// opt-in default. `GAIA_V9N_HITGATE=0` still forces it off for A/B.
fn hitgate_on() -> bool {
    match std::env::var("GAIA_V9N_HITGATE").ok().as_deref() {
        Some("0") => false,
        _ => true,
    }
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
    let tag = "v9n-barres-forensics";
    let sky_reject = sky_history_reject();
    let gate_on = hitgate_on();
    eprintln!("[{tag}] GAIA_V7_SKY_HISTORY reject={sky_reject} GAIA_V9N_HITGATE gate_on={gate_on}");

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
    let n_repeats = env_u32("GAIA_V9N_TARGET_REPEATS", 32); // "many target draws"
    let gamma = evidence_clamp_gamma();

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let val_cam = all.iter().find(|(pn, _)| *pn == "orbit_-20").unwrap().1.clone();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = std::env::var("GAIA_V9N_BARRES_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("rdirect-weights-v9n-last.bin"));
    let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("read {wpath:?}: {e}"));
    let ema = deserialize_weights(&bytes).unwrap_or_else(|| panic!("deserialize {wpath:?}"));
    eprintln!("[{tag}] loaded {wpath:?} ({} bytes)", bytes.len());
    eprintln!("[{tag}] bar-res {tw}x{th} K={k} steps, ref_frames={ref_frames}, K_DRAWS={k_draws}, target n_repeats={n_repeats}");

    // ---- bar_res_probe pipeline, verbatim (rdirect_train_v9k.rs) ----
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
    let last_depth = &steps_depth[last];
    let last_normal = &steps_normal[last];
    // V9N hit-gated compose, byte-for-byte from `bar_res_probe` (rdirect_train_v9k.rs):
    // this step's OWN evidence composite (not the temporal `evidence_mean`
    // the ceiling uses), demod-log'd against this step's own albedo.
    // gate_on false leaves every px on the existing clamp-only path
    // (byte-identical old probe).
    let last_composite = evidence_composite_frame(&steps_low_e[last], &steps_low_d[last], low_w, low_h, tw, th);
    let mut net_clamped = vec![GVec3::ZERO; n];
    let mut ceiling_dl_all: Vec<[f32; OUTPUT_CHANNELS]> = vec![[0.0; OUTPUT_CHANNELS]; n];
    let mut gated_all = vec![false; n];
    for px in 0..n {
        let dl = GVec3::new(last_dl.data[px * 3], last_dl.data[px * 3 + 1], last_dl.data[px * 3 + 2]);
        let divisor = demod_divisor(last_albedo[px]);
        let ceiling_dl: [f32; OUTPUT_CHANNELS] = evidence_ceiling_demod_log(evidence_ceiling[px], gamma, last_albedo[px]);
        ceiling_dl_all[px] = ceiling_dl;
        if gate_on && last_depth[px] <= 0.0 {
            gated_all[px] = true;
            let e = target_demod_log(last_composite[px], last_albedo[px]);
            net_clamped[px] = undo_log_demod(GVec3::new(e[0], e[1], e[2]), divisor);
            continue;
        }
        let presented_dl = GVec3::new(dl.x.min(ceiling_dl[0]), dl.y.min(ceiling_dl[1]), dl.z.min(ceiling_dl[2]));
        net_clamped[px] = undo_log_demod(presented_dl, divisor);
    }

    let sp = sparkle_resid_per_mpx(&net_clamped, &teacher, tw, th);
    eprintln!("[{tag}] bar-res sparkle {sp:.1}/Mpx (reference: v9n-last training-time PROBE epoch 124 reported 35.8/Mpx at this same res/pose — this run's own trace RNG differs from training's, small drift expected)");

    let flagged = sparkle_flagged_pixels(&net_clamped, &teacher, tw, th);
    eprintln!("[{tag}] {} sparkle-flagged texels located", flagged.len());

    // Cap how many texels get the expensive n_repeats x k_draws-draw
    // target-mean treatment (full-frame render per draw — genuinely
    // expensive at bar-res, no windowed headless integrator entry point
    // exists). Default high enough to cover the small flagged set this
    // sparkle rate implies (~10-40 texels at 640x480).
    let max_texels = env_u32("GAIA_V9N_MAX_TEXELS", 64) as usize;
    let sample: Vec<(u32, u32, f32)> = flagged.iter().take(max_texels).cloned().collect();

    println!("# V9N BAR-RES SPARKLE CLASS FORENSICS — bar-res {tw}x{th}, val_cam=orbit_-20, weights={wpath:?}, gate_on={gate_on}");
    println!("# domain: dl = demod-log (loss space); lin = linear radiance (undo_log_demod). Luminance = 0.2126R+0.7152G+0.0722B.");
    println!("# columns: px err_lin(sparkle) hit(depth>0) depth | albedo | evidence_lin(E+D composite, this step) | net_raw_dl net_composed_lin(post-gate/clamp) | teacher_lin | ceiling_dl | target_dl_mean/target_lin_of_dl_mean(n2n-equivalent, n_repeats x K={k_draws} draws) | nbr3x3_hit(row-major, self=X)");

    let mut rows: Vec<serde_json::Value> = Vec::new();

    for &(x, y, e) in &sample {
        let px = (y * tw + x) as usize;
        let albedo = last_albedo[px];
        let divisor = demod_divisor(albedo);
        let cap = ceiling_dl_all[px];
        let is_hit = last_depth[px] > 0.0;
        let gated = gated_all[px];

        // 3x3 neighborhood hit-flags (silhouette-edge check): true=hit, false=no-hit.
        let mut nbr_hit = [[false; 3]; 3];
        for (dyi, dy) in (-1i32..=1).enumerate() {
            for (dxi, dx) in (-1i32..=1).enumerate() {
                let nx = (x as i32 + dx).clamp(0, tw as i32 - 1) as u32;
                let ny = (y as i32 + dy).clamp(0, th as i32 - 1) as u32;
                let npx = (ny * tw + nx) as usize;
                nbr_hit[dyi][dxi] = last_depth[npx] > 0.0;
            }
        }
        let edge_adjacent = nbr_hit.iter().flatten().any(|&h| h != is_hit);

        // ---- n2n-target-equivalent: n_repeats x k_draws reconstruction (this pose, last step) ----
        let mut target_dl_sum = GVec3::ZERO;
        let mut linear_grand_sum = GVec3::ZERO;
        let mut n_draws_total = 0u32;
        for r in 0..n_repeats {
            let mut radiance_sum = GVec3::ZERO;
            for kd in 0..k_draws {
                let seed = DRAW_B_SEED_BASE + (last as u32) * 257 + 11 + kd * 9973 + r * 1_000_003;
                let np_b = IntegratorParams { spp: 1, seed, ..IntegratorParams::default() };
                // Full-frame trace is expensive; no windowed headless
                // integrator entry point exists, so this measures the FULL
                // frame per draw and reads back just the flagged pixel
                // (same cost shape as the v9h forensics tool).
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
        let net_lin = net_clamped[px];
        let teacher_lin = teacher[px];
        let evidence_lin = last_composite[px];

        let nbr_str = format!(
            "[{}{}{} / {}{}{} / {}{}{}]",
            if nbr_hit[0][0] { "H" } else { "." }, if nbr_hit[0][1] { "H" } else { "." }, if nbr_hit[0][2] { "H" } else { "." },
            if nbr_hit[1][0] { "H" } else { "." }, "X", if nbr_hit[1][2] { "H" } else { "." },
            if nbr_hit[2][0] { "H" } else { "." }, if nbr_hit[2][1] { "H" } else { "." }, if nbr_hit[2][2] { "H" } else { "." },
        );

        println!(
            "px=({x:>3},{y:>3}) err_lin={e:.4} hit={is_hit} gated={gated} depth={:.4} | albedo=({:.4},{:.4},{:.4}) | evidence_lin=({:.4},{:.4},{:.4}) | net_raw_dl=({:.4},{:.4},{:.4}) net_composed_lin=({:.4},{:.4},{:.4}) | teacher_lin=({:.4},{:.4},{:.4}) | ceiling_dl=({:.4},{:.4},{:.4}) | target_dl_mean=({:.4},{:.4},{:.4}) target_lin_of_dl_mean=({:.4},{:.4},{:.4}) target_lin_direct={:?}draw=({:.4},{:.4},{:.4}) | nbr3x3_hit={nbr_str} edge_adjacent={edge_adjacent}",
            last_depth[px],
            albedo.x, albedo.y, albedo.z,
            evidence_lin.x, evidence_lin.y, evidence_lin.z,
            raw_dl.x, raw_dl.y, raw_dl.z,
            net_lin.x, net_lin.y, net_lin.z,
            teacher_lin.x, teacher_lin.y, teacher_lin.z,
            cap[0], cap[1], cap[2],
            target_dl_mean.x, target_dl_mean.y, target_dl_mean.z,
            target_lin_of_dl_mean.x, target_lin_of_dl_mean.y, target_lin_of_dl_mean.z,
            n_draws_total,
            target_lin_direct.x, target_lin_direct.y, target_lin_direct.z,
        );
        std::io::stdout().flush().ok();

        rows.push(serde_json::json!({
            "px": [x, y], "err_lin_lum": e,
            "hit": is_hit, "gated": gated, "depth": last_depth[px],
            "albedo": [albedo.x, albedo.y, albedo.z],
            "evidence_lin": [evidence_lin.x, evidence_lin.y, evidence_lin.z],
            "net_raw_dl": [raw_dl.x, raw_dl.y, raw_dl.z],
            "net_composed_lin": [net_lin.x, net_lin.y, net_lin.z],
            "teacher_lin": [teacher_lin.x, teacher_lin.y, teacher_lin.z],
            "ceiling_dl": cap,
            "target_dl_mean": [target_dl_mean.x, target_dl_mean.y, target_dl_mean.z],
            "target_lin_of_dl_mean": [target_lin_of_dl_mean.x, target_lin_of_dl_mean.y, target_lin_of_dl_mean.z],
            "target_lin_direct": [target_lin_direct.x, target_lin_direct.y, target_lin_direct.z],
            "n_draws_total": n_draws_total,
            "nbr3x3_hit": nbr_hit,
            "edge_adjacent": edge_adjacent,
            "normal": [last_normal[px].x, last_normal[px].y, last_normal[px].z],
        }));
    }

    // ---- class summary aggregation ----
    let n_hit = rows.iter().filter(|r| r["hit"].as_bool().unwrap()).count();
    let n_nohit = rows.len() - n_hit;
    let n_edge = rows.iter().filter(|r| r["edge_adjacent"].as_bool().unwrap()).count();
    let albedo_lum: Vec<f64> = rows.iter().map(|r| {
        let a: Vec<f64> = r["albedo"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]
    }).collect();
    let depths: Vec<f64> = rows.iter().map(|r| r["depth"].as_f64().unwrap()).collect();
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    let minmax = |v: &[f64]| (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (albedo_min, albedo_max) = minmax(&albedo_lum);
    let (depth_min, depth_max) = minmax(&depths);

    println!("\n# ---- CLASS VERDICT over {} sampled flagged texels (of {} total flagged) ----", rows.len(), flagged.len());
    println!("# hit (depth>0): {n_hit}/{} | no-hit (depth<=0): {n_nohit}/{}", rows.len(), rows.len());
    println!("# edge-adjacent (>=1 of 8 neighbors differs in hit-flag): {n_edge}/{}", rows.len());
    println!("# albedo luminance: mean={:.4} min={:.4} max={:.4}", mean(&albedo_lum), albedo_min, albedo_max);
    println!("# depth: mean={:.4} min={:.4} max={:.4}", mean(&depths), depth_min, depth_max);
    let verdict = if n_hit == rows.len() {
        "ALL sampled sparkle texels are HIT px (depth>0) — the hit-gate correctly excludes no-hit px from sparkling; the bar-res sparkle class is the CURE4/CURE5 clamp-and-overshoot machinery still acting on HIT pixels, NOT the no-hit/sky class the gate targeted."
    } else if n_nohit == rows.len() {
        "ALL sampled sparkle texels are NO-HIT px (depth<=0) — the gate is NOT holding at bar-res despite holding its class at training-crop res; investigate gate condition mismatch at 640x480."
    } else {
        "MIXED hit/no-hit sparkle texels — both classes contribute at bar-res; gate holds partially."
    };
    println!("# VERDICT: {verdict}");

    let out_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scratch/v9n-barres-forensics.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&serde_json::json!({
        "tag": tag, "weights": wpath.to_string_lossy(), "bar_res": [tw, th], "gate_on": gate_on,
        "k_draws": k_draws, "n_repeats": n_repeats,
        "sparkle_per_mpx_this_run": sp, "n_flagged": flagged.len(), "n_sampled": rows.len(),
        "texels": rows,
        "class_summary": {
            "n_hit": n_hit, "n_nohit": n_nohit, "n_edge_adjacent": n_edge,
            "albedo_lum_mean": mean(&albedo_lum), "albedo_lum_min": albedo_min, "albedo_lum_max": albedo_max,
            "depth_mean": mean(&depths), "depth_min": depth_min, "depth_max": depth_max,
        },
        "verdict": verdict,
    })).unwrap()).unwrap();
    eprintln!("[{tag}] wrote {out_path:?}");
    std::io::stderr().flush().ok();
}
