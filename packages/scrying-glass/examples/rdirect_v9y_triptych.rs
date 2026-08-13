//! V9Y PIXEL-TRUTH TRIPTYCH — writes THREE fixed-name 640x480 PNGs from
//! the SAME bar-res pose so ground truth is directly eyeballable:
//!   scratch/v9y-<tag>-teacher.png = converged-teacher tracer target
//!                                   (default 512spp chunked accumulation,
//!                                   matches training; independent of
//!                                   weights, identical for best/last)
//!   scratch/v9y-<tag>-net.png     = v9y net's own composed bar-res output
//!   scratch/v9y-<tag>-diff.png    = per-px |net-teacher| luminance heatmap
//!                                   (black=0, white>=0.15, same SPARK_DELTA)
//!
//! Forked from `examples/rdirect_v9w2_triptych.rs`, re-matched to v9y's
//! ACTUAL training config (scratch/v9y-train.log). v9y shares v9w2's V9V
//! input clamp and V9U converged-teacher setup but DIFFERS on the V9P/V9Q
//! dark-hit caps:
//!
//!  1. INPUT CLAMP (V9V) — v9y trained with `GAIA_V9V_INPUT_CLAMP=0.10`
//!     (task mandate; matches v9w2's own value). Default here: 0.10.
//!  2. DARK-HIT COMPOSE (V9P/V9Q) — v9y's log prints
//!     `GAIA_V9P_DARK_CAP_DELTA=0` and `GAIA_V9Q_DESPECKLE_DELTA=0` (BOTH
//!     ZEROED this run — differs from v9w2, which had both at 0.10!).
//!     With both deltas <=0.0, dark-hit px take neither the V9P nor V9Q
//!     branch and fall straight through to the ordinary evidence ceiling,
//!     same as v9n. Defaults here: 0.0/0.0 (byte-matches v9y, NOT v9w2).
//!  3. TEACHER (V9U) — v9y trained with `GAIA_V9_TEACHER_SPP=512
//!     chunk=100` (scratch/v9y-train.log), same convention as v9w2. Ported
//!     `render_converged_lin` as this tool's teacher when
//!     `GAIA_V9_TEACHER_SPP>0` (default HERE: 512).
//!
//! Env: GAIA_V9Y_WEIGHTS (default data/rdirect-weights-v9y-last.bin —
//! i.e. the LAST checkpoint; pass the BEST checkpoint's path explicitly
//! for that eval), GAIA_V9Y_TAG (default "eval" — output filename infix,
//! e.g. "best"/"last" -> scratch/v9y-best-*.png / scratch/v9y-last-*.png),
//! GAIA_V9Y_SKIP_TEACHER=1 (skip writing the teacher PNG to disk — the
//! teacher render is pose/weights-independent, so a second run against a
//! different checkpoint at the SAME pose need not re-emit an identical
//! file; the teacher is still rendered and used for the printed
//! sparkle/resid either way), GAIA_V9N_HITGATE / GAIA_V9O_DARK_ALB /
//! GAIA_V9P_DARK_CAP_DELTA / GAIA_V9Q_DESPECKLE_DELTA / GAIA_V9V_INPUT_CLAMP
//! (override for A/B, matching v9k/v9p/v9w2 tool conventions),
//! GAIA_V9_TEACHER_SPP / GAIA_V9_TEACHER_CHUNK (converged-teacher
//! accumulation), GAIA_V7_SKY_HISTORY=reject (matches training; NOT
//! defaulted on here — set it explicitly to replicate the v9y run).
//!
//! GAIA_V9V_MOTION is intentionally NOT reproduced here (see v9w2's doc
//! for the full argument): static pose => mv=0 is the trained
//! bar-res-probe behavior, not a mismatch.
//!
//! Run (BEST checkpoint):
//!   GAIA_V7_SKY_HISTORY=reject GAIA_V9Y_WEIGHTS=data/rdirect-weights-v9y.bin \
//!     GAIA_V9Y_TAG=best GAIA_V9V_INPUT_CLAMP=0.10 \
//!     cargo run --release -j2 --example rdirect_v9y_triptych
//! Run (LAST checkpoint, teacher file already have from the best run):
//!   GAIA_V7_SKY_HISTORY=reject GAIA_V9Y_WEIGHTS=data/rdirect-weights-v9y-last.bin \
//!     GAIA_V9Y_TAG=last GAIA_V9Y_SKIP_TEACHER=1 GAIA_V9V_INPUT_CLAMP=0.10 \
//!     cargo run --release -j2 --example rdirect_v9y_triptych
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
use scrying_glass::rdirect_unet::cpu::{Img, deserialize_weights, weights_sha256};
use scrying_glass::rdirect_unet::MOTION_VECTOR_CHANNELS;
use scrying_glass::scene::{Camera, RenderScene};

const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;
const DRAW_A_SEED_BASE: u32 = 0x7abc;
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
fn env_f32(n: &str, d: f32) -> f32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_str(n: &str, d: &str) -> String {
    std::env::var(n).unwrap_or_else(|_| d.to_string())
}
fn env_bool(n: &str) -> bool {
    matches!(std::env::var(n).ok().as_deref(), Some("1") | Some("true"))
}

/// COPIED from `rdirect_v9p_barres_forensics.rs::hitgate_on` — v9y trained
/// with `GAIA_V9N_HITGATE=true`, default here is TRUE to match.
fn hitgate_on() -> bool {
    match std::env::var("GAIA_V9N_HITGATE").ok().as_deref() {
        Some("0") => false,
        _ => true,
    }
}
/// v9y trained with `GAIA_V9O_DARK_ALB=0.0001` (the function's own default
/// in `rdirect_train_v9k.rs`, unchanged here).
fn v9o_dark_alb() -> f32 {
    env_f32("GAIA_V9O_DARK_ALB", 1.0e-4)
}
/// v9y trained with `GAIA_V9P_DARK_CAP_DELTA=0` — ZEROED this run (differs
/// from v9w2's 0.10!). Default here: 0.0 (byte-matches v9y's train log).
fn v9p_dark_cap_delta() -> f32 {
    env_f32("GAIA_V9P_DARK_CAP_DELTA", 0.0)
}
/// v9y trained with `GAIA_V9Q_DESPECKLE_DELTA=0` — ZEROED this run
/// (differs from v9w2's 0.10!). Default here: 0.0 (byte-matches v9y's
/// train log). With both V9P and V9Q deltas <=0.0, dark-hit px fall
/// straight through to the ordinary evidence ceiling (v9n byte-identical).
fn v9q_despeckle_delta() -> f32 {
    env_f32("GAIA_V9Q_DESPECKLE_DELTA", 0.0)
}
/// Shared despeckle-cap primitive, ported verbatim from
/// `rdirect_train_v9k.rs::despeckle_cap_lin` — given the FULL FRAME's
/// pre-cap composed LINEAR radiance (frozen) and one dark-class pixel's
/// index, returns `neigh_max_lin + delta` over its up-to-8 in-bounds 3x3
/// neighbors, self excluded. (Dead code at v9y's own defaults since both
/// deltas are 0; kept for A/B override parity with v9w2/v9k.)
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

/// V9V INPUT FIREFLY CLAMP, ported verbatim from
/// `rdirect_train_v9k.rs::v9v_input_clamp_delta` — v9y trained with
/// `GAIA_V9V_INPUT_CLAMP=0.10` (task mandate, matches v9w2's own value).
/// Default HERE: 0.10 (NOT the shared function's byte-identical-disable
/// default of 0.0, which would silently mismatch this eval).
fn v9v_input_clamp_delta() -> f32 {
    env_f32("GAIA_V9V_INPUT_CLAMP", 0.10)
}
/// Ported verbatim from `rdirect_train_v9k.rs::apply_input_clamp`. `base`
/// layout (`pixel_features_split`): [0..12)=E's 2x2x3 demod-log taps,
/// [12..24)=D's 2x2x3 demod-log taps, tap stride 3 (one f32 per channel).
/// `delta<=0.0` is a no-op.
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
                }
            }
        }
    }
}

/// V9U CONVERGED-TEACHER SPP — v9y trained with `GAIA_V9_TEACHER_SPP=512
/// chunk=100`. Default HERE: 512/100 (matches v9y's OWN training judge —
/// NOT 0, which would fall back to the old noisy `ref_frames` average).
fn env_teacher_spp() -> u32 {
    env_u32("GAIA_V9_TEACHER_SPP", 512)
}
fn env_teacher_chunk() -> u32 {
    env_u32("GAIA_V9_TEACHER_CHUNK", 100)
}
/// Ported verbatim from `rdirect_train_v9k.rs::render_converged_lin` —
/// byte-for-byte `rdirect_reference.rs::render_converged`'s algorithm
/// (equal-size chunks of `trace_headless_split` at spp:1, summed in linear
/// radiance, weighted by chunk size so the result is the exact overall
/// mean regardless of chunking).
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

/// Per-channel 3x3-neighborhood MEDIAN of a demod-log evidence buffer,
/// COPIED VERBATIM from `rdirect_train_v9k.rs::median3x3_dl`.
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

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}
fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h as f32 }
}

/// `sparkle_resid_per_mpx`'s OWN detection, unchanged (SPARK_DELTA=0.15,
/// linear-luminance err, 3x3 local peak).
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

// ── PNG writing, borrowed from `rdirect_v9_eval_640.rs::write_png` ──
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}
fn write_png_rgb_bytes(bytes: &[u8], w: u32, h: u32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(bytes).unwrap();
    eprintln!("[v9y-triptych] wrote {}", path.display());
}
fn img_to_srgb_bytes(img: &[GVec3], exposure: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(img.len() * 3);
    for px in img {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    bytes
}
fn main() {
    // GAIA_V9_TRIPTYCH_PREFIX: output filename + log prefix, default "v9y"
    // (byte-identical to the pre-generalization hardcoded behavior).
    let prefix = env_str("GAIA_V9_TRIPTYCH_PREFIX", "v9y");
    let tag = env_str("GAIA_V9Y_TAG", "eval");
    let skip_teacher = env_bool("GAIA_V9Y_SKIP_TEACHER");
    let sky_reject = sky_history_reject();
    let gate_on = hitgate_on();
    let dark_alb = v9o_dark_alb();
    let dark_cap_delta = v9p_dark_cap_delta();
    let despeckle_delta = v9q_despeckle_delta();
    let input_clamp = v9v_input_clamp_delta();
    let teacher_spp = env_teacher_spp();
    let teacher_chunk = env_teacher_chunk();
    eprintln!(
        "[{prefix}-triptych:{tag}] GAIA_V7_SKY_HISTORY reject={sky_reject} GAIA_V9N_HITGATE gate_on={gate_on} GAIA_V9O_DARK_ALB={dark_alb} GAIA_V9P_DARK_CAP_DELTA={dark_cap_delta} GAIA_V9Q_DESPECKLE_DELTA={despeckle_delta} GAIA_V9V_INPUT_CLAMP={input_clamp} GAIA_V9_TEACHER_SPP={teacher_spp} chunk={teacher_chunk}"
    );

    let Some((device, queue)) = headless_device() else {
        panic!("[{prefix}-triptych:{tag}] no GPU");
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
    let gamma = evidence_clamp_gamma();

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let val_cam = all.iter().find(|(pn, _)| *pn == "orbit_-20").unwrap().1.clone();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    // GAIA_V9_TRIPTYCH_WEIGHTS (new, preferred) -> GAIA_V9Y_WEIGHTS (old
    // name, still honored) -> compiled default. Old callers unaffected.
    let wpath = std::env::var("GAIA_V9_TRIPTYCH_WEIGHTS")
        .or_else(|_| std::env::var("GAIA_V9Y_WEIGHTS"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("rdirect-weights-v9y-last.bin"));
    let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("read {wpath:?}: {e}"));
    let ema = deserialize_weights(&bytes).unwrap_or_else(|| panic!("deserialize {wpath:?}"));
    eprintln!("[{prefix}-triptych:{tag}] loaded {wpath:?} ({} bytes) sha256={}", bytes.len(), weights_sha256(&ema));
    eprintln!("[{prefix}-triptych:{tag}] bar-res {tw}x{th} K={k} steps, ref_frames={ref_frames}");

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
    // V9U: converged-teacher when teacher_spp>0 (default 512 here, matches
    // v9y's OWN training judge), else the old ref_frames average.
    let teacher: Vec<GVec3> = if teacher_spp > 0 {
        render_converged_lin(
            &device, &queue, &bvh, &val_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, teacher_spp, teacher_chunk,
        )
    } else {
        let (e_full, d_full) = trace_headless_split(
            &device, &queue, &bvh, &val_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
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
    for s in 0..k as usize {
        let mut img = Img::zeros(th as usize, tw as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS);
        for ty in 0..th {
            for tx in 0..tw {
                let px = (ty * tw + tx) as usize;
                let mut base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                    &steps_low_e[s], &steps_low_d[s], low_w, low_h, tw, th, tx, ty,
                    steps_albedo[s][px], steps_normal[s][px], steps_depth[s][px], Vec2::ZERO,
                );
                // V9V input firefly clamp — matches training's bar_res_probe.
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
    // V9N/V9O/V9P/V9Q compose, byte-for-byte from `bar_res_probe`
    // (rdirect_train_v9k.rs) — TWO-PASS: precap_lin frozen first (no-hit
    // evidence passthrough / V9Q dark-hit kept-uncapped / V9P median cap /
    // ordinary evidence ceiling), THEN V9Q's post-compose despeckle cap
    // applied against precap_lin's own 3x3 neighborhood at dark-hit px.
    // At v9y's own defaults (both deltas 0) the V9P/V9Q branches below are
    // dead and every dark-hit px falls through to the ordinary ceiling.
    let last_composite = evidence_composite_frame(&steps_low_e[last], &steps_low_d[last], low_w, low_h, tw, th);
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

    let flagged = sparkle_flagged_pixels(&net_clamped, &teacher, tw, th);
    let sp = (flagged.len() as f64) * 1.0e6 / (tw as f64 * th as f64);
    // `rmse_lin`, byte-for-byte from `rdirect_train_v9k.rs` — PER-CHANNEL
    // RGB RMSE over the whole frame (NOT luminance-only), the exact metric
    // training's own PROBE line prints as `resid`.
    let resid = {
        let mut s = 0.0f64;
        for px in 0..n {
            let d = net_clamped[px] - teacher[px];
            s += (d.x * d.x + d.y * d.y + d.z * d.z) as f64;
        }
        (s / (n as f64 * 3.0)).sqrt()
    };
    eprintln!("[{prefix}-triptych:{tag}] {} sparkle-flagged texels ({sp:.1}/Mpx) resid={resid:.4} at {tw}x{th}", flagged.len());
    println!("[{prefix}-triptych:{tag}] weights={wpath:?} flagged={} sparkle={sp:.1}/Mpx resid={resid:.4}", flagged.len());

    let out_dir = env_str("GAIA_V9_EVAL_OUT", "scratch");

    // (a) teacher — converged tracer target, plain sRGB. Weights/pose-
    // independent (same val_cam every run) — skip the disk write when
    // GAIA_V9Y_SKIP_TEACHER=1 (e.g. a second run against a different
    // checkpoint at the identical pose that already has the file).
    if !skip_teacher {
        let teacher_bytes = img_to_srgb_bytes(&teacher, 1.0);
        write_png_rgb_bytes(&teacher_bytes, tw, th, Path::new(&out_dir).join(format!("{prefix}-{tag}-teacher.png")).as_path());
    }

    // (b) net — v9y's own composed bar-res output, plain sRGB.
    let net_bytes = img_to_srgb_bytes(&net_clamped, 1.0);
    write_png_rgb_bytes(&net_bytes, tw, th, Path::new(&out_dir).join(format!("{prefix}-{tag}-net.png")).as_path());

    // (c) diff — per-px |net-teacher| LINEAR luminance heatmap, grayscale
    // packed as RGB (no sRGB curve — this is an error magnitude map, not a
    // radiance image): black=0, white>=SPARK_DELTA (0.15), same scale the
    // sparkle criterion itself uses.
    let mut diff_bytes = Vec::with_capacity(n * 3);
    for px in 0..n {
        let e = (lum(net_clamped[px]) - lum(teacher[px])).abs();
        let v = ((e / SPARK_DELTA).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        diff_bytes.push(v);
        diff_bytes.push(v);
        diff_bytes.push(v);
    }
    write_png_rgb_bytes(&diff_bytes, tw, th, Path::new(&out_dir).join(format!("{prefix}-{tag}-diff.png")).as_path());

    std::io::stdout().flush().ok();
    std::io::stderr().flush().ok();
}
