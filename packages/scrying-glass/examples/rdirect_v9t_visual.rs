//! V9T VISUAL EVIDENCE — plain + sparkle-marked PNGs of the finished v9t
//! run's composed bar-res (640x480) output, for Pascal's own eye-judgment.
//! Adapted (minimal) from `examples/rdirect_v9p_barres_forensics.rs`: same
//! bar-res compose pipeline (K=3 recurrent settle + v9n hit-gate + v9p
//! dark-cap + normal-ceiling branches, byte-for-byte from
//! `rdirect_train_v9k.rs::bar_res_probe`) and the same `sparkle_flagged_
//! pixels` criterion — but drops the forensics/JSON/per-texel n2n-target
//! reconstruction machinery entirely and instead WRITES PNGS (PNG writing
//! borrowed from `examples/rdirect_v9_eval_640.rs::write_png`).
//!
//! v9t trained with `GAIA_V9N_HITGATE=true GAIA_V9O_DARK_ALB=0.0001
//! GAIA_V9P_DARK_CAP_DELTA=0 GAIA_V9Q_DESPECKLE_DELTA=0` (scratch/v9t-train.log
//! lines 9-11) — "no caps": the v9p/v9q dark-hit caps are OFF (delta<=0
//! falls through to the ordinary evidence-ceiling branch), so this tool's
//! defaults match that exactly (dark_cap_delta default is 0 HERE, unlike
//! the v9p forensics tool's 0.10 default — v9t simply never used it).
//!
//! Env: GAIA_V9T_WEIGHTS (required — path to the .bin to eval),
//! GAIA_V9T_TAG (default "eval" — output filename suffix),
//! GAIA_V9N_HITGATE / GAIA_V9O_DARK_ALB / GAIA_V9P_DARK_CAP_DELTA
//! (override for A/B, matching the v9p tool's own convention),
//! GAIA_V7_SKY_HISTORY=reject (matches training; NOT defaulted on here —
//! set it explicitly to replicate the v9t run).
//!
//! Run: GAIA_V7_SKY_HISTORY=reject GAIA_V9T_WEIGHTS=data/rdirect-weights-v9t.bin \
//!      GAIA_V9T_TAG=best cargo run --release -j2 --example rdirect_v9t_visual
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
const RING_RADIUS: f32 = 5.0;
const RING_THICKNESS: f32 = 1.0; // pixels get marked if |dist-RING_RADIUS| <= this

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

/// COPIED from `rdirect_v9p_barres_forensics.rs::hitgate_on` — v9t trained
/// with `GAIA_V9N_HITGATE=true` (scratch/v9t-train.log line 9), default
/// here is TRUE to match.
fn hitgate_on() -> bool {
    match std::env::var("GAIA_V9N_HITGATE").ok().as_deref() {
        Some("0") => false,
        _ => true,
    }
}
/// COPIED from the v9p tool — v9t trained with `GAIA_V9O_DARK_ALB=0.0001`
/// (scratch/v9t-train.log line 10).
fn v9o_dark_alb() -> f32 {
    env_f32("GAIA_V9O_DARK_ALB", 1.0e-4)
}
/// COPIED from the v9p tool, DEFAULT CHANGED to 0.0 (v9t's own value,
/// scratch/v9t-train.log line 11: "GAIA_V9P_DARK_CAP_DELTA=0" — "no caps").
/// The v9p tool's 0.10 default does NOT apply to v9t; do not reuse it.
fn v9p_dark_cap_delta() -> f32 {
    env_f32("GAIA_V9P_DARK_CAP_DELTA", 0.0)
}

/// Per-channel 3x3-neighborhood MEDIAN of a demod-log evidence buffer,
/// COPIED VERBATIM from the v9p tool (itself copied from
/// `rdirect_train_v9k.rs::median3x3_dl`).
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
/// linear-luminance err, 3x3 local peak) — returns the flagged (x,y,err)
/// list, same as the v9p forensics tool.
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
    eprintln!("[v9t-visual] wrote {}", path.display());
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
/// Circles every flagged (x,y) with a pure-red RING_RADIUS-px ring (outline
/// only, not filled) drawn into an already-sRGB byte buffer.
fn mark_flagged(bytes: &mut [u8], w: u32, h: u32, flagged: &[(u32, u32, f32)]) {
    let (wi, hi) = (w as i32, h as i32);
    let r = RING_RADIUS.ceil() as i32 + 1;
    for &(fx, fy, _) in flagged {
        let (cx, cy) = (fx as i32, fy as i32);
        for dy in -r..=r {
            for dx in -r..=r {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if (dist - RING_RADIUS).abs() > RING_THICKNESS {
                    continue;
                }
                let (px, py) = (cx + dx, cy + dy);
                if px < 0 || px >= wi || py < 0 || py >= hi {
                    continue;
                }
                let idx = ((py as usize) * w as usize + px as usize) * 3;
                bytes[idx] = 255;
                bytes[idx + 1] = 0;
                bytes[idx + 2] = 0;
            }
        }
    }
}

fn main() {
    let tag = env_str("GAIA_V9T_TAG", "eval");
    let sky_reject = sky_history_reject();
    let gate_on = hitgate_on();
    let dark_alb = v9o_dark_alb();
    let dark_cap_delta = v9p_dark_cap_delta();
    eprintln!(
        "[v9t-visual:{tag}] GAIA_V7_SKY_HISTORY reject={sky_reject} GAIA_V9N_HITGATE gate_on={gate_on} GAIA_V9O_DARK_ALB={dark_alb} GAIA_V9P_DARK_CAP_DELTA={dark_cap_delta}"
    );

    let Some((device, queue)) = headless_device() else {
        panic!("[v9t-visual:{tag}] no GPU");
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
    let wpath = std::env::var("GAIA_V9T_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("rdirect-weights-v9t.bin"));
    let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("read {wpath:?}: {e}"));
    let ema = deserialize_weights(&bytes).unwrap_or_else(|| panic!("deserialize {wpath:?}"));
    eprintln!("[v9t-visual:{tag}] loaded {wpath:?} ({} bytes) sha256={}", bytes.len(), weights_sha256(&ema));
    eprintln!("[v9t-visual:{tag}] bar-res {tw}x{th} K={k} steps, ref_frames={ref_frames}");

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
    // V9T compose, byte-for-byte from `bar_res_probe` (rdirect_train_v9k.rs),
    // same three branches the v9p forensics tool implements — nohit-gate /
    // dark-cap (disabled here, delta=0, "no caps") / normal-ceiling.
    let last_composite = evidence_composite_frame(&steps_low_e[last], &steps_low_d[last], low_w, low_h, tw, th);
    let last_evidence_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n)
        .map(|px| target_demod_log(last_composite[px], last_albedo[px]))
        .collect();
    let median_dl = median3x3_dl(&last_evidence_dl, tw, th);
    let mut net_clamped = vec![GVec3::ZERO; n];
    for px in 0..n {
        let dl = GVec3::new(last_dl.data[px * 3], last_dl.data[px * 3 + 1], last_dl.data[px * 3 + 2]);
        let divisor = demod_divisor(last_albedo[px]);
        let is_nohit = last_depth[px] <= 0.0;
        if gate_on && is_nohit {
            let e = last_evidence_dl[px];
            net_clamped[px] = undo_log_demod(GVec3::new(e[0], e[1], e[2]), divisor);
            continue;
        }
        let is_dark_hit = !is_nohit && dark_alb > 0.0 && lum(last_albedo[px]) <= dark_alb;
        if gate_on && is_dark_hit && dark_cap_delta > 0.0 {
            let m = median_dl[px];
            let cap = [m[0] + dark_cap_delta, m[1] + dark_cap_delta, m[2] + dark_cap_delta];
            let capped = GVec3::new(dl.x.min(cap[0]), dl.y.min(cap[1]), dl.z.min(cap[2]));
            net_clamped[px] = undo_log_demod(capped, divisor);
            continue;
        }
        let ceiling_dl: [f32; OUTPUT_CHANNELS] = evidence_ceiling_demod_log(evidence_ceiling[px], gamma, last_albedo[px]);
        let presented_dl = GVec3::new(dl.x.min(ceiling_dl[0]), dl.y.min(ceiling_dl[1]), dl.z.min(ceiling_dl[2]));
        net_clamped[px] = undo_log_demod(presented_dl, divisor);
    }

    let flagged = sparkle_flagged_pixels(&net_clamped, &teacher, tw, th);
    let sp = (flagged.len() as f64) * 1.0e6 / (tw as f64 * th as f64);
    eprintln!("[v9t-visual:{tag}] {} sparkle-flagged texels ({sp:.1}/Mpx) at {tw}x{th}", flagged.len());
    println!("[v9t-visual:{tag}] weights={wpath:?} flagged={} sparkle={sp:.1}/Mpx", flagged.len());

    let out_dir = env_str("GAIA_V9_EVAL_OUT", "scratch");
    let plain_bytes = img_to_srgb_bytes(&net_clamped, 1.0);
    write_png_rgb_bytes(&plain_bytes, tw, th, Path::new(&out_dir).join(format!("v9t-eval-{tag}.png")).as_path());

    let mut marked_bytes = plain_bytes.clone();
    mark_flagged(&mut marked_bytes, tw, th, &flagged);
    write_png_rgb_bytes(&marked_bytes, tw, th, Path::new(&out_dir).join(format!("v9t-eval-{tag}-marked.png")).as_path());

    std::io::stdout().flush().ok();
    std::io::stderr().flush().ok();
}
