//! R-DIRECT v9-body 640×480 EVAL (FINISH ATOM step 2) — loads the trainer's
//! BEST checkpoint (`data/rdirect-weights-v9body.bin`) and runs it at NATIVE
//! render resolution (640×480, `UnetConfig::default()`) on a HELD-OUT pose
//! ("orbit_-20", never in the training camera set — see
//! `rdirect_train_v9.rs::main`'s `val_seq`) through the SAME measurement the
//! trainer's `run_monitor` uses (`sparkle_resid_per_mpx`/`rmse_lin`/
//! `highlight_ratio`, copied verbatim below — no lib change).
//!
//! TWO fixes applied here that the trainer's own `settle()` does NOT have
//! (found during the autopsy, see `scratch/v9-autopsy.md`):
//!   1. UNDO-LOG-DEMOD: the conv net's raw output is demod-log radiance
//!      (`rdirect_unet.rs` module doc: "Output channel count (RGB demod-log
//!      radiance)"), but `rdirect_train_v9.rs::settle()` returns it AS-IS
//!      and compares it directly against the LINEAR teacher — an
//!      apples-to-oranges space mismatch. This file inverts it
//!      (`undo_log_demod`, duplicated math from `rdirect.rs` — that fn is
//!      private, not touched) before every metric/PNG, both with and
//!      without the clamp, so numbers are honest.
//!   2. v7's STRUCTURAL EVIDENCE CLAMP, INFERENCE-ONLY: `presented_dl =
//!      min(net_dl, ceiling_dl)` per channel, `ceiling_dl =
//!      evidence_ceiling_demod_log(local_max_3x3(temporal-mean evidence),
//!      gamma, hi_albedo)` — `rdirect.rs`'s OWN exported fns
//!      (`evidence_clamp_gamma`/`local_max_3x3`/`evidence_composite_frame`/
//!      `evidence_ceiling_demod_log`), the same ones
//!      `direct_render_sequence_hist_split` (v8d's real inference path)
//!      wires internally — v9's `cpu::UnetWeights` has no clamp act, so
//!      this is bolted on at the eval boundary, NOT inside the net.
//!
//! Env: GAIA_V9_EVAL_W/H (640/480), GAIA_V9_EVAL_K (3, matches trainer
//! val_seq step count), GAIA_V9_EVAL_REF (64, teacher frames),
//! GAIA_V7_CLAMP_GAMMA (1.5 default), GAIA_V9_EVAL_OUT (scratch/).
//!
//! Run: cargo run --release -j2 --example rdirect_v9_eval_640

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, split_aov, trace_headless_aov, trace_headless_split,
};
use scrying_glass::rdirect::{
    CamPose, HIST_FEATURES_SPLIT, INPUT_FEATURES_SPLIT, OUTPUT_CHANNELS, bilinear_vec3,
    evidence_clamp_gamma, evidence_ceiling_demod_log, evidence_composite_frame,
    hist_features_split, local_max_3x3, pixel_features_split, sky_history_reject,
};
use scrying_glass::rdirect_unet::cpu::{Img, deserialize_weights, weights_sha256};
use scrying_glass::rdirect_unet::MOTION_VECTOR_CHANNELS;
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;
// v8d parity seed families (rdirect_train_v9.rs) — reused so the input
// evidence draw is the SAME kind of draw the net was trained to denoise.
const DRAW_A_SEED_BASE: u32 = 0x7abc;
// Numerical floor under albedo before dividing — VIII-1's `ALBEDO_DEMOD_EPS`
// (`rdirect.rs::ALBEDO_DEMOD_EPS`, duplicated: the fn that uses it,
// `demod_divisor`, is private).
const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_str(n: &str, d: &str) -> String {
    std::env::var(n).unwrap_or_else(|_| d.to_string())
}

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}
fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h as f32 }
}

/// Duplicated from `rdirect.rs` (private there) — divisor for demod-log.
fn demod_divisor(albedo: GVec3) -> GVec3 {
    if albedo.length_squared() > NO_HIT_ALBEDO_THRESHOLD_SQ {
        albedo + GVec3::splat(ALBEDO_DEMOD_EPS)
    } else {
        GVec3::ONE
    }
}
/// Duplicated from `rdirect.rs` (private there) — inverse of `log_demod`:
/// `presented = (exp(dl)-1).max(0) * divisor`. THE fix for gap #1 above.
fn undo_log_demod(dl: GVec3, divisor: GVec3) -> GVec3 {
    let expm1 = GVec3::new(dl.x.exp() - 1.0, dl.y.exp() - 1.0, dl.z.exp() - 1.0);
    GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * divisor
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}
fn write_png(img: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[v9eval] wrote {}", path.display());
}

// ── the trainer's OWN metric formulas, copied verbatim (rdirect_train_v9.rs)
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

/// Reproject step s-1 into step s (v8d/`rdirect_train_v9.rs::reproject_prev`,
/// unchanged math) — needed to feed `build_input_img`'s recurrent history
/// channels the same way the trainer's `settle` does.
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
                let s = bilinear_vec3(prev_out_dl, fx, fy, pw, ph);
                ([s.x, s.y, s.z], 1.0, mv)
            } else {
                ([0.0; 3], 0.0, [0.0; 2])
            }
        }
    }
}

struct Step {
    low_e: Vec<GVec3>,
    low_d: Vec<GVec3>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
}

fn build_input_img(
    steps: &[Step], cams: &[CamPose], step_idx: usize, prev_out_dl: Option<&[GVec3]>,
    low_w: u32, low_h: u32, tw: u32, th: u32, sky_reject: bool,
) -> Img {
    let step = &steps[step_idx];
    let mut img = Img::zeros(th as usize, tw as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS);
    for ty in 0..th {
        for tx in 0..tw {
            let px = (ty * tw + tx) as usize;
            let base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                &step.low_e, &step.low_d, low_w, low_h, tw, th, tx, ty,
                step.albedo[px], step.normal[px], step.depth[px], Vec2::ZERO,
            );
            let (prev_dl, valid, mv) = match (step_idx, prev_out_dl) {
                (0, _) | (_, None) => ([0.0f32; 3], 0.0f32, [0.0f32; 2]),
                (_, Some(prev)) => {
                    let prev_step = &steps[step_idx - 1];
                    reproject_prev(
                        &cams[step_idx], step.depth[px], step.normal[px], tx, ty, tw, th,
                        &cams[step_idx - 1], prev, &prev_step.depth, &prev_step.normal, tw, th,
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
    let sky_reject = sky_history_reject();
    let tw = env_u32("GAIA_V9_EVAL_W", 640);
    let th = env_u32("GAIA_V9_EVAL_H", 480);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V9_EVAL_K", 3); // matches trainer val_seq step count
    let ref_frames = env_u32("GAIA_V9_EVAL_REF", 64);
    let highlight_pctl = 0.05f32;
    let gamma = evidence_clamp_gamma();
    let out_dir = env_str("GAIA_V9_EVAL_OUT", "scratch");

    eprintln!("[v9eval] {tw}x{th} K={k} ref_frames={ref_frames} gamma={gamma} sky_reject={sky_reject}");

    let Some((device, queue)) = headless_device() else { panic!("[v9eval] no GPU"); };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris: Vec<LeafTriangle> = scene.leaf_triangles();
    let bvh = Bvh::build(&base_tris, &BvhParams::default());

    // HELD-OUT pose: "orbit_-20" — the trainer's own `val_seq` camera,
    // never in `train_cams` (front/wide/orbit_+20/mirror).
    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let base_cam = all.iter().find(|(n, _)| *n == "orbit_-20").unwrap().1.clone();

    // Render K steps at native res, pan_step=0 (matches trainer's val_seq
    // call: `render_pose_seq(..., pan_step=0.0, evidence_spp=1, ...)`).
    let t0 = Instant::now();
    let n = (tw * th) as usize;
    let mut steps = Vec::with_capacity(k as usize);
    let mut cams = Vec::with_capacity(k as usize);
    for step in 0..k {
        let cam = base_cam;
        cams.push(cam_pose(&cam, tw, th));
        let np_a = IntegratorParams { spp: 1, seed: DRAW_A_SEED_BASE + step * 131 + 5, ..IntegratorParams::default() };
        let (low_e, low_d) = trace_headless_split(
            &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np_a,
        );
        let (albedo, normal, depth) = split_aov(&trace_headless_aov(
            &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
        ));
        steps.push(Step { low_e, low_d, albedo, normal, depth });
    }
    // Converged teacher, LAST step's camera (`settle`'s contract: compare
    // against the last step of the sequence).
    let (e_full, d_full) = trace_headless_split(
        &device, &queue, &bvh, &base_cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
        &IntegratorParams::default(),
    );
    let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();
    eprintln!("[v9eval] rendered {k} steps + {ref_frames}-frame teacher in {:.1}s", t0.elapsed().as_secs_f64());

    // Load the BEST checkpoint (fully-conv — forwards at any res).
    // GAIA_V9_EVAL_TAG (default "v9body", the STAGE 2 run's tag) selects
    // which trainer's checkpoint to eval — `data/rdirect-weights-<tag>.bin`,
    // the SAME naming convention every v9/v9b/v9c trainer writes to.
    let eval_tag = env_str("GAIA_V9_EVAL_TAG", "v9body");
    let wpath = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("data/rdirect-weights-{eval_tag}.bin"));
    let bytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("[v9eval] read {}: {e}", wpath.display()));
    let mut net = deserialize_weights(&bytes).unwrap_or_else(|| panic!("[v9eval] deserialize {}", wpath.display()));
    eprintln!("[v9eval] checkpoint={} sha256={} (trained render {}x{})", wpath.display(), weights_sha256(&net), net.config.render_w, net.config.render_h);
    // Fully-convolutional (module doc, `rdirect_unet.rs`): the saved
    // `config.render_w/h` is just the LABEL the trainer stamped (its own
    // wall-budget res, 128x72); `forward`'s only use of it is an assert on
    // the input Img's dims (`rdirect_unet.rs:1074`), not a tensor-shape
    // constant — override to the eval's native 640x480 (out_refine was
    // built `None` at train time since output==render there; keeping
    // output_w/h == render_w/h here preserves that, so no missing-refine
    // panic).
    net.config.render_w = tw as usize;
    net.config.render_h = th as usize;
    net.config.output_w = tw as usize;
    net.config.output_h = th as usize;
    let net = net;

    // Recurrent forward through the K steps (mirrors `history_forward`/
    // `settle`), keeping EVERY step's raw demod-log Img (for the clamp's
    // temporal-mean evidence accumulator) and the reprojected chain.
    let mut chain_dl: Vec<Vec<GVec3>> = Vec::with_capacity(k as usize);
    let mut out_imgs: Vec<Img> = Vec::with_capacity(k as usize);
    for s in 0..k as usize {
        let prev: Option<&[GVec3]> = if s == 0 { None } else { Some(&chain_dl[s - 1]) };
        let input = build_input_img(&steps, &cams, s, prev, low_w, low_h, tw, th, sky_reject);
        let (out, _cache) = net.forward(&input);
        chain_dl.push(img_to_vec3(&out));
        out_imgs.push(out);
    }
    eprintln!("[v9eval] forward chain done in {:.1}s total", t0.elapsed().as_secs_f64());

    // Temporal-mean evidence accumulator across the K steps (v8d/`rdirect.rs
    // ::EvidenceAccum` contract, reimplemented inline — that struct is pub
    // but its `.sum` field is private, easier to fold manually here).
    let mut evidence_sum = vec![GVec3::ZERO; n];
    for s in &steps {
        let composite = evidence_composite_frame(&s.low_e, &s.low_d, low_w, low_h, tw, th);
        for (acc, c) in evidence_sum.iter_mut().zip(composite.iter()) {
            *acc += *c;
        }
    }
    let inv_k = 1.0 / (k.max(1) as f32);
    let evidence_mean: Vec<GVec3> = evidence_sum.iter().map(|&s| s * inv_k).collect();
    let evidence_ceiling = local_max_3x3(&evidence_mean, tw, th); // ceiling in LINEAR evidence space

    // Last step's raw demod-log output — the settled frame the monitor grades.
    let last_dl = &out_imgs[k as usize - 1];
    let last_albedo = &steps[k as usize - 1].albedo;

    let mut net_no_clamp = vec![GVec3::ZERO; n];
    let mut net_clamped = vec![GVec3::ZERO; n];
    for px in 0..n {
        let dl = GVec3::new(last_dl.data[px * 3], last_dl.data[px * 3 + 1], last_dl.data[px * 3 + 2]);
        let divisor = demod_divisor(last_albedo[px]);
        net_no_clamp[px] = undo_log_demod(dl, divisor);

        let ceiling_dl: [f32; OUTPUT_CHANNELS] = evidence_ceiling_demod_log(evidence_ceiling[px], gamma, last_albedo[px]);
        let presented_dl = GVec3::new(dl.x.min(ceiling_dl[0]), dl.y.min(ceiling_dl[1]), dl.z.min(ceiling_dl[2]));
        net_clamped[px] = undo_log_demod(presented_dl, divisor);
    }

    let sp_nc = sparkle_resid_per_mpx(&net_no_clamp, &teacher, tw, th);
    let rs_nc = rmse_lin(&net_no_clamp, &teacher);
    let hl_nc = highlight_ratio(&net_no_clamp, &teacher, highlight_pctl);
    let sp_c = sparkle_resid_per_mpx(&net_clamped, &teacher, tw, th);
    let rs_c = rmse_lin(&net_clamped, &teacher);
    let hl_c = highlight_ratio(&net_clamped, &teacher, highlight_pctl);

    let spark_target = 16.0f64;
    let resid_gate = 0.035f64;
    let bar_nc = sp_nc < spark_target && rs_nc < resid_gate;
    let bar_c = sp_c < spark_target && rs_c < resid_gate;

    println!("[v9eval] === 640x480 EVAL, held-out pose orbit_-20, checkpoint {} ===", wpath.display());
    println!("[v9eval] WITHOUT clamp: sparkle={sp_nc:.1}/Mpx resid={rs_nc:.4} highlight_ratio={hl_nc:.3} | bar(sp<16,resid<0.035)={bar_nc}");
    println!("[v9eval] WITH    clamp: sparkle={sp_c:.1}/Mpx resid={rs_c:.4} highlight_ratio={hl_c:.3}  gamma={gamma} | bar(sp<16,resid<0.035)={bar_c}");
    if !bar_nc && !bar_c {
        println!("[v9eval] NO BAR CLAIM — neither path meets sp<16 AND resid<0.035; numbers reported as measured, vs-bar is honestly FAILED both ways.");
    }
    std::io::stdout().flush().ok();

    // GAIA_V9_EVAL_OUT_PREFIX default "v9" — preserves the ORIGINAL filenames
    // (v9-eval-teacher.png etc, FINISH ATOM step 2) regardless of which
    // checkpoint tag is loaded; pass a different prefix (e.g. "v9c") to avoid
    // clobbering when evaling a different trainer's checkpoint.
    let out_prefix = env_str("GAIA_V9_EVAL_OUT_PREFIX", "v9");
    write_png(&teacher, tw, th, 1.0, Path::new(&out_dir).join(format!("{out_prefix}-eval-teacher.png")).as_path());
    write_png(&net_clamped, tw, th, 1.0, Path::new(&out_dir).join(format!("{out_prefix}-eval-v9clamped.png")).as_path());
    write_png(&net_no_clamp, tw, th, 1.0, Path::new(&out_dir).join(format!("{out_prefix}-eval-v9noclamp.png")).as_path());
}
