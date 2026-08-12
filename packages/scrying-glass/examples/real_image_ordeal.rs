//! THE REAL-IMAGE ORDEAL (Architect, 2026-07-18: "REAL OR BLACK").
//!
//! THE REAL IMAGE BAR: the app presents a neural frame ONLY when the shipped
//! weights pass this ordeal. The bar models HIS eye — zero visible sparkle at
//! stillness, no ghost trails under motion, and an image that is genuinely the
//! converged teacher (not a smeared guess). It runs the RECURRENT N2 net over
//! f(seed) validation poses, STILL and PAN, and measures three IRON quantities:
//!
//!   1. resid_still  — RMSE(linear) of the settled still frame vs the converged
//!                     teacher. "Is it the real image?"  ≤ RESID_BAR.
//!   2. sparkle_still — isolated-bright-pixel count per megapixel on the settled
//!                     still frame (a pixel whose luminance jumps past its 3×3
//!                     neighbourhood median by SPARK_DELTA). "The dots." ≤ SPARKLE_BAR.
//!   3. tvar_still   — mean per-pixel temporal variance of luminance across the
//!                     settled tail frames. A converged running mean → ~0.
//!                     "Does it stop shimmering when he holds still?" ≤ TVAR_BAR.
//!   And under motion:
//!   4. resid_move   — RMSE of a mid-pan frame vs ITS OWN teacher. ≤ RESID_MOVE_BAR.
//!   5. ghost_excess — resid_move(history ON) − resid_move(history OFF). History
//!                     must not smear motion. ≤ GHOST_BAR.
//!
//! PASS ⇔ all five under bar. On PASS it writes the sidecar stamp beside the
//! weights (`<weights>.stamp`) that `main.rs`'s rig-build gate verifies; on FAIL
//! it writes NO stamp (or removes a stale one) and prints the exact residual
//! distance to each bar. The thresholds below are IRON — do NOT soften them to
//! make a net pass; a failing net earns honest BLACK.
//!
//! Run: cargo run -p scrying-glass --release --example real_image_ordeal
//!   GAIA_ORDEAL_WEIGHTS=v2|v3|<path>   (default v3)

use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
    trace_headless_split,
};
use scrying_glass::rdirect::{
    CamPose, EvidenceAccum, HistFrame, HistFrameSplit, HIST_FEATURES_SPLIT, INPUT_FEATURES,
    INPUT_FEATURES_SPLIT, Mlp, clamp_evidence_lin, deserialize_weights, direct_render_sequence_hist,
    direct_render_sequence_hist_split, evidence_clamp_gamma, evidence_composite_frame,
    hist_features, hist_features_split, local_max_3x3, pixel_features, pixel_features_split,
    sky_history_reject, stamp_pass_text, stamp_path_for,
};
// V9 WIRING ATOM, ITEM 2 (2026-07-21) — the body-generic ordeal door: a
// U-Net-format weights file (GAIARD9 magic, `rdirect_unet::cpu::deserialize_weights`)
// runs through the CPU-trainable twin (cross-platform, no target_os=macos gate —
// same choice `examples/rdirect_v9_eval_640.rs` already made for its own 640x480
// eval, so this file keeps compiling everywhere the Mlp path already did; the
// GPU MPSGraph tensor path, `rdirect_unet::UnetLive`, is mac-only and is the
// ITEM 3 live-present-path body, not this correctness gate).
use scrying_glass::rdirect_unet::cpu::{
    Img as UnetImg, UnetWeights, deserialize_weights as unet_deserialize_weights,
};
use scrying_glass::rdirect_unet::MOTION_VECTOR_CHANNELS;
// STAMP-DAY PARITY FIX (task mandate, 2026-07-26): the v9 EYE-TEST WINDOW
// (main.rs `NetPresent::new_v9`) presents through TWO compose surgeries this
// ordeal door (a67942cc, 07-21) predates and did NOT apply: the V9V input
// firefly clamp (`gather_v9`'s WGSL E/D tap median-of-4+delta ceiling,
// `rdirect.rs::v9v_input_clamp_delta`) and the V9N hit-gate compose
// (`rdirect_v9_compose::compose_cpu_reference`, the exact CPU reference the
// live GPU pass is proven parity-equal to). v9y (the checkpoint under
// ordeal) was TRAINED with both active (scratch/v9y-train.log,
// GAIA_V9V_INPUT_CLAMP=0.10 / GAIA_V9N_HITGATE=true) — running it through the
// old bare compose would test a DIFFERENT act than the one that ever runs
// live, violating the 07-23 ruling ("probe compose must equal live
// compose", HANDOFF.md §07-23). Both ported/reused here, byte-identical to
// the live path's own sources (not reimplemented): `v9v_input_clamp_delta`
// (env getter) + `apply_input_clamp` (verbatim port, same as
// `examples/rdirect_v9y_triptych.rs` in v9body) for the clamp; the SHARED
// `compose_cpu_reference` (relocated from main.rs into the lib for exactly
// this reuse, `rdirect_v9_compose.rs`) for the hit-gate.
use scrying_glass::rdirect::v9v_input_clamp_delta;
use scrying_glass::rdirect_v9_compose::compose_cpu_reference;
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

// ── IRON BARS (do not soften — the bar models HIS eye) ──────────────────────
const RESID_BAR: f64 = 0.035; // settled still RMSE vs teacher (v2 held-out ~.031-.035)
const SPARKLE_BAR: f64 = 40.0; // isolated bright px / megapixel at stillness
const SPARK_DELTA: f32 = 0.15; // linear luminance jump over 3×3 median = a "dot"
const TVAR_BAR: f64 = 5.0e-4; // mean temporal luminance variance over settled tail
const RESID_MOVE_BAR: f64 = 0.060; // mid-pan RMSE vs per-frame teacher
const GHOST_BAR: f64 = 0.012; // history must not raise motion RMSE beyond this

const DEPTH_TOL: f32 = 0.05; // reprojection depth guard (light-fix temporal.y band)
const NORMAL_THRESH: f32 = 0.85; // reprojection normal guard (light-fix temporal.z)

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose {
        eye: cam.eye,
        right,
        up,
        forward,
        half_tan: (cam.fov_y_radians * 0.5).tan(),
        aspect: w as f32 / h as f32,
    }
}

struct FrameBufs {
    low: Vec<GVec3>,
    // N5 split radiance (E, D) — populated only when the weights are the 39-in
    // split net; empty otherwise.
    low_e: Vec<GVec3>,
    low_d: Vec<GVec3>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    cam: CamPose,
}

#[allow(clippy::too_many_arguments)]
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    cam: &Camera,
    seed: u32,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    is_split: bool,
) -> FrameBufs {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    let np = IntegratorParams { spp: 1, seed, ..IntegratorParams::default() };
    let low = resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1,
        &np, None,
    ));
    let (low_e, low_d) = if is_split {
        trace_headless_split(
            device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let (albedo, normal, depth) = split_aov(&trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
    ));
    FrameBufs { low, low_e, low_d, albedo, normal, depth, cam: cam_pose(cam, target_w, target_h) }
}

fn render_teacher(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    cam: &Camera,
    ref_frames: u32,
    target_w: u32,
    target_h: u32,
) -> Vec<GVec3> {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        ref_frames, &IntegratorParams::default(), None,
    ))
}

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Isolated INVENTED bright dots per megapixel: pixels where the net's
/// luminance exceeds the CONVERGED TEACHER's by more than SPARK_DELTA AND that
/// excess is a strict local maximum of the signed error over the 3×3
/// neighbourhood — i.e. a firefly the net hallucinated that is NOT in the real
/// image. Measured against the teacher (not the image's own texture), so a
/// converged surface with real high-frequency detail scores ZERO and
/// teacher-vs-teacher is exactly 0. This is the dot the Architect sees.
fn sparkle_resid_per_mpx(net: &[GVec3], teacher: &[GVec3], w: u32, h: u32) -> f64 {
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
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if err(x + dx, y + dy) >= e {
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

/// Mean per-pixel temporal variance of luminance across a set of frames.
fn temporal_variance(frames: &[Vec<GVec3>]) -> f64 {
    if frames.len() < 2 {
        return 0.0;
    }
    let n = frames[0].len();
    let m = frames.len() as f64;
    let mut acc = 0.0f64;
    for i in 0..n {
        let mut s = 0.0f64;
        let mut s2 = 0.0f64;
        for f in frames {
            let l = lum(f[i]) as f64;
            s += l;
            s2 += l * l;
        }
        let mean = s / m;
        acc += (s2 / m - mean * mean).max(0.0);
    }
    acc / n as f64
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn write_panel(panel: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in panel {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[ordeal] wrote {}", path.display());
}

/// Single-frame (history OFF) render — the no-memory baseline for ghost_excess.
fn render_single(mlp: &Mlp, f: &FrameBufs, tw: u32, th: u32, low_w: u32, low_h: u32) -> Vec<GVec3> {
    let motion = vec![Vec2::ZERO; (tw * th) as usize];
    let n = (tw * th) as usize;
    let mut out = vec![GVec3::ZERO; n];
    let in_dim = mlp.layer_dims()[0].0 as usize;
    let is_split = in_dim == HIST_FEATURES_SPLIT;
    let uses_hist = in_dim == INPUT_FEATURES + 4;
    // v7e evidence clamp: same ceiling source as direct_render_sequence_hist_split,
    // identical act (only meaningful for the split net; empty low_e/low_d for
    // the non-split nets makes it a zero-evidence no-op ceiling, harmless since
    // this branch isn't hit for those nets' clamp path below).
    let gamma = evidence_clamp_gamma();
    // no-memory baseline: only THIS frame's evidence (matches its "no
    // history" nature — unlike the recurrent path, there is no prior step
    // to temporally average with).
    let evidence_max = if is_split {
        local_max_3x3(&evidence_composite_frame(&f.low_e, &f.low_d, low_w, low_h, tw, th), tw, th)
    } else {
        Vec::new()
    };
    for ty in 0..th {
        for tx in 0..tw {
            let i = (ty * tw + tx) as usize;
            let albedo = f.albedo[i];
            let dl = if is_split {
                let base = pixel_features_split(
                    &f.low_e, &f.low_d, low_w, low_h, tw, th, tx, ty, albedo, f.normal[i], f.depth[i], motion[i],
                );
                mlp.forward(&hist_features_split(&base, [0.0; 3], 0.0))
            } else {
                let base = pixel_features(
                    &f.low, low_w, low_h, tw, th, tx, ty, albedo, f.normal[i], f.depth[i], motion[i],
                );
                if uses_hist {
                    mlp.forward(&hist_features(&base, [0.0; 3], 0.0))
                } else {
                    mlp.forward(&base)
                }
            };
            // undo log-demod
            let div = if albedo.length_squared() > 1e-8 { albedo + GVec3::splat(1e-3) } else { GVec3::ONE };
            let expm1 = GVec3::new(dl[0].exp() - 1.0, dl[1].exp() - 1.0, dl[2].exp() - 1.0);
            let net_lin = GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * div;
            out[i] = if is_split { clamp_evidence_lin(net_lin, evidence_max[i], gamma) } else { net_lin };
        }
    }
    out
}

/// Render a still/pan SEQUENCE through the recurrent net, branching on whether
/// the weights are the N5 split net (39-in, two radiance channels) or the
/// v3/v5 single-channel net (27-in).
#[allow(clippy::too_many_arguments)]
fn sequence_render(
    mlp: &Mlp,
    bufs: &[FrameBufs],
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    is_split: bool,
) -> Vec<Vec<GVec3>> {
    if is_split {
        let hf: Vec<HistFrameSplit> = bufs
            .iter()
            .map(|b| HistFrameSplit {
                low_e: &b.low_e, low_d: &b.low_d, low_w, low_h,
                hi_albedo: &b.albedo, hi_normal: &b.normal, hi_depth: &b.depth,
                target_w, target_h, cam: b.cam,
            })
            .collect();
        direct_render_sequence_hist_split(mlp, &hf, DEPTH_TOL, NORMAL_THRESH)
    } else {
        let hf: Vec<HistFrame> = bufs
            .iter()
            .map(|b| HistFrame {
                low_radiance: &b.low, low_w, low_h,
                hi_albedo: &b.albedo, hi_normal: &b.normal, hi_depth: &b.depth,
                target_w, target_h, cam: b.cam,
            })
            .collect();
        direct_render_sequence_hist(mlp, &hf, DEPTH_TOL, NORMAL_THRESH)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// V9 WIRING ATOM, ITEM 2 (2026-07-21) — U-NET BODY ORDEAL PATH. Same
// STILL/PAN loop shape as the Mlp path above, same IRON bars, same stamp
// machinery (`stamp_path_for`/`stamp_pass_text`, the VERDICT block below is
// a straight duplicate of `main`'s own); only the per-frame forward +
// feature layout differ. `reproject_prev_unet`/`build_input_img_unet` are
// `examples/rdirect_train_v9c.rs`'s `reproject_prev`/`build_input_img`
// (also duplicated once already into `rdirect_v9_eval_640.rs`) ported here
// a third time — both source fns are private to their own binaries, so
// this is the SAME house duplication convention those two files already
// established for this exact code, not a new pattern.
// ─────────────────────────────────────────────────────────────────────────

/// `reproject_prev` (`rdirect_train_v9c.rs`, also in `rdirect_v9_eval_640.rs`),
/// unchanged math — reprojects step s-1's screen into step s, returning the
/// sampled demod-log history triple + validity + the RAW screen-space
/// motion vector (pixels/frame) the v9 body's 2 extra input channels carry.
#[allow(clippy::too_many_arguments)]
fn reproject_prev_unet(
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

/// `build_input_img` (`rdirect_train_v9c.rs`) — the 41-channel input tensor
/// for one step (39 = v7 `hist_features_split` + 2 motion-vector channels),
/// sourced from a `FrameBufs` — the SAME struct/render call the Mlp N5-
/// V9V INPUT FIREFLY CLAMP, ported verbatim from
/// `rdirect_train_v9k.rs::apply_input_clamp` (also duplicated in
/// `examples/rdirect_v9y_triptych.rs` in v9body — third house copy of this
/// exact code, same reason: no cross-crate/example sharing for trainer
/// internals). `base` layout (`pixel_features_split`): [0..12)=E's 2x2x3
/// demod-log taps, [12..24)=D's 2x2x3 demod-log taps, tap stride 3 (one f32
/// per channel). `delta<=0.0` is a no-op — IRON LAW byte-identical to the
/// pre-clamp gather (matches `v9v_input_clamp_delta`'s own 0.0 default).
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

/// V9N HIT-GATE COMPOSE, applied to a whole frame's RAW net demod-log output
/// — thin adapter from this file's `FrameBufs`/`Vec<GVec3>` shapes to
/// `compose_cpu_reference`'s flat-f32 buffer contract (the exact function
/// `main.rs`'s `NetPresent::v9_hitgate_compose` calls for `GAIA_V9_COMPOSE=cpu`,
/// relocated to the lib so this ordeal reuses it verbatim instead of a fourth
/// hand-synced copy). `aov` needs only (albedo.xyz, depth) per px — the
/// [4..8) lane is unused by the reference, left zeroed. `ed` needs the
/// low-res E/D taps as (rgb, count) — this file's `FrameBufs.low_e/low_d`
/// are already temporally-resolved (spp=1, `trace_headless_split`'s own
/// sum/max(count,1) divide), so count=1.0 reproduces the identical value
/// through the reference's own max(count,1) divide (a no-op distinguishment,
/// not an approximation). Output stays demod-log (same domain as `dl_vec`);
/// the caller's existing demod-undo + evidence-clamp loop is unchanged.
fn v9_hitgate_dl(dl_vec: &[GVec3], buf: &FrameBufs, low_w: u32, low_h: u32, tw: u32, th: u32, hitgate: bool) -> Vec<GVec3> {
    let n = (tw * th) as usize;
    if !hitgate {
        return dl_vec.to_vec();
    }
    let mut aov = vec![0.0f32; n * 8];
    for i in 0..n {
        aov[i * 8] = buf.albedo[i].x;
        aov[i * 8 + 1] = buf.albedo[i].y;
        aov[i * 8 + 2] = buf.albedo[i].z;
        aov[i * 8 + 3] = buf.depth[i];
    }
    let lc = (low_w * low_h) as usize;
    let mut ed = vec![0.0f32; lc * 8];
    for j in 0..lc {
        ed[j * 8] = buf.low_e[j].x;
        ed[j * 8 + 1] = buf.low_e[j].y;
        ed[j * 8 + 2] = buf.low_e[j].z;
        ed[j * 8 + 3] = 1.0;
        ed[j * 8 + 4] = buf.low_d[j].x;
        ed[j * 8 + 5] = buf.low_d[j].y;
        ed[j * 8 + 6] = buf.low_d[j].z;
        ed[j * 8 + 7] = 1.0;
    }
    let mut net_out = vec![0.0f32; n * 3];
    for i in 0..n {
        net_out[i * 3] = dl_vec[i].x;
        net_out[i * 3 + 1] = dl_vec[i].y;
        net_out[i * 3 + 2] = dl_vec[i].z;
    }
    let gated = compose_cpu_reference(&net_out, &aov, &ed, low_w, low_h, tw, th, hitgate);
    (0..n).map(|i| GVec3::new(gated[i * 3], gated[i * 3 + 1], gated[i * 3 + 2])).collect()
}

/// split path above already uses (`render_frame(..., is_split=true)`); only
/// the consumer differs, there is no separate render path for this body.
#[allow(clippy::too_many_arguments)]
fn build_input_img_unet(
    bufs: &[FrameBufs],
    step_idx: usize,
    prev_out_dl: Option<&[GVec3]>,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
    sky_reject: bool,
) -> UnetImg {
    let buf = &bufs[step_idx];
    let mut img = UnetImg::zeros(th as usize, tw as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS);
    for ty in 0..th {
        for tx in 0..tw {
            let px = (ty * tw + tx) as usize;
            let mut base: [f32; INPUT_FEATURES_SPLIT] = pixel_features_split(
                &buf.low_e, &buf.low_d, low_w, low_h, tw, th, tx, ty,
                buf.albedo[px], buf.normal[px], buf.depth[px], Vec2::ZERO,
            );
            apply_input_clamp(&mut base, v9v_input_clamp_delta());
            let (prev_dl, valid, mv) = match (step_idx, prev_out_dl) {
                (0, _) | (_, None) => ([0.0f32; 3], 0.0f32, [0.0f32; 2]),
                (_, Some(prev)) => {
                    let prev_buf = &bufs[step_idx - 1];
                    reproject_prev_unet(
                        &buf.cam, buf.depth[px], buf.normal[px], tx, ty, tw, th,
                        &prev_buf.cam, prev, &prev_buf.depth, &prev_buf.normal, tw, th,
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

fn dl_img_to_vec3(dl: &UnetImg) -> Vec<GVec3> {
    let mut v = vec![GVec3::ZERO; dl.h * dl.w];
    for y in 0..dl.h {
        for x in 0..dl.w {
            v[y * dl.w + x] = GVec3::new(dl.at(y, x, 0), dl.at(y, x, 1), dl.at(y, x, 2));
        }
    }
    v
}

/// Recurrent forward through a still/pan SEQUENCE, U-Net body — the same
/// contract as `sequence_render` above (one `Vec<GVec3>` per frame, LINEAR,
/// evidence-clamped, demod-undone) but sourced from the CPU-trainable
/// twin's whole-image forward instead of a per-pixel `Mlp::forward` loop.
/// Clamp mechanics reuse the SAME `EvidenceAccum`/`clamp_evidence_lin`
/// (`rdirect.rs`) the Mlp path's `direct_render_sequence_hist_split` calls
/// internally — byte-identical clamp act across both bodies, gamma=1.5
/// default (`evidence_clamp_gamma()`), incremental temporal-mean per frame
/// (v8d/Mlp-path contract, NOT v9c trainer's own per-pose-whole-mean
/// simplification — the ordeal door matches the SHIPPED inference clamp,
/// not the trainer's training-time approximation of it).
#[allow(clippy::too_many_arguments)]
fn sequence_render_unet(
    net: &UnetWeights,
    bufs: &[FrameBufs],
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    sky_reject: bool,
    hitgate: bool,
) -> Vec<Vec<GVec3>> {
    let gamma = evidence_clamp_gamma();
    let n = (target_w * target_h) as usize;
    let mut accum = EvidenceAccum::new(target_w, target_h);
    let mut chain_dl: Vec<Vec<GVec3>> = Vec::with_capacity(bufs.len());
    let mut outs: Vec<Vec<GVec3>> = Vec::with_capacity(bufs.len());
    for s in 0..bufs.len() {
        let prev: Option<&[GVec3]> = if s == 0 { None } else { Some(&chain_dl[s - 1]) };
        let input = build_input_img_unet(bufs, s, prev, low_w, low_h, target_w, target_h, sky_reject);
        let (out_img, _cache) = net.forward(&input);
        let dl_raw = dl_img_to_vec3(&out_img);
        // GAIA_V9N_HITGATE parity (task mandate): the live path stores the
        // GATED dl into recurrent history (`history.swap` reads
        // `out_dl_padded`, sourced from `net_out_gated`, `main.rs` resolve
        // stage), never the net's raw pre-gate output — chain_dl below must
        // match or the reprojected history diverges from what ever runs live.
        let dl_vec = v9_hitgate_dl(&dl_raw, &bufs[s], low_w, low_h, target_w, target_h, hitgate);

        let frame_composite = evidence_composite_frame(&bufs[s].low_e, &bufs[s].low_d, low_w, low_h, target_w, target_h);
        accum.push(&frame_composite);
        let evidence_max = accum.ceiling();

        let mut out_rgb = vec![GVec3::ZERO; n];
        for i in 0..n {
            let albedo = bufs[s].albedo[i];
            let divisor = if albedo.length_squared() > 1e-8 { albedo + GVec3::splat(1e-3) } else { GVec3::ONE };
            let dl = dl_vec[i];
            let expm1 = GVec3::new(dl.x.exp() - 1.0, dl.y.exp() - 1.0, dl.z.exp() - 1.0);
            let net_lin = GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * divisor;
            out_rgb[i] = clamp_evidence_lin(net_lin, evidence_max[i], gamma);
        }
        chain_dl.push(dl_vec);
        outs.push(out_rgb);
    }
    outs
}

/// U-Net no-history baseline (`ghost_excess`'s single-frame comparator) —
/// mirrors `render_single` above: THIS frame's own evidence only (no
/// temporal accumulation — there is no prior step for a no-memory render).
fn render_single_unet(net: &UnetWeights, buf: &FrameBufs, tw: u32, th: u32, low_w: u32, low_h: u32, hitgate: bool) -> Vec<GVec3> {
    let gamma = evidence_clamp_gamma();
    let n = (tw * th) as usize;
    let single = std::slice::from_ref(buf);
    let input = build_input_img_unet(single, 0, None, low_w, low_h, tw, th, false);
    let (out_img, _cache) = net.forward(&input);
    let dl_raw = dl_img_to_vec3(&out_img);
    let dl_vec = v9_hitgate_dl(&dl_raw, buf, low_w, low_h, tw, th, hitgate);
    let frame_composite = evidence_composite_frame(&buf.low_e, &buf.low_d, low_w, low_h, tw, th);
    let evidence_max = local_max_3x3(&frame_composite, tw, th);
    let mut out = vec![GVec3::ZERO; n];
    for i in 0..n {
        let albedo = buf.albedo[i];
        let divisor = if albedo.length_squared() > 1e-8 { albedo + GVec3::splat(1e-3) } else { GVec3::ONE };
        let dl = dl_vec[i];
        let expm1 = GVec3::new(dl.x.exp() - 1.0, dl.y.exp() - 1.0, dl.z.exp() - 1.0);
        let net_lin = GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * divisor;
        out[i] = clamp_evidence_lin(net_lin, evidence_max[i], gamma);
    }
    out
}

/// The U-Net body's own STILL/PAN ordeal loop — a straight duplicate of
/// `main`'s Mlp loop + VERDICT block (same bars, same stamp machinery),
/// swapping `sequence_render`/`render_single` for the `_unet` variants
/// above. Terminates the process on every path (mirrors `main`'s own
/// `std::process::exit` calls), so its return type is `!`.
#[allow(clippy::too_many_arguments)]
fn run_unet_ordeal(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    wpath: &Path,
    wbytes: &[u8],
    net: &UnetWeights,
    target_w: u32,
    target_h: u32,
    low_w: u32,
    low_h: u32,
    still_len: u32,
    tail: u32,
    pan_len: u32,
    pan_step: f32,
    ref_still: u32,
    ref_move: u32,
    exposure: f32,
    poses: &[(&str, Camera)],
) -> ! {
    let sky_reject = sky_history_reject();
    println!("[ordeal] GAIA_V7_SKY_HISTORY reject={sky_reject} (mandate expects true — set GAIA_V7_SKY_HISTORY=reject)");
    // STAMP-DAY PARITY FIX: GAIA_V9N_HITGATE / GAIA_V9V_INPUT_CLAMP, read
    // once and applied identically to every still/pan/single render below —
    // the SAME two env vars `main.rs`'s `NetPresent::new_v9` reads for the
    // live eye-test window (byte-identical matcher). Unset/off reproduces
    // the pre-parity-fix ordeal exactly (IRON, no default change for
    // Mlp-body weights or any U-Net checkpoint that was NOT trained with
    // these features).
    let hitgate = matches!(std::env::var("GAIA_V9N_HITGATE").as_deref(), Ok("1" | "true" | "on"));
    let input_clamp = v9v_input_clamp_delta();
    println!(
        "[ordeal] GAIA_V9N_HITGATE={hitgate} GAIA_V9V_INPUT_CLAMP={input_clamp} (compose parity vs the live eye-test window — set both to match a checkpoint's own training config, e.g. v9y: hitgate=1 clamp=0.10)"
    );

    let t0 = Instant::now();
    let mut resid_still = 0.0f64;
    let mut sparkle_still = 0.0f64;
    let mut sparkle_teacher = 0.0f64;
    let mut tvar_still = 0.0f64;
    let mut resid_move = 0.0f64;
    let mut ghost_excess = 0.0f64;
    let mut shot_still: Option<(Vec<GVec3>, Vec<GVec3>)> = None;
    let mut shot_move: Option<(Vec<GVec3>, Vec<GVec3>)> = None;

    for (pname, cam) in poses {
        let still_bufs: Vec<FrameBufs> = (0..still_len)
            .map(|k| render_frame(device, queue, base_tris, scene, cam, 0x5eed + k * 101 + 7, low_w, low_h, target_w, target_h, true))
            .collect();
        let teacher_still = render_teacher(device, queue, base_tris, scene, cam, ref_still, target_w, target_h);
        let outs = sequence_render_unet(net, &still_bufs, low_w, low_h, target_w, target_h, sky_reject, hitgate);
        let settled = outs.last().unwrap();
        let r = rmse(settled, &teacher_still) as f64;
        let sp = sparkle_resid_per_mpx(settled, &teacher_still, target_w, target_h);
        let spt = sparkle_resid_per_mpx(&teacher_still, &teacher_still, target_w, target_h);
        let tv = temporal_variance(&outs[(still_len - tail) as usize..]);
        println!("[ordeal] {pname} STILL: resid={r:.5} sparkle={sp:.1}/Mpx (teacher {spt:.1}) tvar={tv:.3e}");
        resid_still += r;
        sparkle_still += sp;
        sparkle_teacher += spt;
        tvar_still += tv;
        if shot_still.is_none() {
            shot_still = Some((settled.clone(), teacher_still.clone()));
        }

        let mut pan_cams = Vec::new();
        for k in 0..pan_len {
            let mut c = cam.clone();
            c.yaw += pan_step * k as f32;
            pan_cams.push(c);
        }
        let pan_bufs: Vec<FrameBufs> = pan_cams
            .iter()
            .enumerate()
            .map(|(k, c)| render_frame(device, queue, base_tris, scene, c, 0x1234 + k as u32 * 97 + 3, low_w, low_h, target_w, target_h, true))
            .collect();
        let pan_outs = sequence_render_unet(net, &pan_bufs, low_w, low_h, target_w, target_h, sky_reject, hitgate);
        let mid = (pan_len - 1) as usize;
        let teacher_mid = render_teacher(device, queue, base_tris, scene, &pan_cams[mid], ref_move, target_w, target_h);
        let rm_hist = rmse(&pan_outs[mid], &teacher_mid) as f64;
        let single = render_single_unet(net, &pan_bufs[mid], target_w, target_h, low_w, low_h, hitgate);
        let rm_single = rmse(&single, &teacher_mid) as f64;
        let ge = rm_hist - rm_single;
        println!("[ordeal] {pname} PAN mid: resid_hist={rm_hist:.5} resid_single={rm_single:.5} ghost_excess={ge:+.5}");
        resid_move += rm_hist;
        ghost_excess = ghost_excess.max(ge);
        if shot_move.is_none() {
            shot_move = Some((pan_outs[mid].clone(), teacher_mid.clone()));
        }
    }

    let np = poses.len() as f64;
    resid_still /= np;
    sparkle_still /= np;
    sparkle_teacher /= np;
    tvar_still /= np;
    resid_move /= np;

    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("proof/neural-live");
    let sp = std::env::var("GAIA_ORDEAL_SHOT").unwrap_or_else(|_| "s25-unet".into());
    if let Some((net_img, teacher)) = &shot_still {
        write_panel(net_img, target_w, target_h, exposure, &proof.join(format!("{sp}-still.png")));
        write_panel(teacher, target_w, target_h, exposure, &proof.join(format!("{sp}-still-teacher.png")));
    }
    if let Some((net_img, teacher)) = &shot_move {
        write_panel(net_img, target_w, target_h, exposure, &proof.join(format!("{sp}-moving.png")));
        write_panel(teacher, target_w, target_h, exposure, &proof.join(format!("{sp}-moving-teacher.png")));
    }

    let p_resid = resid_still <= RESID_BAR;
    let p_spark = sparkle_still <= SPARKLE_BAR;
    let p_tvar = tvar_still <= TVAR_BAR;
    let p_rmove = resid_move <= RESID_MOVE_BAR;
    let p_ghost = ghost_excess <= GHOST_BAR;
    let pass = p_resid && p_spark && p_tvar && p_rmove && p_ghost;

    println!("\n[ordeal] ===== REAL-IMAGE BAR (U-NET BODY) ===== ({:.1}s)", t0.elapsed().as_secs_f64());
    let row = |name: &str, v: f64, bar: f64, ok: bool, lower: bool| {
        let dist = if lower { v - bar } else { bar - v };
        println!(
            "  {name:<14} {v:>12.5}  bar {bar:>10.5}  {}  (distance to bar {:+.5})",
            if ok { "PASS" } else { "FAIL" },
            dist
        );
    };
    row("resid_still", resid_still, RESID_BAR, p_resid, true);
    row("sparkle_still", sparkle_still, SPARKLE_BAR, p_spark, true);
    row("tvar_still", tvar_still, TVAR_BAR, p_tvar, true);
    row("resid_move", resid_move, RESID_MOVE_BAR, p_rmove, true);
    row("ghost_excess", ghost_excess, GHOST_BAR, p_ghost, true);
    println!("  (teacher sparkle {sparkle_teacher:.1}/Mpx — reference floor)");

    let stamp = stamp_path_for(wpath);
    if pass {
        let metrics = [
            ("resid_still", resid_still),
            ("sparkle_still", sparkle_still),
            ("tvar_still", tvar_still),
            ("resid_move", resid_move),
            ("ghost_excess", ghost_excess),
        ];
        std::fs::write(&stamp, stamp_pass_text(wbytes, &metrics)).unwrap();
        println!("\n[ordeal] VERDICT: PASS — stamp written {} — weights may present.", stamp.display());
        std::process::exit(0);
    } else {
        let _ = std::fs::remove_file(&stamp);
        println!("\n[ordeal] VERDICT: FAIL — NO stamp — the window is BLACK by law (real or black).");
        std::process::exit(1);
    }
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[ordeal] no GPU adapter");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let target_w = env_u32("GAIA_ORDEAL_W", 480);
    let target_h = env_u32("GAIA_ORDEAL_H", 360);
    let low_w = target_w / 2;
    let low_h = target_h / 2;
    let still_len = env_u32("GAIA_ORDEAL_STILL", 10);
    let tail = env_u32("GAIA_ORDEAL_TAIL", 4).min(still_len);
    let pan_len = env_u32("GAIA_ORDEAL_PAN", 6);
    let pan_step = env_f32("GAIA_ORDEAL_PANSTEP", 0.004); // rad/frame ≈ 0.23°, a slow pan
    let ref_still = env_u32("GAIA_ORDEAL_REF_STILL", 96);
    let ref_move = env_u32("GAIA_ORDEAL_REF_MOVE", 48);
    let exposure = env_f32("GAIA_RDIRECT_EXPOSURE", 1.6);

    // weights selection + path
    let sel = std::env::var("GAIA_ORDEAL_WEIGHTS").unwrap_or_else(|_| "v3".to_string());
    let wrel = match sel.as_str() {
        "v1" => "data/rdirect-weights-v1.bin".to_string(),
        "v2" => "data/rdirect-weights-v2.bin".to_string(),
        "v3" => "data/rdirect-weights-v3.bin".to_string(),
        "v4" => "data/rdirect-weights-v4.bin".to_string(),
        "v5" => "data/rdirect-weights-v5.bin".to_string(),
        "v6" => "data/rdirect-weights-v6.bin".to_string(),
        "v7" => "data/rdirect-weights-v7.bin".to_string(),
        "v8" => "data/rdirect-weights-v8.bin".to_string(),
        other => other.to_string(),
    };
    let wpath: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join(&wrel);
    let wbytes = match std::fs::read(&wpath) {
        Ok(b) => b,
        Err(e) => {
            println!("[ordeal] FAIL: cannot read weights {wrel}: {e} — no stamp, present BLACK.");
            std::process::exit(2);
        }
    };

    let val_poses = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| val_poses.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let poses = [("orbit_-20", find("orbit_-20")), ("orbit_+40", find("orbit_+40"))];

    // V9 WIRING ATOM, ITEM 2 — BODY-GENERIC ORDEAL DOOR: body selection is
    // the weights FORMAT itself (magic bytes), never an env "mode" flag
    // (DESIGN LAW, CLAUDE.md ★ stack: "body selection = weights format/
    // manifest, never a mode/fallback framing"). Try the U-Net magic
    // (GAIARD9) first — `cpu::deserialize_weights` returns `None` on any
    // non-matching header, including the Mlp's own GAIARDR1 magic, so this
    // is a cheap, safe probe that falls through untouched for every v1-v8
    // Mlp weights file that already passes/fails this same gate today.
    if let Some(mut unet_net) = unet_deserialize_weights(&wbytes) {
        // Fully-convolutional body (module doc, `rdirect_unet.rs`): the
        // checkpoint's own `config.render_w/h` is just the trainer's wall-
        // budget training res label, not a hard tensor-shape constant
        // (`rdirect_v9_eval_640.rs` makes the identical override for its
        // own 640x480 eval) — run the ordeal at ITS OWN resolution
        // (GAIA_ORDEAL_W/H), same as the Mlp path above.
        unet_net.config.render_w = target_w as usize;
        unet_net.config.render_h = target_h as usize;
        unet_net.config.output_w = target_w as usize;
        unet_net.config.output_h = target_h as usize;
        println!(
            "[ordeal] weights={wrel} format=U-NET (GAIARD9) in_channels={} widths={:?} n_scales={} res {target_w}x{target_h} still={still_len} pan={pan_len}",
            unet_net.config.in_channels, unet_net.config.widths, unet_net.config.n_scales
        );
        run_unet_ordeal(
            &device, &queue, &base_tris, &scene, &wpath, &wbytes, &unet_net,
            target_w, target_h, low_w, low_h, still_len, tail, pan_len, pan_step,
            ref_still, ref_move, exposure, &poses,
        ); // run_unet_ordeal itself exits the process on every path (PASS/FAIL) — diverges (`-> !`)
    }

    let mlp = deserialize_weights(&wbytes).expect("weights parse");
    let in_dim = mlp.layer_dims()[0].0 as usize;
    let is_split = in_dim == HIST_FEATURES_SPLIT;
    println!(
        "[ordeal] weights={wrel} in_dim={in_dim} ({}) res {target_w}x{target_h} still={still_len} pan={pan_len}",
        if is_split { "N5 split recurrent (39)" }
        else if in_dim == INPUT_FEATURES + 4 { "N2 recurrent (27)" }
        else { "v2 current-frame (23) — no memory" }
    );

    let t0 = Instant::now();
    let mut resid_still = 0.0f64;
    let mut sparkle_still = 0.0f64;
    let mut sparkle_teacher = 0.0f64;
    let mut tvar_still = 0.0f64;
    let mut resid_move = 0.0f64;
    let mut ghost_excess = 0.0f64;
    let mut shot_still: Option<(Vec<GVec3>, Vec<GVec3>)> = None; // (net, teacher)
    let mut shot_move: Option<(Vec<GVec3>, Vec<GVec3>)> = None;

    for (pname, cam) in &poses {
        // ── STILL: same camera, fresh seed each frame ──────────────────────
        let still_bufs: Vec<FrameBufs> = (0..still_len)
            .map(|k| render_frame(&device, &queue, &base_tris, &scene, cam, 0x5eed + k * 101 + 7, low_w, low_h, target_w, target_h, is_split))
            .collect();
        let teacher_still = render_teacher(&device, &queue, &base_tris, &scene, cam, ref_still, target_w, target_h);
        let outs = sequence_render(&mlp, &still_bufs, low_w, low_h, target_w, target_h, is_split);
        let settled = outs.last().unwrap();
        let r = rmse(settled, &teacher_still) as f64;
        let sp = sparkle_resid_per_mpx(settled, &teacher_still, target_w, target_h);
        let spt = sparkle_resid_per_mpx(&teacher_still, &teacher_still, target_w, target_h);
        let tv = temporal_variance(&outs[(still_len - tail) as usize..]);
        println!("[ordeal] {pname} STILL: resid={r:.5} sparkle={sp:.1}/Mpx (teacher {spt:.1}) tvar={tv:.3e}");
        resid_still += r;
        sparkle_still += sp;
        sparkle_teacher += spt;
        tvar_still += tv;
        if shot_still.is_none() {
            shot_still = Some((settled.clone(), teacher_still.clone()));
        }

        // ── PAN: yaw drifts each frame; per-frame teacher ─────────────────
        let mut pan_cams = Vec::new();
        for k in 0..pan_len {
            let mut c = cam.clone();
            c.yaw += pan_step * k as f32;
            pan_cams.push(c);
        }
        let pan_bufs: Vec<FrameBufs> = pan_cams
            .iter()
            .enumerate()
            .map(|(k, c)| render_frame(&device, &queue, &base_tris, &scene, c, 0x1234 + k as u32 * 97 + 3, low_w, low_h, target_w, target_h, is_split))
            .collect();
        let pan_outs = sequence_render(&mlp, &pan_bufs, low_w, low_h, target_w, target_h, is_split);
        // mid-pan frame (history has built up but camera is moving = ghost test)
        let mid = (pan_len - 1) as usize;
        let teacher_mid = render_teacher(&device, &queue, &base_tris, &scene, &pan_cams[mid], ref_move, target_w, target_h);
        let rm_hist = rmse(&pan_outs[mid], &teacher_mid) as f64;
        let single = render_single(&mlp, &pan_bufs[mid], target_w, target_h, low_w, low_h);
        let rm_single = rmse(&single, &teacher_mid) as f64;
        let ge = rm_hist - rm_single;
        println!("[ordeal] {pname} PAN mid: resid_hist={rm_hist:.5} resid_single={rm_single:.5} ghost_excess={ge:+.5}");
        resid_move += rm_hist;
        ghost_excess = ghost_excess.max(ge); // WORST ghost across poses
        if shot_move.is_none() {
            shot_move = Some((pan_outs[mid].clone(), teacher_mid.clone()));
        }
    }

    let np = poses.len() as f64;
    resid_still /= np;
    sparkle_still /= np;
    sparkle_teacher /= np;
    tvar_still /= np;
    resid_move /= np;

    // ── proof PNGs ──────────────────────────────────────────────────────────
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("proof/neural-live");
    // N5 writes the s25 panels (both eyes: net vs teacher); older nets keep s21.
    let sp = std::env::var("GAIA_ORDEAL_SHOT").unwrap_or_else(|_| if is_split { "s25".into() } else { "s21".into() });
    if let Some((net, teacher)) = &shot_still {
        write_panel(net, target_w, target_h, exposure, &proof.join(format!("{sp}-still.png")));
        write_panel(teacher, target_w, target_h, exposure, &proof.join(format!("{sp}-still-teacher.png")));
    }
    if let Some((net, teacher)) = &shot_move {
        write_panel(net, target_w, target_h, exposure, &proof.join(format!("{sp}-moving.png")));
        write_panel(teacher, target_w, target_h, exposure, &proof.join(format!("{sp}-moving-teacher.png")));
    }

    // ── VERDICT ──────────────────────────────────────────────────────────────
    let p_resid = resid_still <= RESID_BAR;
    let p_spark = sparkle_still <= SPARKLE_BAR;
    let p_tvar = tvar_still <= TVAR_BAR;
    let p_rmove = resid_move <= RESID_MOVE_BAR;
    let p_ghost = ghost_excess <= GHOST_BAR;
    let pass = p_resid && p_spark && p_tvar && p_rmove && p_ghost;

    println!("\n[ordeal] ===== REAL-IMAGE BAR ===== ({:.1}s)", t0.elapsed().as_secs_f64());
    let row = |name: &str, v: f64, bar: f64, ok: bool, lower: bool| {
        let dist = if lower { v - bar } else { bar - v };
        println!(
            "  {name:<14} {v:>12.5}  bar {bar:>10.5}  {}  (distance to bar {:+.5})",
            if ok { "PASS" } else { "FAIL" },
            dist
        );
    };
    row("resid_still", resid_still, RESID_BAR, p_resid, true);
    row("sparkle_still", sparkle_still, SPARKLE_BAR, p_spark, true);
    row("tvar_still", tvar_still, TVAR_BAR, p_tvar, true);
    row("resid_move", resid_move, RESID_MOVE_BAR, p_rmove, true);
    row("ghost_excess", ghost_excess, GHOST_BAR, p_ghost, true);
    println!("  (teacher sparkle {sparkle_teacher:.1}/Mpx — reference floor)");

    let stamp = stamp_path_for(&wpath);
    if pass {
        let metrics = [
            ("resid_still", resid_still),
            ("sparkle_still", sparkle_still),
            ("tvar_still", tvar_still),
            ("resid_move", resid_move),
            ("ghost_excess", ghost_excess),
        ];
        std::fs::write(&stamp, stamp_pass_text(&wbytes, &metrics)).unwrap();
        println!("\n[ordeal] VERDICT: PASS — stamp written {} — weights may present.", stamp.display());
    } else {
        // Remove any stale stamp so a previously-passing file cannot present.
        let _ = std::fs::remove_file(&stamp);
        println!("\n[ordeal] VERDICT: FAIL — NO stamp — the window is BLACK by law (real or black).");
        std::process::exit(1);
    }
}
