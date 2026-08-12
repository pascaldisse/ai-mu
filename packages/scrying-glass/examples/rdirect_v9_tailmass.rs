//! v9e forensics step 1b (`scratch/v9-autopsy.md` v9e section): instrument
//! the K=8-averaged noise2noise TARGETS at v9c's own pool poses, in the
//! EXACT domain `rdirect_train_v9c.rs`'s loss computes in (verified by
//! reading that file: `target_dl` is `target_demod_log(mean of K linear
//! draws, albedo)` — the K-average happens in LINEAR radiance, then the
//! mean is demod-log-encoded ONCE; the loss then does plain per-pixel MSE
//! of `presented_dl` (net output, clamped) against this `target_dl` — so
//! the loss's own arithmetic runs in DEMOD-LOG space, and that is the
//! space this tool measures tail mass in).
//!
//! For each of N pool poses (same anchors/orbit/jitter/k_draws/evidence_spp
//! as `rdirect_train_v9c.rs`'s main-pose draws — front x2, wide x1, single
//! `k=3` step 0 per pose to keep this cheap), computes per-pixel luminance
//! of `target_dl`, a local-mean via a 7x7 box filter (no existing
//! `local_mean` helper in the crate — `local_max_3x3` is a MAX, not a mean,
//! so this file adds its own), and reports:
//!   - fraction of pixels with lum > k*local_mean for k in {3,10,30}
//!   - max/mean ratio of the whole-image luminance
//! averaged over all sampled poses, plus per-pose lines to `stdout`.

use std::io::Write;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, trace_headless_split};
use scrying_glass::rdirect::{OUTPUT_CHANNELS, target_demod_log};
use scrying_glass::scene::{Camera, RenderScene};
use scrying_glass::denoiser_dataset::{camera_at, orbit_camera};

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
fn rand_uniform(rng: &mut Rng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next() >> 11) as f64 / (1u64 << 53) as f64) as f32;
    lo + (hi - lo) * u
}

const DRAW_B_SEED_BASE: u32 = 0xB222; // v9c parity — noise2noise label draws

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(n: &str, d: f32) -> f32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

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

fn lum(c: [f32; OUTPUT_CHANNELS]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// 7x7 box-filter LOCAL MEAN (not max — `local_max_3x3` in `rdirect.rs` is
/// a different statistic, used for the evidence ceiling, not this).
fn local_mean_7x7(lum_img: &[f32], w: u32, h: u32) -> Vec<f32> {
    let (w, h) = (w as i32, h as i32);
    let r = 3i32;
    let mut out = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (sx, sy) = (x + dx, y + dy);
                    if sx >= 0 && sx < w && sy >= 0 && sy < h {
                        sum += lum_img[(sy * w + sx) as usize] as f64;
                        n += 1;
                    }
                }
            }
            out[(y * w + x) as usize] = (sum / n.max(1) as f64) as f32;
        }
    }
    out
}

fn main() {
    let run_tag = "v9-tailmass";
    let Some((device, queue)) = headless_device() else {
        panic!("[{run_tag}] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();
    let bvh = Bvh::build(&base_tris, &BvhParams::default());

    // Exactly v9c's config (rdirect_train_v9c.rs main()).
    let tw = env_u32("GAIA_V9_W", 128);
    let th = env_u32("GAIA_V9_H", 72);
    let k_draws = env_u32("GAIA_V9_K", 8);
    let evidence_spp = env_u32("GAIA_V9_TAILMASS_SPP", 1); // v9c's pool-draw evidence_spp
    let pool_orbit_deg = env_f32("GAIA_V9C_POOL_ORBIT_DEG", 30.0);
    let pool_jitter = env_f32("GAIA_V9C_POOL_JITTER", 0.6);
    let n_poses = env_u32("GAIA_V9_TAILMASS_N", 12); // poses to sample (front x2/3, wide x1/3 cadence)
    let seed = env_u32("GAIA_V9_TAILMASS_SEED", 0x5eed_c0de) as u64;

    let fov_deg = params.fov_y_degrees;
    let front_eye = params.camera_position;
    let front_pivot = [0.0f32, 2.0, 0.0];
    let wide_eye = [-4.5f32, 8.5, 33.0];
    let wide_pivot = [-5.5f32, 2.0, 15.5];

    eprintln!(
        "[{run_tag}] {tw}x{th} K={k_draws} evidence_spp={evidence_spp} pool_orbit_deg={pool_orbit_deg} pool_jitter={pool_jitter} n_poses={n_poses}"
    );
    eprintln!("[{run_tag}] DOMAIN: loss computes target_dl = target_demod_log(mean_linear(K draws), albedo) — measuring tail mass IN THIS demod-log target, matching the loss's own arithmetic (verified by source read, see file header)");

    let mut rng = Rng(0xd15e_ed00_08f0_0dc0 ^ seed);
    let n = (tw * th) as usize;

    let mut agg_frac3 = 0.0f64;
    let mut agg_frac10 = 0.0f64;
    let mut agg_frac30 = 0.0f64;
    let mut agg_maxmean = 0.0f64;
    let mut n_samples = 0u32;

    for i in 0..n_poses {
        // Cadence matches v9c: 2 front-pivot draws per 1 wide-anchor draw.
        let (eye, pivot) = if i % 3 == 2 { (wide_eye, wide_pivot) } else { (front_eye, front_pivot) };
        let (cam, yaw) = pool_camera(eye, pivot, fov_deg, pool_orbit_deg, pool_jitter, &mut rng);

        let (_low_e, _low_d) = trace_headless_split(
            &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw / 2, th / 2, 1,
            &IntegratorParams { spp: evidence_spp, seed: 0x7abc, ..IntegratorParams::default() },
        );
        let (albedo, _normal, _depth) = scrying_glass::integrator::split_aov(&scrying_glass::integrator::trace_headless_aov(
            &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
        ));

        let mut radiance_sum = vec![GVec3::ZERO; n];
        for kd in 0..k_draws {
            let np_b = IntegratorParams { spp: evidence_spp, seed: DRAW_B_SEED_BASE + 11 + kd * 9973, ..IntegratorParams::default() };
            let (e_b, d_b) = trace_headless_split(&device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, 1, &np_b);
            for px in 0..n {
                radiance_sum[px] += e_b[px] + d_b[px];
            }
        }
        let inv_k = 1.0 / (k_draws.max(1) as f32);
        let target_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n).map(|px| target_demod_log(radiance_sum[px] * inv_k, albedo[px])).collect();
        let lum_img: Vec<f32> = target_dl.iter().map(|c| lum(*c)).collect();

        let mean_lum = (lum_img.iter().map(|v| *v as f64).sum::<f64>() / n as f64).max(1e-9);
        let max_lum = lum_img.iter().cloned().fold(0.0f32, f32::max);
        let local_mean = local_mean_7x7(&lum_img, tw, th);

        let mut c3 = 0u32;
        let mut c10 = 0u32;
        let mut c30 = 0u32;
        for px in 0..n {
            let lm = local_mean[px].max(1e-6);
            let v = lum_img[px];
            if v > 3.0 * lm {
                c3 += 1;
            }
            if v > 10.0 * lm {
                c10 += 1;
            }
            if v > 30.0 * lm {
                c30 += 1;
            }
        }
        let frac3 = c3 as f64 / n as f64;
        let frac10 = c10 as f64 / n as f64;
        let frac30 = c30 as f64 / n as f64;
        let maxmean = max_lum as f64 / mean_lum;

        println!(
            "[{run_tag}] pose {i} anchor={} yaw={yaw:.1} mean_lum={mean_lum:.4} max_lum={max_lum:.4} max/mean={maxmean:.2} frac(>3xlocal)={:.5} frac(>10xlocal)={:.5} frac(>30xlocal)={:.5}",
            if i % 3 == 2 { "wide" } else { "front" }, frac3, frac10, frac30
        );
        std::io::stdout().flush().ok();

        agg_frac3 += frac3;
        agg_frac10 += frac10;
        agg_frac30 += frac30;
        agg_maxmean += maxmean;
        n_samples += 1;
    }

    let nf = n_samples.max(1) as f64;
    println!(
        "[{run_tag}] AGGREGATE over {n_samples} pool poses: mean frac(>3xlocal)={:.5} mean frac(>10xlocal)={:.5} mean frac(>30xlocal)={:.5} mean max/mean={:.2}",
        agg_frac3 / nf, agg_frac10 / nf, agg_frac30 / nf, agg_maxmean / nf
    );
}
