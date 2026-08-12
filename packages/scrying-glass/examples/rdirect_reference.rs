//! TRUE CONVERGED GROUND TRUTH — offline reference renderer for the
//! bar-res (640x480) `orbit_-20` held-out pose (same pose + scene +
//! integrator `rdirect_v9_eval_640.rs`/`rdirect_train_v9k.rs` use as
//! "teacher"). Unlike those files' `ref_frames` teacher (default 64 spp,
//! used only as an eval yardstick), this renders MANY MORE samples and is
//! the first attempt at an actually-converged (no visible speckle)
//! reference image for this scene/pose.
//!
//! Accumulates in LINEAR space (chunked `trace_headless_split` calls,
//! equal chunk size => plain running mean of chunk means), presents with
//! the SAME tone/gamma as the eval PNGs (`linear_to_srgb`, exposure 1.0,
//! `rdirect_v9_eval_640.rs::write_png` verbatim).
//!
//! Env: GAIA_REF_SPP (total samples per pixel, default 512 — IRON law),
//! GAIA_REF_CHUNK (samples per progress chunk, default 100),
//! GAIA_REF_W/H (640/480), GAIA_REF_OUT (scratch/).
//!
//! Run: cargo run --release --example rdirect_reference
//!      nice -n19 cargo run --release --example rdirect_reference   # long runs

use std::path::Path;
use std::time::Instant;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, trace_headless_split};
use scrying_glass::scene::{LeafTriangle, RenderScene};

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_str(n: &str, d: &str) -> String {
    std::env::var(n).unwrap_or_else(|_| d.to_string())
}
fn env_f32_opt(n: &str) -> Option<f32> {
    std::env::var(n).ok().and_then(|v| v.parse().ok())
}

// ── verbatim from rdirect_v9_eval_640.rs (same presentation) ──────────────
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
    eprintln!("[ref] wrote {}", path.display());
}

/// Renders `total_spp` samples of the held-out `orbit_-20` pose at `w`x`h`,
/// in equal chunks of `chunk` samples (progress print after every chunk),
/// accumulating the teacher (`e+d`) in linear space. Chunk means are
/// weighted by chunk size (last chunk may be short) so the result is the
/// exact overall mean regardless of how it was chunked.
#[allow(clippy::too_many_arguments)]
fn render_converged(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    cam: &scrying_glass::scene::Camera,
    sun: &scrying_glass::scene::SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    w: u32,
    h: u32,
    total_spp: u32,
    chunk: u32,
    t0: &Instant,
) -> Vec<GVec3> {
    let n = (w * h) as usize;
    let mut sum = vec![GVec3::ZERO; n];
    let mut done = 0u32;
    while done < total_spp {
        let this_chunk = chunk.min(total_spp - done);
        let params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
        let (e, d) = trace_headless_split(
            device, queue, bvh, cam, sun, sky_top, sky_horizon, w, h, this_chunk, &params,
        );
        for i in 0..n {
            sum[i] += (e[i] + d[i]) * (this_chunk as f32);
        }
        done += this_chunk;
        eprintln!(
            "[ref] {done}/{total_spp} spp done ({:.1}s elapsed)",
            t0.elapsed().as_secs_f64()
        );
    }
    let inv = 1.0 / (total_spp.max(1) as f32);
    sum.into_iter().map(|s| s * inv).collect()
}

fn main() {
    let tw = env_u32("GAIA_REF_W", 640);
    let th = env_u32("GAIA_REF_H", 480);
    let total_spp = env_u32("GAIA_REF_SPP", 512);
    let chunk = env_u32("GAIA_REF_CHUNK", 100);
    let out_dir = env_str("GAIA_REF_OUT", "scratch");

    eprintln!("[ref] {tw}x{th} spp={total_spp} chunk={chunk} out={out_dir}");

    let Some((device, queue)) = headless_device() else { panic!("[ref] no GPU"); };
    let mut params = scrying_glass::denoiser_dataset::naruko_params();
    // MAGIC CRYSTAL PART 1 — EMISSION SWEEP: multiplies the emissive radiance
    // of every scene material (`chain.color * emission_intensity` at every
    // per-triangle emission derivation in `RenderScene::from_ecs`) at THIS
    // reference-scene build only — a local scale on this example's own
    // `SceneParameters`, never touching world data or any other caller's
    // params. IRON default 1.0 reproduces the OLD hardcoded
    // `naruko_params().emission_intensity` exactly (byte-identical: `x * 1.0`
    // is a no-op in IEEE754 for these finite values).
    let emission_scale = env_f32_opt("GAIA_REF_EMISSION_SCALE").unwrap_or(1.0);
    params.emission_intensity *= emission_scale;
    eprintln!(
        "[ref] emission_intensity: {} (base 2.5 x scale {})",
        params.emission_intensity, emission_scale
    );
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    // DIAGNOSTIC INSTRUMENTATION (not a renderer change): `leaf_triangles()`
    // alone is the STATIC partition only — entities carrying a `behavior`
    // or `body` component (the orbiting show-light orbs, kami_orb, any
    // rigid body) are split into `Dynamics` and are ABSENT from this tool's
    // BVH unless explicitly merged in. The live server always merges both
    // partitions per frame (see main.rs `dynamic_leaf_triangles`); this
    // offline tool historically did not. Default ON so the "converged
    // truth" actually matches what a player sees at t=0 (dynamics at their
    // authored BIND pose, since nothing here ever ticks); set
    // GAIA_REF_INCLUDE_DYNAMICS=0 to reproduce the OLD static-only behavior.
    let include_dynamics = env_u32("GAIA_REF_INCLUDE_DYNAMICS", 1) != 0;
    let mut base_tris: Vec<LeafTriangle> = scene.leaf_triangles();
    if include_dynamics {
        let dyn_tris = scene.dynamic_leaf_triangles();
        eprintln!("[ref] +{} dynamic-partition triangles merged (behavior/body entities, bind pose)", dyn_tris.len());
        base_tris.extend(dyn_tris);
    } else {
        eprintln!("[ref] GAIA_REF_INCLUDE_DYNAMICS=0 — static partition ONLY (legacy behavior, orbs/bodies OMITTED)");
    }
    let bvh = Bvh::build(&base_tris, &BvhParams::default());

    // SAME held-out pose as rdirect_v9_eval_640.rs / bar_res_probe, UNLESS
    // overridden below (IRON: pose is a param w/ default = orbit_-20).
    // Override env (any subset; unset fields keep the orbit_-20 default):
    //   GAIA_REF_EYE_X/Y/Z, GAIA_REF_YAW (radians), GAIA_REF_PITCH (radians)
    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let mut base_cam = all.iter().find(|(n, _)| *n == "orbit_-20").unwrap().1.clone();
    let mut pose_note = "orbit_-20 (default)".to_string();
    let ex = env_f32_opt("GAIA_REF_EYE_X");
    let ey = env_f32_opt("GAIA_REF_EYE_Y");
    let ez = env_f32_opt("GAIA_REF_EYE_Z");
    let ryaw = env_f32_opt("GAIA_REF_YAW");
    let rpitch = env_f32_opt("GAIA_REF_PITCH");
    if ex.is_some() || ey.is_some() || ez.is_some() || ryaw.is_some() || rpitch.is_some() {
        if let Some(x) = ex { base_cam.eye.x = x; }
        if let Some(y) = ey { base_cam.eye.y = y; }
        if let Some(z) = ez { base_cam.eye.z = z; }
        if let Some(y) = ryaw { base_cam.yaw = y; }
        if let Some(p) = rpitch { base_cam.pitch = p; }
        pose_note = format!(
            "override eye=({:.2},{:.2},{:.2}) yaw={:.3} pitch={:.3}",
            base_cam.eye.x, base_cam.eye.y, base_cam.eye.z, base_cam.yaw, base_cam.pitch
        );
    }
    eprintln!("[ref] pose: {pose_note}");

    let t0 = Instant::now();
    let teacher = render_converged(
        &device, &queue, &bvh, &base_cam, &scene.sun, scene.sky_top, scene.sky_horizon,
        tw, th, total_spp, chunk, &t0,
    );
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("[ref] {total_spp}spp reference done in {elapsed:.1}s");

    // MAGIC CRYSTAL PART 2 diagnostic: percentile histogram of the converged
    // teacher's per-pixel LINEAR max-channel radiance (pre-tonemap) — read to
    // pick a sane GAIA_NEE_CLAMP value (e.g. p99.9) instead of guessing. Cheap
    // (one sort over w*h floats), always printed, never gated — pure stderr
    // diagnostic, no output-file change.
    {
        let mut lum: Vec<f32> = teacher.iter().map(|c| c.x.max(c.y).max(c.z)).collect();
        lum.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| -> f32 {
            let idx = ((lum.len() as f64 - 1.0) * p).round() as usize;
            lum[idx.min(lum.len() - 1)]
        };
        eprintln!(
            "[ref] linear max-channel radiance histogram: p50={:.3} p90={:.3} p99={:.3} p99.9={:.3} p99.99={:.3} p99.999={:.6} max={:.6}",
            pct(0.50), pct(0.90), pct(0.99), pct(0.999), pct(0.9999), pct(0.99999), lum[lum.len() - 1]
        );
        for thr in [150.0f32, 200.0, 300.0, 500.0, 1000.0, 5000.0] {
            let n = lum.iter().filter(|&&v| v > thr).count();
            eprintln!("[ref]   px > {thr:.0}: {n}");
        }
        // Same percentiles EXCLUDING near-124.99+ pixels (direct hits on an
        // emitter's own face — the un-clamped `tri.emission` term, never
        // touched by `clamp_nee_radiance`): isolates INDIRECTLY-lit surfaces
        // (ambient+sun+GI+NEE), where the NEE shadow-ray spikes actually live.
        let mut indirect: Vec<f32> = lum.iter().copied().filter(|&v| v < 124.99).collect();
        if !indirect.is_empty() {
            indirect.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ipct = |p: f64| -> f32 {
                let idx = ((indirect.len() as f64 - 1.0) * p).round() as usize;
                indirect[idx.min(indirect.len() - 1)]
            };
            eprintln!(
                "[ref] INDIRECT-only (excludes direct emitter-face hits, {} px): p50={:.3} p90={:.3} p99={:.3} p99.9={:.3} p99.99={:.3} max={:.3}",
                indirect.len(), ipct(0.50), ipct(0.90), ipct(0.99), ipct(0.999), ipct(0.9999), indirect[indirect.len()-1]
            );
        }
    }

    let out_path = Path::new(&out_dir).join(format!("reference-{total_spp}spp.png"));
    write_png(&teacher, tw, th, 1.0, &out_path);
    println!("[ref] {total_spp}spp -> {} ({elapsed:.1}s)", out_path.display());

    // MAGIC CRYSTAL PART 2 diagnostic: optional raw LINEAR dump (f32 rgb,
    // row-major, no header) for offline firefly analysis (a true
    // local-neighborhood outlier test needs pre-tonemap radiance — sRGB
    // compresses highlights and hides spike magnitude). Off by default (a
    // 640x480 dump is ~3.5MB; never written unless asked).
    if env_u32("GAIA_REF_DUMP_LINEAR", 0) != 0 {
        let lin_path = Path::new(&out_dir).join(format!("reference-{total_spp}spp.lin"));
        let mut bytes = Vec::with_capacity(teacher.len() * 12);
        for px in &teacher {
            bytes.extend_from_slice(&px.x.to_le_bytes());
            bytes.extend_from_slice(&px.y.to_le_bytes());
            bytes.extend_from_slice(&px.z.to_le_bytes());
        }
        std::fs::write(&lin_path, &bytes).unwrap();
        eprintln!("[ref] wrote {} ({}x{} f32 rgb)", lin_path.display(), tw, th);
    }

    // OLD teacher quality (K=8-equivalent) side-by-side, same machinery,
    // freshly rendered so both PNGs come from this file's own path.
    if total_spp != 8 {
        let t1 = Instant::now();
        let old = render_converged(
            &device, &queue, &bvh, &base_cam, &scene.sun, scene.sky_top, scene.sky_horizon,
            tw, th, 8, 8, &t1,
        );
        let elapsed8 = t1.elapsed().as_secs_f64();
        let out8 = Path::new(&out_dir).join("reference-8spp.png");
        write_png(&old, tw, th, 1.0, &out8);
        println!("[ref] 8spp -> {} ({elapsed8:.1}s)", out8.display());
    }
}
