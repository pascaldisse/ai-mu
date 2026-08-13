//! V9-WIRE LEVER 2 — NUMERIC VERIFY (task mandate, 2026-07-26): does the
//! split E/D radiance trace's own sum (`radiance_ed`, `integrate_split` in
//! integrator.wgsl) equal the composite path-tracer's own sum (`radiance`,
//! `integrate`), per pixel, over the SAME RNG draws (same camera, sun, sky,
//! seed, samples_before progression)?
//!
//! Both kernels run the byte-identical bounce loop (BVH walk, GGX/diffuse
//! lobe sampling, Russian roulette, `urand(pixel, sample, dim)` hashes) —
//! `radiance` accumulates every contribution into ONE running sum `L`;
//! `radiance_ed` accumulates the SAME contributions into TWO running sums
//! (`e_acc` before the chain's first diffuse/rough scatter, `d_acc` after),
//! then this harness (mirroring what `integrate_split`'s WGSL itself does
//! per-sample: `e_sum + med.xyz + med.w*rp.e`, `d_sum + med.w*rp.d`) sums
//! `e+d` and compares it to `radiance`'s own total. Floating-point addition
//! is not associative, so `e+d` regroups the SAME terms in a different order
//! than `L`'s single running sum — this harness measures exactly how far
//! that regrouping drifts (expectation: fp-associativity-close, not
//! necessarily bit-identical).
//!
//! Run at multiple spp (1, 4, 16) over multiple frames (progressing
//! `samples_before`, so the RNG draws actually exercised span many distinct
//! (pixel, sample, dim) hashes, many bounce depths, many diffuse/specular
//! chain lengths) at the naruko "front" pose (the same world/pose every
//! other v7/v9 probe in this file uses).
//!
//! RESULT (2026-07-26, naruko "front", 48x32 low-res grid, N=1536):
//! E+D != composite for ~30/1536 px (~2%) at EVERY spp/frame shape tried,
//! max_abs_diff ~3.7e-2 (NOT sub-ULP fp noise). Root-caused via two
//! isolation probes (both in `main` below): (1) composite-vs-composite
//! (the SAME `radiance()`/`integrate()` kernel run on two independently-
//! constructed `Integrator` instances) is BIT-IDENTICAL (0.0 diff) — rules
//! out GPU/pipeline/buffer nondeterminism entirely; (2) zeroing the sun
//! (`SunLight.intensity = 0`) makes the mismatch vanish COMPLETELY (0/1536,
//! max_abs_diff 0.0) at the SAME max_bounces=0 case that showed 30/1536
//! mismatches with the sun on. Conclusion: the divergence lives in the sun
//! next-event branch's `occluded(p + n*eps, sun_dir, eps, INF)` shadow-ray
//! test specifically — `radiance`/`radiance_ed` are two SEPARATELY COMPILED
//! WGSL functions with source-identical sun-branch code, but the surrounding
//! code differs (one running accumulator vs `e_acc`/`d_acc` + a `diffuse_seen`
//! bookkeeping bool), which can lead the shader compiler to fuse/reassociate
//! the `p = o + d*hit.t` / `p + n*eps` bias arithmetic differently between
//! the two functions — a few ULPs of drift in the shadow-ray origin, which
//! is enough to FLIP the boolean occlusion result exactly at grazing self-
//! shadow terminators (classic shadow-acne territory), turning a sub-ULP
//! numeric drift into an all-or-nothing loss of the FULL sun term at those
//! specific pixels. This is a genuine (if narrow, terminator-only) pre-
//! existing divergence between the two kernels — NOT associativity-close in
//! the naive sense, though its ROOT is still floating-point non-
//! associativity (amplified through a boolean gate rather than staying a
//! small numeric delta). Filed honestly; not fixed here (out of this atom's
//! scope — LEVER 2 below does not depend on the answer either way, see the
//! downstream-use finding).
//!
//! DOWNSTREAM-USE FINDING (separate from the numeric question, see
//! `main.rs`'s `v9_trace_reuse` field doc): `FeatureGatherV9::encode` (the
//! v9 gather `main.rs::resolve_frame_v9`/`_async` drive) takes `accum_ed` +
//! `aov` as its ONLY trace inputs — it never reads the composite `net_accum`
//! buffer `integrator.dispatch`/`radiance` fills. So for `is_v9`, the
//! composite dispatch's OWN output is dead regardless of whether E+D equals
//! it numerically; `GAIA_V9_TRACE_REUSE` skips that dispatch outright rather
//! than deriving anything from E+D. This harness's parity number is reported
//! anyway (task mandate) as the honest cross-check that the two kernels really
//! are the SAME transport, just bucketed differently — not two silently
//! diverged implementations that happen to both go unread today.

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::integrator::{
    headless_device, resolve, trace_headless, trace_headless_split, IntegratorParams,
};
use scrying_glass::scene::RenderScene;

fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &scrying_glass::scene::Camera,
    sun: &scrying_glass::scene::SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    w: u32,
    h: u32,
    frames: u32,
    spp: u32,
    label: &str,
) {
    let params = IntegratorParams { spp, seed: 0x7abc + 5, ..IntegratorParams::default() };

    // Composite: `integrate()` / `radiance()`.
    let composite_raw = trace_headless(device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h, frames, &params, None);
    let composite = resolve(&composite_raw);

    // Split: `integrate_split()` / `radiance_ed()`. `trace_headless_split`
    // already divides sum/count exactly like `resolve()` does for the
    // composite accum, over the SAME frame/samples_before progression.
    let (e, d) = trace_headless_split(device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h, frames, &params);

    assert_eq!(composite.len(), e.len());
    assert_eq!(composite.len(), d.len());

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_abs = 0f64;
    let mut worst_px = 0usize;
    for i in 0..composite.len() {
        let sum_ed = e[i] + d[i];
        let diff = (sum_ed - composite[i]).abs();
        let dmax = diff.x.max(diff.y).max(diff.z);
        if dmax > max_abs {
            max_abs = dmax;
            worst_px = i;
        }
        let mag = composite[i].x.abs().max(composite[i].y.abs()).max(composite[i].z.abs()).max(1e-6);
        max_rel = max_rel.max(dmax / mag);
        sum_abs += dmax as f64;
    }
    let mean_abs = sum_abs / (composite.len() as f64);
    println!(
        "[v9-ed-composite-parity] {label} spp={spp} frames={frames} N={} \
         max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e} mean_abs_diff={mean_abs:.6e} \
         worst_px={worst_px} composite={:?} e+d={:?}",
        composite.len(),
        composite[worst_px],
        e[worst_px] + d[worst_px],
    );
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v9-ed-composite-parity] SKIP — no GPU adapter");
        return;
    };

    let params = naruko_params();
    let world_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let front = law_poses(&params)
        .into_iter()
        .find(|(n, _)| *n == "front")
        .expect("front pose")
        .1;

    let (w, h) = (48u32, 32u32); // low-res trace grid, matches the v9 pipeline's own low_w/low_h scale

    // Multiple spp/frame shapes: 1spp x1 frame (the live per-frame case),
    // higher spp/more frames to exercise deeper bounce chains and more
    // distinct (pixel, sample, dim) RNG draws.
    run_case(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, 1, "1frame");
    run_case(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, 16, "1frame-16spp");
    run_case(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 4, 4, "4frame-4spp");
    run_case(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 16, 1, "16frame-1spp");

    // DEBUG: max_bounces=0 isolates the primary-hit-only terms (emissive +
    // sun next-event + sky miss) — diffuse_seen can never flip, so every
    // contribution should land in e_acc alone and e+d should equal L exactly
    // (same single-term sum, no regrouping at all). If THIS already mismatches,
    // the bug is in the shared per-hit term computation, not the bucket split.
    let p0 = IntegratorParams { spp: 1, max_bounces: 0, seed: 0x7abc + 5, ..IntegratorParams::default() };
    let composite_raw = trace_headless(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, &p0, None);
    let composite0 = resolve(&composite_raw);
    let (e0, d0) = trace_headless_split(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, &p0);
    let mut max0 = 0f32;
    let mut worst0 = 0usize;
    for i in 0..composite0.len() {
        let diff = (e0[i] + d0[i] - composite0[i]).abs();
        let dmax = diff.x.max(diff.y).max(diff.z);
        if dmax > max0 { max0 = dmax; worst0 = i; }
    }
    println!(
        "[v9-ed-composite-parity] DEBUG max_bounces=0 max_abs_diff={max0:.6e} worst_px={worst0} \
         composite={:?} e={:?} d={:?} e+d={:?}",
        composite0[worst0], e0[worst0], d0[worst0], e0[worst0] + d0[worst0],
    );
    let mut n_mismatch = 0usize;
    let mut shown = 0usize;
    for i in 0..composite0.len() {
        let diff = (e0[i] + d0[i] - composite0[i]).abs();
        let dmax = diff.x.max(diff.y).max(diff.z);
        if dmax > 1e-4 {
            n_mismatch += 1;
            if shown < 8 {
                println!(
                    "[v9-ed-composite-parity] DEBUG mismatch px={i} composite={:?} e={:?} d={:?}",
                    composite0[i], e0[i], d0[i]
                );
                shown += 1;
            }
        }
    }
    println!("[v9-ed-composite-parity] DEBUG max_bounces=0 mismatches(>1e-4)={n_mismatch}/{}", composite0.len());

    // DEBUG: dump the first 8 pixels raw (composite, e, d) unconditionally at
    // spp=1/frame=1 to see the general shape of any mismatch (not just the
    // single worst pixel).
    let params1 = IntegratorParams { spp: 1, seed: 0x7abc + 5, ..IntegratorParams::default() };
    let craw = trace_headless(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, &params1, None);
    let c1 = resolve(&craw);
    let (e1, d1) = trace_headless_split(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, &params1);
    for i in 0..8 {
        println!("[v9-ed-composite-parity] DEBUG px={i} composite={:?} e={:?} d={:?} e+d={:?}", c1[i], e1[i], d1[i], e1[i] + d1[i]);
    }

    // DEBUG: composite-vs-composite (SAME `radiance()` kernel, TWO SEPARATE
    // `Integrator`/pipeline/buffer instances) — isolates whether the earlier
    // mismatch is about `radiance` vs `radiance_ed` specifically, or about
    // running the SAME shader twice on two independently-constructed GPU
    // objects (pipeline/buffer nondeterminism unrelated to the E/D split).
    let craw_a = trace_headless(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, &p0, None);
    let ca = resolve(&craw_a);
    let craw_b = trace_headless(&device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1, &p0, None);
    let cb = resolve(&craw_b);
    let mut max_cc = 0f32;
    for i in 0..ca.len() {
        let diff = (ca[i] - cb[i]).abs();
        max_cc = max_cc.max(diff.x.max(diff.y).max(diff.z));
    }
    println!("[v9-ed-composite-parity] DEBUG composite-vs-composite (2 separate Integrator instances) max_abs_diff={max_cc:.6e}");

    // DEBUG: sun OFF (intensity=0) at max_bounces=0 — isolates whether the
    // mismatch survives without the sun next-event term at all (pointing at
    // the emissive term instead) or disappears (pointing at the sun/occluded
    // branch specifically).
    let mut sun_off = scene.sun.clone();
    sun_off.intensity = 0.0;
    let craw_s = trace_headless(&device, &queue, &bvh, &front, &sun_off, scene.sky_top, scene.sky_horizon, w, h, 1, &p0, None);
    let cs = resolve(&craw_s);
    let (es, ds) = trace_headless_split(&device, &queue, &bvh, &front, &sun_off, scene.sky_top, scene.sky_horizon, w, h, 1, &p0);
    let mut max_s = 0f32;
    let mut n_mismatch_s = 0usize;
    for i in 0..cs.len() {
        let diff = (es[i] + ds[i] - cs[i]).abs();
        let dmax = diff.x.max(diff.y).max(diff.z);
        max_s = max_s.max(dmax);
        if dmax > 1e-4 { n_mismatch_s += 1; }
    }
    println!("[v9-ed-composite-parity] DEBUG sun-off max_bounces=0 max_abs_diff={max_s:.6e} mismatches(>1e-4)={n_mismatch_s}/{}", cs.len());
}
