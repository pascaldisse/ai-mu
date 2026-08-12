//! V9 WIRING ATOM, ITEM 3 (2026-07-21) — FRAME WALL, v7 body vs v9 body.
//!
//! TEACHER/BENCHMARK SURFACE (same "measuring instrument, not a gate"
//! precedent as `examples/onepath_budget.rs`'s own header) — offscreen,
//! N-frame, median+p95 full-frame wall for BOTH bodies at 640×480 render
//! res (output res = render res, IRON param default), a PANNING camera
//! (so history/reprojection is exercised, not a degenerate static-camera
//! best case).
//!
//! v7 body: the EXACT production per-frame GPU sequence `main.rs`'s
//! `NetPresent::net_present_frame` runs for `is_v7` weights — same types,
//! same calls, GPU-resident end to end (no CPU round-trip between trace and
//! net): `Integrator::dispatch_split` (evidence) + `dispatch_aov` (gbuffer)
//! -> `FeatureGatherHistSplit::encode` (GPU gather + history reprojection,
//! `HistoryBuffers` ping-pong) -> `RdirectLive::forward_shared` (the real
//! zero-copy pooled MPSGraph forward). This harness does NOT go through
//! `main.rs`/`NetPresent` itself (untouched, still v7-only) — it calls the
//! SAME production types this example imports directly, which is the
//! honest way to measure "the render loop's act" without touching the
//! gated present path (see the doc block below on why NOT touching
//! `main.rs` was the deliberate scope decision for this atom).
//!
//! v9 body: `UnetLive::forward_gpu_ms`/`forward_cpu_roundtrip` (the SAME
//! GPU tensor-path forward ITEM 1 wired a trained-weights loader for) fed
//! by a CPU-BUILT 41-channel feature image — there is NO WGSL gather
//! shader for the v9 body's channel layout (39 `hist_features_split` + 2
//! real motion-vector channels) yet; `build_input_img_unet` below is the
//! SAME feature-build ITEM 2's ordeal-door path uses (itself ported from
//! `examples/rdirect_train_v9c.rs`, now duplicated a 4th time — the
//! consistent house pattern for this exact code, every prior copy cited
//! inline). This is a REAL, DISCLOSED asymmetry, not hidden: v7's gather is
//! GPU-native and zero-copy; v9's is CPU-computed and re-uploaded every
//! frame. The table below reports BOTH the full wall (apples-to-apples
//! "what it costs today") AND `UnetLive::forward_gpu_ms` alone (the
//! GPU-only forward ITEM 1's parity test already exercises), so a reader
//! can see the ceiling a future GPU-native v9 gather could reach.
//!
//! WHY `main.rs`/`NetPresent` IS NOT TOUCHED THIS ATOM (scope decision,
//! disclosed): the REAL IMAGE BAR gate (`verify_stamp`, "NO env override")
//! structurally refuses to present ANY unstamped weights, v9c included —
//! wiring `NetPresent` to accept a U-Net-format checkpoint would add real
//! surface area to a 5000-line, actively-shared, v7-serving production file
//! for a body that cannot legally present through it yet (no stamp exists,
//! and manufacturing one to unblock a perf run would violate the same "NO
//! env override" law the gate exists to enforce). This harness gets the
//! honest number without that risk; full `NetPresent` integration is the
//! disclosed remaining gap for a follow-up atom, once a v9 stamp exists.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_frame_wall
//!   GAIA_WALL_N (40 default), GAIA_WALL_WARMUP (8), GAIA_WALL_W/H (640/480)

use std::path::Path;
use std::time::Instant;

use glam::Vec2;
use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{Integrator, IntegratorParams, IntegratorUniform, headless_device, split_aov, trace_headless_aov};
use scrying_glass::rdirect::{
    CamPose, HIST_FEATURES_SPLIT, INPUT_FEATURES_SPLIT, deserialize_weights, hist_features_split,
    pixel_features_split,
};
use scrying_glass::rdirect_gather::{FeatureGatherHistSplit, FeatureGatherV9, Fp16Packer, HistoryBuffers};
use scrying_glass::rdirect_live::RdirectLive;
use scrying_glass::rdirect_unet::cpu::{Img as UnetImg, deserialize_weights as unet_deserialize_weights};
use scrying_glass::rdirect_unet::{MOTION_VECTOR_CHANNELS, UnetLive};
use scrying_glass::scene::{Camera, RenderScene};

const V7_DEPTH_TOL: f32 = 0.05; // == main.rs's own V7_DEPTH_TOL
const V7_NORMAL_THRESH: f32 = 0.85; // == main.rs's own V7_NORMAL_THRESH

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h.max(1) as f32 }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn p95(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() as f64) * 0.95) as usize;
    v[i.min(v.len() - 1)]
}

/// Synchronous GPU->host f32 readback (copy + map + wait) — the v9-gpu-gather
/// row's own "glue" cost this atom measures honestly (no zero-copy MTLBuffer
/// bridge between wgpu's device and `UnetLive`'s own separate system Metal
/// device exists yet — the disclosed remaining gap, see the doc's own gaps
/// section). Same pattern `tests/rdirect_gather_v9_ordeal.rs`'s `read_f32` /
/// `v7_present_parity_probe.rs`'s own copy use.
fn read_f32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wall v9-gpu-gather readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9-gpu-gather copy") });
    enc.copy_buffer_to_buffer(buf, 0, &readback, 0, bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    enc.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback chan").expect("map");
    let mapped = readback.get_mapped_range(..).expect("mapped");
    let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    v
}

// ─────────────────────────────────────────────────────────────────────────
// v9 BODY: CPU feature build (ITEM 2's `build_input_img_unet`, itself
// ported from `rdirect_train_v9c.rs`'s `build_input_img`/`reproject_prev` —
// duplicated here a 4th time, same house convention every prior copy
// already established, both source fns private to their own binaries).
// ─────────────────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn reproject_prev_unet(
    cur_cam: &CamPose, cur_depth: f32, cur_normal: glam::Vec3, tx: u32, ty: u32, tw: u32, th: u32,
    prev_cam: &CamPose, prev_out_dl: &[glam::Vec3], prev_depth: &[f32], prev_normal: &[glam::Vec3],
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
                let depth_ok = (dist_prev - prev_d).abs() <= V7_DEPTH_TOL * dist_prev.max(1e-4);
                let normal_ok = cur_normal.dot(prev_normal[pj]) >= V7_NORMAL_THRESH;
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

struct UnetStep {
    low_e: Vec<glam::Vec3>,
    low_d: Vec<glam::Vec3>,
    albedo: Vec<glam::Vec3>,
    normal: Vec<glam::Vec3>,
    depth: Vec<f32>,
    cam: CamPose,
}

#[allow(clippy::too_many_arguments)]
fn build_input_img_unet(
    steps: &[UnetStep], step_idx: usize, prev_out_dl: Option<&[glam::Vec3]>,
    low_w: u32, low_h: u32, tw: u32, th: u32, sky_reject: bool,
) -> UnetImg {
    let step = &steps[step_idx];
    let mut img = UnetImg::zeros(th as usize, tw as usize, HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS);
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
                    reproject_prev_unet(
                        &step.cam, step.depth[px], step.normal[px], tx, ty, tw, th,
                        &prev_step.cam, prev, &prev_step.depth, &prev_step.normal, tw, th, sky_reject,
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

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[wall] no GPU adapter");
    };
    let target_w = env_u32("GAIA_WALL_W", 640);
    let target_h = env_u32("GAIA_WALL_H", 480);
    let low_w = target_w / 2;
    let low_h = target_h / 2;
    let n_frames = env_u32("GAIA_WALL_N", 40) as usize;
    let n_warmup = env_u32("GAIA_WALL_WARMUP", 8) as usize;
    let pan_step = 0.004f32; // rad/frame, same slow pan the real-image ordeal uses
    let sky_reject = scrying_glass::rdirect::sky_history_reject();

    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();
    let bvh = Bvh::build(&base_tris, &BvhParams::default());

    let val_poses = scrying_glass::denoiser_dataset::law_poses(&params);
    let base_cam = val_poses.iter().find(|(n, _)| *n == "orbit_-20").unwrap().1.clone();
    let mut pan_cams: Vec<Camera> = Vec::with_capacity(n_warmup + n_frames);
    for k in 0..(n_warmup + n_frames) {
        let mut c = base_cam.clone();
        c.yaw += pan_step * k as f32;
        pan_cams.push(c);
    }

    println!("[wall] res {target_w}x{target_h} frames={n_frames} warmup={n_warmup} sky_reject={sky_reject}");

    // ─────────────────────────── v7 BODY ───────────────────────────
    let v7_bytes = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v7.bin"))
        .expect("committed v7 weights (the stamped default)");
    let mlp7 = deserialize_weights(&v7_bytes).expect("v7 weights parse");
    assert_eq!(mlp7.layer_dims()[0].0 as usize, HIST_FEATURES_SPLIT, "v7 weights must be the 39-in split net");
    let n = (target_w * target_h) as usize;
    let live7 = RdirectLive::from_wgpu_queue(&device, &queue, &v7_bytes, n).expect("RdirectLive on the wgpu Metal device");
    let feats7 = live7.feature_buffer().expect("pooled feature buffer");
    let hist_gather = FeatureGatherHistSplit::new(&device);
    let mut history = HistoryBuffers::new(&device, target_w, target_h);
    // Dummy vec4-padded "this frame's out_dl" source for the history swap's
    // buffer-to-buffer copy. Production packs the net's REAL output into this
    // shape first (`EvidenceEngine::encode_pack`, main.rs) — that pack pass +
    // the evidence-accumulate/clamp/demod passes it sits beside are smaller
    // auxiliary compute dispatches that run AFTER the timed region below in
    // production (the n0d perf doc's own dominant-cost split is trace+net);
    // skipped here so this harness doesn't need to re-import `EvidenceEngine`
    // for a WALL-timing-only measurement. `history.swap`'s own cost is a
    // buffer-to-buffer copy sized by BYTES, not content — this dummy buffer
    // is the same size (`out_dl_bytes(n)`) as the real one, so the copy's
    // timing contribution is representative even though its content is not
    // meaningful (disclosed simplification, not hidden).
    let out_dl_dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wall v7 out_dl dummy (swap-cost only, not real net output)"),
        size: FeatureGatherHistSplit::out_dl_bytes(n).max(1),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);
    let split_buf = integrator.make_split_buffer(&device, low_w, low_h);
    let aov_buf = integrator.make_aov_buffer(&device, target_w, target_h);
    let compute_bg_low = integrator.compute_bind_group(&device, &integrator.make_accum(&device, low_w, low_h));
    // dispatch_aov shares the SAME @group(0) layout as the composite integrate
    // pipeline (main.rs comment: "A SECOND compute pipeline (own bind group
    // layout at @group(1))") — it needs a validly-sized accum buffer bound at
    // @group(0) even though its own shader entry never writes through it.
    // Allocated ONCE outside the timed loop (production keeps `self.net_accum`
    // persistent too — only the BIND GROUP wrapper is rebuilt per frame there).
    let native_accum_buf = integrator.make_accum(&device, target_w, target_h);
    let compute_bg_native = integrator.compute_bind_group(&device, &native_accum_buf);
    let split_bg = integrator.split_bind_group(&device, &split_buf);
    let aov_bg = integrator.aov_bind_group(&device, &aov_buf);
    let np = IntegratorParams { spp: 1, ..IntegratorParams::default() };

    let mut v7_walls: Vec<f64> = Vec::with_capacity(n_frames);
    let mut prev_cam7: Option<CamPose> = None;
    for (k, cam) in pan_cams.iter().enumerate() {
        let cur_cam = cam_pose(cam, target_w, target_h);
        let t0 = Instant::now();
        // trace: evidence split (low res) + AOV (native res) — SAME two
        // dispatches `net_present_frame` issues before its gather stage.
        let uni_low = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let uni_native = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v7 trace") });
        enc.clear_buffer(&split_buf, 0, None);
        integrator.dispatch_split(&queue, &mut enc, &uni_low, &compute_bg_low, &split_bg, low_w, low_h);
        integrator.dispatch_aov(&queue, &mut enc, &uni_native, &compute_bg_native, &aov_bg, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // gather: GPU-native hist-split feature build + history reprojection.
        let prev_cam = prev_cam7.unwrap_or(cur_cam);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v7 gather") });
        hist_gather.encode(
            &device, &queue, &mut enc, &split_buf, &aov_buf, feats7,
            &history.prev_out_dl, &history.prev_aov, cur_cam, prev_cam, history.has_prev,
            history.w, history.h, V7_DEPTH_TOL, V7_NORMAL_THRESH, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // forward: the REAL production zero-copy pooled MPSGraph call.
        let _out = live7.forward_shared(n).expect("v7 forward_shared");
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // history swap: GPU-side copy of this frame's own AOV into prev_aov
        // (prev_out_dl copy skipped here — this harness measures WALL cost
        // only, not correctness; `has_prev`/`prev_cam` still advance so the
        // gather's reprojection branch is genuinely exercised every frame
        // after the first, matching production's steady-state cost).
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v7 swap") });
        history.swap(&mut enc, &out_dl_dummy, &aov_buf, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        prev_cam7 = Some(cur_cam);

        if k >= n_warmup {
            v7_walls.push(wall_ms);
        }
    }
    let v7_median = median(&mut v7_walls.clone());
    let v7_p95 = p95(&mut v7_walls.clone());
    println!("[wall] v7  (39-in split, GPU-native gather+forward_shared): median={v7_median:.3}ms p95={v7_p95:.3}ms n={}", v7_walls.len());

    // ─────────────────────────── v9 BODY ───────────────────────────
    let v9_bytes = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v9c.bin"))
        .expect("v9c checkpoint (copied read-only from v9body's own commit, ITEM 2)");
    let mut unet_net = unet_deserialize_weights(&v9_bytes).expect("v9c weights parse (GAIARD9)");
    unet_net.config.render_w = target_w as usize;
    unet_net.config.render_h = target_h as usize;
    unet_net.config.output_w = target_w as usize;
    unet_net.config.output_h = target_h as usize;
    let live9 = UnetLive::from_weights(&unet_net).expect("UnetLive from trained weights (ITEM 1 loader)");

    let mut v9_walls: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9_gpu_only: Vec<f64> = Vec::with_capacity(n_frames);
    let mut chain_dl: Vec<Vec<glam::Vec3>> = Vec::new();
    let mut steps: Vec<UnetStep> = Vec::new();
    for (k, cam) in pan_cams.iter().enumerate() {
        let t0 = Instant::now();
        // trace: SAME two passes (CPU-readback headless helpers here, unlike
        // v7's GPU-resident path above — the disclosed asymmetry the module
        // doc explains: there is no GPU gather shader for v9's channel
        // layout yet, so a CPU-side feature build needs the buffers on the
        // host regardless; using the readback helpers keeps this file's own
        // trace code honest/simple rather than hand-duplicating a GPU-
        // resident-then-immediately-read-back path that would cost the same).
        let (low_e, low_d) = scrying_glass::integrator::trace_headless_split(
            &device, &queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np,
        );
        let (albedo, normal, depth) = split_aov(&trace_headless_aov(
            &device, &queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        ));
        let cur_cam = cam_pose(cam, target_w, target_h);
        steps.push(UnetStep { low_e, low_d, albedo, normal, depth, cam: cur_cam });
        let idx = steps.len() - 1;
        let prev: Option<&[glam::Vec3]> = if idx == 0 { None } else { Some(&chain_dl[idx - 1]) };

        // "gather": CPU-side 41-channel feature build (the disclosed gap).
        let input = build_input_img_unet(&steps, idx, prev, low_w, low_h, target_w, target_h, sky_reject);

        // forward: the REAL GPU tensor-path forward (ITEM 1's loader).
        let out = live9.forward_cpu_roundtrip(&input.data).expect("v9 forward_cpu_roundtrip");
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = live9.forward_gpu_ms(&input.data).expect("v9 forward_gpu_ms");

        let mut dl_vec = vec![glam::Vec3::ZERO; (target_w * target_h) as usize];
        for i in 0..dl_vec.len() {
            dl_vec[i] = glam::Vec3::new(out[i * 3], out[i * 3 + 1], out[i * 3 + 2]);
        }
        chain_dl.push(dl_vec);

        if k >= n_warmup {
            v9_walls.push(wall_ms);
            v9_gpu_only.push(gpu_ms);
        }
    }
    let v9_median = median(&mut v9_walls.clone());
    let v9_p95 = p95(&mut v9_walls.clone());
    let v9_gpu_median = median(&mut v9_gpu_only.clone());
    let v9_gpu_p95 = p95(&mut v9_gpu_only.clone());
    println!(
        "[wall] v9  (41-in U-Net, CPU-built gather + forward_cpu_roundtrip): median={v9_median:.3}ms p95={v9_p95:.3}ms n={}",
        v9_walls.len()
    );
    println!(
        "[wall] v9  GPU-only forward_gpu_ms (no trace/gather/readback, ITEM 1's own split): median={v9_gpu_median:.3}ms p95={v9_gpu_p95:.3}ms"
    );

    // ─────────────────────── v9 BODY, GPU-NATIVE GATHER ───────────────────────
    // V9-WIRE ATOM (2026-07-21): the CPU per-pixel feature-build loop above is
    // replaced by `FeatureGatherV9::encode` (`gather_v9`, GPU-native, mirrors
    // the v7 production `FeatureGatherHistSplit::encode` house pattern one
    // section up) — SAME two GPU trace dispatches v7 issues (`dispatch_split`
    // + `dispatch_aov`), then ONE compute dispatch builds the whole 41-wide
    // input tensor on the GPU. The remaining disclosed asymmetry vs v7: there
    // is NO zero-copy MTLBuffer bridge yet between wgpu's device and
    // `UnetLive`'s own separate system Metal device (`from_weights`/
    // `from_system` both call `MTLCreateSystemDefaultDevice` directly) — so
    // this row still pays ONE GPU->host readback of the gathered tensor
    // (`read_f32`, timed, IN the wall) before `forward_cpu_roundtrip` re-
    // uploads it fp16 on `UnetLive`'s own device. Full zero-copy wiring
    // (mirroring `RdirectLive::from_wgpu_queue`'s own pooled-buffer bridge)
    // is the disclosed remaining gap — see the doc's gaps section.
    let gather9 = FeatureGatherV9::new(&device);
    let mut history9 = HistoryBuffers::new(&device, target_w, target_h);
    let split_buf9 = integrator.make_split_buffer(&device, low_w, low_h);
    let aov_buf9 = integrator.make_aov_buffer(&device, target_w, target_h);
    let compute_bg_low9 = integrator.compute_bind_group(&device, &integrator.make_accum(&device, low_w, low_h));
    let native_accum_buf9 = integrator.make_accum(&device, target_w, target_h);
    let compute_bg_native9 = integrator.compute_bind_group(&device, &native_accum_buf9);
    let split_bg9 = integrator.split_bind_group(&device, &split_buf9);
    let aov_bg9 = integrator.aov_bind_group(&device, &aov_buf9);
    let feat41_bytes = FeatureGatherV9::feature_bytes(n);
    let feats9_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wall v9-gpu-gather feats41"),
        size: feat41_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut v9g_walls: Vec<f64> = Vec::with_capacity(n_frames);
    // Decomposition (per the task's own "if >16.67ms, decompose" ask):
    // trace (dispatch_split+dispatch_aov), gather (gather_v9 dispatch),
    // readback (GPU->host copy+map), forward (forward_cpu_roundtrip).
    let mut v9g_trace: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9g_gather: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9g_readback: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9g_forward: Vec<f64> = Vec::with_capacity(n_frames);
    let mut prev_cam9g: Option<CamPose> = None;
    for (k, cam) in pan_cams.iter().enumerate() {
        let cur_cam = cam_pose(cam, target_w, target_h);
        let t0 = Instant::now();
        let uni_low = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let uni_native = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9g trace") });
        enc.clear_buffer(&split_buf9, 0, None);
        integrator.dispatch_split(&queue, &mut enc, &uni_low, &compute_bg_low9, &split_bg9, low_w, low_h);
        integrator.dispatch_aov(&queue, &mut enc, &uni_native, &compute_bg_native9, &aov_bg9, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t_trace = t0.elapsed().as_secs_f64() * 1000.0;

        let prev_cam = prev_cam9g.unwrap_or(cur_cam);
        let t1 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9g gather") });
        gather9.encode(
            &device, &queue, &mut enc, &split_buf9, &aov_buf9, &feats9_buf, &history9.prev_out_dl,
            &history9.prev_aov, cur_cam, prev_cam, history9.has_prev, history9.w, history9.h,
            V7_DEPTH_TOL, V7_NORMAL_THRESH, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t_gather = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let host_feats = read_f32(&device, &queue, &feats9_buf, feat41_bytes);
        let t_readback = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = Instant::now();
        let out = live9.forward_cpu_roundtrip(&host_feats).expect("v9 forward_cpu_roundtrip (gpu-gather row)");
        let t_forward = t3.elapsed().as_secs_f64() * 1000.0;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // History swap: THIS row's swap carries the REAL net output (unlike
        // the v7 section's dummy swap-cost-only buffer above) — cheap enough
        // (one small buffer_init + a GPU-side copy) not to need excluding
        // from realism, and untimed here (same convention v7's own section
        // uses: the wall stops at the real work, swap is next-frame prep).
        let out_dl_padded: Vec<[f32; 4]> = (0..n).map(|p| [out[p * 3], out[p * 3 + 1], out[p * 3 + 2], 0.0]).collect();
        let out_dl_buf9 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wall v9g real out_dl (for next-frame history)"),
            contents: bytemuck::cast_slice(&out_dl_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9g swap") });
        history9.swap(&mut enc, &out_dl_buf9, &aov_buf9, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        prev_cam9g = Some(cur_cam);

        if k >= n_warmup {
            v9g_walls.push(wall_ms);
            v9g_trace.push(t_trace);
            v9g_gather.push(t_gather);
            v9g_readback.push(t_readback);
            v9g_forward.push(t_forward);
        }
    }
    let v9g_median = median(&mut v9g_walls.clone());
    let v9g_p95 = p95(&mut v9g_walls.clone());
    println!(
        "[wall] v9  (41-in U-Net, GPU-native gather_v9 + readback + forward_cpu_roundtrip): median={v9g_median:.3}ms p95={v9g_p95:.3}ms n={}",
        v9g_walls.len()
    );
    println!(
        "[wall] v9g decomposition medians: trace={:.3}ms gather={:.3}ms readback={:.3}ms forward={:.3}ms",
        median(&mut v9g_trace.clone()), median(&mut v9g_gather.clone()), median(&mut v9g_readback.clone()), median(&mut v9g_forward.clone())
    );
    println!(
        "[wall] v9g decomposition p95s:     trace={:.3}ms gather={:.3}ms readback={:.3}ms forward={:.3}ms",
        p95(&mut v9g_trace.clone()), p95(&mut v9g_gather.clone()), p95(&mut v9g_readback.clone()), p95(&mut v9g_forward.clone())
    );

    // ───────────────────── v9 BODY, GPU GATHER + FAST FORWARD ─────────────────────
    // The v9g row above still calls `forward_cpu_roundtrip` for its forward
    // step (`runWithMTLCommandQueue_feeds_targetTensors_targetOperations`,
    // recompiles/re-encodes the graph every call — ITEM 1's own doc names
    // this the "~34ms CPU encode/wait around a ~6ms GPU forward" n0e gap).
    // This row swaps in `forward_gpu_with_output` (this atom's own addition
    // to `rdirect_unet.rs`, SAME persistent-compiled-executable path
    // `forward_gpu_ms` already used for ITEM 1's own "GPU-only" timing row,
    // now also returning the real output) — the closest this atom gets to
    // "gather (GPU) + forward (persistent executable) + glue only", the
    // ceiling the task's own framing ("GPU forward alone is 4.75ms") points at.
    let mut v9gf_walls: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9gf_gather: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9gf_readback: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9gf_forward: Vec<f64> = Vec::with_capacity(n_frames);
    // GPU-only sub-split of `v9gf_forward` (the `MTLCommandBuffer`
    // GPUEndTime-GPUStartTime `forward_gpu_with_output` returns, SAME number
    // `forward_gpu_ms` reports) — the gap between this and `v9gf_forward`'s
    // own CPU wall-clock is the fp16 conversion glue (host f32->f16 input
    // write, 12.6M scalar element writes at 640x480x41, + f16->f32 output
    // read), not GPU time.
    let mut v9gf_forward_gpu_only: Vec<f64> = Vec::with_capacity(n_frames);
    let mut history9f = HistoryBuffers::new(&device, target_w, target_h);
    let mut prev_cam9gf: Option<CamPose> = None;
    for (k, cam) in pan_cams.iter().enumerate() {
        let cur_cam = cam_pose(cam, target_w, target_h);
        let t0 = Instant::now();
        let uni_low = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let uni_native = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9gf trace") });
        enc.clear_buffer(&split_buf9, 0, None);
        integrator.dispatch_split(&queue, &mut enc, &uni_low, &compute_bg_low9, &split_bg9, low_w, low_h);
        integrator.dispatch_aov(&queue, &mut enc, &uni_native, &compute_bg_native9, &aov_bg9, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let prev_cam = prev_cam9gf.unwrap_or(cur_cam);
        let t1 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9gf gather") });
        gather9.encode(
            &device, &queue, &mut enc, &split_buf9, &aov_buf9, &feats9_buf, &history9f.prev_out_dl,
            &history9f.prev_aov, cur_cam, prev_cam, history9f.has_prev, history9f.w, history9f.h,
            V7_DEPTH_TOL, V7_NORMAL_THRESH, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t_gather = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let host_feats = read_f32(&device, &queue, &feats9_buf, feat41_bytes);
        let t_readback = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = Instant::now();
        let (gpu_only_ms, out) = live9.forward_gpu_with_output(&host_feats).expect("v9 forward_gpu_with_output");
        let t_forward = t3.elapsed().as_secs_f64() * 1000.0;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let out_dl_padded: Vec<[f32; 4]> = (0..n).map(|p| [out[p * 3], out[p * 3 + 1], out[p * 3 + 2], 0.0]).collect();
        let out_dl_buf9f = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wall v9gf real out_dl (for next-frame history)"),
            contents: bytemuck::cast_slice(&out_dl_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9gf swap") });
        history9f.swap(&mut enc, &out_dl_buf9f, &aov_buf9, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        prev_cam9gf = Some(cur_cam);

        if k >= n_warmup {
            v9gf_walls.push(wall_ms);
            v9gf_gather.push(t_gather);
            v9gf_readback.push(t_readback);
            v9gf_forward.push(t_forward);
            v9gf_forward_gpu_only.push(gpu_only_ms);
        }
    }
    let v9gf_median = median(&mut v9gf_walls.clone());
    let v9gf_p95 = p95(&mut v9gf_walls.clone());
    println!(
        "[wall] v9  (41-in U-Net, GPU-native gather_v9 + readback + forward_gpu_with_output): median={v9gf_median:.3}ms p95={v9gf_p95:.3}ms n={}",
        v9gf_walls.len()
    );
    println!(
        "[wall] v9gf decomposition medians: gather={:.3}ms readback={:.3}ms forward(cpu-wall)={:.3}ms forward(gpu-only)={:.3}ms fp16-glue={:.3}ms",
        median(&mut v9gf_gather.clone()), median(&mut v9gf_readback.clone()), median(&mut v9gf_forward.clone()),
        median(&mut v9gf_forward_gpu_only.clone()),
        median(&mut v9gf_forward.clone()) - median(&mut v9gf_forward_gpu_only.clone())
    );
    println!(
        "[wall] v9gf decomposition p95s:     gather={:.3}ms readback={:.3}ms forward(cpu-wall)={:.3}ms forward(gpu-only)={:.3}ms",
        p95(&mut v9gf_gather.clone()), p95(&mut v9gf_readback.clone()), p95(&mut v9gf_forward.clone()), p95(&mut v9gf_forward_gpu_only.clone())
    );

    // ───────────────── v9 BODY, ZERO-COPY BRIDGE (this atom) ─────────────────
    // V9-WIRE ATOM (2026-07-21, ZERO-COPY BRIDGE): the v9gf row above still
    // pays a `read_f32` GPU->host readback of the whole 41-channel gathered
    // tensor (~50MB/frame) plus a host-side `f16::from_f32` scalar loop over
    // 12.6M elements inside `forward_gpu_with_output`'s input write — both
    // exist ONLY because `UnetLive::from_weights`/`from_system` open their
    // OWN `MTLCreateSystemDefaultDevice()` handle, separate from wgpu's own
    // Metal device. This row uses `UnetLive::from_wgpu_queue` (mirrors
    // `RdirectLive::from_wgpu_queue` exactly) + `Fp16Packer` (a GPU compute
    // dispatch, `pack2x16float`) to write the gather's f32 output directly
    // into the SAME MTLBuffer the graph's forward reads — no readback, no
    // CPU conversion loop, `forward_gpu_bridged` reads only the (~14x
    // smaller) output back.
    let live9_bridge =
        UnetLive::from_wgpu_queue(&device, &queue, &unet_net).expect("UnetLive::from_wgpu_queue (the zero-copy bridge)");
    let feature_buf_u32 = live9_bridge.feature_buf_u32().expect("from_wgpu_queue built the bridge").clone();
    let packer = Fp16Packer::new(&device);
    let n_in = (target_w * target_h) as usize * (HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS);

    let mut v9zc_walls: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_trace: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_gather: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_pack: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_forward: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_forward_gpu_only: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_encode_commit: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_wait: Vec<f64> = Vec::with_capacity(n_frames);
    let mut v9zc_readout: Vec<f64> = Vec::with_capacity(n_frames);
    let mut history9zc = HistoryBuffers::new(&device, target_w, target_h);
    let mut prev_cam9zc: Option<CamPose> = None;
    for (k, cam) in pan_cams.iter().enumerate() {
        let cur_cam = cam_pose(cam, target_w, target_h);
        let t0 = Instant::now();
        let uni_low = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let uni_native = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
            integrator.node_count, integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9zc trace") });
        enc.clear_buffer(&split_buf9, 0, None);
        integrator.dispatch_split(&queue, &mut enc, &uni_low, &compute_bg_low9, &split_bg9, low_w, low_h);
        integrator.dispatch_aov(&queue, &mut enc, &uni_native, &compute_bg_native9, &aov_bg9, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t_trace = t0.elapsed().as_secs_f64() * 1000.0;

        let prev_cam = prev_cam9zc.unwrap_or(cur_cam);
        let t1 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9zc gather") });
        gather9.encode(
            &device, &queue, &mut enc, &split_buf9, &aov_buf9, &feats9_buf, &history9zc.prev_out_dl,
            &history9zc.prev_aov, cur_cam, prev_cam, history9zc.has_prev, history9zc.w, history9zc.h,
            V7_DEPTH_TOL, V7_NORMAL_THRESH, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t_gather = t1.elapsed().as_secs_f64() * 1000.0;

        // pack: GPU-side f32->fp16, straight into the bridged MTLBuffer
        // `live9_bridge`'s own forward reads — no CPU touches this tensor.
        let t2 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9zc pack") });
        packer.encode(&device, &mut enc, &feats9_buf, &feature_buf_u32, n_in);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t_pack = t2.elapsed().as_secs_f64() * 1000.0;

        // forward: zero-copy — no input write, Metal's same-queue commit-
        // order guarantee (see `UnetLive::from_wgpu_queue`'s own doc) is why
        // the pack's `queue.submit` above (already completed, we polled) is
        // enough ordering without an explicit fence.
        let t3 = Instant::now();
        let (gpu_only_ms, out) = live9_bridge.forward_gpu_bridged().expect("forward_gpu_bridged");
        let t_forward = t3.elapsed().as_secs_f64() * 1000.0;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t_encode_commit = live9_bridge.last_encode_commit_ms();
        let t_wait = live9_bridge.last_wait_ms();
        let t_readout = live9_bridge.last_readout_ms();

        let out_dl_padded: Vec<[f32; 4]> = (0..n).map(|p| [out[p * 3], out[p * 3 + 1], out[p * 3 + 2], 0.0]).collect();
        let out_dl_buf9zc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wall v9zc real out_dl (for next-frame history)"),
            contents: bytemuck::cast_slice(&out_dl_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wall v9zc swap") });
        history9zc.swap(&mut enc, &out_dl_buf9zc, &aov_buf9, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        prev_cam9zc = Some(cur_cam);

        if k >= n_warmup {
            v9zc_walls.push(wall_ms);
            v9zc_trace.push(t_trace);
            v9zc_gather.push(t_gather);
            v9zc_pack.push(t_pack);
            v9zc_forward.push(t_forward);
            v9zc_forward_gpu_only.push(gpu_only_ms);
            v9zc_encode_commit.push(t_encode_commit);
            v9zc_wait.push(t_wait);
            v9zc_readout.push(t_readout);
        }
    }
    let v9zc_median = median(&mut v9zc_walls.clone());
    let v9zc_p95 = p95(&mut v9zc_walls.clone());
    println!(
        "[wall] v9  (41-in U-Net, ZERO-COPY BRIDGE: GPU gather + GPU pack + forward_gpu_bridged): median={v9zc_median:.3}ms p95={v9zc_p95:.3}ms n={}",
        v9zc_walls.len()
    );
    println!(
        "[wall] v9zc decomposition medians: trace={:.3}ms gather={:.3}ms pack={:.3}ms forward(wall)={:.3}ms forward(gpu-only)={:.3}ms",
        median(&mut v9zc_trace.clone()), median(&mut v9zc_gather.clone()), median(&mut v9zc_pack.clone()),
        median(&mut v9zc_forward.clone()), median(&mut v9zc_forward_gpu_only.clone())
    );
    println!(
        "[wall] v9zc decomposition p95s:     trace={:.3}ms gather={:.3}ms pack={:.3}ms forward(wall)={:.3}ms forward(gpu-only)={:.3}ms",
        p95(&mut v9zc_trace.clone()), p95(&mut v9zc_gather.clone()), p95(&mut v9zc_pack.clone()),
        p95(&mut v9zc_forward.clone()), p95(&mut v9zc_forward_gpu_only.clone())
    );
    println!(
        "[wall] v9zc forward(wall) sub-split medians: encode+commit={:.3}ms wait={:.3}ms readout={:.3}ms (sum {:.3}ms vs forward(wall) {:.3}ms)",
        median(&mut v9zc_encode_commit.clone()), median(&mut v9zc_wait.clone()), median(&mut v9zc_readout.clone()),
        median(&mut v9zc_encode_commit.clone()) + median(&mut v9zc_wait.clone()) + median(&mut v9zc_readout.clone()),
        median(&mut v9zc_forward.clone())
    );

    println!("\n[wall] ===== SUMMARY (640x480 unless GAIA_WALL_W/H overridden), {n_frames} frames after {n_warmup} warmup =====");
    println!("  v7            full wall (production GPU-native gather+forward)          median {v7_median:>8.3} ms  p95 {v7_p95:>8.3} ms");
    println!("  v9-cpu-gather full wall (CPU-built gather + GPU forward)                 median {v9_median:>8.3} ms  p95 {v9_p95:>8.3} ms");
    println!("  v9-gpu-gather full wall (GPU-native gather + readback + GPU forward)     median {v9g_median:>8.3} ms  p95 {v9g_p95:>8.3} ms");
    println!("  v9-gpu-gather+fast-forward (GPU gather + readback + persistent-exec fwd) median {v9gf_median:>8.3} ms  p95 {v9gf_p95:>8.3} ms");
    println!("  v9-zero-copy-bridge (GPU gather + GPU pack + forward_gpu_bridged)        median {v9zc_median:>8.3} ms  p95 {v9zc_p95:>8.3} ms");
    println!("  v9 GPU-only forward alone (no trace/gather/readback)                     median {v9_gpu_median:>8.3} ms  p95 {v9_gpu_p95:>8.3} ms");
    println!("  60fps budget: 16.67 ms");
}
