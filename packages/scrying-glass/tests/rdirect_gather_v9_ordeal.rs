//! V9-WIRE ATOM (2026-07-21) — GPU-NATIVE GATHER PARITY, the v9 body's
//! 41-in layout (`FeatureGatherV9`/`gather_v9`, `rdirect_gather.rs` /
//! `rdirect_gather_split.wgsl`) vs the CPU reference feature build
//! (`rdirect_train_v9c.rs::build_input_img`/`reproject_prev`, also
//! duplicated as `rdirect_frame_wall.rs::build_input_img_unet`/
//! `reproject_prev_unet` — SAME house convention: the source fns are
//! private to their own binaries, so a byte-for-byte port lives here too).
//!
//! Same shape as the house `n0b_gather_and_shared_forward_match_cpu` GATE A
//! (`tests/rdirect_gather_ordeals.rs`, tolerance 1e-4 abs — GPU vs CPU f32
//! div+ln is last-ULP class; a breach means a real wiring bug, not float
//! drift) but exercises BOTH v9-specific branches n0b's 23-in/no-history
//! case never touches:
//!   - the recurrent history slots (idx 35-38, `hist_features_split`'s own
//!     reprojection/validity gate — same math Stage 2's `gather_hist_split`
//!     already carries, re-used byte-for-byte by `gather_v9`)
//!   - the REAL motion-vector slots (idx 39-40) — v9-only, zero in v7.
//!
//! Two-frame PANNING sequence (front pose, then a small +1.5° yaw step) so
//! frame 1 exercises a genuinely non-degenerate reprojection (not the
//! trivial same-pixel identity a static camera would give every valid
//! pixel). History is seeded with a DETERMINISTIC SYNTHETIC "previous net
//! output" (not a real net forward — this test isolates the GATHER, not
//! the forward kernel, matching this atom's own scope) fed identically to
//! both the GPU (`HistoryBuffers::swap`) and CPU (`reproject_prev`'s own
//! `prev_out_dl` parameter) sides, so the recurrent reprojection math is
//! exercised under a controlled, known-identical input on both sides.
//!
//! Skips (does not fail) when no Metal/GPU device is present — same house
//! convention as `rdirect_gather_ordeals.rs`.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use glam::{Vec2, Vec3};
use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::integrator::{
    headless_device, split_aov, trace_headless_aov, Integrator, IntegratorParams, IntegratorUniform,
};
use scrying_glass::rdirect::{
    bilinear_vec3, hist_features_split, pixel_features_split, sky_history_reject, CamPose,
    HIST_FEATURES_SPLIT,
};
use scrying_glass::rdirect_gather::{FeatureGatherHistSplit, FeatureGatherV9, HistoryBuffers};
use scrying_glass::rdirect_unet::MOTION_VECTOR_CHANNELS;
use scrying_glass::scene::{Camera, RenderScene};

const DEPTH_TOL: f32 = 0.05; // == main.rs's own V7_DEPTH_TOL (v9 gather reuses the v7 house values)
const NORMAL_THRESH: f32 = 0.85; // == main.rs's own V7_NORMAL_THRESH
const FEATURES_V9: usize = HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS; // 41

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose {
        eye: cam.eye,
        right,
        up,
        forward,
        half_tan: (cam.fov_y_radians * 0.5).tan(),
        aspect: w as f32 / h.max(1) as f32,
    }
}

fn read_f32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("v9 gather ordeal readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
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

/// Mirrors `accum_ed`'s layout (2 vec4 cells/px = 8 f32/px: E sum+count,
/// D sum+count) — same decode `v7_present_parity_probe.rs::split_from_cells`
/// uses, duplicated here per the same house convention (private per binary).
fn split_from_cells(cells: &[f32], n: usize) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut e = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        let ec = &cells[i * 8..i * 8 + 4];
        let dc = &cells[i * 8 + 4..i * 8 + 8];
        let ecount = ec[3].max(1.0);
        let dcount = dc[3].max(1.0);
        e.push(Vec3::new(ec[0] / ecount, ec[1] / ecount, ec[2] / ecount));
        d.push(Vec3::new(dc[0] / dcount, dc[1] / dcount, dc[2] / dcount));
    }
    (e, d)
}

/// Byte-for-byte port of `rdirect_train_v9c.rs::reproject_prev` /
/// `rdirect_frame_wall.rs::reproject_prev_unet` — both private to their own
/// binaries, so a third copy here is the established house convention, not
/// a new pattern. Returns (prev_dl, valid, mv) — mv is ONLY nonzero on the
/// `ok` (valid=1) branch, matching the CPU reference's own contract exactly
/// (this is the exact behavior `gather_v9` mirrors on the GPU side).
#[allow(clippy::too_many_arguments)]
fn reproject_prev_ref(
    cur_cam: &CamPose,
    cur_depth: f32,
    cur_normal: Vec3,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    prev_cam: &CamPose,
    prev_out_dl: &[Vec3],
    prev_depth: &[f32],
    prev_normal: &[Vec3],
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
                let s = bilinear_vec3(prev_out_dl, fx, fy, pw, ph);
                ([s.x, s.y, s.z], 1.0, mv)
            } else {
                ([0.0; 3], 0.0, [0.0; 2])
            }
        }
    }
}

#[test]
fn v9_gather_matches_cpu_reference() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v9-gather] SKIP — no GPU adapter");
        return;
    };

    let params = naruko_params();
    let world_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);

    let (low_w, low_h, target_w, target_h) = (48u32, 32u32, 96u32, 64u32);
    let n = (target_w * target_h) as usize;
    let low_n = (low_w * low_h) as usize;
    let sky_reject = sky_history_reject();

    let front = law_poses(&params).into_iter().find(|(n, _)| *n == "front").expect("front pose").1;
    let mut panned = front;
    panned.yaw += 1.5f32.to_radians();
    let cams: Vec<Camera> = vec![front, panned];

    let gather9 = FeatureGatherV9::new(&device);
    let mut history = HistoryBuffers::new(&device, target_w, target_h);

    let mut gpu_feats_per_frame: Vec<Vec<f32>> = Vec::new();
    // Per-step CPU-side owned buffers: (low_e, low_d, albedo, normal, depth, cam).
    type OwnedStep = (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<f32>, CamPose);
    let mut owned: Vec<OwnedStep> = Vec::new();
    // The SAME synthetic "previous net output" fed to both GPU history and
    // the CPU reference's own `prev_out_dl` — deterministic, not a real net.
    let mut synth_history: Vec<(Vec<Vec3>, CamPose)> = Vec::new();

    for (idx, cam) in cams.iter().enumerate() {
        let np = IntegratorParams { spp: 1, seed: 0x9c41 + (idx as u32 * 197), ..IntegratorParams::default() };
        let accum_ed = integrator.make_split_buffer(&device, low_w, low_h);
        let compute_bg = integrator.compute_bind_group(&device, &integrator.make_accum(&device, low_w, low_h));
        let split_bg = integrator.split_bind_group(&device, &accum_ed);
        let uniform = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, integrator.node_count,
            integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("v9-gather trace split") });
        integrator.dispatch_split(&queue, &mut enc, &uniform, &compute_bg, &split_bg, low_w, low_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let aov_raw = trace_headless_aov(
            &device, &queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        );
        let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("v9-gather aov(native)"),
            contents: bytemuck::cast_slice(&aov_raw),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let cur_cam = cam_pose(cam, target_w, target_h);
        let prev_cam = history.prev_cam.unwrap_or(cur_cam);

        // ── GPU: gather_v9 (41-in) ──
        let feat_bytes = FeatureGatherV9::feature_bytes(n);
        let feats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v9-gather feats41"),
            size: feat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("v9-gather dispatch") });
        gather9.encode(
            &device, &queue, &mut enc, &accum_ed, &aov_buf, &feats_buf, &history.prev_out_dl,
            &history.prev_aov, cur_cam, prev_cam, history.has_prev, history.w, history.h,
            DEPTH_TOL, NORMAL_THRESH, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let gpu_feats = read_f32(&device, &queue, &feats_buf, feat_bytes);
        assert_eq!(gpu_feats.len(), n * FEATURES_V9, "gpu feature tensor length");
        gpu_feats_per_frame.push(gpu_feats);

        // CPU-side low_e/low_d re-derived from the SAME dispatched accum_ed
        // bytes (bit-identical RNG samples to what the GPU gather itself
        // just read) — same technique `v7_present_parity_probe.rs` uses.
        let low_e_d = read_f32(&device, &queue, &accum_ed, (low_n as u64) * 32);
        let (low_e, low_d) = split_from_cells(&low_e_d, low_n);
        let (hi_albedo, hi_normal, hi_depth) = split_aov(&aov_raw);
        owned.push((low_e, low_d, hi_albedo, hi_normal, hi_depth, cur_cam));

        // Deterministic synthetic "previous net output" (demod-log space,
        // NOT a real net forward — isolates the gather from the forward
        // kernel, this atom's own scope) — SAME values fed to the GPU
        // history buffer (via `swap`) and stashed for the CPU reference's
        // own `prev_out_dl` next iteration.
        let synth: Vec<[f32; 4]> = (0..n)
            .map(|p| {
                let x = (p as f32 * 0.017).sin() * 0.4 + 0.4;
                let y = (p as f32 * 0.031 + 1.7).sin() * 0.3 + 0.3;
                let z = (p as f32 * 0.043 + 3.1).sin() * 0.35 + 0.35;
                [x.max(0.0), y.max(0.0), z.max(0.0), 0.0]
            })
            .collect();
        let synth_vec3: Vec<Vec3> = synth.iter().map(|v| Vec3::new(v[0], v[1], v[2])).collect();
        let synth_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("v9-gather synthetic prev out_dl"),
            contents: bytemuck::cast_slice(&synth),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("v9-gather swap") });
        history.swap(&mut enc, &synth_buf, &aov_buf, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        synth_history.push((synth_vec3, cur_cam));
    }

    // ── CPU reference: same 41-wide feature build, per step ──
    let mut max_all = 0f32;
    let mut max_base = 0f32; // idx 0-34 (split base, motion slot dead-zero)
    let mut max_hist = 0f32; // idx 35-38 (reprojected history)
    let mut max_mv = 0f32; // idx 39-40 (real motion vector)
    let mut nonzero_valid_px = 0u32;
    for (idx, (low_e, low_d, albedo, normal, depth, cur_cam)) in owned.iter().enumerate() {
        let mut cpu_feats = Vec::with_capacity(n * FEATURES_V9);
        for ty in 0..target_h {
            for tx in 0..target_w {
                let px = (ty * target_w + tx) as usize;
                let base = pixel_features_split(
                    low_e, low_d, low_w, low_h, target_w, target_h, tx, ty, albedo[px], normal[px],
                    depth[px], Vec2::ZERO,
                );
                let (prev_dl, valid, mv) = if idx == 0 {
                    ([0.0f32; 3], 0.0f32, [0.0f32; 2])
                } else {
                    let (prev_out, prev_cam) = &synth_history[idx - 1];
                    let (_, _, _, prev_normal, prev_depth, _) = &owned[idx - 1];
                    reproject_prev_ref(
                        cur_cam, depth[px], normal[px], tx, ty, target_w, target_h, prev_cam, prev_out,
                        prev_depth, prev_normal, target_w, target_h, sky_reject,
                    )
                };
                if valid > 0.5 {
                    nonzero_valid_px += 1;
                }
                let feat39 = hist_features_split(&base, prev_dl, valid);
                let mut feat41 = [0.0f32; FEATURES_V9];
                feat41[..HIST_FEATURES_SPLIT].copy_from_slice(&feat39);
                feat41[HIST_FEATURES_SPLIT] = mv[0];
                feat41[HIST_FEATURES_SPLIT + 1] = mv[1];
                cpu_feats.extend_from_slice(&feat41);
            }
        }

        let gpu = &gpu_feats_per_frame[idx];
        assert_eq!(gpu.len(), cpu_feats.len(), "frame {idx} feature tensor length");
        let mut frame_max = 0f32;
        for p in 0..n {
            for k in 0..FEATURES_V9 {
                let d = (gpu[p * FEATURES_V9 + k] - cpu_feats[p * FEATURES_V9 + k]).abs();
                frame_max = frame_max.max(d);
                max_all = max_all.max(d);
                if k < HIST_FEATURES_SPLIT - 4 {
                    max_base = max_base.max(d);
                } else if k < HIST_FEATURES_SPLIT {
                    max_hist = max_hist.max(d);
                } else {
                    max_mv = max_mv.max(d);
                }
            }
        }
        eprintln!("[v9-gather] frame {idx} N={n} x {FEATURES_V9} feat · max abs {frame_max:.3e}");
    }

    eprintln!(
        "[v9-gather] OVERALL max abs {max_all:.3e} (base[0-34] {max_base:.3e} · hist[35-38] {max_hist:.3e} · mv[39-40] {max_mv:.3e}) valid_px_frame1={nonzero_valid_px}"
    );
    // Sanity: the pan step must actually exercise SOME valid reprojected
    // history (else this test would trivially pass on an all-zero history
    // branch and never touch idx 35-40 at all).
    assert!(nonzero_valid_px > 0, "pan sequence produced zero valid reprojected pixels — history/mv branch untested");
    // Same house tolerance as `n0b_gather_and_shared_forward_match_cpu`
    // GATE A (`rdirect_gather_ordeals.rs`): GPU vs CPU f32 div+ln+bilinear
    // is last-ULP class; a breach at this magnitude is a real wiring bug
    // (wrong tap/packing/offset/reprojection sign), not float drift.
    assert!(max_all < 1.0e-4, "gather_v9 vs CPU reference abs {max_all:.3e} >= 1e-4");
}

/// Cross-check: `gather_v9`'s idx 0-38 must be byte-identical (same math, no
/// new drift) to Stage 2's already-proven `gather_hist_split` (39-in) for
/// the SAME inputs — the v9 entry is additive (new mv slots only), never a
/// reimplementation of the shared history math it copies.
#[test]
fn v9_gather_hist_slots_match_stage2_gather() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v9-gather] SKIP — no GPU adapter");
        return;
    };

    let params = naruko_params();
    let world_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);

    let (low_w, low_h, target_w, target_h) = (48u32, 32u32, 96u32, 64u32);
    let n = (target_w * target_h) as usize;

    let front = law_poses(&params).into_iter().find(|(n, _)| *n == "front").expect("front pose").1;
    let mut panned = front;
    panned.yaw += 1.5f32.to_radians();

    let np = IntegratorParams { spp: 1, seed: 0x1357, ..IntegratorParams::default() };
    let accum_ed = integrator.make_split_buffer(&device, low_w, low_h);
    let compute_bg = integrator.compute_bind_group(&device, &integrator.make_accum(&device, low_w, low_h));
    let split_bg = integrator.split_bind_group(&device, &accum_ed);
    let uniform = IntegratorUniform::build(
        &panned, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, integrator.node_count,
        integrator.tri_count, 0, &np, None,
    );
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stage2-vs-v9 trace") });
    integrator.dispatch_split(&queue, &mut enc, &uniform, &compute_bg, &split_bg, low_w, low_h);
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let aov_raw = trace_headless_aov(
        &device, &queue, &bvh, &panned, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
    );
    let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stage2-vs-v9 aov"),
        contents: bytemuck::cast_slice(&aov_raw),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let cur_cam = cam_pose(&panned, target_w, target_h);
    let prev_cam = cam_pose(&front, target_w, target_h);
    // A non-trivial (all-zero) prev history buffer so the reprojection
    // branch is actually exercised, identical bytes fed to BOTH gathers.
    let prev_out: Vec<[f32; 4]> = (0..n)
        .map(|p| [(p as f32 * 0.013).sin().abs(), (p as f32 * 0.021).cos().abs(), 0.2, 0.0])
        .collect();
    let prev_out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stage2-vs-v9 prev out_dl"),
        contents: bytemuck::cast_slice(&prev_out),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let prev_aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stage2-vs-v9 prev aov"),
        contents: bytemuck::cast_slice(&aov_raw), // reuse this frame's own AOV as a stand-in "previous" AOV
        usage: wgpu::BufferUsages::STORAGE,
    });

    let hist39 = FeatureGatherHistSplit::new(&device);
    let feat39_bytes = FeatureGatherHistSplit::feature_bytes(n);
    let feats39_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stage2 feats39"),
        size: feat39_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stage2 gather") });
    hist39.encode(
        &device, &queue, &mut enc, &accum_ed, &aov_buf, &feats39_buf, &prev_out_buf, &prev_aov_buf,
        cur_cam, prev_cam, true, target_w, target_h, DEPTH_TOL, NORMAL_THRESH, low_w, low_h, target_w,
        target_h,
    );
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let feats39 = read_f32(&device, &queue, &feats39_buf, feat39_bytes);

    let gather9 = FeatureGatherV9::new(&device);
    let feat41_bytes = FeatureGatherV9::feature_bytes(n);
    let feats41_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("v9 feats41"),
        size: feat41_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("v9 gather") });
    gather9.encode(
        &device, &queue, &mut enc, &accum_ed, &aov_buf, &feats41_buf, &prev_out_buf, &prev_aov_buf,
        cur_cam, prev_cam, true, target_w, target_h, DEPTH_TOL, NORMAL_THRESH, low_w, low_h, target_w,
        target_h,
    );
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let feats41 = read_f32(&device, &queue, &feats41_buf, feat41_bytes);

    let mut max_diff = 0f32;
    for p in 0..n {
        for k in 0..HIST_FEATURES_SPLIT {
            let a = feats39[p * HIST_FEATURES_SPLIT + k];
            let b = feats41[p * FEATURES_V9 + k];
            max_diff = max_diff.max((a - b).abs());
        }
    }
    eprintln!("[v9-gather] idx 0-38 vs Stage2 gather_hist_split: max abs {max_diff:.3e}");
    // Exact same math, same inputs — this must be bit-identical modulo FMA
    // reassociation noise (both entries run on the SAME GPU/driver, same
    // op sequence up to idx 38) — an exact-match bar, not a derived one.
    assert_eq!(max_diff, 0.0, "gather_v9 idx 0-38 diverged from gather_hist_split (same math, same inputs)");
}
