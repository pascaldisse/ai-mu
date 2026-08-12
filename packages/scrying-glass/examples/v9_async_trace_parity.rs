//! V9-WIRE LEVER 1 — NUMERIC PARITY (task mandate, 2026-07-26): does removing
//! the two mid-trace `device.poll(wait_indefinitely())` calls between the v9
//! trace stage's sub-dispatches (`resolve_frame_v9`/`_async`'s new
//! `GAIA_NATIVE_ASYNC_TRACE` gate) change the trace stage's OWN output?
//!
//! Deliberately does NOT drive a live ticking world/server (that confounds
//! any multi-process screenshot diff with independent wall-clock animation
//! phase — verified empirically: two live `scrying-glass` offscreen servers
//! launched together and captured within ~1-3s of each other still drift by
//! 10-20 world-tick frames apiece, purely from OS scheduling jitter, making
//! pixel-for-pixel PNG comparison meaningless for this specific question).
//! Instead this harness reproduces JUST the trace stage `resolve_frame_v9`
//! runs (`integrator.dispatch` composite + `dispatch_split` + `dispatch_aov`,
//! same calls, same bind groups) for ONE fixed camera pose/uniform, twice, on
//! two independent buffer sets — "A" with the OLD unconditional polls between
//! submits, "B" with them removed (mirroring the new `async_trace` gate) —
//! and reads back `net_accum_ed` (what `FeatureGatherV9`/gather actually
//! consumes) + `net_aov` to the CPU for a bit-exact compare. Since all three
//! dispatches ride the SAME wgpu queue (FIFO command-buffer ordering is a
//! wgpu/Metal guarantee), removing a CPU-side `poll()` between two `submit()`
//! calls cannot reorder or alter what the GPU executes — it only changes when
//! the CPU host blocks waiting for it. This harness measures the actual
//! buffer bytes, not just recites that argument.
//!
//! Net/gather/compose/demod stages are UNTOUCHED by `GAIA_NATIVE_ASYNC_TRACE`
//! (pure functions of the trace stage's own output buffers) — if the trace
//! output is bit-identical, so is everything downstream; this harness does
//! not need to re-run them to make that claim.

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::integrator::{headless_device, Integrator, IntegratorParams, IntegratorUniform};
use scrying_glass::scene::RenderScene;

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity copy") });
    enc.copy_buffer_to_buffer(buf, 0, &rb, 0, size);
    let (tx, rx) = std::sync::mpsc::channel();
    enc.map_buffer_on_submit(&rb, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback chan").expect("map readback");
    let mapped = rb.get_mapped_range(..).expect("mapped readback");
    let out = mapped.to_vec();
    drop(mapped);
    rb.unmap();
    out
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v9-async-trace-parity] SKIP — no GPU adapter");
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

    // Mirrors the live rig's own low_w/low_h/target_w/target_h scale (v9's
    // resolve_frame_v9 uses e.g. 640x480 canvas -> low-res trace grid; here a
    // smaller grid is fine, this harness only checks BUFFER BYTES agree, not
    // real-time performance).
    let (low_w, low_h, target_w, target_h) = (64u32, 40u32, 128u32, 80u32);
    let ip = IntegratorParams { spp: 1, seed: 0x7abc + 5, ..IntegratorParams::default() };
    let uni_low = IntegratorUniform::build(
        &front, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
        0, 0, 0, &ip, None,
    );
    let uni_target = IntegratorUniform::build(
        &front, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        0, 0, 0, &ip, None,
    );

    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);
    // uni_low/uni_target need the REAL node/tri counts (IntegratorUniform::build's
    // node_count/tri_count args) -- rebuild with the integrator's own counts.
    let uni_low = IntegratorUniform::build(
        &front, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
        integrator.node_count, integrator.tri_count, 0, &ip, None,
    );
    let uni_target = IntegratorUniform::build(
        &front, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        integrator.node_count, integrator.tri_count, 0, &ip, None,
    );
    let _ = uni_target; // silence unused-before-rebuild warning path clarity

    let n_low = (low_w as u64) * (low_h as u64);
    let n_tgt = (target_w as u64) * (target_h as u64);
    let accum_cell: u64 = 16; // vec4<f32>

    // Two independent buffer sets: A (polled, mirrors the OLD `resolve_frame_v9`
    // unconditionally), B (unpolled between trace1/trace2, mirrors the NEW
    // `async_trace=true` path). `net_accum` (composite) is allocated per the
    // real rig's own shape too (bound but its CONTENTS are irrelevant to this
    // harness's question — `dispatch_split`/`dispatch_aov` need SOME buffer at
    // that binding for bind-group validity, exactly as `v9_trace_reuse`'s doc
    // in main.rs explains).
    let make_set = || {
        let net_accum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parity net_accum"),
            size: n_low.max(1) * accum_cell,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let accum_ed = integrator.make_split_buffer(&device, low_w, low_h);
        let aov = integrator.make_aov_buffer(&device, target_w, target_h);
        (net_accum, accum_ed, aov)
    };
    let (net_accum_a, accum_ed_a, aov_a) = make_set();
    let (net_accum_b, accum_ed_b, aov_b) = make_set();

    let run_trace = |net_accum: &wgpu::Buffer, accum_ed: &wgpu::Buffer, aov: &wgpu::Buffer, poll_between: bool| {
        let accum_bg = integrator.compute_bind_group(&device, net_accum);
        let aov_bg = integrator.aov_bind_group(&device, aov);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity trace1") });
        enc.clear_buffer(net_accum, 0, None);
        integrator.dispatch(&queue, &mut enc, &uni_low, &accum_bg, low_w, low_h);
        enc.clear_buffer(accum_ed, 0, None);
        let split_bg = integrator.split_bind_group(&device, accum_ed);
        integrator.dispatch_split(&queue, &mut enc, &uni_low, &accum_bg, &split_bg, low_w, low_h);
        queue.submit(Some(enc.finish()));
        if poll_between {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity trace2 aov") });
        integrator.dispatch_aov(&queue, &mut enc, &uni_target, &accum_bg, &aov_bg, target_w, target_h);
        queue.submit(Some(enc.finish()));
        if poll_between {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
    };

    // A: OLD behavior (poll between every submit — matches `async_trace=false`).
    run_trace(&net_accum_a, &accum_ed_a, &aov_a, true);
    // B: NEW behavior under `GAIA_NATIVE_ASYNC_TRACE=1` (no poll between the
    // trace1/trace2 submits — the ONE poll that stays in the real code, before
    // gather reads these buffers, happens in `readback` below instead, which
    // is exactly as load-bearing/present in both paths).
    run_trace(&net_accum_b, &accum_ed_b, &aov_b, false);

    let ed_a = readback(&device, &queue, &accum_ed_a, n_low * accum_cell * 2);
    let ed_b = readback(&device, &queue, &accum_ed_b, n_low * accum_cell * 2);
    let aov_bytes_a = readback(&device, &queue, &aov_a, n_tgt * accum_cell * 2);
    let aov_bytes_b = readback(&device, &queue, &aov_b, n_tgt * accum_cell * 2);

    assert_eq!(ed_a.len(), ed_b.len());
    assert_eq!(aov_bytes_a.len(), aov_bytes_b.len());

    let ed_diff = ed_a.iter().zip(ed_b.iter()).filter(|(x, y)| x != y).count();
    let aov_diff = aov_bytes_a.iter().zip(aov_bytes_b.iter()).filter(|(x, y)| x != y).count();

    // Also diff as f32 for a human-readable max-abs-diff (bytes matching is
    // the real bar; this is just legible).
    let ed_a_f: &[f32] = bytemuck::cast_slice(&ed_a);
    let ed_b_f: &[f32] = bytemuck::cast_slice(&ed_b);
    let mut max_abs = 0f32;
    for (x, y) in ed_a_f.iter().zip(ed_b_f.iter()) {
        max_abs = max_abs.max((x - y).abs());
    }

    println!(
        "[v9-async-trace-parity] net_accum_ed bytes_total={} bytes_diff={} (max_abs_diff_f32={max_abs:.6e}) \
         net_aov bytes_total={} bytes_diff={}",
        ed_a.len(), ed_diff, aov_bytes_a.len(), aov_diff,
    );
    if ed_diff == 0 && aov_diff == 0 {
        println!("[v9-async-trace-parity] PASS — trace stage output is BIT-IDENTICAL with/without the mid-trace polls");
    } else {
        println!("[v9-async-trace-parity] FAIL — trace stage output DIFFERS with/without the mid-trace polls");
        std::process::exit(1);
    }
}
