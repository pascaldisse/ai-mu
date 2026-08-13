//! V9-WIRE GPU FUSE (task mandate, 2026-07-25) — PROBE-EQUALS-LIVE PARITY.
//! `rdirect_v9_compose::V9ComposePass` (the fused WGSL compute compose,
//! `GAIA_V9_COMPOSE=gpu`) vs `compose_cpu_reference` (the CPU readback+loop
//! path, `GAIA_V9_COMPOSE=cpu` — the IRON default `main.rs`'s
//! `resolve_frame_v9` still calls when the env var is unset). SAME synthetic
//! frame fed to both, byte-for-byte comparable — the "renders the same
//! frame both ways" law, isolated to the compose stage (the stage that
//! actually moved, not re-tracing/re-inferring the whole pipeline, which
//! would only add non-reproducible net/RNG noise to the comparison).
//!
//! The synthetic frame deliberately exercises every branch the CPU
//! reference and the WGSL kernel both carry:
//!   - hit px (depth>0, net's raw output must pass through unchanged)
//!   - no-hit px (depth<=0, bilinear evidence composite + demod-log)
//!   - near-zero albedo (the alb_sq<=NO_HIT_SQ divisor==1 branch)
//!   - zero-count E/D cells (the max(count,1) floor)
//!   - `hitgate` on AND off (off = pure passthrough, both sides)
//!
//! Skips (does not fail) when no Metal/GPU device is present — same house
//! convention as `tests/rdirect_gather_v9_ordeal.rs`.

#![cfg(target_os = "macos")]

use wgpu::util::DeviceExt;

use scrying_glass::integrator::headless_device;
use scrying_glass::rdirect_v9_compose::{compose_cpu_reference, V9ComposePass};

fn read_f32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("v9 compose parity readback"),
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
    rx.recv().expect("v9 compose parity readback channel").expect("v9 compose parity map readback");
    let mapped = readback.get_mapped_range(..).expect("v9 compose parity mapped readback");
    let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    v
}

/// Deterministic pseudo-random unit float in [0,1) — xorshift32, seeded per
/// call site so the synthetic frame is fully reproducible (no external
/// `rand` dependency, no wall-clock/RNG state).
fn hashf(seed: u32) -> f32 {
    let mut x = seed ^ 0x9E37_79B9;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x as f64 / u32::MAX as f64) as f32
}

fn run_case(hitgate: bool) {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v9-compose-parity] no GPU device available -- SKIP");
        return;
    };

    // low != target (4x) so the bilinear evidence resample is genuinely
    // exercised, not a degenerate 1:1 tap.
    let low_w = 32u32;
    let low_h = 24u32;
    let target_w = 128u32;
    let target_h = 96u32;
    let n = (target_w * target_h) as usize;
    let low_n = (low_w * low_h) as usize;

    // net_out: [n,3] synthetic raw net demod-log output (only load-bearing
    // on hit px / hitgate-off, but populated everywhere).
    let net_out: Vec<f32> = (0..n * 3).map(|i| hashf(i as u32 * 7 + 1) * 2.0).collect();

    // aov: [n,8] = 2 vec4/px: [albedo.xyz, depth], [normal.xyz, 0].
    let mut aov = vec![0f32; n * 8];
    for p in 0..n {
        let miss = p % 3 != 0; // 2/3 no-hit, 1/3 hit -- both branches well exercised
        let near_zero_albedo = p % 7 == 0; // exercises the alb_sq<=NO_HIT_SQ branch
        let (r, g, b) = if near_zero_albedo {
            (0.0, 0.0, 0.0)
        } else {
            (hashf(p as u32 * 3 + 11), hashf(p as u32 * 3 + 12), hashf(p as u32 * 3 + 13))
        };
        aov[p * 8] = r;
        aov[p * 8 + 1] = g;
        aov[p * 8 + 2] = b;
        aov[p * 8 + 3] = if miss { 0.0 } else { 1.0 + hashf(p as u32 * 5 + 21) * 10.0 };
        aov[p * 8 + 4] = hashf(p as u32 * 9 + 31) * 2.0 - 1.0;
        aov[p * 8 + 5] = hashf(p as u32 * 9 + 32) * 2.0 - 1.0;
        aov[p * 8 + 6] = hashf(p as u32 * 9 + 33) * 2.0 - 1.0;
        aov[p * 8 + 7] = 0.0;
    }

    // accum_ed: [low_n,8] = E vec4 (rgb,count) + D vec4 (rgb,count).
    let mut ed = vec![0f32; low_n * 8];
    for j in 0..low_n {
        let zero_count = j % 5 == 0; // exercises the max(count,1) floor on both E and D
        ed[j * 8] = hashf(j as u32 * 13 + 41) * 3.0;
        ed[j * 8 + 1] = hashf(j as u32 * 13 + 42) * 3.0;
        ed[j * 8 + 2] = hashf(j as u32 * 13 + 43) * 3.0;
        ed[j * 8 + 3] = if zero_count { 0.0 } else { 1.0 + hashf(j as u32 * 13 + 44) * 8.0 };
        ed[j * 8 + 4] = hashf(j as u32 * 17 + 51) * 3.0;
        ed[j * 8 + 5] = hashf(j as u32 * 17 + 52) * 3.0;
        ed[j * 8 + 6] = hashf(j as u32 * 17 + 53) * 3.0;
        ed[j * 8 + 7] = if zero_count { 0.0 } else { 1.0 + hashf(j as u32 * 17 + 54) * 8.0 };
    }

    // CPU reference -- the SAME oracle main.rs's GAIA_V9_COMPOSE=cpu default
    // path calls (relocated verbatim, not reimplemented -- see the module
    // doc on `compose_cpu_reference`).
    let cpu = compose_cpu_reference(&net_out, &aov, &ed, low_w, low_h, target_w, target_h, hitgate);

    // GPU fused pass (GAIA_V9_COMPOSE=gpu path).
    let net_out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parity net_out"),
        contents: bytemuck::cast_slice(&net_out),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parity aov"),
        contents: bytemuck::cast_slice(&aov),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ed_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parity accum_ed"),
        contents: bytemuck::cast_slice(&ed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gated_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity gated out"),
        size: (n as u64) * 12,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let compose = V9ComposePass::new(&device);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("v9 compose parity") });
    compose.encode(
        &device, &queue, &mut enc, &net_out_buf, &aov_buf, &ed_buf, &gated_buf, n as u32, low_w, low_h, target_w,
        target_h, hitgate,
    );
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let gpu = read_f32(&device, &queue, &gated_buf, (n as u64) * 12);

    assert_eq!(cpu.len(), gpu.len(), "cpu/gpu output length mismatch");
    let mut max_diff = [0f32; 3];
    let mut max_diff_overall = 0f32;
    for p in 0..n {
        for c in 0..3 {
            let a = cpu[p * 3 + c];
            let b = gpu[p * 3 + c];
            let d = (a - b).abs();
            max_diff[c] = max_diff[c].max(d);
            max_diff_overall = max_diff_overall.max(d);
        }
    }
    eprintln!(
        "[v9-compose-parity] hitgate={hitgate} n={n} low={low_w}x{low_h} target={target_w}x{target_h} \
         max_abs_diff r={:.3e} g={:.3e} b={:.3e} overall={:.3e}",
        max_diff[0], max_diff[1], max_diff[2], max_diff_overall
    );
    // LAW (task mandate): <=1e-5 linear, or justify a precision delta
    // honestly. Both sides run plain f32 storage/arithmetic (no f16 in this
    // stage), so the bar is the tight one -- a breach means a real
    // divergence, not a storage-precision tradeoff.
    const TOL: f32 = 1.0e-5;
    assert!(
        max_diff_overall <= TOL,
        "v9 compose GPU vs CPU diverged beyond {TOL:e}: max_abs_diff={max_diff_overall:e} (r={:e} g={:e} b={:e})",
        max_diff[0],
        max_diff[1],
        max_diff[2]
    );
}

#[test]
fn v9_compose_gpu_matches_cpu_hitgate_on() {
    run_case(true);
}

#[test]
fn v9_compose_gpu_matches_cpu_hitgate_off() {
    run_case(false);
}
