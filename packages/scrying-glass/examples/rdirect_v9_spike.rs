//! v9-body STAGE 1 PERF SPIKE — HANDOFF.md → v9-body lane, "Stage 1 —
//! shape-parametric multi-scale body + perf spike". Measures forward WALL
//! (host-side, `Instant`-timed, `runWithMTLCommandQueue` blocking round-
//! trip — the same honest bring-up class of number N0.a's own spike used
//! before its own S1/S2 GPU-only optimizations) of `rdirect_unet::UnetLive`
//! at 2-3 width configs, UNTRAINED (He-init) weights, OFFSCREEN (own system
//! Metal device — no window, per CLAUDE.md → WINDOW BAN).
//!
//! Budget: net ≤~9ms = 16.67 − rays ~7.6
//! [source: docs/perf/2026-07-18-neural-live-n0.md].
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_v9_spike
//!      2>&1 | tee scratch/v9-spike.log

#[cfg(target_os = "macos")]
fn main() {
    use scrying_glass::rdirect_unet::{UnetConfig, UnetLive};
    use std::time::Instant;

    const SEED: u64 = 0x00d1_5eed;
    const WARMUP: usize = 8;
    const SAMPLES: usize = 40;
    const BUDGET_MS: f64 = 9.07; // 16.67 - 7.6, docs/perf/2026-07-18-neural-live-n0.md

    let render_w = 640usize;
    let render_h = 480usize;

    // 3 width configs, all n_scales=3 (full/2/4), M1-sized — NOT DLSS
    // widths (dlss4-bringup/REPORT.md: one DLSS-documented-width block
    // alone is 73-83ms clean, 8-9x the WHOLE budget).
    let configs: Vec<(&str, UnetConfig)> = vec![
        (
            "A-tiny  widths=[12,20,32]",
            UnetConfig {
                widths: vec![12, 20, 32],
                render_w,
                render_h,
                ..Default::default()
            },
        ),
        (
            "B-small widths=[16,28,44]",
            UnetConfig {
                widths: vec![16, 28, 44],
                render_w,
                render_h,
                ..Default::default()
            },
        ),
        (
            "C-med   widths=[24,40,64]",
            UnetConfig {
                widths: vec![24, 40, 64],
                render_w,
                render_h,
                ..Default::default()
            },
        ),
        // Stretch probe (C-med's GPU-only median left >2x budget headroom) —
        // finds the real ceiling instead of stopping at the first PASS.
        (
            "D-large widths=[40,72,112]",
            UnetConfig {
                widths: vec![40, 72, 112],
                render_w,
                render_h,
                ..Default::default()
            },
        ),
        // CAPACITY ROUND (task mandate): the 2 candidates being weighed to
        // close resid 0.0527->0.035 by widening the body — measured BEFORE
        // training (speed law: pick the WIDEST candidate whose gpu_median
        // stays <=~7.0ms, C-med [24,40,64]=4.61ms gpu is the reference).
        (
            "E-cap1  widths=[28,48,80]",
            UnetConfig {
                widths: vec![28, 48, 80],
                render_w,
                render_h,
                ..Default::default()
            },
        ),
        (
            "F-cap2  widths=[32,56,96]",
            UnetConfig {
                widths: vec![32, 56, 96],
                render_w,
                render_h,
                ..Default::default()
            },
        ),
    ];

    println!("v9-body STAGE 1 SPIKE — {}x{} render/output res, seed={:#x}", render_w, render_h, SEED);
    println!(
        "{:<28} {:>10} {:>12} {:>11} {:>9} {:>11} {:>9} {:>8}",
        "config", "params", "in_ch", "wall_med_ms", "wall_p95", "gpu_med_ms", "gpu_p95", "budget(gpu)"
    );

    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();

    for (name, cfg) in &configs {
        let net = match UnetLive::from_system(cfg, SEED) {
            Ok(n) => n,
            Err(e) => {
                println!("{:<28} BUILD FAILED: {}", name, e);
                continue;
            }
        };
        let n_in = cfg.input_elems();
        // Deterministic pseudo-random input (not the RNG the net's own
        // weights use — a separate stream, avoids any accidental identity
        // correlation) so every timed call touches real, non-degenerate
        // data all the way through the graph.
        let mut seed = SEED ^ 0x5151_5151_5151_5151;
        let input: Vec<f32> = (0..n_in)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((seed >> 33) as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect();

        // Warmup — first calls pay one-time graph-compile/dispatch cost.
        let mut last_out: Vec<f32> = Vec::new();
        for _ in 0..WARMUP {
            match net.forward_cpu_roundtrip(&input) {
                Ok(o) => last_out = o,
                Err(e) => {
                    println!("{:<28} FORWARD FAILED (warmup): {}", name, e);
                    continue;
                }
            }
        }

        let mut samples_ms: Vec<f64> = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let t0 = Instant::now();
            match net.forward_cpu_roundtrip(&input) {
                Ok(o) => {
                    last_out = o;
                    samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                Err(e) => println!("{:<28} FORWARD FAILED (sample): {}", name, e),
            }
        }
        if samples_ms.is_empty() {
            continue;
        }
        samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples_ms[samples_ms.len() / 2];
        let p95_idx = ((samples_ms.len() as f64 * 0.95) as usize).min(samples_ms.len() - 1);
        let p95 = samples_ms[p95_idx];
        let min = samples_ms[0];

        // GPU-ONLY split (rdirect_live.rs S3 pattern): compiled executable,
        // own command buffer, MTLCommandBuffer GPUStartTime/GPUEndTime — the
        // number that survives once the CPU-side per-frame encode (the wall
        // above) is optimized away (n0e precedent: wall 40ms -> GPU 6ms).
        for _ in 0..WARMUP {
            let _ = net.forward_gpu_ms(&input);
        }
        let mut gpu_ms: Vec<f64> = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            match net.forward_gpu_ms(&input) {
                Ok(ms) => gpu_ms.push(ms),
                Err(e) => println!("{:<28} GPU-SPLIT FAILED (sample): {}", name, e),
            }
        }
        let (gpu_median, gpu_p95) = if gpu_ms.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            gpu_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let gm = gpu_ms[gpu_ms.len() / 2];
            let gi = ((gpu_ms.len() as f64 * 0.95) as usize).min(gpu_ms.len() - 1);
            (gm, gpu_ms[gi])
        };

        // Sanity: output finite, non-degenerate (not silently NaN/all-zero
        // — an untrained He-init net is expected to output noise, not a
        // meaningful image, but it must be REAL noise, not a broken graph).
        let n_finite = last_out.iter().filter(|v| v.is_finite()).count();
        let n_nonzero = last_out.iter().filter(|&&v| v != 0.0).count();

        let _ = min;
        println!(
            "{:<28} {:>10} {:>12} {:>11.2} {:>9.2} {:>11.2} {:>9.2} {:>8}",
            name,
            cfg.approx_param_count(),
            cfg.in_channels,
            median,
            p95,
            gpu_median,
            gpu_p95,
            if gpu_median <= BUDGET_MS { "PASS" } else { "OVER" }
        );
        println!(
            "  sanity: {}/{} finite, {}/{} nonzero (untrained noise expected, not NaN/zero)",
            n_finite,
            last_out.len(),
            n_nonzero,
            last_out.len()
        );

        results.push((name.to_string(), gpu_median, gpu_p95, cfg.approx_param_count()));
    }

    println!();
    println!("(results/, table columns above: wall = runWithMTLCommandQueue CPU roundtrip; gpu = MTLCommandBuffer GPUStartTime/GPUEndTime on a compiled executable, own command buffer — budget verdict is evaluated on GPU-ONLY, matching rdirect_live.rs's own n0e finding that the wall is CPU-encode-dominated, not GPU-bound.)");
    let under_budget: Vec<&(String, f64, f64, usize)> =
        results.iter().filter(|(_, med, _, _)| *med <= BUDGET_MS).collect();
    match under_budget.iter().max_by_key(|(_, _, _, params)| *params) {
        Some((name, med, p95, params)) => println!(
            "LARGEST CONFIG UNDER GPU BUDGET ({:.2}ms): {} (gpu median {:.2}ms p95 {:.2}ms, {} params)",
            BUDGET_MS, name, med, p95, params
        ),
        None => println!(
            "NO CONFIG UNDER GPU BUDGET ({:.2}ms) — all {} configs over.",
            BUDGET_MS,
            results.len()
        ),
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("rdirect_v9_spike: macOS-only (MPSGraph tensor path).");
}
