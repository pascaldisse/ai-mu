//! Parity + stability gates for plane.rs (field/plane, W2).
//! Naive-FFT sizes only (32x32, 64x64) — full-size parity deferred to
//! post-merge fast FFT per field-adversary.md §1.

use field::fft::Fft2;
use field::plane::{
    encode_png, energy, excite, has_nonfinite, wave_step, wave_step_reference, WaveKernel,
    WaveParams,
};
use field::{FieldConfig, Slice};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn run_parity(w: usize, h: usize, steps: usize) -> f32 {
    let cfg = FieldConfig::new(w, h);
    let p = WaveParams { seed: 42, ..WaveParams::default() };
    assert!(p.stable(), "params must satisfy CFL");
    let f = Fft2::new(cfg);
    let k = WaveKernel::new(cfg, &p);

    let mut cur_fft = Slice::zeros(cfg);
    let mut prev_fft = Slice::zeros(cfg);
    excite(&mut cur_fft, &mut prev_fft, w / 2, h / 2, 1.0, p.seed);

    let mut cur_ref = cur_fft.clone();
    let mut prev_ref = prev_fft.clone();

    let mut max_diff = 0.0f32;
    for _ in 0..steps {
        let next_fft = wave_step(&f, &k, &cur_fft, &prev_fft);
        let next_ref = wave_step_reference(&p, &cur_ref, &prev_ref);
        let diff = max_abs_diff(&next_fft.data, &next_ref.data);
        if diff > max_diff {
            max_diff = diff;
        }
        prev_fft = cur_fft;
        cur_fft = next_fft;
        prev_ref = cur_ref;
        cur_ref = next_ref;
    }
    max_diff
}

#[test]
fn parity_32x32_200_steps() {
    let max_diff = run_parity(32, 32, 200);
    eprintln!("parity_32x32_200_steps: max|fft-reference| = {:e}", max_diff);
    assert!(max_diff <= 1e-3, "max diff {} exceeds 1e-3 (32x32)", max_diff);
}

#[test]
fn parity_64x64_200_steps() {
    let max_diff = run_parity(64, 64, 200);
    eprintln!("parity_64x64_200_steps: max|fft-reference| = {:e}", max_diff);
    assert!(max_diff <= 1e-3, "max diff {} exceeds 1e-3 (64x64)", max_diff);
}

#[test]
fn cfl_stable_gate() {
    let p = WaveParams::default();
    assert!(p.stable(), "default params must be CFL-stable");

    let unstable = WaveParams { dt: 2.0, ..WaveParams::default() };
    assert!(!unstable.stable(), "dt=2.0 (courant=2.0) must be unstable");
}

#[test]
fn energy_monotone_decay_with_damping() {
    // Matches plane-v0's own convention (tests/field.rs energy_decays_with_damping):
    // the impulsive excite causes a brief non-monotone transient in the discrete
    // energy proxy; the GATE is the settled trend, sampled in checkpoints after
    // the strike, same as source port (checks e50 > e500, not every step).
    let cfg = FieldConfig::new(32, 32);
    let p = WaveParams { seed: 7, damping: 0.05, ..WaveParams::default() };
    assert!(p.stable());
    let f = Fft2::new(cfg);
    let k = WaveKernel::new(cfg, &p);

    let mut cur = Slice::zeros(cfg);
    let mut prev = Slice::zeros(cfg);
    excite(&mut cur, &mut prev, 16, 16, 1.0, p.seed);

    let mut checkpoints = Vec::new();
    for step in 1..=500 {
        let next = wave_step(&f, &k, &cur, &prev);
        if step % 50 == 0 {
            checkpoints.push(energy(&p, &next, &cur));
        }
        prev = cur;
        cur = next;
    }

    assert!(checkpoints.len() == 10, "expected 10 checkpoints, got {}", checkpoints.len());
    for i in 1..checkpoints.len() {
        assert!(
            checkpoints[i] <= checkpoints[i - 1] * 1.001,
            "checkpoint energy rose beyond noise margin: [{}]={} -> [{}]={}",
            i - 1,
            checkpoints[i - 1],
            i,
            checkpoints[i]
        );
    }
    assert!(
        *checkpoints.last().unwrap() < checkpoints[0],
        "energy did not decay overall: e50={} e500={}",
        checkpoints[0],
        checkpoints.last().unwrap()
    );
}

#[test]
fn stays_finite_500_steps() {
    let cfg = FieldConfig::new(32, 32);
    let p = WaveParams { seed: 3, ..WaveParams::default() };
    assert!(p.stable());
    let f = Fft2::new(cfg);
    let k = WaveKernel::new(cfg, &p);

    let mut cur = Slice::zeros(cfg);
    let mut prev = Slice::zeros(cfg);
    excite(&mut cur, &mut prev, 16, 16, 1.0, p.seed);

    for step in 0..500 {
        let next = wave_step(&f, &k, &cur, &prev);
        assert!(!has_nonfinite(&next), "nonfinite value at step {}", step);
        prev = cur;
        cur = next;
    }
}

#[test]
fn encode_png_same_seed_byte_identical() {
    let cfg = FieldConfig::new(16, 16);
    let build = || {
        let mut cur = Slice::zeros(cfg);
        let mut prev = Slice::zeros(cfg);
        excite(&mut cur, &mut prev, 8, 8, 1.0, 55);
        encode_png(&cur, 0.06)
    };
    let a = build();
    let b = build();
    assert_eq!(a, b, "encode_png must be byte-identical across two same-seed runs");
}
