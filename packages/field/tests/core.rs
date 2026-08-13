//! W1 core gates: fft.rs radix-2 correctness vs naive oracle, atoms.rs
//! unit-magnitude conjugate-symmetric spectrum properties.

use field::atoms::seeded_atom;
use field::fft::Fft2;
use field::ops::{bind, similarity, unbind};
use field::{naive_dft2, naive_idft2, seed_unit, FieldConfig, Slice, C32};
use std::time::Instant;

fn fill_slice(cfg: FieldConfig, seed: u64) -> Slice {
    let d = cfg.d();
    let data: Vec<f32> = (0..d as u64).map(|i| seed_unit(seed, i)).collect();
    Slice::from_data(cfg, data)
}

fn rel_l2_complex(got: &[C32], want: &[C32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in got.iter().zip(want) {
        let dre = (a.re - b.re) as f64;
        let dim = (a.im - b.im) as f64;
        num += dre * dre + dim * dim;
        den += (b.re as f64) * (b.re as f64) + (b.im as f64) * (b.im as f64);
    }
    num.sqrt() / den.sqrt().max(1e-12)
}

fn rel_l2_real(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in got.iter().zip(want) {
        let d = (*a - *b) as f64;
        num += d * d;
        den += (*b as f64) * (*b as f64);
    }
    num.sqrt() / den.sqrt().max(1e-12)
}

fn check_fft_matches_naive(w: usize, h: usize) {
    let cfg = FieldConfig::new(w, h);
    let fft = Fft2::new(cfg);
    let x = fill_slice(cfg, 11 + (w * h) as u64);

    let got_fwd = fft.forward(&x.data);
    let want_fwd = naive_dft2(cfg, &x.data);
    let rel_fwd = rel_l2_complex(&got_fwd, &want_fwd);
    assert!(rel_fwd <= 1e-3, "forward rel err {rel_fwd} at {w}x{h}");

    let got_inv = fft.inverse(&want_fwd);
    let want_inv = naive_idft2(cfg, &want_fwd);
    let rel_inv = rel_l2_real(&got_inv, &want_inv);
    assert!(rel_inv <= 1e-3, "inverse rel err {rel_inv} at {w}x{h}");
}

#[test]
fn fft_matches_naive_16x16() {
    check_fft_matches_naive(16, 16);
}

#[test]
fn fft_matches_naive_32x32() {
    check_fft_matches_naive(32, 32);
}

#[test]
fn fft_roundtrip() {
    for &(w, h) in &[(16usize, 16usize), (32, 32), (64, 64), (128, 128)] {
        let cfg = FieldConfig::new(w, h);
        let fft = Fft2::new(cfg);
        let x = fill_slice(cfg, 42);
        let spec = fft.forward(&x.data);
        let back = fft.inverse(&spec);
        let maxdiff = x
            .data
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxdiff <= 1e-4, "roundtrip maxdiff {maxdiff} at {w}x{h}");
    }
}

#[test]
fn atom_unit_norm() {
    let cfg = FieldConfig::new(128, 128);
    let fft = Fft2::new(cfg);
    for seed in [1u64, 2, 3, 999] {
        let a = seeded_atom(&fft, cfg, seed);
        let n = a.norm();
        assert!((n - 1.0).abs() <= 1e-3, "seed {seed} norm {n}");
    }
}

#[test]
fn unbind_bind_recovers_x() {
    let cfg = FieldConfig::new(128, 128);
    let fft = Fft2::new(cfg);
    let a = seeded_atom(&fft, cfg, 7);
    let x = fill_slice(cfg, 123);
    let bound = bind(&fft, &a, &x);
    let recovered = unbind(&fft, &bound, &a);
    let rel = rel_l2_real(&recovered.data, &x.data);
    assert!(rel <= 1e-3, "recover rel err {rel}");
}

#[test]
fn cross_seed_low_similarity() {
    let cfg = FieldConfig::new(128, 128);
    let fft = Fft2::new(cfg);
    for &(sa, sb) in &[(1u64, 2u64), (5, 9), (100, 200)] {
        let a = seeded_atom(&fft, cfg, sa);
        let b = seeded_atom(&fft, cfg, sb);
        let sim = similarity(&a, &b);
        assert!(sim.abs() < 0.1, "seeds {sa},{sb} sim {sim}");
    }
}

#[test]
fn same_seed_bit_identical() {
    let cfg = FieldConfig::new(64, 64);
    let fft = Fft2::new(cfg);
    let a1 = seeded_atom(&fft, cfg, 555);
    let a2 = seeded_atom(&fft, cfg, 555);
    assert_eq!(a1.digest(), a2.digest());
}

#[test]
fn fft_128_timing_informational() {
    let cfg = FieldConfig::new(128, 128);
    let fft = Fft2::new(cfg);
    let x = fill_slice(cfg, 1);
    let iters = 50u32;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = fft.forward(&x.data);
    }
    let elapsed = start.elapsed();
    let per_call_us = elapsed.as_micros() as f64 / iters as f64;
    println!("fft 128x128 forward: {per_call_us:.2} us/call (informational, n={iters})");
}
