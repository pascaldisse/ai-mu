//! W8 — capacity/SNR characterization (2026-08-01).
//!
//! Measures the adversary §2 bound in THIS substrate: bundling K items into
//! d dims degrades SNR ~ sqrt(d/K); store is N×d because one vector can't
//! hold a world. At cfg 64×64 (d=4096), for K in {1,4,16,64,256}:
//!   bundle K distinct seeded atoms -> sim(bundle, member) for all members,
//!   sim(bundle, distractor) for 256 FRESH distractor atoms.
//! Report: member μ, mean|distractor| μ, margin, retrieval accuracy.
//!
//! atoms.rs is todo!() in this tree -> TEST-LOCAL atom helper (spec from
//! src/atoms.rs doc comment): unit-magnitude random-phase spectrum, phases
//! via mix64, conjugate symmetry S[-k] = conj(S[k]), self-conjugate bins
//! (DC + Nyquist) = ±1 from seed. Materialized to a REAL slice via the
//! naive_idft2 formula. atoms.rs mandates the naive O(d²) oracle; at
//! d=4096 that is ~16.7M trig pairs PER ATOM (256 members + 5×256
//! distractors ≈ 1536 atoms ≈ hours). So the helper materializes via a
//! test-local separable cached-twiddle idft2, ALGEBRAICALLY IDENTICAL to
//! lib naive_idft2 (same 1/d-normalized +i-sign 2D DFT, same f64 acc order
//! per output); pinned by test `fast_idft_equiv_naive` (≤1e-6, plus
//! Parseval ||atom||≈1 within 1e-3). Measurement itself runs through the
//! REAL ops::{bundle, similarity}.
//!
//! Determinism: every seed is a fixed constant; mix64 only, no rand.

use field::ops;
use field::{mix64, naive_idft2, C32, FieldConfig, Slice};

const CFG: FieldConfig = FieldConfig { width: 64, height: 64 };
const D: usize = 64 * 64;
const DISTRACTORS: usize = 256;
/// Member atom seeds: nested first-K of 256 (seeds 0..256).
const MEMBERS: usize = 256;
/// Fresh distractor seed base per K: 1_000_000 + k*1000 + i.
const DIST_BASE: u64 = 1_000_000;

// ------------------------------------------------------------------ atoms

/// phase(kx,ky) = f(seed, ky*W+kx) via mix64 ONLY, in [0, TAU).
fn phase(seed: u64, idx: u64) -> f32 {
    let m = mix64(seed ^ mix64(idx));
    ((m >> 11) as f64 / (1u64 << 53) as f64) as f32 * std::f32::consts::TAU
}

/// Unit-magnitude random-phase spectrum, conjugate symmetry, self-conjugate
/// bins (DC + Nyquist combos) = ±1 from seed. |S[k]| == 1 EVERYWHERE ->
/// Parseval: ||atom|| == 1 (1/d-normalized inverse).
fn spectrum_for_atom(cfg: FieldConfig, seed: u64) -> Vec<C32> {
    let (w, h) = (cfg.width, cfg.height);
    let mut spec = vec![C32::ZERO; w * h];
    for ky in 0..h {
        for kx in 0..w {
            let i = ky * w + kx;
            let p = ((h - ky) % h) * w + (w - kx) % w;
            if i < p {
                // canonical: lower index gets the random phase, partner = conj
                let ph = phase(seed, i as u64);
                let s = C32::new(ph.cos(), ph.sin());
                spec[i] = s;
                spec[p] = s.conj();
            } else if i == p {
                // self-conjugate bin: ±1 deterministic from seed
                let m = mix64(seed ^ mix64(i as u64 ^ 0x51DE_C0DE));
                spec[i] = C32::new(if m & 1 == 0 { 1.0 } else { -1.0 }, 0.0);
            }
            // i > p: partner already assigned at iteration p, skip
        }
    }
    spec
}

/// Cached 1D twiddles e^{2πi·k·x/n} as f64 pairs (n ≤ 64 here).
struct Tw {
    re: f64,
    im: f64,
}
fn twiddles(n: usize) -> Vec<Vec<Tw>> {
    (0..n)
        .map(|k| {
            (0..n)
                .map(|x| {
                    let a = 2.0 * std::f64::consts::PI * (k * x) as f64 / n as f64;
                    Tw { re: a.cos(), im: a.sin() }
                })
                .collect()
        })
        .collect()
}

/// Separable cached-twiddle inverse 2D DFT, real output, 1/d normalized.
/// Same math as lib naive_idft2 (sign +i, unnormalized spectrum, 1/d):
/// row pass over kx per (ky,x), then col pass over ky per (x,y), f64 acc.
/// Expected |x - naive_idft2(cfg, spec)| <= ~1e-6 (see equiv test).
fn idft2_fast(cfg: FieldConfig, spec: &[C32]) -> Vec<f32> {
    let (w, h) = (cfg.width, cfg.height);
    assert_eq!(spec.len(), w * h);
    let twx = twiddles(w);
    let twy = twiddles(h);
    // row pass: t[ky][x] = Σ_kx spec[ky][kx] · e^{2πi·kx·x/W}
    let mut t = vec![C32::ZERO; w * h];
    for ky in 0..h {
        for x in 0..w {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for kx in 0..w {
                let s = spec[ky * w + kx];
                let tw = &twx[kx][x];
                re += s.re as f64 * tw.re - s.im as f64 * tw.im;
                im += s.re as f64 * tw.im + s.im as f64 * tw.re;
            }
            t[ky * w + x] = C32::new(re as f32, im as f32);
        }
    }
    // col pass: out[y][x] = (1/d) Σ_ky t[ky][x] · e^{2πi·ky·y/H}, real part
    let inv_d = 1.0 / (w * h) as f64;
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut re = 0.0f64;
            for ky in 0..h {
                let s = t[ky * w + x];
                let tw = &twy[ky][y];
                re += s.re as f64 * tw.re - s.im as f64 * tw.im;
            }
            out[y * w + x] = (re * inv_d) as f32;
        }
    }
    out
}

/// TEST-LOCAL atom helper (src/atoms.rs is todo!() here). Real slice,
/// unit-magnitude spectrum, ||atom|| ≈ 1. Materialization = naive_idft2
/// formula via idft2_fast (equiv pinned ≤1e-6 below).
fn seeded_atom(cfg: FieldConfig, seed: u64) -> Slice {
    Slice::from_data(cfg, idft2_fast(cfg, &spectrum_for_atom(cfg, seed)))
}

// ------------------------------------------------------------ sanity tests

/// Atom helper contract (atoms.rs doc): ||atom|| == 1 within 1e-3; distinct
/// seeds near-orthogonal; fast materialization ≡ lib naive_idft2 within 1e-6.
#[test]
fn atom_helper_sanity() {
    for seed in [0u64, 1, 42, 0xDEAD_BEEF, 123_456_789] {
        let a = seeded_atom(CFG, seed);
        let n = a.norm();
        assert!(
            (n - 1.0).abs() < 1e-3,
            "seed {seed}: ||atom|| = {n}, expected 1 within 1e-3"
        );
        // distinct seeds -> |similarity| small (near-orthogonal)
        let b = seeded_atom(CFG, seed ^ 0xF00D);
        let s = ops::similarity(&a, &b).abs();
        assert!(s < 0.1, "seed {seed}: |sim| to distinct atom = {s}, expect < 0.1");
    }
}

/// Pins the fast materialization to the LIBRARY oracle naive_idft2: same
/// spectrum in -> outputs within 1e-6 (f32 rounding), for multiple seeds.
#[test]
fn fast_idft_equiv_naive() {
    for seed in [0u64, 7, 0xBEEF, 987_654_321] {
        let spec = spectrum_for_atom(CFG, seed);
        let naive = naive_idft2(CFG, &spec);
        let fast = idft2_fast(CFG, &spec);
        let mut max_abs = 0.0f32;
        for (n, f) in naive.iter().zip(&fast) {
            max_abs = max_abs.max((n - f).abs());
        }
        assert!(
            max_abs < 1e-6,
            "seed {seed}: |fast - naive_idft2| max = {max_abs}, expect < 1e-6"
        );
    }
}

// ------------------------------------------------------------ the measurement

/// Main capacity/SNR characterization at cfg 64×64 (d = 4096).
/// K in {1,4,16,64,256}: bundle K members, sim to all members + 256 fresh
/// distractors. Asserts: accuracy 100% at K<=16, >=95% at K<=64,
/// member/distractor separation shrinks ~1/sqrt(K) (fit within factor 2),
/// distractor noise floor flat at ~1/sqrt(d).
#[test]
fn capacity_snr_at_4096() {
    // 256 member atoms, nested first-K per K (distinct seeds 0..256)
    let members: Vec<Slice> = (0..MEMBERS as u64).map(|s| seeded_atom(CFG, s)).collect();

    println!("\n=== W8 capacity/SNR @ 64x64 (d = {D}) ===");
    println!("{:<4} {:>12} {:>16} {:>12} {:>10} {:>10} {:>10}", "K", "member μ", "mean|dist| μ", "margin", "acc%", "min member", "max dist");

    let ks = [1usize, 4, 16, 64, 256];
    let mut margins = Vec::new();
    for &k in &ks {
        let member_refs: Vec<&Slice> = members[..k].iter().collect();
        let bundle = ops::bundle(&member_refs);

        // members: sim(bundle, member) for ALL k members
        let mut member_sims = Vec::with_capacity(k);
        for m in 0..k {
            member_sims.push(ops::similarity(&bundle, &members[m]));
        }

        // distractors: 256 FRESH atoms per K
        let mut dist_abs = Vec::with_capacity(DISTRACTORS);
        let mut dist_signed = Vec::with_capacity(DISTRACTORS);
        for i in 0..DISTRACTORS as u64 {
            let seed = DIST_BASE + k as u64 * 1000 + i;
            let d = seeded_atom(CFG, seed);
            let s = ops::similarity(&bundle, &d);
            dist_signed.push(s);
            dist_abs.push(s.abs());
        }

        let member_mean = member_sims.iter().sum::<f32>() / k as f32;
        let dist_mean = dist_abs.iter().sum::<f32>() / DISTRACTORS as f32;
        let margin = member_mean - dist_mean;
        let min_member = member_sims.iter().cloned().fold(f32::MAX, f32::min);
        let max_dist = dist_signed.iter().cloned().fold(f32::MIN, f32::max);
        let accuracy =
            member_sims.iter().filter(|&&m| m > max_dist).count() as f32 / k as f32 * 100.0;
        margins.push((k, margin));

        println!(
            "{:<4} {:>12.6} {:>16.6} {:>12.6} {:>9.1}% {:>10.6} {:>10.6}",
            k, member_mean, dist_mean, margin, accuracy, min_member, max_dist
        );

        // ---- assertions (adversary §2, measured in THIS substrate) ----
        if k <= 16 {
            assert_eq!(accuracy, 100.0, "K={k}: retrieval accuracy must be 100%");
        }
        if k <= 64 {
            assert!(accuracy >= 95.0, "K={k}: accuracy {accuracy}% < 95%");
        }
        // separation ~ 1/sqrt(K): member μ and margin × sqrt(K) within
        // factor 2 of 1.0
        let member_scaled = member_mean * (k as f32).sqrt();
        assert!(
            (0.5..=2.0).contains(&member_scaled),
            "K={k}: member μ·√K = {member_scaled}, expect within [0.5, 2.0]"
        );
        let margin_scaled = margin * (k as f32).sqrt();
        assert!(
            (0.5..=2.0).contains(&margin_scaled),
            "K={k}: margin·√K = {margin_scaled}, expect within [0.5, 2.0]"
        );
        // distractor floor flat at ~1/sqrt(d): mean|dist|·√d in [0.4, 1.6]
        let floor_scaled = dist_mean * (D as f32).sqrt();
        assert!(
            (0.4..=1.6).contains(&floor_scaled),
            "K={k}: mean|dist|·√d = {floor_scaled}, expect ~0.8 flat floor"
        );
    }

    // monotone shrink of margin with K (K=256 may approach the noise floor)
    for w in margins.windows(2) {
        assert!(
            w[1].1 < w[0].1,
            "margin must shrink with K: {} ({}) !< {} ({})",
            w[0].0, w[0].1, w[1].0, w[1].1
        );
    }
    println!("=== end capacity table ===\n");
}
