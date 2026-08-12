//! FIELD — the unified one-field substrate (V1, 2026-08-01).
//!
//! Adversary-forced unification (docs/research/2026-08-01-field-adversary.md):
//! a hypervector IS a field slice: d = W*H, flattened row-major spatial grid.
//! ONE op-set, FFT-domain: bind = 2D circular convolution · bundle =
//! superposition (+) · probe = correlation (matched filter) · render =
//! reshape + amplitude->pixel. The wave plane = a fixed-kernel bind.
//! Store = N×d RAM-RESIDENT matrix; SSD = cold journal ONLY (never paged
//! inside a frame). Cadence: field tick @120fps floor; learning amortized.
//!
//! Law (ENTROPY.md): ultradeterminism. state = f(seed). No rand, no clock,
//! no nondeterministic iteration. All f32 persisted as bit patterns.
//!
//! FROZEN CONTRACT: this file + module signatures are owned by the archon.
//! Workers replace module INTERNALS, never signatures.
//! Ownership map -> docs/design/2026-08-01-field-contract.md

pub mod atoms;
pub mod fft;
pub mod journal;
pub mod ops;
pub mod plane;
pub mod store;
pub mod surface;
pub mod world;

pub type Scalar = f32;

// ---------------------------------------------------------------- determinism

/// Deterministic bit-mixer (SplitMix64 finalizer). Canonical, from plane-v0.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// seed -> deterministic value in [-1, 1]. Canonical, from plane-v0.
#[inline]
pub fn seed_unit(seed: u64, tag: u64) -> f32 {
    let m = mix64(seed ^ mix64(tag));
    ((m >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
}

/// FNV-1a 64 over bytes. Canonical digest for determinism gates.
#[inline]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ------------------------------------------------------------------- geometry

/// Field configuration. IRON: parameters with defaults, no hardcodes.
/// Both dims MUST be powers of two (radix-2 FFT).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldConfig {
    pub width: usize,
    pub height: usize,
}

impl FieldConfig {
    pub fn new(width: usize, height: usize) -> Self {
        let c = FieldConfig { width, height };
        assert!(c.valid(), "FieldConfig dims must be powers of two >= 2");
        c
    }
    /// d = W*H: the ONE dimensionality. A hypervector is a field slice.
    #[inline]
    pub fn d(&self) -> usize {
        self.width * self.height
    }
    pub fn valid(&self) -> bool {
        self.width >= 2
            && self.height >= 2
            && self.width.is_power_of_two()
            && self.height.is_power_of_two()
    }
}

impl Default for FieldConfig {
    fn default() -> Self {
        FieldConfig { width: 128, height: 128 }
    }
}

/// A field slice = ONE hypervector = ONE W×H grid, flattened row-major.
/// This is the single tensor type of the engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Slice {
    pub cfg: FieldConfig,
    pub data: Vec<Scalar>,
}

impl Slice {
    pub fn zeros(cfg: FieldConfig) -> Self {
        Slice { cfg, data: vec![0.0; cfg.d()] }
    }
    pub fn from_data(cfg: FieldConfig, data: Vec<Scalar>) -> Self {
        assert_eq!(data.len(), cfg.d());
        Slice { cfg, data }
    }
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.cfg.width + x
    }
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt() as f32
    }
    /// Bit-exact digest (f32 bit patterns, row-major order).
    pub fn digest(&self) -> u64 {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        fnv1a(&bytes)
    }
}

// ------------------------------------------------------------------- complex

/// Minimal complex f32. No external deps — sovereign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C32 {
    pub re: f32,
    pub im: f32,
}

impl C32 {
    pub const ZERO: C32 = C32 { re: 0.0, im: 0.0 };
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        C32 { re, im }
    }
    #[inline]
    pub fn mul(self, o: C32) -> C32 {
        C32 { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    #[inline]
    pub fn conj(self) -> C32 {
        C32 { re: self.re, im: -self.im }
    }
    #[inline]
    pub fn add(self, o: C32) -> C32 {
        C32 { re: self.re + o.re, im: self.im + o.im }
    }
    #[inline]
    pub fn scale(self, s: f32) -> C32 {
        C32 { re: self.re * s, im: self.im * s }
    }
    #[inline]
    pub fn abs(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

// -------------------------------------------------- naive DFT (test oracle)

/// Naive O(d^2) 2D DFT. TEST ORACLE ONLY — correctness reference for fft::Fft2.
/// Spectrum layout: spec[ky * W + kx], forward sign -i, unnormalized.
pub fn naive_dft2(cfg: FieldConfig, real: &[Scalar]) -> Vec<C32> {
    let (w, h) = (cfg.width, cfg.height);
    assert_eq!(real.len(), w * h);
    let mut out = vec![C32::ZERO; w * h];
    for ky in 0..h {
        for kx in 0..w {
            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;
            for y in 0..h {
                for x in 0..w {
                    let ang = -2.0 * std::f64::consts::PI
                        * ((kx * x) as f64 / w as f64 + (ky * y) as f64 / h as f64);
                    let v = real[y * w + x] as f64;
                    acc_re += v * ang.cos();
                    acc_im += v * ang.sin();
                }
            }
            out[ky * w + kx] = C32::new(acc_re as f32, acc_im as f32);
        }
    }
    out
}

/// Naive O(d^2) inverse 2D DFT, real part. TEST ORACLE ONLY. 1/d normalized.
pub fn naive_idft2(cfg: FieldConfig, spec: &[C32]) -> Vec<Scalar> {
    let (w, h) = (cfg.width, cfg.height);
    assert_eq!(spec.len(), w * h);
    let inv_d = 1.0f64 / (w * h) as f64;
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for ky in 0..h {
                for kx in 0..w {
                    let ang = 2.0 * std::f64::consts::PI
                        * ((kx * x) as f64 / w as f64 + (ky * y) as f64 / h as f64);
                    let s = spec[ky * w + kx];
                    acc += s.re as f64 * ang.cos() - s.im as f64 * ang.sin();
                }
            }
            out[y * w + x] = (acc * inv_d) as f32;
        }
    }
    out
}
