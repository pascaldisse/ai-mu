//! 2D FFT over field slices. OWNED BY W1 (core).
//!
//! SKELETON STATE: Fft2 currently delegates to the naive O(d^2) DFT oracle so
//! every other lane runs TODAY at small dims. W1 replaces internals with an
//! iterative radix-2 Cooley-Tukey (rows then cols), power-of-two dims only,
//! precomputed twiddles, zero allocation in the transform hot path beyond the
//! output buffer. SIGNATURES FROZEN. Contract:
//!   forward/inverse must match naive_dft2/naive_idft2 within rel 1e-3 at
//!   16x16 and 32x32 (tests/core.rs), and inverse(forward(x)) ~= x <= 1e-4.
//! No external FFT crate — sovereign, deterministic.

use crate::{naive_dft2, naive_idft2, C32, FieldConfig, Scalar};

pub struct Fft2 {
    pub cfg: FieldConfig,
}

impl Fft2 {
    pub fn new(cfg: FieldConfig) -> Self {
        assert!(cfg.valid());
        Fft2 { cfg }
    }

    /// real slice data -> full complex spectrum, layout spec[ky*W + kx],
    /// forward sign -i, unnormalized.
    pub fn forward(&self, real: &[Scalar]) -> Vec<C32> {
        // W1: replace with radix-2. Naive oracle keeps the wave unblocked.
        naive_dft2(self.cfg, real)
    }

    /// full complex spectrum -> real slice data (imaginary parts discarded).
    /// 1/d normalized: inverse(forward(x)) == x.
    pub fn inverse(&self, spec: &[C32]) -> Vec<Scalar> {
        // W1: replace with radix-2.
        naive_idft2(self.cfg, spec)
    }
}
