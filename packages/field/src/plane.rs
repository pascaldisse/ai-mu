//! THE PLANE AS BIND. OWNED BY W2. SIGNATURES FROZEN.
//!
//! Port of plane-v0 (commit 0587c8d3, packages/plane) into the unified field:
//! the damped-wave FD step IS a fixed-kernel bind. plane-v0 step:
//!   next = 2*cur - prev + k*lap(cur) - dd*(cur - prev)
//!     k = (c*dt/dx)^2, dd = damping*dt, lap = 5-point Laplacian.
//! Regrouped linearly:  next = (2 - dd)*cur + k*lap(cur) - (1 - dd)*prev.
//! On the TORUS (periodic wrap — V1 boundary law; plane-v0 Dirichlet was V0
//! scaffold) lap is circulant, so in the Fourier basis:
//!   next = ifft( fft(cur) ∘ K1 ) + a_prev * prev
//!   K1(kx,ky) = (2 - dd) + k * Lhat(kx,ky)
//!   Lhat(kx,ky) = 2cos(2π kx/W) + 2cos(2π ky/H) - 4     (real)
//!   a_prev = -(1 - dd)
//! wave_step_reference = the SAME stencil computed directly with periodic
//! wrap. PARITY GATE (tests/plane_parity.rs): |fft_step - reference_step|max
//! <= 1e-3 over >= 200 steps after one excite, at 32x32 and 64x64 (naive-FFT
//! sizes; full 128x128/512x512 parity re-run post-merge on fast FFT).
//! Also port: CFL stability test, energy-decay test, nonfinite guard,
//! deterministic to_gray/encode_png (grayscale, plane-v0 quantisation).

use crate::fft::Fft2;
use crate::{C32, FieldConfig, Scalar, Slice};

/// Wave parameters. Defaults IDENTICAL to plane-v0 Params.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveParams {
    pub c: f32,
    pub dt: f32,
    pub damping: f32,
    pub dx: f32,
    pub seed: u64,
    pub range: f32,
}

impl Default for WaveParams {
    fn default() -> Self {
        WaveParams { c: 1.0, dt: 0.5, damping: 0.02, dx: 1.0, seed: 0, range: 0.06 }
    }
}

impl WaveParams {
    pub fn courant(&self) -> f32 {
        self.c * self.dt / self.dx
    }
    pub fn stable(&self) -> bool {
        self.courant() <= std::f32::consts::FRAC_1_SQRT_2
    }
}

/// The fixed propagator kernel: K1 spectrum + prev coefficient.
pub struct WaveKernel {
    pub cfg: FieldConfig,
    pub k1_spec: Vec<C32>,
    pub a_prev: f32,
}

impl WaveKernel {
    /// Build K1 analytically (NOT via FFT of a stencil image — exact spectrum).
    pub fn new(cfg: FieldConfig, p: &WaveParams) -> Self {
        let _ = (cfg, p);
        todo!("W2: K1(kx,ky) = (2-dd) + k*Lhat, a_prev = -(1-dd)")
    }
}

/// One wave tick = ONE bind by the fixed kernel + prev superposition.
pub fn wave_step(f: &Fft2, k: &WaveKernel, cur: &Slice, prev: &Slice) -> Slice {
    let _ = (f, k, cur, prev);
    todo!("W2: ifft(fft(cur) ∘ K1) + a_prev*prev")
}

/// Direct 5-point stencil with periodic wrap. Parity oracle for wave_step.
pub fn wave_step_reference(p: &WaveParams, cur: &Slice, prev: &Slice) -> Slice {
    let _ = (p, cur, prev);
    todo!("W2: direct periodic stencil, same regrouped coefficients")
}

/// Gaussian strike, ported from plane-v0: sigma=2, r=8, displacement strike
/// (cur AND prev += amp*g), jitter = 1 + 0.1*seed_unit(seed, (x<<32)|y).
/// V1 difference: TORUS WRAP instead of edge clip.
pub fn excite(
    cur: &mut Slice,
    prev: &mut Slice,
    x: usize,
    y: usize,
    amplitude: Scalar,
    seed: u64,
) {
    let _ = (cur, prev, x, y, amplitude, seed);
    todo!("W2: port plane-v0 excite with periodic wrap")
}

/// Energy proxy, ported from plane-v0 (kinetic from cur-prev + potential).
pub fn energy(p: &WaveParams, cur: &Slice, prev: &Slice) -> f64 {
    let _ = (p, cur, prev);
    todo!("W2: port plane-v0 energy")
}

pub fn has_nonfinite(s: &Slice) -> bool {
    s.data.iter().any(|v| !v.is_finite())
}

/// Render = probe of the spatial marginal: reshape + amplitude->pixel.
/// Deterministic quantisation identical to plane-v0 to_gray.
pub fn to_gray(s: &Slice, range: f32, buf: &mut Vec<u8>) {
    let _ = (s, range, buf);
    todo!("W2: port plane-v0 to_gray")
}

/// 8-bit grayscale PNG in memory (png crate, as plane-v0 encode_png).
pub fn encode_png(s: &Slice, range: f32) -> Vec<u8> {
    let _ = (s, range);
    todo!("W2: port plane-v0 encode_png")
}
