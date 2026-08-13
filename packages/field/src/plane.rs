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
        let dd = p.damping * p.dt;
        let k = p.courant() * p.courant();
        let (w, h) = (cfg.width, cfg.height);
        let mut k1_spec = vec![C32::ZERO; cfg.d()];
        for ky in 0..h {
            for kx in 0..w {
                let lhat = 2.0 * (2.0 * std::f32::consts::PI * kx as f32 / w as f32).cos()
                    + 2.0 * (2.0 * std::f32::consts::PI * ky as f32 / h as f32).cos()
                    - 4.0;
                let k1 = (2.0 - dd) + k * lhat;
                k1_spec[ky * w + kx] = C32::new(k1, 0.0);
            }
        }
        let a_prev = -(1.0 - dd);
        WaveKernel { cfg, k1_spec, a_prev }
    }
}

/// One wave tick = ONE bind by the fixed kernel + prev superposition.
pub fn wave_step(f: &Fft2, k: &WaveKernel, cur: &Slice, prev: &Slice) -> Slice {
    assert_eq!(cur.cfg, k.cfg);
    assert_eq!(prev.cfg, k.cfg);
    let spec = f.forward(&cur.data);
    let prod: Vec<C32> = spec.iter().zip(&k.k1_spec).map(|(s, k1)| s.mul(*k1)).collect();
    let conv = f.inverse(&prod);
    let mut out = vec![0.0f32; cur.cfg.d()];
    for i in 0..out.len() {
        out[i] = conv[i] + k.a_prev * prev.data[i];
    }
    Slice::from_data(cur.cfg, out)
}

/// Direct 5-point stencil with periodic wrap. Parity oracle for wave_step.
pub fn wave_step_reference(p: &WaveParams, cur: &Slice, prev: &Slice) -> Slice {
    assert_eq!(cur.cfg, prev.cfg);
    let cfg = cur.cfg;
    let (w, h) = (cfg.width, cfg.height);
    let dd = p.damping * p.dt;
    let k = p.courant() * p.courant();
    let a_prev = -(1.0 - dd);
    let mut out = vec![0.0f32; cfg.d()];
    for y in 0..h {
        let ym = if y == 0 { h - 1 } else { y - 1 };
        let yp = if y == h - 1 { 0 } else { y + 1 };
        for x in 0..w {
            let xm = if x == 0 { w - 1 } else { x - 1 };
            let xp = if x == w - 1 { 0 } else { x + 1 };
            let i = y * w + x;
            let c0 = cur.data[i];
            let lap = cur.data[y * w + xm] + cur.data[y * w + xp] + cur.data[ym * w + x]
                + cur.data[yp * w + x]
                - 4.0 * c0;
            out[i] = (2.0 - dd) * c0 + k * lap + a_prev * prev.data[i];
        }
    }
    Slice::from_data(cfg, out)
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
    assert_eq!(cur.cfg, prev.cfg);
    let cfg = cur.cfg;
    let (w, h) = (cfg.width as i64, cfg.height as i64);
    let sigma = 2.0f32;
    let r = 8i64;
    let jitter = 1.0 + 0.1 * crate::seed_unit(seed, (x as u64) << 32 | y as u64);
    let amp = amplitude * jitter;
    for dy in -r..=r {
        for dx in -r..=r {
            let px = ((x as i64 + dx) % w + w) % w;
            let py = ((y as i64 + dy) % h + h) % h;
            let d2 = (dx * dx + dy * dy) as f32;
            let g = (-d2 / (2.0 * sigma * sigma)).exp();
            let i = py as usize * cfg.width + px as usize;
            cur.data[i] += amp * g;
            prev.data[i] += amp * g;
        }
    }
}

/// Energy proxy, ported from plane-v0 (kinetic from cur-prev + potential).
pub fn energy(p: &WaveParams, cur: &Slice, prev: &Slice) -> f64 {
    let inv_dt = 1.0 / p.dt as f64;
    let mut e = 0.0f64;
    for i in 0..cur.data.len() {
        let u = cur.data[i] as f64;
        let v = (cur.data[i] - prev.data[i]) as f64 * inv_dt;
        e += v * v + u * u;
    }
    e
}

pub fn has_nonfinite(s: &Slice) -> bool {
    s.data.iter().any(|v| !v.is_finite())
}

/// Render = probe of the spatial marginal: reshape + amplitude->pixel.
/// Deterministic quantisation identical to plane-v0 to_gray.
pub fn to_gray(s: &Slice, range: f32, buf: &mut Vec<u8>) {
    buf.clear();
    buf.reserve(s.data.len());
    let range = range.max(1e-9);
    for &v in &s.data {
        let t = (v / range).clamp(-1.0, 1.0);
        let g = ((t * 0.5 + 0.5) * 255.0 + 0.5) as u8;
        buf.push(g);
    }
}

/// 8-bit grayscale PNG in memory (png crate, as plane-v0 encode_png).
pub fn encode_png(s: &Slice, range: f32) -> Vec<u8> {
    let mut gray = Vec::new();
    to_gray(s, range, &mut gray);
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, s.cfg.width as u32, s.cfg.height as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().expect("png header");
        w.write_image_data(&gray).expect("png data");
    }
    out
}
