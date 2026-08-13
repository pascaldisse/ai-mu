//! 2D FFT over field slices. OWNED BY W1 (core).
//!
//! Iterative radix-2 Cooley-Tukey (decimation-in-time), rows then columns,
//! power-of-two dims only, twiddles + bit-reversal permutations precomputed
//! once in Fft2::new. No external FFT crate — sovereign, deterministic.
//! Contract:
//!   forward/inverse must match naive_dft2/naive_idft2 within rel 1e-3 at
//!   16x16 and 32x32 (tests/core.rs), and inverse(forward(x)) ~= x <= 1e-4.

use crate::{C32, FieldConfig, Scalar};
use std::cell::RefCell;

pub struct Fft2 {
    pub cfg: FieldConfig,
    perm_w: Vec<usize>,
    perm_h: Vec<usize>,
    tw_w: Vec<C32>,
    tw_h: Vec<C32>,
    // Reusable strided-column scratch (interior mutability: forward/inverse
    // stay &self per frozen signature). No allocation beyond this + the
    // output buffer once warmed up.
    scratch: RefCell<Vec<C32>>,
}

impl Fft2 {
    pub fn new(cfg: FieldConfig) -> Self {
        assert!(cfg.valid());
        let perm_w = bit_reverse_perm(cfg.width);
        let perm_h = bit_reverse_perm(cfg.height);
        let tw_w = twiddle_table(cfg.width);
        let tw_h = twiddle_table(cfg.height);
        let scratch_len = cfg.width.max(cfg.height);
        Fft2 {
            cfg,
            perm_w,
            perm_h,
            tw_w,
            tw_h,
            scratch: RefCell::new(vec![C32::ZERO; scratch_len]),
        }
    }

    /// real slice data -> full complex spectrum, layout spec[ky*W + kx],
    /// forward sign -i, unnormalized.
    pub fn forward(&self, real: &[Scalar]) -> Vec<C32> {
        assert_eq!(real.len(), self.cfg.d());
        let mut buf: Vec<C32> = real.iter().map(|&v| C32::new(v, 0.0)).collect();
        self.transform2d(&mut buf, false);
        buf
    }

    /// full complex spectrum -> real slice data (imaginary parts discarded).
    /// 1/d normalized: inverse(forward(x)) == x.
    pub fn inverse(&self, spec: &[C32]) -> Vec<Scalar> {
        assert_eq!(spec.len(), self.cfg.d());
        let mut buf: Vec<C32> = spec.to_vec();
        self.transform2d(&mut buf, true);
        let inv_d = 1.0f32 / self.cfg.d() as f32;
        buf.iter().map(|c| c.re * inv_d).collect()
    }

    /// Separable 2D transform: 1D FFT along each row (x axis), then along
    /// each column (y axis). Unnormalized both directions; caller applies
    /// 1/d once on the inverse path. Order of axes does not affect the
    /// final value (finite double-sum is order independent).
    fn transform2d(&self, buf: &mut [C32], invert: bool) {
        let (w, h) = (self.cfg.width, self.cfg.height);
        for y in 0..h {
            let row = &mut buf[y * w..y * w + w];
            fft1d_inplace(row, &self.perm_w, &self.tw_w, invert);
        }
        let mut scratch = self.scratch.borrow_mut();
        for x in 0..w {
            for y in 0..h {
                scratch[y] = buf[y * w + x];
            }
            fft1d_inplace(&mut scratch[..h], &self.perm_h, &self.tw_h, invert);
            for y in 0..h {
                buf[y * w + x] = scratch[y];
            }
        }
    }
}

/// Bit-reversal permutation for a length-n (power of two, n>=2) sequence.
fn bit_reverse_perm(n: usize) -> Vec<usize> {
    let bits = n.trailing_zeros();
    (0..n).map(|i| ((i as u32).reverse_bits() >> (32 - bits)) as usize).collect()
}

/// Precomputed forward twiddle table: tw[k] = exp(-2*pi*i*k/n), k in 0..n/2.
/// Every stage's twiddle exp(-2*pi*i*j/m) for sub-fft size m equals
/// tw[j*(n/m)], so one table of size n/2 serves all stages.
fn twiddle_table(n: usize) -> Vec<C32> {
    let half = n / 2;
    (0..half)
        .map(|k| {
            let ang = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            C32::new(ang.cos() as f32, ang.sin() as f32)
        })
        .collect()
}

/// In-place iterative radix-2 decimation-in-time FFT. `perm` = bit-reversal
/// permutation for len(buf), `tw` = precomputed forward twiddle table for
/// len(buf). `invert` uses conjugated twiddles (unnormalized inverse); caller
/// normalizes by 1/n separately.
fn fft1d_inplace(buf: &mut [C32], perm: &[usize], tw: &[C32], invert: bool) {
    let n = buf.len();
    for i in 0..n {
        let j = perm[i];
        if j > i {
            buf.swap(i, j);
        }
    }
    let mut m = 2usize;
    while m <= n {
        let half = m / 2;
        let stride = n / m;
        let mut start = 0usize;
        while start < n {
            for j in 0..half {
                let base = tw[j * stride];
                let t = if invert { base.conj() } else { base };
                let a = buf[start + j];
                let bt = buf[start + j + half].mul(t);
                buf[start + j] = a.add(bt);
                buf[start + j + half] = C32::new(a.re - bt.re, a.im - bt.im);
            }
            start += m;
        }
        m <<= 1;
    }
}
