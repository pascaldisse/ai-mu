//! The ONE op-set. FROZEN + WORKING (archon-written, thin).
//! bind = 2D circular convolution (FFT-domain multiply).
//! unbind = correlation (multiply by conjugate spectrum) — exact inverse for
//! unit-magnitude-spectrum atoms (atoms.rs guarantees this).
//! bundle = superposition. similarity = cosine. probe lives in store.rs.
//! W1 may optimize internals (e.g. cached spectra); semantics frozen.

use crate::fft::Fft2;
use crate::Slice;

/// bind(a, b) = ifft(fft(a) ∘ fft(b)). Circular convolution on the torus.
/// The wave propagator IS this op with a fixed kernel (plane.rs).
pub fn bind(f: &Fft2, a: &Slice, b: &Slice) -> Slice {
    assert_eq!(a.cfg, b.cfg);
    let sa = f.forward(&a.data);
    let sb = f.forward(&b.data);
    let prod: Vec<_> = sa.iter().zip(&sb).map(|(x, y)| x.mul(*y)).collect();
    Slice::from_data(a.cfg, f.inverse(&prod))
}

/// unbind(a, key) = ifft(fft(a) ∘ conj(fft(key))). Correlation / matched filter.
pub fn unbind(f: &Fft2, a: &Slice, key: &Slice) -> Slice {
    assert_eq!(a.cfg, key.cfg);
    let sa = f.forward(&a.data);
    let sk = f.forward(&key.data);
    let prod: Vec<_> = sa.iter().zip(&sk).map(|(x, y)| x.mul(y.conj())).collect();
    Slice::from_data(a.cfg, f.inverse(&prod))
}

/// bundle = superposition = interference. Plain sum, no normalization.
pub fn bundle(slices: &[&Slice]) -> Slice {
    assert!(!slices.is_empty());
    let cfg = slices[0].cfg;
    let mut out = Slice::zeros(cfg);
    for s in slices {
        assert_eq!(s.cfg, cfg);
        for (o, v) in out.data.iter_mut().zip(&s.data) {
            *o += *v;
        }
    }
    out
}

/// Cosine similarity. Zero-norm guard -> 0.0.
pub fn similarity(a: &Slice, b: &Slice) -> f32 {
    assert_eq!(a.cfg, b.cfg);
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.data.iter().zip(&b.data) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}
