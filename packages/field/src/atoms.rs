//! Seeded atoms. OWNED BY W1 (core). SIGNATURES FROZEN.
//!
//! seeded_atom(f, cfg, seed) -> real Slice whose spectrum has UNIT MAGNITUDE
//! everywhere (random phase from mix64) with conjugate symmetry enforced so
//! the atom is real: S[(-kx mod W, -ky mod H)] = conj(S[kx, ky]);
//! self-conjugate bins (DC + Nyquist combos) = ±1 deterministic from seed.
//! Consequences (tested in tests/core.rs):
//!   - ||atom|| == 1 (Parseval, 1/d-normalized inverse) within 1e-3.
//!   - unbind(bind(a, x), a) recovers x within 1e-3 rel (EXACT inverse
//!     property of unit-magnitude spectra).
//!   - distinct seeds -> |similarity| small (near-orthogonal directions).
//! Determinism: phase(kx,ky) = f(seed, ky*W+kx) via mix64 ONLY. No rand.

use crate::fft::Fft2;
use crate::{mix64, C32, FieldConfig, Slice};

/// Deterministic unit-magnitude-spectrum random-phase atom. state = f(seed).
pub fn seeded_atom(f: &Fft2, cfg: FieldConfig, seed: u64) -> Slice {
    debug_assert_eq!(f.cfg, cfg);
    let (w, h) = (cfg.width, cfg.height);
    let d = cfg.d();
    let mut spec = vec![C32::ZERO; d];
    let mut done = vec![false; d];
    for ky in 0..h {
        for kx in 0..w {
            let i = ky * w + kx;
            if done[i] {
                continue;
            }
            let mkx = (w - kx) % w;
            let mky = (h - ky) % h;
            let j = mky * w + mkx;
            if i == j {
                // self-conjugate bin (DC / Nyquist combo): must stay real.
                let bit = mix64(seed ^ mix64(i as u64)) & 1;
                let val = if bit == 0 { 1.0 } else { -1.0 };
                spec[i] = C32::new(val, 0.0);
                done[i] = true;
            } else {
                let m = mix64(seed ^ mix64(i as u64));
                let ang = (m as f64 / u64::MAX as f64) * 2.0 * std::f64::consts::PI;
                let (s, c) = ang.sin_cos();
                let val = C32::new(c as f32, s as f32);
                spec[i] = val;
                spec[j] = val.conj();
                done[i] = true;
                done[j] = true;
            }
        }
    }
    let data = f.inverse(&spec);
    Slice::from_data(cfg, data)
}
