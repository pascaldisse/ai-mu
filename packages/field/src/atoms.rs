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
use crate::{FieldConfig, Slice};

/// Deterministic unit-magnitude-spectrum random-phase atom. state = f(seed).
pub fn seeded_atom(_f: &Fft2, _cfg: FieldConfig, _seed: u64) -> Slice {
    // W1 implements per module doc. todo!() until then.
    todo!("W1: unit-magnitude random-phase spectrum atom, conjugate-symmetric")
}
