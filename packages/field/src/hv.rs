//! Hyperdimensional vectors and the VSA compose ops: `random_vec`, `bind`,
//! `bundle`, `similarity`, `permute`.
//!
//! # Design choice — bind = element-wise bipolar multiplication (MAP-B /
//! Binary Spatter Code), not circular convolution
//!
//! HRR (Plate 1995) binds via circular convolution and unbinds via
//! *approximate* correlation with the involution of the key — exact recovery
//! needs an FFT round-trip (O(d log d)) and the inverse is only approximate
//! for non-unitary keys. MAP (Multiply-Add-Permute, Gayler 1998) / Binary
//! Spatter Codes (Kanerva) bind by an element-wise product instead:
//!
//! - **Self-inverse, exact**: for bipolar `key_i ∈ {-1,+1}`, `key_i * key_i
//!   = 1` for every lane, so `bind(bind(a, key), key) = a` bit-for-bit — no
//!   stored inverse, no "approximately" in the recovery guarantee.
//! - **O(d), embarrassingly parallel, SIMD-friendly** — no FFT, no
//!   permutation state, associative *and* commutative.
//! - Matches the field-as-substrate framing (§ FIELD.md "one consistent
//!   vector space"): the same element-wise lane is both the compose op and
//!   the storage unit, no transform between representation and disk layout.
//!
//! Trade-off (recorded, not hidden): circular convolution self-binds
//! `bind(a, a)` to a smeared vector distinct from `a`, which HRR literature
//! sometimes uses for "distance" encoding; MAP-B has no such property. V0
//! does not need it — recorded here so a later V1 knows what was traded away.

use bytemuck::cast_slice;
use seed::hash_seq;

/// Fixed hypervector dimensionality for this substrate (V0).
pub const DIM: usize = 8192;

/// A deterministic bipolar hypervector, `hash(seed, index)` in every lane.
///
/// § ENTROPY.md "THERE IS NO RANDOMNESS": the vector is `f(seed)`, not a
/// rolled die — same seed, same bytes, forever, on every platform (the
/// SplitMix64 finalizer in `seed::hash_seq` is integer-only, no
/// platform-dependent float rounding).
///
/// Each lane is the low bit of an independently-keyed avalanche hash: after
/// SplitMix64's finalizing mix, every output bit is uniform and
/// pairwise-independent of the others, so taking bit 0 directly is unbiased
/// and avoids the truncation/rounding a `f32` threshold would introduce at
/// the boundary.
pub fn random_vec(field_seed: u64) -> Vec<f32> {
    (0..DIM as u64)
        .map(|i| {
            let bits = hash_seq(field_seed, &[i]);
            if bits & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Bind two vectors by element-wise (Hadamard) product.
///
/// Self-inverse for bipolar operands: `bind(bind(a, k), k) == a` exactly
/// when `k` is bipolar (every lane ±1), since `k_i * k_i = 1`. `unbind` is
/// therefore just `bind` again — see [`unbind`].
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "bind: dimension mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// Unbind: identical operation to [`bind`] — the self-inverse property of
/// bipolar element-wise multiplication means there is no separate inverse
/// operator to implement.
pub fn unbind(bound: &[f32], key: &[f32]) -> Vec<f32> {
    bind(bound, key)
}

/// Superpose (bundle) a set of vectors: element-wise sum, then L2-renormalize.
///
/// Renormalizing keeps composed vectors on a comparable scale for cosine
/// similarity regardless of how many items were bundled together. An empty
/// input, or a set that cancels to the zero vector, yields the zero vector
/// (similarity against it is defined as 0.0 — see [`similarity`]).
///
/// # Panics
/// If any two vectors differ in length, or any is not [`DIM`]-long... no —
/// `bundle` accepts any consistent length; callers pick the dimension.
pub fn bundle(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return vec![0.0; DIM];
    }
    let dim = vecs[0].len();
    let mut sum = vec![0.0f32; dim];
    for v in vecs {
        assert_eq!(v.len(), dim, "bundle: dimension mismatch");
        for (s, x) in sum.iter_mut().zip(v) {
            *s += x;
        }
    }
    renormalize(&mut sum);
    sum
}

fn renormalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity in `[-1, 1]`. `0.0` if either vector has zero norm.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "similarity: dimension mismatch");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Cyclic-shift a vector by `shift` lanes (a role/position compose op, e.g.
/// for encoding sequences: `bind(permute(item, position), role)`). Provided
/// alongside bind/bundle as a third compose primitive; V0's required tests
/// exercise bind/unbind, not this.
pub fn permute(v: &[f32], shift: usize) -> Vec<f32> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let shift = shift % n;
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&v[n - shift..]);
    out.extend_from_slice(&v[..n - shift]);
    out
}

/// Cast a `[f32]` slice to raw bytes for the mmap slab (zero-copy).
pub(crate) fn as_bytes(v: &[f32]) -> &[u8] {
    cast_slice(v)
}
