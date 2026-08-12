//! speak-back s1 — TRUE inverse of sealed speak-in (cosmo-speak@c5ef614e).
//!
//! speak-in: word → 3 channels (fnv1a+mix64) → 8×8 grid cell center →
//! Gaussian excite, amp signature ∈ [0.4,1.0].
//! → true speak-back = 64 cell-energy readout → word-signature correlation
//! → argmax. O(W·H + |L|·64). No dense-atom cosine, no ML.
//!
//! Mirrors speak.rs constants exactly (CHANNELS=64, GRID=8, WORD_CHANNELS=3,
//! AMP_MIN=0.4). Deterministic: no rand, no clock, f64 fixed-order sums.

use crate::{mix64, seed_unit, Scalar};

pub const CHANNELS: usize = 64;
const GRID: usize = 8;
pub const WORD_CHANNELS: usize = 3;
const AMP_MIN: f32 = 0.4;

// ------------------------------------------------------- speak-in mirror

/// fnv1a-64 (mirror of field crate's fnv1a).
/// AUDIT FIX (kimi, 2026-08-01): field crate's prime is 0x0000_0100_0000_01b3
/// (a NON-STANDARD FNV prime — 0x100000001b3, not standard 0x1000001b3).
/// This mirror previously used the standard prime → every word_seed/channel
/// diverged from real speak-in: s0 decoded the WRONG channels (gates stayed
/// green because synth data used the same broken mirror — self-consistent
/// only). Fixed to the field crate's actual prime; parity is pinned by the
/// test `prime_parity_vs_field_truth` below.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn word_seed(word: &str) -> u64 {
    fnv1a(word.as_bytes())
}

/// word → its semantic channel indices (mirror of speak.rs::word_channels).
pub fn word_channels(word: &str) -> Vec<usize> {
    let h = word_seed(word);
    (0..WORD_CHANNELS)
        .map(|i| (mix64(h ^ mix64(i as u64)) % CHANNELS as u64) as usize)
        .collect()
}

/// Deterministic per-(word, channel) amplitude (mirror of speak.rs).
pub fn channel_amp(seed: u64, channel: usize) -> f32 {
    let u = seed_unit(seed, channel as u64) * 0.5 + 0.5;
    AMP_MIN + (1.0 - AMP_MIN) * u
}

/// channel → (x, y) cell center (mirror of speak.rs::channel_center).
pub fn channel_center(w: usize, h: usize, channel: usize) -> (usize, usize) {
    assert!(channel < CHANNELS);
    let cw = w / GRID;
    let chh = h / GRID;
    assert!(cw >= 1 && chh >= 1, "plane {w}x{h} too small for C64 grid");
    (channel % GRID * cw + cw / 2, channel / GRID * chh + chh / 2)
}

// ------------------------------------------------------- cell readout 觀

/// Plane → 64 cell energies (sum of squares per 8×8 cell, fixed row-major
/// order → deterministic f64 accumulation).
pub fn cell_energies(plane: &[Scalar], w: usize, h: usize) -> [f64; CHANNELS] {
    assert_eq!(plane.len(), w * h);
    let cw = w / GRID;
    let chh = h / GRID;
    let mut e = [0.0f64; CHANNELS];
    for y in 0..h {
        let gy = (y / chh).min(GRID - 1);
        for x in 0..w {
            let gx = (x / cw).min(GRID - 1);
            let v = plane[y * w + x] as f64;
            e[gy * GRID + gx] += v * v;
        }
    }
    e
}

// ------------------------------------------------------- decode

/// Word signature over C64: expected energy pattern — amp² at its channels
/// (channel collisions accumulate, mirroring repeated excites).
pub fn word_signature(word: &str) -> [f64; CHANNELS] {
    let seed = word_seed(word);
    let mut sig = [0.0f64; CHANNELS];
    for ch in word_channels(word) {
        let a = channel_amp(seed, ch) as f64;
        sig[ch] += a * a;
    }
    sig
}

fn norm(v: &[f64; CHANNELS]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cosine of signature vs measured cell energies (fixed order, f64).
pub fn signature_score(sig: &[f64; CHANNELS], energies: &[f64; CHANNELS]) -> f64 {
    let ns = norm(sig);
    let ne = norm(energies);
    if ns == 0.0 || ne == 0.0 {
        return 0.0;
    }
    let dot: f64 = sig.iter().zip(energies).map(|(a, b)| a * b).sum();
    dot / (ns * ne)
}

/// Field plane → top-k (word, score), sorted desc, tie-break lower index.
/// threshold: min score to emit (0.0 = emit all).
pub fn cell_speak_back(
    plane: &[Scalar],
    w: usize,
    h: usize,
    lexicon: &[&str],
    threshold: f64,
    top_k: usize,
) -> Vec<(String, f64)> {
    let e = cell_energies(plane, w, h);
    let mut scored: Vec<(usize, f64)> = lexicon
        .iter()
        .enumerate()
        .map(|(i, wd)| (i, signature_score(&word_signature(wd), &e)))
        .filter(|(_, s)| *s >= threshold)
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(top_k);
    scored.into_iter().map(|(i, s)| (lexicon[i].to_string(), s)).collect()
}

/// Top-1 or None below threshold.
pub fn cell_decode_top1(
    plane: &[Scalar],
    w: usize,
    h: usize,
    lexicon: &[&str],
    threshold: f64,
) -> Option<String> {
    cell_speak_back(plane, w, h, lexicon, threshold, 1)
        .into_iter()
        .next()
        .map(|(w, _)| w)
}

// ------------------------------------------------------- synthetic excite

/// Emulate speak-in's Gaussian strike (σ=2, r=8) at (cx, cy), amp scaled —
/// synthetic stand-in for Op::Excite so gates exercise the REAL spatial
/// form (local blob), not dense atoms.
pub fn gaussian_excite(plane: &mut [Scalar], w: usize, h: usize, cx: usize, cy: usize, amp: f32) {
    const R: isize = 8;
    const SIGMA2: f32 = 2.0 * 2.0;
    for dy in -R..=R {
        for dx in -R..=R {
            let x = cx as isize + dx;
            let y = cy as isize + dy;
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                continue;
            }
            let d2 = (dx * dx + dy * dy) as f32;
            plane[y as usize * w + x as usize] += amp * (-d2 / (2.0 * SIGMA2)).exp();
        }
    }
}

/// Emulate the full speak-in of one word onto a plane (blob per channel).
pub fn synth_speak(plane: &mut [Scalar], w: usize, h: usize, word: &str, scale: f32) {
    let seed = word_seed(word);
    for ch in word_channels(word) {
        let (cx, cy) = channel_center(w, h, ch);
        gaussian_excite(plane, w, h, cx, cy, scale * channel_amp(seed, ch));
    }
}
