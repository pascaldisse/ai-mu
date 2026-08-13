//! speak-back s0 — inverse of speak-in.
//!
//! Synthetic field state + semantic anchor lexicon → nearest-anchor word.
//! Deterministic, zero external deps, zero rand, zero I/O.

use std::collections::HashMap;

pub mod cell_decode;

pub type Scalar = f32;

// ---------------------------------------------------------------- determinism

#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub fn seed_unit(seed: u64, tag: u64) -> Scalar {
    let m = mix64(seed ^ mix64(tag));
    ((m >> 11) as f64 / (1u64 << 53) as f64) as Scalar * 2.0 - 1.0
}

// ---------------------------------------------------------------- field slice

#[derive(Clone, Debug, PartialEq)]
pub struct Slice {
    pub data: Vec<Scalar>,
}

impl Slice {
    pub fn zeros(d: usize) -> Self {
        Slice { data: vec![0.0; d] }
    }

    pub fn from_data(data: Vec<Scalar>) -> Self {
        Slice { data }
    }

    pub fn d(&self) -> usize {
        self.data.len()
    }

    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt()
    }

    pub fn cosine(&self, other: &Slice) -> Scalar {
        assert_eq!(self.d(), other.d());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in self.data.iter().zip(&other.data) {
            dot += *x as f64 * *y as f64;
            na += *x as f64 * *x as f64;
            nb += *y as f64 * *y as f64;
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as Scalar
    }
}

// ---------------------------------------------------------------- semantic atom

/// Deterministic unit-norm hypervector seeded from `seed`.
pub fn atom(d: usize, seed: u64) -> Slice {
    let mut data: Vec<Scalar> = (0..d as u64).map(|i| seed_unit(seed, i)).collect();
    let n = data.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        for v in &mut data {
            *v = (*v as f64 * inv) as Scalar;
        }
    }
    Slice::from_data(data)
}

// ---------------------------------------------------------------- lexicon

pub struct Lexicon {
    pub d: usize,
    pub words: Vec<String>,
    pub anchors: Vec<Slice>,
    pub threshold: Scalar,
}

impl Lexicon {
    pub fn new(d: usize, words: &[&str], seed_base: u64, threshold: Scalar) -> Self {
        let mut anchors = Vec::with_capacity(words.len());
        for (i, _) in words.iter().enumerate() {
            anchors.push(atom(d, seed_base.wrapping_add(i as u64)));
        }
        Lexicon {
            d,
            words: words.iter().map(|w| w.to_string()).collect(),
            anchors,
            threshold,
        }
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn anchor(&self, word: &str) -> Option<&Slice> {
        self.words.iter().position(|w| w == word).map(|i| &self.anchors[i])
    }
}

// ---------------------------------------------------------------- speak-back

/// Excite a semantic channel: state += amp * anchor(word).
pub fn speak(state: &mut Slice, lex: &Lexicon, word: &str, amp: Scalar) {
    let a = lex.anchor(word).expect("word not in lexicon");
    assert_eq!(state.d(), a.d());
    for (s, v) in state.data.iter_mut().zip(&a.data) {
        *s += amp * *v;
    }
}

/// Decode field state to top-k (word, score) pairs, sorted desc.
/// Tie-break: lower word index first.
pub fn speak_back(state: &Slice, lex: &Lexicon, top_k: usize) -> Vec<(String, Scalar)> {
    let mut scored: Vec<(usize, Scalar)> = lex
        .anchors
        .iter()
        .enumerate()
        .map(|(i, a)| (i, state.cosine(a)))
        .filter(|(_, s)| *s >= lex.threshold)
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(top_k);
    scored.into_iter().map(|(i, s)| (lex.words[i].clone(), s)).collect()
}

/// Decode top-1 word, or None if nothing crosses threshold.
pub fn decode_top1(state: &Slice, lex: &Lexicon) -> Option<String> {
    speak_back(state, lex, 1).into_iter().next().map(|(w, _)| w)
}

// ---------------------------------------------------------------- diagnostics

/// Score all words, threshold ignored.
pub fn score_all(state: &Slice, lex: &Lexicon) -> HashMap<String, Scalar> {
    lex.anchors
        .iter()
        .enumerate()
        .map(|(i, a)| (lex.words[i].clone(), state.cosine(a)))
        .collect()
}
