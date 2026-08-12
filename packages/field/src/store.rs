//! N×d RAM-RESIDENT store. FROZEN + WORKING minimal (archon-written).
//! The field is NOT one vector (capacity: SNR ~ sqrt(d/K)) — it is N slots.
//! Adversary law: hot set RESIDENT in RAM; SSD = cold journal ONLY; NO
//! mmap demand-paging inside a frame. This struct is plain Vec<f32> — resident
//! by construction. W3 hardens: probe top-k perf, tests, docs; semantics frozen.
//!
//! Slot convention (world.rs): slot 0 = plane cur, slot 1 = plane prev,
//! slots 2.. = free. probe() scans slots 2.. only.

use crate::ops::similarity;
use crate::{FieldConfig, Scalar, Slice};

/// Reserved slots: 0 = wave cur, 1 = wave prev.
pub const RESERVED_SLOTS: usize = 2;

pub struct Store {
    pub cfg: FieldConfig,
    pub n_slots: usize,
    data: Vec<Scalar>, // row-major N×d, resident
}

impl Store {
    pub fn new(cfg: FieldConfig, n_slots: usize) -> Self {
        assert!(n_slots >= RESERVED_SLOTS);
        Store { cfg, n_slots, data: vec![0.0; n_slots * cfg.d()] }
    }

    #[inline]
    pub fn row(&self, slot: usize) -> &[Scalar] {
        let d = self.cfg.d();
        &self.data[slot * d..(slot + 1) * d]
    }

    #[inline]
    pub fn row_mut(&mut self, slot: usize) -> &mut [Scalar] {
        let d = self.cfg.d();
        &mut self.data[slot * d..(slot + 1) * d]
    }

    pub fn write(&mut self, slot: usize, s: &Slice) {
        assert_eq!(s.cfg, self.cfg);
        self.row_mut(slot).copy_from_slice(&s.data);
    }

    pub fn read(&self, slot: usize) -> Slice {
        Slice::from_data(self.cfg, self.row(slot).to_vec())
    }

    /// Content-addressed probe = attention over slots. Cosine, top_k results
    /// sorted desc, DETERMINISTIC tie-break: lower slot index first.
    /// Scans slots RESERVED_SLOTS.. only.
    pub fn probe(&self, key: &Slice, top_k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = (RESERVED_SLOTS..self.n_slots)
            .map(|i| (i, similarity(&self.read(i), key)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        scored.truncate(top_k);
        scored
    }

    /// Bit-exact digest over all rows (f32 bit patterns) — determinism gates.
    pub fn digest(&self) -> u64 {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        crate::fnv1a(&bytes)
    }
}
