//! N×d RAM-RESIDENT store. FROZEN + WORKING minimal (archon-written).
//! The field is NOT one vector (capacity: SNR ~ sqrt(d/K)) — it is N slots.
//! Adversary law: hot set RESIDENT in RAM; SSD = cold journal ONLY; NO
//! mmap demand-paging inside a frame. This struct is plain Vec<f32> — resident
//! by construction. W3 hardens: probe top-k perf, tests, docs; semantics frozen.
//!
//! Slot convention (world.rs): slot 0 = plane cur, slot 1 = plane prev,
//! slots 2.. = free. probe() scans slots 2.. only.

use crate::{FieldConfig, Scalar, Slice};
use std::cell::RefCell;

/// Reserved slots: 0 = wave cur, 1 = wave prev.
pub const RESERVED_SLOTS: usize = 2;

/// Squared L2 norm with the EXACT accumulation of ops::similarity (f64,
/// sequential, `(*x as f64) * (*x as f64)` per element) so cached norms keep
/// probe results bit-identical to recomputing them in the similarity loop.
#[inline]
fn sqnorm_f64(row: &[Scalar]) -> f64 {
    let mut na = 0.0f64;
    for x in row {
        na += (*x as f64) * (*x as f64);
    }
    na
}

pub struct Store {
    pub cfg: FieldConfig,
    pub n_slots: usize,
    data: Vec<Scalar>, // row-major N×d, resident
    // PERF (field/perf): per-slot cached squared norms for probe(). Values
    // bit-identical to the na accumulator in ops::similarity. write()
    // refreshes eagerly; row_mut() marks the slot stale (caller mutates
    // behind our back) and probe() recomputes lazily. Interior mutability
    // keeps probe() &self per the frozen signature.
    norms: RefCell<Vec<f64>>,
    stale: RefCell<Vec<bool>>,
}

impl Store {
    pub fn new(cfg: FieldConfig, n_slots: usize) -> Self {
        assert!(n_slots >= RESERVED_SLOTS);
        Store {
            cfg,
            n_slots,
            data: vec![0.0; n_slots * cfg.d()],
            norms: RefCell::new(vec![0.0; n_slots]), // zero rows -> norm 0.0 exactly
            stale: RefCell::new(vec![false; n_slots]),
        }
    }

    #[inline]
    pub fn row(&self, slot: usize) -> &[Scalar] {
        let d = self.cfg.d();
        &self.data[slot * d..(slot + 1) * d]
    }

    #[inline]
    pub fn row_mut(&mut self, slot: usize) -> &mut [Scalar] {
        let d = self.cfg.d();
        self.stale.borrow_mut()[slot] = true; // norm cache invalidated
        &mut self.data[slot * d..(slot + 1) * d]
    }

    pub fn write(&mut self, slot: usize, s: &Slice) {
        assert_eq!(s.cfg, self.cfg);
        self.row_mut(slot).copy_from_slice(&s.data);
        // Eager norm refresh (same value lazy recompute would produce).
        self.norms.borrow_mut()[slot] = sqnorm_f64(&s.data);
        self.stale.borrow_mut()[slot] = false;
    }

    pub fn read(&self, slot: usize) -> Slice {
        Slice::from_data(self.cfg, self.row(slot).to_vec())
    }

    /// Content-addressed probe = attention over slots. Cosine, top_k results
    /// sorted desc, DETERMINISTIC tie-break: lower slot index first.
    /// Scans slots RESERVED_SLOTS.. only.
    ///
    /// PERF (field/perf): bit-exact rewrite of the naive scan. Old path did
    /// read(i) (a d-sized copy) + full similarity() per slot — ~80 ms/frame
    /// at N=4096,d=16K. Now: no copies, key norm hoisted once per call, row
    /// norms served from the cache, and the similarity() zero-norm guard
    /// (`na==0.0 || nb==0.0 -> 0.0`) applied as an early-out. Every f64
    /// accumulation keeps the exact order/precision of ops::similarity, so
    /// returned sims are bit-identical to the old implementation.
    pub fn probe(&self, key: &Slice, top_k: usize) -> Vec<(usize, f32)> {
        // Old code asserted cfg equality inside similarity() on the FIRST
        // scanned slot; preserve that (and only that) panic condition.
        if self.n_slots > RESERVED_SLOTS {
            assert_eq!(key.cfg, self.cfg);
        }
        // Key squared norm hoisted out of the slot loop (same f64 sequential
        // accumulation as similarity's nb -> same bits).
        let nb = sqnorm_f64(&key.data);
        let mut norms = self.norms.borrow_mut();
        let mut stale = self.stale.borrow_mut();
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.n_slots - RESERVED_SLOTS);
        for i in RESERVED_SLOTS..self.n_slots {
            let row = self.row(i);
            if stale[i] {
                norms[i] = sqnorm_f64(row);
                stale[i] = false;
            }
            let na = norms[i];
            // similarity() guard first — bit-exact early-out for zero rows.
            let sim = if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                let mut dot = 0.0f64;
                for (x, y) in row.iter().zip(&key.data) {
                    dot += (*x as f64) * (*y as f64);
                }
                (dot / (na.sqrt() * nb.sqrt())) as f32
            };
            scored.push((i, sim));
        }
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
