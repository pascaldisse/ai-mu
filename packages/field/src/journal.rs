//! SSD COLD JOURNAL. OWNED BY W3. SIGNATURES FROZEN.
//! Adversary law: SSD = cold backing ONLY. Append-only op log; replay =
//! bit-exact state reconstruction (ENTROPY.md: seed + journal = the world).
//! NO write/flush inside the frame hot path: append() buffers in RAM,
//! flush() is explicit, called off-cadence.
//! Binary format v1 (LE): magic "FLDJ" u32, version u32=1, w u32, h u32,
//! params (c,dt,damping,dx as f32-bits u32 · seed u64 · range f32-bits u32),
//! n_slots u32, then ops: tag u8 + payload. All f32 AS BIT PATTERNS (u32)
//! -> bit-exact roundtrip. Truncated trailing op on read = io error.

use crate::plane::WaveParams;
use crate::FieldConfig;
use std::io;
use std::path::Path;

/// The op alphabet. Everything that mutates the world goes through this.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// store[slot] = atoms::seeded_atom(seed)          tag 1
    SeedAtom { slot: u32, seed: u64 },
    /// plane::excite(slot0, slot1, x, y, amp)          tag 2
    Excite { x: u32, y: u32, amp_bits: u32 },
    /// count wave ticks                                 tag 3
    Step { count: u32 },
    /// store[dst] = bind(store[a], store[b])           tag 4
    Bind { dst: u32, a: u32, b: u32 },
    /// store[dst] = bundle(store[srcs...])             tag 5
    Bundle { dst: u32, srcs: Vec<u32> },
    /// store[slot] = raw f32-bits row                  tag 6
    WriteRaw { slot: u32, data_bits: Vec<u32> },
}

pub struct Journal {
    buf: Vec<u8>,
    path: std::path::PathBuf,
}

impl Journal {
    /// Create file, write header immediately, buffer ops thereafter.
    pub fn create(
        path: &Path,
        cfg: FieldConfig,
        params: &WaveParams,
        n_slots: usize,
    ) -> io::Result<Self> {
        let _ = (path, cfg, params, n_slots);
        todo!("W3: header write per format v1")
    }

    /// Buffer one op in RAM. NEVER touches disk (hot-path safe).
    pub fn append(&mut self, op: &Op) {
        let _ = op;
        todo!("W3: serialize into self.buf")
    }

    /// Flush buffered ops to disk. Off-cadence only.
    pub fn flush(&mut self) -> io::Result<()> {
        todo!("W3: append buf to file, clear buf")
    }

    /// Read a journal back: (cfg, params, n_slots, ops). Bit-exact.
    pub fn read_all(path: &Path) -> io::Result<(FieldConfig, WaveParams, usize, Vec<Op>)> {
        let _ = path;
        todo!("W3: parse header + ops, error on truncation")
    }
}
