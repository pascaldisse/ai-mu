//! WORLD = the composed one-field state. OWNED BY W6. SIGNATURES FROZEN.
//! World = Store (N×d resident) + fixed WaveKernel + step_index. The plane
//! LIVES IN the store: slot 0 = cur, slot 1 = prev (store::RESERVED_SLOTS).
//! tick() = ONE bind by the fixed kernel (the 120fps hot path: no alloc
//! beyond scratch, no I/O, no journal flush). apply(Op) = the ONLY mutation
//! door — journal replay is `for op { world.apply(op) }` = bit-exact.

use crate::atoms;
use crate::fft::Fft2;
use crate::journal::Op;
use crate::ops;
use crate::plane;
use crate::plane::{WaveKernel, WaveParams};
use crate::store::Store;
use crate::{FieldConfig, Slice};

/// Recoverable op-application failure. A malformed journal (e.g. a `寫`/WriteRaw
/// whose payload length != d = w*h) is INPUT, not a bug — replay must return it,
/// never panic. Producer side rejects it too (fieldc emit.s, code -5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyError {
    /// WriteRaw payload length mismatch: got vs required d.
    WriteRawLen { slot: u32, got: usize, want: usize },
}

impl core::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ApplyError::WriteRawLen { slot, got, want } => write!(
                f,
                "WriteRaw slot {slot}: payload is {got} bits, must be d = {want} bits"
            ),
        }
    }
}

impl std::error::Error for ApplyError {}

pub struct World {
    pub cfg: FieldConfig,
    pub params: WaveParams,
    pub fft: Fft2,
    pub kernel: WaveKernel,
    pub store: Store,
    pub step_index: u64,
}

impl World {
    pub fn new(cfg: FieldConfig, params: WaveParams, n_slots: usize) -> Self {
        let fft = Fft2::new(cfg);
        let kernel = WaveKernel::new(cfg, &params);
        let store = Store::new(cfg, n_slots);
        World { cfg, params, fft, kernel, store, step_index: 0 }
    }

    /// The single mutation door. Semantics per journal::Op docs.
    /// Panics on malformed ops; prefer [`World::try_apply`] for untrusted input.
    pub fn apply(&mut self, op: &Op) {
        self.try_apply(op).expect("World::apply on malformed op")
    }

    /// Fallible mutation door: malformed input → Err, never panic.
    pub fn try_apply(&mut self, op: &Op) -> Result<(), ApplyError> {
        match op {
            Op::SeedAtom { slot, seed } => {
                // store[slot] = atoms::seeded_atom(seed)
                let atom = atoms::seeded_atom(&self.fft, self.cfg, *seed);
                self.store.write(*slot as usize, &atom);
            }
            Op::Excite { x, y, amp_bits } => {
                // plane::excite(slot0, slot1, x, y, amp) — read rows, mutate, write back.
                let mut cur = self.store.read(0);
                let mut prev = self.store.read(1);
                plane::excite(
                    &mut cur,
                    &mut prev,
                    *x as usize,
                    *y as usize,
                    f32::from_bits(*amp_bits),
                    self.params.seed,
                );
                self.store.write(0, &cur);
                self.store.write(1, &prev);
            }
            Op::Step { count } => {
                for _ in 0..*count {
                    self.tick();
                }
            }
            Op::Bind { dst, a, b } => {
                // store[dst] = bind(store[a], store[b])
                let sa = self.store.read(*a as usize);
                let sb = self.store.read(*b as usize);
                let out = ops::bind(&self.fft, &sa, &sb);
                self.store.write(*dst as usize, &out);
            }
            Op::Bundle { dst, srcs } => {
                // store[dst] = bundle(store[srcs...]) — superposition
                let rows: Vec<Slice> = srcs.iter().map(|s| self.store.read(*s as usize)).collect();
                let refs: Vec<&Slice> = rows.iter().collect();
                let out = ops::bundle(&refs);
                self.store.write(*dst as usize, &out);
            }
            Op::WriteRaw { slot, data_bits } => {
                // store[slot] = raw f32-bits row
                if data_bits.len() != self.cfg.d() {
                    return Err(ApplyError::WriteRawLen {
                        slot: *slot,
                        got: data_bits.len(),
                        want: self.cfg.d(),
                    });
                }
                let row = self.store.row_mut(*slot as usize);
                for (dst, bits) in row.iter_mut().zip(data_bits) {
                    *dst = f32::from_bits(*bits);
                }
            }
        }
        Ok(())
    }

    /// One field tick = wave_step on slots 0/1. HOT PATH: 120fps floor.
    /// slot1 <- old cur, slot0 <- next, step_index += 1. Zero I/O, zero
    /// journal calls. NOTE: allocs here are the 3 d-sized Slices the frozen
    /// wave_step(&Slice,&Slice)->Slice signature forces (cur/prev inputs +
    /// next output); no scratch slot exists in the frozen World struct —
    /// post-merge W2 can fuse the copies inside wave_step if the floor needs it.
    pub fn tick(&mut self) {
        let cur = self.store.read(0);
        let prev = self.store.read(1);
        let next = plane::wave_step(&self.fft, &self.kernel, &cur, &prev);
        self.store.write(1, &cur); // prev <- old cur
        self.store.write(0, &next); // cur <- next
        self.step_index += 1;
    }

    /// Bit-exact replay: new world, apply all ops. Determinism gate target.
    pub fn replay(cfg: FieldConfig, params: WaveParams, n_slots: usize, ops: &[Op]) -> Self {
        World::try_replay(cfg, params, n_slots, ops).expect("World::replay on malformed journal")
    }

    /// Fallible replay: a malformed journal returns Err instead of panicking.
    pub fn try_replay(
        cfg: FieldConfig,
        params: WaveParams,
        n_slots: usize,
        ops: &[Op],
    ) -> Result<Self, ApplyError> {
        let mut w = World::new(cfg, params, n_slots);
        for op in ops {
            w.try_apply(op)?;
        }
        Ok(w)
    }

    pub fn cur(&self) -> Slice {
        self.store.read(0)
    }

    /// Bit-exact digest: store digest ^ step_index mix — determinism gates.
    pub fn digest(&self) -> u64 {
        self.store.digest() ^ crate::mix64(self.step_index)
    }

    /// Render = the probe of the spatial marginal (plane::encode_png of cur).
    pub fn render_png(&self) -> Vec<u8> {
        crate::plane::encode_png(&self.cur(), self.params.range)
    }
}
