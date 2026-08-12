//! WORLD = the composed one-field state. OWNED BY W6. SIGNATURES FROZEN.
//! World = Store (N×d resident) + fixed WaveKernel + step_index. The plane
//! LIVES IN the store: slot 0 = cur, slot 1 = prev (store::RESERVED_SLOTS).
//! tick() = ONE bind by the fixed kernel (the 120fps hot path: no alloc
//! beyond scratch, no I/O, no journal flush). apply(Op) = the ONLY mutation
//! door — journal replay is `for op { world.apply(op) }` = bit-exact.

use crate::fft::Fft2;
use crate::journal::Op;
use crate::plane::{WaveKernel, WaveParams};
use crate::store::Store;
use crate::{FieldConfig, Slice};

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
        let _ = (cfg, params, n_slots);
        todo!("W6: build fft + kernel + zeroed store")
    }

    /// The single mutation door. Semantics per journal::Op docs.
    pub fn apply(&mut self, op: &Op) {
        let _ = op;
        todo!("W6: match op -> atoms/plane/ops/store calls")
    }

    /// One field tick = wave_step on slots 0/1. HOT PATH: 120fps floor.
    pub fn tick(&mut self) {
        todo!("W6: next = wave_step(cur, prev); prev<-cur; cur<-next; step_index+=1")
    }

    /// Bit-exact replay: new world, apply all ops. Determinism gate target.
    pub fn replay(cfg: FieldConfig, params: WaveParams, n_slots: usize, ops: &[Op]) -> Self {
        let _ = (cfg, params, n_slots, ops);
        todo!("W6: World::new + apply each op in order")
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
