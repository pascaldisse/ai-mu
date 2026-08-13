//! FIELD RUNTIME — Task 2 (FIELD CORE WIRING). One clean composition API
//! for an app render loop (120Hz target): wraps World (tick/store/render) +
//! Journal (append/replay) behind ONE struct. Additive only — does not
//! touch the frozen signatures of plane.rs/store.rs/journal.rs/world.rs.
//!
//! Contract:
//!   FieldRuntime::new(cfg, params, n_slots, journal_path) -- fresh world +
//!     fresh journal file (header written immediately, per Journal::create).
//!   tick()            -- ONE wave_step. HOT PATH: no I/O beyond a RAM
//!                         journal append (Journal::append never touches
//!                         disk — see journal.rs). Target well under 8.33ms.
//!   excite(x,y,amp)   -- Gaussian strike via World::apply (the one
//!                         mutation door) + journal record. Off the tick
//!                         cadence in practice, but I/O-free either way.
//!   probe(key, top_k) -- content-addressed read over free slots. No I/O.
//!   flush()           -- EXPLICIT off-cadence disk write of buffered ops.
//!                         Never call this from the hot tick loop.
//!   FieldRuntime::replay(path, resume_journal_path) -- bit-exact
//!     reconstruction: reads the op log, rebuilds World via World::replay
//!     (world.rs's own determinism gate target), then opens a fresh
//!     journal at resume_journal_path so the resumed session can keep
//!     appending. The source journal at `path` is left untouched.
//!
//! Determinism (ENTROPY.md): every mutating call (tick, excite) is ALSO an
//! appended Op, so replay(path) after a live session reproduces cur()/
//! digest() bit-exactly (tests/runtime.rs: live == replay).

use std::io;
use std::path::{Path, PathBuf};

use crate::journal::{Journal, Op};
use crate::plane::WaveParams;
use crate::world::World;
use crate::{FieldConfig, Slice};

pub struct FieldRuntime {
    pub world: World,
    journal: Journal,
    journal_path: PathBuf,
}

impl FieldRuntime {
    /// Fresh runtime: new World (all slots zero) + a fresh journal file at
    /// `journal_path` (created/truncated now, header written immediately).
    pub fn new(
        cfg: FieldConfig,
        params: WaveParams,
        n_slots: usize,
        journal_path: &Path,
    ) -> io::Result<Self> {
        let journal = Journal::create(journal_path, cfg, &params, n_slots)?;
        let world = World::new(cfg, params, n_slots);
        Ok(FieldRuntime { world, journal, journal_path: journal_path.to_path_buf() })
    }

    /// ONE field tick = ONE wave_step (World::tick — slot0/slot1 bind by the
    /// fixed kernel). 120fps-floor hot path: zero disk I/O (journal.append
    /// buffers in RAM only; call flush() off-cadence).
    #[inline]
    pub fn tick(&mut self) {
        self.world.tick();
        self.journal.append(&Op::Step { count: 1 });
    }

    /// Gaussian strike at (x, y). Routed through World::apply (the single
    /// mutation door, world.rs) so live state and journal replay agree by
    /// construction; also appends the op for replay.
    pub fn excite(&mut self, x: usize, y: usize, amplitude: f32) {
        let op = Op::Excite { x: x as u32, y: y as u32, amp_bits: amplitude.to_bits() };
        self.world.apply(&op);
        self.journal.append(&op);
    }

    /// Content-addressed probe over free slots (store::RESERVED_SLOTS..).
    /// No I/O, no journal entry (probe is a read, not a mutation).
    pub fn probe(&self, key: &Slice, top_k: usize) -> Vec<(usize, f32)> {
        self.world.store.probe(key, top_k)
    }

    /// Current plane state (slot 0). Copy — off the hot tick path, for
    /// render/inspection.
    pub fn cur(&self) -> Slice {
        self.world.cur()
    }

    /// Render = grayscale PNG of cur (plane::encode_png, deterministic
    /// quantisation). Off the hot path — call at display cadence, not tick.
    pub fn render_png(&self) -> Vec<u8> {
        self.world.render_png()
    }

    /// Bit-exact digest of world state (store ^ step_index) — determinism
    /// gates / replay-equality checks.
    pub fn digest(&self) -> u64 {
        self.world.digest()
    }

    /// Explicit off-cadence journal flush: buffered ops -> disk. NEVER call
    /// this inside the tick loop.
    pub fn flush(&mut self) -> io::Result<()> {
        self.journal.flush()
    }

    /// Path of the journal this runtime is currently appending to.
    pub fn journal_path(&self) -> &Path {
        &self.journal_path
    }

    /// Bit-exact replay: read the op log at `path` (Journal::read_all),
    /// rebuild World by replaying every op (World::replay), then open a
    /// FRESH journal at `resume_journal_path` for continued appends. The
    /// source journal at `path` is left untouched (read-only access).
    pub fn replay(path: &Path, resume_journal_path: &Path) -> io::Result<Self> {
        let (cfg, params, n_slots, ops) = Journal::read_all(path)?;
        let world = World::replay(cfg, params, n_slots, &ops);
        let journal = Journal::create(resume_journal_path, cfg, &params, n_slots)?;
        Ok(FieldRuntime {
            world,
            journal,
            journal_path: resume_journal_path.to_path_buf(),
        })
    }
}
