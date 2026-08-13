//! GATE-ONLY tool (fieldlang-asm integration gate, narigo archon).
//! Usage: cargo run --example replay_fldj -- file.fldj [...]
//! journal read_all -> World::replay -> print fnv1a-based World digest.
use std::path::Path;

fn main() {
    for arg in std::env::args().skip(1) {
        let (cfg, params, n_slots, ops) = field::journal::Journal::read_all(Path::new(&arg))
            .expect("read_all failed");
        let w = field::world::World::replay(cfg, params, n_slots, &ops);
        println!("{} digest={:016x}", arg, w.digest());
    }
}
