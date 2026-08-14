//! GATE-ONLY tool (fieldlang-asm integration gate, narigo archon).
//! Usage: cargo run --example replay_fldj -- file.fldj [...]
//! journal read_all -> World::replay -> print fnv1a-based World digest.
//! FLDJ_REF=1 → Step を `plane::wave_step_reference`(5点 stencil oracle, plane.rs:93-113)で
//! 執行し、slot0 の f32 cell を一行一値で吐く(fieldrun Q20 との parity 実測用・§15)。
use std::path::Path;

fn main() {
    let refmode = std::env::var("FLDJ_REF").ok().as_deref() == Some("1");
    for arg in std::env::args().skip(1) {
        let (cfg, params, n_slots, ops) = field::journal::Journal::read_all(Path::new(&arg))
            .expect("read_all failed");
        if refmode {
            let mut w = field::world::World::new(cfg, params, n_slots);
            for op in &ops {
                match op {
                    field::journal::Op::Step { count } => {
                        for _ in 0..*count {
                            let cur = w.store.read(0);
                            let prev = w.store.read(1);
                            let next =
                                field::plane::wave_step_reference(&w.params, &cur, &prev);
                            w.store.write(1, &cur);
                            w.store.write(0, &next);
                            w.step_index += 1;
                        }
                    }
                    other => w.apply(other),
                }
            }
            for v in w.store.row(0) {
                println!("{:.12e}", v);
            }
            continue;
        }
        let w = field::world::World::replay(cfg, params, n_slots, &ops);
        println!("{} digest={:016x}", arg, w.digest());
    }
}
