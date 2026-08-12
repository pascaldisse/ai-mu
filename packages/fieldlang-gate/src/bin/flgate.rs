//! ============================================================
//! GATE TOOL — NOT THE COMPILER. See lib.rs banner.
//! flgate = acceptance oracle driver:
//!   flgate ref    <in.fld> <out.fldj>  — reference .fldj via FROZEN Rust
//!                                        journal writer (byte oracle)
//!   flgate digest <in.fldj>            — Journal::read_all + World::replay,
//!                                        print fnv1a-family world digest
//! ============================================================

use field::journal::Journal;
use field::world::World;
use std::path::Path;
use std::process::exit;

fn usage() -> ! {
    eprintln!("flgate (GATE TOOL — not the compiler)");
    eprintln!("  flgate ref    <in.fld> <out.fldj>   reference .fldj via frozen journal");
    eprintln!("  flgate digest <in.fldj>             replay + print world digest");
    exit(2);
}

fn cmd_ref(src_path: &str, out_path: &str) {
    let src = std::fs::read_to_string(src_path).unwrap_or_else(|e| {
        eprintln!("read {}: {}", src_path, e);
        exit(1);
    });
    let prog = fieldlang_gate::parse(&src).unwrap_or_else(|e| {
        eprintln!("{}: {}", src_path, e);
        exit(1);
    });
    let path = Path::new(out_path);
    let mut j = Journal::create(path, prog.cfg, &prog.params, prog.n_slots).unwrap_or_else(|e| {
        eprintln!("create {}: {}", out_path, e);
        exit(1);
    });
    for op in &prog.ops {
        j.append(op);
    }
    j.flush().unwrap_or_else(|e| {
        eprintln!("flush {}: {}", out_path, e);
        exit(1);
    });
    println!(
        "ref {} -> {} ({}x{} slots={} seed={} ops={})",
        src_path,
        out_path,
        prog.cfg.width,
        prog.cfg.height,
        prog.n_slots,
        prog.params.seed,
        prog.ops.len()
    );
}

fn cmd_digest(journal_path: &str) {
    let (cfg, params, n_slots, ops) =
        Journal::read_all(Path::new(journal_path)).unwrap_or_else(|e| {
            eprintln!("read_all {}: {}", journal_path, e);
            exit(1);
        });
    let world = World::replay(cfg, params, n_slots, &ops);
    println!(
        "{} cfg={}x{} slots={} seed={} ops={} steps={} digest={:016x}",
        journal_path,
        cfg.width,
        cfg.height,
        n_slots,
        params.seed,
        ops.len(),
        world.step_index,
        world.digest()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_, cmd, a, b] if cmd == "ref" => cmd_ref(a, b),
        [_, cmd, a] if cmd == "digest" => cmd_digest(a),
        _ => usage(),
    }
}
