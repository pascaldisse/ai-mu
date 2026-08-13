//! Task 2 (FIELD CORE WIRING) — FieldRuntime tests. DETERMINISTIC ONLY.
//! Temp files under CARGO_TARGET_TMPDIR (target/), deterministic names.

use field::atoms;
use field::plane::WaveParams;
use field::runtime::FieldRuntime;
use field::store::RESERVED_SLOTS;
use field::FieldConfig;
use std::path::{Path, PathBuf};

const TMP: &str = env!("CARGO_TARGET_TMPDIR");

fn fresh_dir(name: &str) -> PathBuf {
    let d = Path::new(TMP).join(name);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn jpath(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    let _ = std::fs::remove_file(&p);
    p
}

fn cfg() -> FieldConfig {
    FieldConfig::new(32, 32) // small dims: naive-FFT oracle keeps tests fast
}

#[test]
fn tick_advances_and_journals_ram_only() {
    let dir = fresh_dir("rt_tick");
    let path = jpath(&dir, "rt_tick.fldj");
    let mut rt = FieldRuntime::new(cfg(), WaveParams::default(), 8, &path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 48, "create writes header only");
    assert_eq!(rt.world.step_index, 0);
    for _ in 0..5 {
        rt.tick();
    }
    assert_eq!(rt.world.step_index, 5);
    // tick() never touches disk (journal.append buffers in RAM only)
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 48, "tick must not touch disk");
    rt.flush().unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 48, "flush must grow the file");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn excite_mutates_and_probe_skips_reserved_slots() {
    let dir = fresh_dir("rt_excite");
    let path = jpath(&dir, "rt_excite.fldj");
    let mut rt = FieldRuntime::new(cfg(), WaveParams::default(), 8, &path).unwrap();
    let before = rt.digest();
    rt.excite(4, 4, 0.3);
    assert_ne!(rt.digest(), before, "excite must change world state");

    let key = atoms::seeded_atom(&rt.world.fft, cfg(), 7);
    let hits = rt.probe(&key, 4);
    for (slot, _) in &hits {
        assert!(*slot >= RESERVED_SLOTS, "probe must not return reserved plane slots");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn replay_reproduces_live_session_bit_exact() {
    let dir = fresh_dir("rt_replay");
    let src = jpath(&dir, "rt_live.fldj");
    let resume = jpath(&dir, "rt_resume.fldj");

    let mut live = FieldRuntime::new(cfg(), WaveParams::default(), 8, &src).unwrap();
    live.excite(3, 5, 0.2);
    for _ in 0..12 {
        live.tick();
    }
    live.excite(10, 10, -0.1);
    for _ in 0..8 {
        live.tick();
    }
    let live_digest = live.digest();
    let live_step = live.world.step_index;
    live.flush().unwrap();

    let replayed = FieldRuntime::replay(&src, &resume).unwrap();
    assert_eq!(replayed.world.step_index, live_step, "replay step_index must match live");
    assert_eq!(replayed.digest(), live_digest, "replay digest must be bit-exact vs live");
    assert_eq!(replayed.cur().digest(), live.cur().digest(), "replay cur() bit-exact vs live");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&resume);
}

#[test]
fn replay_is_deterministic_across_two_runs() {
    let dir = fresh_dir("rt_replay_determ");
    let src = jpath(&dir, "rt_src.fldj");
    let r1 = jpath(&dir, "rt_r1.fldj");
    let r2 = jpath(&dir, "rt_r2.fldj");

    let mut live = FieldRuntime::new(cfg(), WaveParams::default(), 8, &src).unwrap();
    for i in 0..20 {
        if i % 5 == 0 {
            live.excite(i % cfg().width, (i * 3) % cfg().height, 0.05 * (i as f32 + 1.0));
        }
        live.tick();
    }
    live.flush().unwrap();

    let a = FieldRuntime::replay(&src, &r1).unwrap();
    let b = FieldRuntime::replay(&src, &r2).unwrap();
    assert_eq!(a.digest(), b.digest(), "replay must be bit-exact deterministic");
    assert_eq!(a.world.step_index, b.world.step_index);

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&r1);
    let _ = std::fs::remove_file(&r2);
}
