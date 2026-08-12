//! W6 world tests — apply/tick/replay digests. Owned by W6.
//! COMPILE gate now (world.rs internals are live); runtime green post-merge
//! when atoms.rs/plane.rs bodies land (they are todo!() until then).

use field::atoms;
use field::journal::Op;
use field::ops;
use field::plane::WaveParams;
use field::store::RESERVED_SLOTS;
use field::world::World;
use field::{FieldConfig, Slice};

fn cfg() -> FieldConfig {
    FieldConfig::new(32, 32) // small dims: naive-FFT oracle keeps tests fast
}

fn world() -> World {
    World::new(cfg(), WaveParams::default(), 16)
}

fn atom(w: &World, seed: u64) -> Slice {
    atoms::seeded_atom(&w.fft, w.cfg, seed)
}

// ---------------------------------------------------------------- apply

#[test]
fn seed_atom_writes_slot() {
    let mut w = world();
    assert_eq!(w.step_index, 0);
    w.apply(&Op::SeedAtom { slot: 2, seed: 7 });
    let s = w.store.read(2);
    assert_ne!(s.digest(), Slice::zeros(cfg()).digest(), "atom must be nonzero");
    assert!(s.data.iter().all(|v| v.is_finite()));
    assert_eq!(w.store.read(3).digest(), Slice::zeros(cfg()).digest(), "other slots untouched");
}

#[test]
fn write_raw_roundtrip_is_bit_exact() {
    let mut w = world();
    // deterministic pattern: f32 bit patterns of the index
    let bits: Vec<u32> = (0..w.cfg.d() as u32).map(f32::from_bits).map(f32::to_bits).collect();
    w.apply(&Op::WriteRaw { slot: 5, data_bits: bits.clone() });
    let row = w.store.row(5);
    assert_eq!(row.len(), w.cfg.d());
    for (i, &v) in row.iter().enumerate() {
        assert_eq!(v.to_bits(), bits[i], "bit-exact write at {i}");
    }
}

#[test]
fn bind_matches_ops_bind() {
    let mut w = world();
    w.apply(&Op::SeedAtom { slot: 2, seed: 1 });
    w.apply(&Op::SeedAtom { slot: 3, seed: 2 });
    w.apply(&Op::Bind { dst: 4, a: 2, b: 3 });
    let a = w.store.read(2);
    let b = w.store.read(3);
    let expect = ops::bind(&w.fft, &a, &b);
    assert_eq!(w.store.read(4).digest(), expect.digest(), "bind dst == ops::bind");
}

#[test]
fn bundle_is_superposition() {
    let mut w = world();
    w.apply(&Op::SeedAtom { slot: 2, seed: 3 });
    w.apply(&Op::SeedAtom { slot: 3, seed: 4 });
    w.apply(&Op::Bundle { dst: 4, srcs: vec![2, 3] });
    let a = w.store.read(2);
    let b = w.store.read(3);
    let expect = ops::bundle(&[&a, &b]);
    assert_eq!(w.store.read(4).digest(), expect.digest(), "bundle dst == ops::bundle");
    // and dst == elementwise sum
    let got = w.store.read(4);
    for i in 0..got.data.len() {
        assert_eq!(got.data[i].to_bits(), (a.data[i] + b.data[i]).to_bits(), "sum at {i}");
    }
}

#[test]
fn excite_mutates_plane_and_leaves_free_slots() {
    let mut w = world();
    let before = w.store.digest();
    w.apply(&Op::Excite { x: 0, y: 0, amp_bits: 0.1f32.to_bits() });
    assert_ne!(w.store.digest(), before, "excite must change the plane");
    // free slots 2.. still zero
    for slot in RESERVED_SLOTS..w.store.n_slots {
        assert_eq!(w.store.read(slot).digest(), Slice::zeros(cfg()).digest());
    }
}

// ---------------------------------------------------------------- tick

#[test]
fn tick_swaps_slots_and_advances_index() {
    let mut w = world();
    w.apply(&Op::Excite { x: 0, y: 0, amp_bits: 0.1f32.to_bits() });
    let cur0 = w.store.read(0).digest();
    let prev0 = w.store.read(1).digest();
    w.tick();
    assert_eq!(w.step_index, 1);
    // slot1 <- old cur (bit-exact swap)
    assert_eq!(w.store.read(1).digest(), cur0, "prev must become old cur");
    // slot0 changed by the wave step
    assert_ne!(w.store.read(0).digest(), cur0);
    let _ = prev0; // old prev was consumed by wave_step; only cur0 is reproducible here
}

#[test]
fn step_op_counts_ticks() {
    let mut w = world();
    w.apply(&Op::Step { count: 5 });
    assert_eq!(w.step_index, 5);
    w.apply(&Op::Step { count: 3 });
    assert_eq!(w.step_index, 8);
}

// ---------------------------------------------------------------- replay + digest

#[test]
fn replay_matches_live_apply() {
    let ops = vec![
        Op::SeedAtom { slot: 2, seed: 11 },
        Op::Excite { x: 4, y: 4, amp_bits: (-0.2f32).to_bits() },
        Op::Step { count: 3 },
        Op::Bind { dst: 5, a: 2, b: 2 },
        Op::Bundle { dst: 6, srcs: vec![2, 5] },
        Op::Step { count: 2 },
    ];
    let mut live = World::new(cfg(), WaveParams::default(), 16);
    for op in &ops {
        live.apply(op);
    }
    let replay = World::replay(cfg(), WaveParams::default(), 16, &ops);
    assert_eq!(live.step_index, replay.step_index);
    assert_eq!(live.digest(), replay.digest(), "live == replay (bit-exact)");
}

#[test]
fn replay_is_deterministic() {
    let ops = vec![
        Op::SeedAtom { slot: 2, seed: 42 },
        Op::Excite { x: 8, y: 8, amp_bits: 0.05f32.to_bits() },
        Op::Step { count: 10 },
        Op::WriteRaw { slot: 3, data_bits: (0..(cfg().d() as u32)).map(f32::from_bits).map(f32::to_bits).collect() },
    ];
    let r1 = World::replay(cfg(), WaveParams::default(), 16, &ops);
    let r2 = World::replay(cfg(), WaveParams::default(), 16, &ops);
    assert_eq!(r1.digest(), r2.digest(), "replay must be bit-exact deterministic");
}

#[test]
fn digest_mixes_step_index() {
    // same store, same step_index via different op shapes -> same digest
    let a = World::replay(cfg(), WaveParams::default(), 16, &[
        Op::SeedAtom { slot: 2, seed: 9 },
        Op::Step { count: 2 },
    ]);
    let b = World::replay(cfg(), WaveParams::default(), 16, &[
        Op::SeedAtom { slot: 2, seed: 9 },
        Op::Step { count: 1 },
        Op::Step { count: 1 },
    ]);
    assert_eq!(a.step_index, 2);
    assert_eq!(a.digest(), b.digest(), "digest depends on (store, step_index) only");

    // same store but different step_index -> digest MUST differ (mix64)
    let c = World::replay(cfg(), WaveParams::default(), 16, &[Op::SeedAtom { slot: 2, seed: 9 }]);
    assert_eq!(c.store.digest(), a.store.digest(), "stores identical");
    assert_ne!(c.digest(), a.digest(), "digest must mix step_index");
}

// ---------------------------------------------------------------- probe

#[test]
fn probe_scans_free_slots_only_and_is_deterministic() {
    let mut w = world();
    w.apply(&Op::SeedAtom { slot: 2, seed: 5 });
    w.apply(&Op::SeedAtom { slot: 3, seed: 6 });
    let key = atom(&w, 99);
    let r1 = w.store.probe(&key, 4);
    let r2 = w.store.probe(&key, 4);
    assert_eq!(r1, r2, "probe deterministic");
    assert!(!r1.is_empty());
    assert!(r1.len() <= 4);
    for (slot, _score) in &r1 {
        assert!(*slot >= RESERVED_SLOTS, "probe must not scan reserved plane slots");
    }
}
