//! ULTRADETERMINISM GATES — W5. OWNED BY W5. tests/determinism.rs ONLY.
//! Law (ENTROPY.md): state = f(seed). No rand, no clock. Verifies:
//!   (1) live apply of identical op sequence on two Worlds -> digest bit-equal
//!   (2) journal record -> flush -> read_all -> World::replay -> digest ==
//!       live digest (bit-exact replay, ENTROPY.md core claim)
//!   (3) render_png bytes identical across two independent live runs
//!   (4) digest DIFFERS when seed differs (anti-vacuous: gates can't pass
//!       by both sides being all-zero/no-op)
//!   (5) Slice::digest stable: seeded_atom(same seed) twice -> same digest
//!
//! Fixture: FieldConfig 32x32 (naive-DFT-friendly), n_slots 16.
//! Op sequence: SeedAtom x8 (slots 2..10) + Excite x3 (wave plane) +
//! Step{100} + Bind(dst=10,a=2,b=3) + Bundle(dst=11,srcs=[4,5,6,7]).
//!
//! KNOWN (2026-08-01): world.rs/plane.rs/journal.rs/atoms.rs/fft.rs bodies
//! are todo!() until their owning waves land. This file must COMPILE against
//! the frozen signatures; runtime is UNVERIFIED-until-merge (see report).
//!
//! Journal file: deterministic path under target/ (never /tmp, never rand).

use field::atoms;
use field::fft::Fft2;
use field::journal::{Journal, Op};
use field::plane::WaveParams;
use field::world::World;
use field::FieldConfig;

use std::path::Path;

const W: usize = 32;
const H: usize = 32;
const N_SLOTS: usize = 16;

fn cfg() -> FieldConfig {
    FieldConfig::new(W, H)
}

fn params() -> WaveParams {
    WaveParams::default()
}

/// The frozen op fixture. `seed_base` shifts all SeedAtom seeds so the same
/// builder produces a same-shape / different-content sequence for the
/// anti-vacuous seed-difference gate.
fn build_ops(seed_base: u64) -> Vec<Op> {
    let mut ops = Vec::new();
    // (1) SeedAtom x8 into free slots 2..10 (RESERVED_SLOTS = 2: 0=cur,1=prev)
    for i in 0..8u32 {
        ops.push(Op::SeedAtom {
            slot: 2 + i,
            seed: seed_base + i as u64,
        });
    }
    // (2) Excite x3 on the wave plane (slots 0/1, per world::apply contract)
    ops.push(Op::Excite {
        x: 8,
        y: 8,
        amp_bits: 1.0f32.to_bits(),
    });
    ops.push(Op::Excite {
        x: 16,
        y: 20,
        amp_bits: 0.5f32.to_bits(),
    });
    ops.push(Op::Excite {
        x: 24,
        y: 4,
        amp_bits: 0.25f32.to_bits(),
    });
    // (3) Step 100 wave ticks
    ops.push(Op::Step { count: 100 });
    // (4) Bind dst=10 <- a=2, b=3
    ops.push(Op::Bind {
        dst: 10,
        a: 2,
        b: 3,
    });
    // (5) Bundle dst=11 <- srcs=[4,5,6,7]
    ops.push(Op::Bundle {
        dst: 11,
        srcs: vec![4, 5, 6, 7],
    });
    ops
}

fn run_live(seed_base: u64) -> World {
    let mut world = World::new(cfg(), params(), N_SLOTS);
    for op in build_ops(seed_base) {
        world.apply(&op);
    }
    world
}

/// (1) Two independently-built Worlds, identical cfg/params/n_slots, same
/// op sequence applied live -> digest bit-identical.
#[test]
fn live_apply_same_ops_bitexact_digest() {
    let w1 = run_live(1);
    let w2 = run_live(1);
    assert_eq!(
        w1.digest(),
        w2.digest(),
        "live digests diverged for identical op sequence"
    );
}

/// (2) Record the same ops through Journal::create/append/flush, read_all,
/// World::replay -> digest equals the live digest. Bit-exact replay is the
/// ENTROPY.md core claim (seed + journal fully determines state).
#[test]
fn journal_replay_matches_live_digest() {
    let ops = build_ops(1);
    let live = run_live(1);
    let live_digest = live.digest();

    let path = Path::new("target/field_determinism_replay.fldj");
    std::fs::create_dir_all("target").expect("target/ must exist for cargo builds");
    let _ = std::fs::remove_file(path); // deterministic clean slate, no /tmp

    let mut journal = Journal::create(path, cfg(), &params(), N_SLOTS).expect("Journal::create");
    for op in &ops {
        journal.append(op);
    }
    journal.flush().expect("Journal::flush");

    let (r_cfg, r_params, r_n_slots, r_ops) = Journal::read_all(path).expect("Journal::read_all");
    assert_eq!(r_cfg, cfg(), "journal roundtrip changed FieldConfig");
    assert_eq!(
        r_params,
        params(),
        "journal roundtrip changed WaveParams (not bit-exact)"
    );
    assert_eq!(r_n_slots, N_SLOTS, "journal roundtrip changed n_slots");
    assert_eq!(r_ops, ops, "journal roundtrip changed the op sequence");

    let replayed = World::replay(r_cfg, r_params, r_n_slots, &r_ops);
    assert_eq!(
        replayed.digest(),
        live_digest,
        "journal-replayed world digest diverged from live-applied world digest"
    );
}

/// (3) render_png bytes identical across two independent live runs with the
/// same op sequence — render is a pure function of state, state = f(seed).
#[test]
fn render_png_bytes_identical_across_live_runs() {
    let a = run_live(1);
    let b = run_live(1);
    let png_a = a.render_png();
    let png_b = b.render_png();
    assert_eq!(
        png_a, png_b,
        "render_png bytes diverged for identical op sequence"
    );
    assert!(!png_a.is_empty(), "render_png produced empty output");
}

/// (4) Anti-vacuous sanity: digests DIFFER when the seed changes. Guards
/// against the gates passing because both sides are degenerate/all-zero.
#[test]
fn digests_differ_when_seed_differs() {
    let w_seed1 = run_live(1);
    let w_seed2 = run_live(2);
    assert_ne!(
        w_seed1.digest(),
        w_seed2.digest(),
        "digest did not change under a different seed base (vacuous gate)"
    );
}

/// (5) Slice::digest stable: same seeded_atom seed, computed twice -> same
/// digest. state = f(seed), no hidden mutable/clock dependence in atoms.rs.
#[test]
fn slice_digest_stable_for_same_seeded_atom_seed() {
    let f = Fft2::new(cfg());
    let a1 = atoms::seeded_atom(&f, cfg(), 42);
    let a2 = atoms::seeded_atom(&f, cfg(), 42);
    assert_eq!(
        a1.digest(),
        a2.digest(),
        "seeded_atom(seed=42) digest not stable across two calls"
    );
}
