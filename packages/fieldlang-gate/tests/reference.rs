//! ============================================================
//! GATE TOOL — NOT THE COMPILER. Acceptance oracle tests.
//! ============================================================

use field::journal::{Journal, Op};
use field::world::World;
use fieldlang_gate::*;
use std::path::Path;

const TMP: &str = env!("CARGO_TARGET_TMPDIR");

// ------------------------------------------------------------ parse vectors

#[test]
fn defaults_without_header() {
    let p = parse("歩 3\n").unwrap();
    assert_eq!(p.cfg.width, 64);
    assert_eq!(p.cfg.height, 64);
    assert_eq!(p.n_slots, 16);
    assert_eq!(p.params.seed, 42);
    assert_eq!(p.params.c.to_bits(), 0x3F80_0000);
    assert_eq!(p.params.dt.to_bits(), 0x3DCC_CCCD);
    assert_eq!(p.params.damping.to_bits(), 0x3F7F_BE77);
    assert_eq!(p.params.dx.to_bits(), 0x3F80_0000);
    assert_eq!(p.params.range.to_bits(), 0x3F80_0000);
    assert_eq!(p.ops, vec![Op::Step { count: 3 }]);
}

#[test]
fn header_overrides_w_h_slots_seed_only() {
    let p = parse("界 32 32 2 7\n歩 1\n").unwrap();
    assert_eq!((p.cfg.width, p.cfg.height), (32, 32));
    assert_eq!(p.n_slots, 2);
    assert_eq!(p.params.seed, 7);
    // wave params untouched by 界 — driver defaults stand
    assert_eq!(p.params.dt.to_bits(), 0x3DCC_CCCD);
}

#[test]
fn every_op_parses_to_exact_journal_op() {
    let p = parse(
        "種 2 42\n撃 16 16 1065353216\n歩 60\n縛 4 2 3\n束 5 2 2 3\n寫 6 2 0 4294967295\n",
    )
    .unwrap();
    assert_eq!(
        p.ops,
        vec![
            Op::SeedAtom { slot: 2, seed: 42 },
            Op::Excite { x: 16, y: 16, amp_bits: 1065353216 },
            Op::Step { count: 60 },
            Op::Bind { dst: 4, a: 2, b: 3 },
            Op::Bundle { dst: 5, srcs: vec![2, 3] },
            Op::WriteRaw { slot: 6, data_bits: vec![0, 4294967295] },
        ]
    );
}

/// The decimal bit-pattern literals used in the shipped examples MUST equal
/// the intended f32 values — this test is the guard for those magic numbers.
#[test]
fn example_decimal_literals_match_f32_bits() {
    assert_eq!(1065353216u32, 1.0f32.to_bits());
    assert_eq!(1061158912u32, 0.75f32.to_bits());
    let want: Vec<f32> = vec![
        0.5, -0.5, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.1, 0.2, 0.3, 0.4,
    ];
    let got: Vec<u32> = vec![
        1056964608, 3204448256, 1048576000, 3196059648, 0, 0, 0, 0, 1065353216, 3212836864,
        1056964608, 3204448256, 1036831949, 1045220557, 1050253722, 1053609165,
    ];
    let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want_bits);
}

// ------------------------------------------------------------ errors

#[test]
fn errors_carry_line_and_char_col() {
    let e = parse("歩 1\n種 2 x\n").unwrap_err();
    assert_eq!((e.line, e.col), (2, 5), "{}", e); // CJK glyph = ONE char col

    let e = parse("疲 1 2\n").unwrap_err();
    assert_eq!((e.line, e.col), (1, 1), "{}", e);

    let e = parse("種 2 42\n界 8 8 2 1\n").unwrap_err();
    assert_eq!(e.line, 2, "{}", e); // 界 after op

    let e = parse("界 8 8 2 1\n界 8 8 2 1\n").unwrap_err();
    assert_eq!(e.line, 2, "{}", e); // 界 twice
}

#[test]
fn counted_lists_are_exact() {
    assert!(parse("束 5 2 2\n").is_err()); // n=2, one src
    assert!(parse("束 5 1 2 3\n").is_err()); // n=1, two srcs
    assert!(parse("寫 2 3 1 2\n").is_err());
    assert!(parse("束 5 0\n").unwrap().ops
        == vec![Op::Bundle { dst: 5, srcs: vec![] }]); // n=0 lexically legal;
    // bundle() would assert at replay — semantic law, gate-visible, not parse law
}

#[test]
fn range_and_overflow_errors() {
    assert!(parse("歩 18446744073709551616\n").is_err()); // u64 overflow
    assert!(parse("歩 4294967296\n").is_err()); // > u32 max
    assert!(parse("歩 -1\n").is_err()); // no minus sign in v0
    assert!(parse("歩 1.0\n").is_err()); // no float literals in v0
}

#[test]
fn gate_tool_hygiene_rejects_unreplayable_headers() {
    assert!(parse("界 30 32 2 1\n").is_err()); // not power of two
    assert!(parse("界 32 32 1 1\n").is_err()); // < RESERVED_SLOTS
}

#[test]
fn comments_blanks_whitespace() {
    let p = parse("# 注釈\n\n  歩 5   # trailing\n\t\n").unwrap();
    assert_eq!(p.ops, vec![Op::Step { count: 5 }]);
}

// -------------------------------------------------- end-to-end (frozen path)

fn replay_digest(path: &Path) -> u64 {
    let (cfg, params, n_slots, ops) = Journal::read_all(path).unwrap();
    World::replay(cfg, params, n_slots, &ops).digest()
}

#[test]
fn reference_journal_roundtrips_bit_exact_and_replays_deterministically() {
    for (name, src) in [
        ("e1", include_str!("../examples/01_wave.fld")),
        ("e2", include_str!("../examples/02_algebra.fld")),
        ("e3", include_str!("../examples/03_raw.fld")),
    ] {
        let prog = parse(src).unwrap();
        let path = Path::new(TMP).join(format!("gate_{}.fldj", name));
        let _ = std::fs::remove_file(&path);
        let mut j = Journal::create(&path, prog.cfg, &prog.params, prog.n_slots).unwrap();
        for op in &prog.ops {
            j.append(op);
        }
        j.flush().unwrap();

        // frozen read_all must hand back EXACTLY what was written
        let (cfg, params, n_slots, ops) = Journal::read_all(&path).unwrap();
        assert_eq!(cfg, prog.cfg);
        assert_eq!(params, prog.params);
        assert_eq!(n_slots, prog.n_slots);
        assert_eq!(ops, prog.ops);

        // determinism: two replays of the same bytes → same digest
        let d1 = replay_digest(&path);
        let d2 = replay_digest(&path);
        assert_eq!(d1, d2, "{} digest drifted", name);
        println!("{} digest={:016x}", name, d1);
    }
}
