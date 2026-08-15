//! W3: store + cold journal tests. DETERMINISTIC ONLY — no rand, no clock.
//! Temp files under CARGO_TARGET_TMPDIR (target/), deterministic names, cleaned.

use field::journal::{Journal, Op};
use field::plane::WaveParams;
use field::store::{Store, RESERVED_SLOTS};
use field::{FieldConfig, Slice};
use std::path::{Path, PathBuf};

const TMP: &str = env!("CARGO_TARGET_TMPDIR");

fn fresh_dir(name: &str) -> PathBuf {
    let d = Path::new(TMP).join(name);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn jpath(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    let _ = std::fs::remove_file(&p); // deterministic state
    p
}

fn small_cfg() -> FieldConfig {
    FieldConfig::new(4, 4)
}

// ---------------------------------------------------------------- store (a)

#[test]
fn store_roundtrip_bit_exact() {
    let cfg = small_cfg();
    let mut store = Store::new(cfg, 5);
    let bits = [
        f32::NEG_INFINITY.to_bits(),
        (-0.0f32).to_bits(),
        1.5f32.to_bits(),
        f32::NAN.to_bits(),
        f32::MIN_POSITIVE.to_bits(),
    ];
    let mut data = Vec::with_capacity(cfg.d());
    for i in 0..cfg.d() {
        data.push(f32::from_bits(bits[i % bits.len()]));
    }
    let s = Slice::from_data(cfg, data.clone());
    store.write(2, &s);
    let back = store.read(2);
    assert_eq!(back.cfg, cfg);
    let got: Vec<u32> = back.data.iter().map(|v| v.to_bits()).collect();
    let want: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want, "store roundtrip must be bit-exact");
}

#[test]
fn probe_topk_sorted_desc_tiebreak_lower_slot_first() {
    let cfg = small_cfg();
    let n_slots = 6;
    let mut store = Store::new(cfg, n_slots);
    let ones = Slice::from_data(cfg, vec![1.0; cfg.d()]);
    let one_hot = {
        let mut v = vec![0.0; cfg.d()];
        v[0] = 1.0;
        Slice::from_data(cfg, v)
    };
    let alternating = Slice::from_data(
        cfg,
        (0..cfg.d()).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect(),
    );
    // reserved slots 0/1 get IDENTICAL-to-key rows: probe must skip them.
    store.write(0, &ones);
    store.write(1, &ones);
    store.write(2, &ones);
    store.write(3, &ones);
    store.write(4, &one_hot); // sim 0.25
    store.write(5, &alternating); // sim 0.0

    let key = ones.clone();
    let top = store.probe(&key, n_slots);
    // 1.0 (slots 2,3), 0.25 (slot 4), 0.0 (slot 5) — desc, tie lower-slot first
    assert_eq!(top, vec![(2, 1.0), (3, 1.0), (4, 0.25), (5, 0.0)]);
    assert!(top.iter().all(|(i, _)| *i >= RESERVED_SLOTS), "reserved slots leaked");
    // top_k truncation
    assert_eq!(store.probe(&key, 2), vec![(2, 1.0), (3, 1.0)]);
    assert_eq!(store.probe(&key, 0), Vec::<(usize, f32)>::new());
}

// ---------------------------------------------------------------- journal (b)

fn sample_cfg_params() -> (FieldConfig, WaveParams) {
    let cfg = FieldConfig::new(8, 8);
    let params = WaveParams {
        c: 1.5,
        dt: 0.25,
        damping: 0.02,
        dx: 2.0,
        seed: 42,
        range: 0.125,
    };
    (cfg, params)
}

fn sample_ops() -> Vec<Op> {
    vec![
        Op::SeedAtom { slot: 2, seed: 0xDEAD_BEEF_CAFE_F00D },
        Op::Excite { x: 3, y: 5, amp_bits: f32::NAN.to_bits() },
        Op::Step { count: 120 },
        Op::Bind { dst: 4, a: 2, b: 3 },
        Op::Bundle { dst: 5, srcs: vec![2, 3, 4] },
        Op::Bundle { dst: 6, srcs: vec![] }, // empty vec
        Op::WriteRaw {
            slot: 7,
            data_bits: vec![0x3F80_0000, 0xBF80_0000, 0x0000_0000, 0x8000_0000],
        },
        Op::WriteRaw { slot: 8, data_bits: vec![] }, // empty vec
        Op::Excite { x: 0, y: 0, amp_bits: (-0.0f32).to_bits() },
        Op::Step { count: 0 },
    ]
}

#[test]
fn journal_roundtrip_all_ops_partial_eq_and_f32_bits() {
    let dir = fresh_dir("roundtrip");
    let path = jpath(&dir, "journal_all_ops.fldj");
    let (cfg, params) = sample_cfg_params();
    let ops = sample_ops();

    let mut j = Journal::create(&path, cfg, &params, 16).unwrap();
    for op in &ops {
        j.append(op);
    }
    j.flush().unwrap();

    let (rcfg, rparams, rn, rops) = Journal::read_all(&path).unwrap();
    assert_eq!(rcfg, cfg);
    assert_eq!(rparams, params);
    assert_eq!(rn, 16);
    assert_eq!(rops, ops, "every Op variant roundtrips == PartialEq");
    // f32 payloads bit-exact (u32 bit patterns preserved)
    for (a, b) in rops.iter().zip(&ops) {
        match (a, b) {
            (Op::Excite { amp_bits: x, .. }, Op::Excite { amp_bits: y, .. }) => {
                assert_eq!(x, y, "amp_bits not bit-exact")
            }
            (Op::WriteRaw { data_bits: x, .. }, Op::WriteRaw { data_bits: y, .. }) => {
                assert_eq!(x, y, "data_bits not bit-exact")
            }
            _ => {}
        }
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn journal_header_roundtrip() {
    let dir = fresh_dir("header");
    let path = jpath(&dir, "journal_header.fldj");
    let (cfg, _) = sample_cfg_params();
    // unusual cfg + non-default params + odd n_slots
    let cfg2 = FieldConfig::new(4, 2);
    let params2 = WaveParams { c: 0.1, dt: 1.0, damping: 0.5, dx: 0.25, seed: u64::MAX, range: 1.0 };
    let mut j = Journal::create(&path, cfg2, &params2, 7).unwrap();
    j.flush().unwrap(); // header only, no ops
    let (rcfg, rparams, rn, rops) = Journal::read_all(&path).unwrap();
    assert_eq!(rcfg, cfg2);
    assert_eq!(rparams, params2);
    assert_eq!(rn, 7);
    assert!(rops.is_empty());
    let _ = std::fs::remove_file(&path);
    // sanity: cfg2 differs from the other sample cfg
    assert_ne!(cfg2, cfg);
}

#[test]
fn journal_truncated_file_is_error() {
    let dir = fresh_dir("trunc");
    let (cfg, params) = sample_cfg_params();
    let path = jpath(&dir, "journal_trunc.fldj");
    let mut j = Journal::create(&path, cfg, &params, 16).unwrap();
    j.append(&Op::SeedAtom { slot: 2, seed: 7 });
    j.append(&Op::WriteRaw { slot: 3, data_bits: vec![1, 2, 3, 4] });
    j.flush().unwrap();

    let full = std::fs::read(&path).unwrap();
    assert_eq!(Journal::read_all(&path).unwrap().3.len(), 2, "full file parses");
    let n = full.len();

    // cuts inside header or inside an op payload must error
    for cut in [1usize, 3, 47, 49, 60, 53, 55, n - 1] {
        let t = dir.join(format!("trunc_cut{}.fldj", cut));
        std::fs::write(&t, &full[..cut]).unwrap();
        assert!(
            Journal::read_all(&t).is_err(),
            "cut at byte {} must be io::Error (truncation)",
            cut
        );
        let _ = std::fs::remove_file(&t);
    }
    // header alone (48B) is a valid empty-op journal
    let h = dir.join("trunc_header_only.fldj");
    std::fs::write(&h, &full[..48]).unwrap();
    let r = Journal::read_all(&h).unwrap();
    assert!(r.3.is_empty());
    let _ = std::fs::remove_file(&h);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn journal_append_buffers_ram_only_flush_writes() {
    let dir = fresh_dir("append");
    let (cfg, params) = sample_cfg_params();
    let path = jpath(&dir, "journal_append.fldj");
    let mut j = Journal::create(&path, cfg, &params, 16).unwrap();
    // header written immediately at create()
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 48, "create must write header now");

    let ops = sample_ops();
    for op in &ops {
        j.append(op);
    }
    // append() NEVER touches disk
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 48, "append touched disk");

    j.flush().unwrap();
    let after = std::fs::metadata(&path).unwrap().len();
    assert!(after > 48, "flush must grow the file");

    // buffered again after flush
    j.append(&Op::Step { count: 1 });
    assert_eq!(std::fs::metadata(&path).unwrap().len(), after, "append touched disk post-flush");
    j.flush().unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > after);

    let (_, _, _, rops) = Journal::read_all(&path).unwrap();
    let mut want = ops.clone();
    want.push(Op::Step { count: 1 });
    assert_eq!(rops, want, "file replay == buffered sequence");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn journal_identical_sequences_byte_identical_files() {
    let dir = fresh_dir("bytes");
    let (cfg, params) = sample_cfg_params();
    let ops = sample_ops();
    let p1 = jpath(&dir, "journal_a.fldj");
    let p2 = jpath(&dir, "journal_b.fldj");

    let mut j1 = Journal::create(&p1, cfg, &params, 16).unwrap();
    let mut j2 = Journal::create(&p2, cfg, &params, 16).unwrap();
    // j1: single flush. j2: interleaved flush boundaries — must NOT change bytes.
    for (i, op) in ops.iter().enumerate() {
        j1.append(op);
        j2.append(op);
        if i % 3 == 1 {
            j2.flush().unwrap();
        }
    }
    j1.flush().unwrap();
    j2.flush().unwrap();

    assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
    // both replay to the same op sequence
    assert_eq!(Journal::read_all(&p1).unwrap().3, ops);
    assert_eq!(Journal::read_all(&p2).unwrap().3, ops);
    let _ = std::fs::remove_file(&p1);
    let _ = std::fs::remove_file(&p2);
}

/// atom9 互換: header n_slots が 0/1 の journal = malformed。
/// read_all は panic せず InvalidData Err。native fieldrun の rc=9 と同一判定。
#[test]
fn read_all_rejects_n_slots_below_reserved() {
    let dir = fresh_dir("nslots");
    let (cfg, params) = sample_cfg_params();
    for bad in [0u32, 1u32] {
        let path = jpath(&dir, &format!("nslots_{bad}.fldj"));
        let mut j = Journal::create(&path, cfg, &params, 2).unwrap();
        j.append(&Op::Step { count: 1 });
        j.flush().unwrap();
        // header off44 = n_slots(u32 LE)を破壊 = wire malformed
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[44..48].copy_from_slice(&bad.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let e = Journal::read_all(&path).expect_err("n_slots < 2 must be Err");
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{bad}: {e}");
    }
}
