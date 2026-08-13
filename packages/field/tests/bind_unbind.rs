//! V0 acceptance tests: determinism, exact bind/unbind, and bind/unbind
//! recovery at scale through the persisted mmap store (10k stored items).

use field::{bind, bundle, permute, random_vec, similarity, unbind, Field, DIM};
use tempfile::tempdir;

#[test]
fn random_vec_is_deterministic_same_seed() {
    let a = random_vec(0xC0FFEE);
    let b = random_vec(0xC0FFEE);
    assert_eq!(a, b, "same seed must yield byte-identical vectors");
    assert_eq!(a.len(), DIM);
}

#[test]
fn random_vec_differs_across_seeds_and_is_bipolar() {
    let a = random_vec(1);
    let b = random_vec(2);
    assert_ne!(a, b);
    assert!(a.iter().all(|&x| x == 1.0 || x == -1.0));
    // Independently-seeded random bipolar vectors in d=8192 are
    // near-orthogonal with overwhelming probability (expected |sim| ~
    // 1/sqrt(d) ~ 0.011); this is a sanity bound, not a tight one.
    let sim = similarity(&a, &b);
    assert!(sim.abs() < 0.1, "unrelated vectors should be near-orthogonal, got {sim}");
}

#[test]
fn bind_is_exact_self_inverse_for_bipolar_vectors() {
    let key = random_vec(11);
    let value = random_vec(22);
    let bound = bind(&value, &key);
    let recovered = unbind(&bound, &key);
    // Bipolar element-wise product: key_i * key_i == 1 for every lane, so
    // this must be exact, not merely close.
    assert_eq!(recovered, value);
    // The vectors are bit-exact (asserted above); `similarity`'s two sqrt
    // round-trips (norm computed, then multiplied back) are not, so compare
    // with an epsilon rather than exact float equality.
    assert!((similarity(&recovered, &value) - 1.0).abs() < 1e-5);
}

#[test]
fn bundle_and_similarity_basic_shape() {
    let a = random_vec(1);
    let b = random_vec(2);
    let bundled = bundle(&[a.clone(), b.clone()]);
    // The bundle should look at least a little like each ingredient
    // (superposition), not be orthogonal to either.
    assert!(similarity(&bundled, &a) > 0.2);
    assert!(similarity(&bundled, &b) > 0.2);
    // Self-similarity is always 1.
    assert!((similarity(&a, &a) - 1.0).abs() < 1e-6);
    // Empty bundle -> zero vector -> similarity defined as 0.0.
    let empty = bundle(&[]);
    assert_eq!(similarity(&empty, &a), 0.0);
}

#[test]
fn permute_is_a_bijection_recoverable_by_inverse_shift() {
    let v = random_vec(7);
    let shifted = permute(&v, 3);
    assert_ne!(shifted, v);
    let back = permute(&shifted, v.len() - 3);
    assert_eq!(back, v);
}

/// The required V0 acceptance test: bind/unbind recovers the original value
/// with >0.9 cosine similarity, checked through the persisted mmap store at
/// 10,000 stored items — this exercises `store()`'s growth/remap and reads
/// the recovered vector back out through the *public* `get()`/`cleanup()`
/// surface (not a private-field bypass), so it proves the on-disk bytes
/// round-trip, not just the in-memory algebra.
#[test]
fn bind_unbind_recovers_at_10k_scale() {
    const N: u64 = 10_000;
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("field-10k.bin");
    let mut field = Field::open(&path).expect("open field");

    // Independent key/value streams per item: distinct domains of the same
    // deterministic hash (§ ENTROPY.md — no randomness, state = f(seed)).
    let seed_key = |i: u64| 0x4B45_5900_0000_0000u64 ^ i; // "KEY"-tagged
    let seed_val = |i: u64| 0x5641_4C00_0000_0000u64 ^ i; // "VAL"-tagged

    for i in 0..N {
        let key = random_vec(seed_key(i));
        let value = random_vec(seed_val(i));
        let composed = bind(&value, &key);
        let idx = field.store(&format!("item{i}"), composed).expect("store");
        assert_eq!(idx, i);
    }
    assert_eq!(field.len(), N);
    // store() succeeding for all N with capacity doubling from 64 past
    // 8192 *is* the growth/remap test.

    // Cheap check (O(dim) per item via `get()`, no full-table scan): sample
    // densely with a stride that lands inside every growth-doubling region
    // (64, 128, 256, ..., 8192, 10000).
    let sample: Vec<u64> = (0..N).step_by(37).collect();
    assert!(sample.len() > 200);

    let mut min_sim = f32::INFINITY;
    let mut composed_by_index = std::collections::HashMap::new();
    for &i in &sample {
        let key = random_vec(seed_key(i));
        let value = random_vec(seed_val(i));
        let expected_name = format!("item{i}");

        // Fetch the persisted bytes back by id — proves the mmap round-trip,
        // not a locally-recomputed stand-in.
        let (name, composed) = field.get(i).expect("record exists");
        assert_eq!(name, expected_name);

        let recovered = unbind(&composed, &key);
        let sim = similarity(&recovered, &value);
        min_sim = min_sim.min(sim);
        assert!(sim > 0.9, "item {i}: bind/unbind recovery sim {sim} <= 0.9");
        composed_by_index.insert(i, composed);
    }
    eprintln!(
        "bind_unbind_recovers_at_10k_scale: {} samples via get(), min similarity {:.6}",
        sample.len(),
        min_sim
    );

    // Cleanup memory (O(len * dim) full-table scan — expensive by nature of
    // a V0 linear-scan store, so exercised on a handful of spot checks
    // rather than every sample): first, last, and each capacity-doubling
    // boundary (64, 128, 256, 512, 1024, 2048, 4096, 8192).
    let mut spot_checks: Vec<u64> = vec![0, 1, 63, 64, 65, N - 1];
    let mut cap = 64u64;
    while cap < N {
        spot_checks.push(cap);
        cap *= 2;
    }
    for &i in &spot_checks {
        let expected_name = format!("item{i}");
        let composed = composed_by_index.get(&i).cloned().unwrap_or_else(|| {
            let key = random_vec(seed_key(i));
            let value = random_vec(seed_val(i));
            bind(&value, &key)
        });
        let m = field.cleanup(&composed).expect("nonempty field");
        assert_eq!(m.name, expected_name, "cleanup must find record {i} among {N} stored");
        assert!(m.similarity > 0.999);
    }
    eprintln!("cleanup spot checks ok: {} full-table scans", spot_checks.len());
}

#[test]
fn determinism_reopen_yields_identical_bytes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("field-reopen.bin");

    let v0 = random_vec(999);
    {
        let mut field = Field::open(&path).expect("open");
        field.store("only", v0.clone()).expect("store");
    } // dropped: file persists

    let field2 = Field::open(&path).expect("reopen");
    assert_eq!(field2.len(), 1);
    let m = field2.cleanup(&v0).expect("match");
    assert_eq!(m.name, "only");
    // Bytes round-trip exactly through the mmap (that's the point of this
    // test); `similarity`'s sqrt round-trip is not bit-exact, hence epsilon.
    assert!(
        (m.similarity - 1.0).abs() < 1e-5,
        "reopened bytes must match exactly, sim={}",
        m.similarity
    );
}
