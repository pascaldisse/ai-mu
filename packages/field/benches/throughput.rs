//! Plain-timing bench (no criterion — keeps the dep tree light under the
//! `-j2` nice'd build constraint). Run: `cargo bench -p field --bench throughput`
//! or directly: `cargo run -p field --release --bin throughput_bench`-equiv.
//!
//! `harness = false` in Cargo.toml, so this is a plain `fn main()` that
//! measures store()/probe() latency+throughput at 10k items and prints the
//! numbers to stdout.

use std::time::Instant;

use field::{bind, random_vec, Field};
use tempfile::tempdir;

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx]
}

fn main() {
    const N: u64 = 10_000;
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bench-field.bin");
    let mut field = Field::open(&path).expect("open");

    // --- random_vec generation ---
    let t0 = Instant::now();
    let vecs: Vec<Vec<f32>> = (0..N).map(random_vec).collect();
    let gen_elapsed = t0.elapsed();
    println!(
        "random_vec: {} vecs in {:.3}ms -> {:.0} vecs/sec",
        N,
        gen_elapsed.as_secs_f64() * 1000.0,
        N as f64 / gen_elapsed.as_secs_f64()
    );

    // --- store() ---
    let keys: Vec<Vec<f32>> = (0..N).map(|i| random_vec(i ^ 0xA5A5_A5A5)).collect();
    let mut store_us: Vec<f64> = Vec::with_capacity(N as usize);
    let t1 = Instant::now();
    for i in 0..N as usize {
        let composed = bind(&vecs[i], &keys[i]);
        let s = Instant::now();
        field
            .store(&format!("item{i}"), composed)
            .expect("store");
        store_us.push(s.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let store_elapsed = t1.elapsed();
    store_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "store: {} ops in {:.3}ms -> {:.0} ops/sec, p50 {:.2}us, p99 {:.2}us",
        N,
        store_elapsed.as_secs_f64() * 1000.0,
        N as f64 / store_elapsed.as_secs_f64(),
        percentile(&store_us, 0.50),
        percentile(&store_us, 0.99),
    );

    // --- probe() over the full 10k-item slab ---
    const PROBES: usize = 50;
    let mut probe_us: Vec<f64> = Vec::with_capacity(PROBES);
    let t2 = Instant::now();
    for p in 0..PROBES {
        let composed = bind(&vecs[p], &keys[p]);
        let s = Instant::now();
        let matches = field.probe(&composed);
        probe_us.push(s.elapsed().as_secs_f64() * 1_000_000.0);
        assert_eq!(matches[0].name, format!("item{p}"));
    }
    let probe_elapsed = t2.elapsed();
    probe_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "probe (scan {} records): {} ops in {:.3}ms -> {:.0} ops/sec, p50 {:.2}us, p99 {:.2}us",
        N,
        PROBES,
        probe_elapsed.as_secs_f64() * 1000.0,
        PROBES as f64 / probe_elapsed.as_secs_f64(),
        percentile(&probe_us, 0.50),
        percentile(&probe_us, 0.99),
    );

    // --- cleanup() (top-1 only, same scan cost as probe minus the sort/alloc) ---
    let mut cleanup_us: Vec<f64> = Vec::with_capacity(PROBES);
    let t3 = Instant::now();
    for p in 0..PROBES {
        let composed = bind(&vecs[p], &keys[p]);
        let s = Instant::now();
        let m = field.cleanup(&composed).unwrap();
        cleanup_us.push(s.elapsed().as_secs_f64() * 1_000_000.0);
        assert_eq!(m.name, format!("item{p}"));
    }
    let cleanup_elapsed = t3.elapsed();
    cleanup_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "cleanup (scan {} records): {} ops in {:.3}ms -> {:.0} ops/sec, p50 {:.2}us, p99 {:.2}us",
        N,
        PROBES,
        cleanup_elapsed.as_secs_f64() * 1000.0,
        PROBES as f64 / cleanup_elapsed.as_secs_f64(),
        percentile(&cleanup_us, 0.50),
        percentile(&cleanup_us, 0.99),
    );
}
