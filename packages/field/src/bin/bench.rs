//! field-bench — 120fps floor gate. OWNED BY W6.
//! argv (all optional, defaults in parens — IRON: params not hardcodes):
//!   --width (128) --height (128) --slots (4096) --probe-keys (1)
//!   --frames (1000) --render-every (12)
//! Loop per frame: world.tick() + store.probe(key) + render every Nth.
//! Report: ms/frame mean/p50/p99, steps/s, PASS/FAIL vs 8.33ms (120fps).
//! Timing via Instant = harness only; SIM state stays f(seed).
//! Hand-rolled argv parse (no clap): `--flag value` or `--flag=value`.

use field::atoms;
use field::plane::WaveParams;
use field::world::World;
use field::{FieldConfig, Slice};
use std::time::Instant;

const FLOOR_MS: f64 = 1000.0 / 120.0; // 8.33ms — the 120fps floor.

struct BenchArgs {
    width: usize,
    height: usize,
    slots: usize,
    probe_keys: usize,
    frames: usize,
    render_every: usize,
}

impl Default for BenchArgs {
    fn default() -> Self {
        BenchArgs {
            width: 128,
            height: 128,
            slots: 4096,
            probe_keys: 1,
            frames: 1000,
            render_every: 12,
        }
    }
}

fn parse_uint(argv: &[String], i: &mut usize, flag: &str, inline: Option<&str>) -> usize {
    let raw = match inline {
        Some(v) => v.to_string(),
        None => {
            *i += 1;
            argv.get(*i)
                .unwrap_or_else(|| panic!("{flag} requires a value"))
                .clone()
        }
    };
    raw.parse::<usize>()
        .unwrap_or_else(|_| panic!("{flag}: invalid integer '{raw}'"))
}

fn print_usage() {
    eprintln!(
        "field-bench — 120fps floor gate\n\
         usage: field-bench [--width W] [--height H] [--slots N] [--probe-keys K]\n\
         \x20\x20\x20\x20\x20\x20\x20 [--frames N] [--render-every N] [--help]\n\
         defaults: --width 128 --height 128 --slots 4096 --probe-keys 1\n\
         \x20\x20\x20\x20\x20\x20\x20 --frames 1000 --render-every 12"
    );
}

fn parse_args() -> BenchArgs {
    let mut a = BenchArgs::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) => (f, Some(v)),
            None => (arg.as_str(), None),
        };
        match flag {
            "--width" => a.width = parse_uint(&argv, &mut i, "--width", inline),
            "--height" => a.height = parse_uint(&argv, &mut i, "--height", inline),
            "--slots" => a.slots = parse_uint(&argv, &mut i, "--slots", inline),
            "--probe-keys" => a.probe_keys = parse_uint(&argv, &mut i, "--probe-keys", inline),
            "--frames" => a.frames = parse_uint(&argv, &mut i, "--frames", inline),
            "--render-every" => a.render_every = parse_uint(&argv, &mut i, "--render-every", inline),
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => panic!("unknown flag '{other}' (see --help)"),
        }
        i += 1;
    }
    assert!(a.probe_keys >= 1, "--probe-keys must be >= 1");
    assert!(a.frames >= 1, "--frames must be >= 1");
    a
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty());
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn main() {
    let a = parse_args();

    let cfg = FieldConfig::new(a.width, a.height); // asserts pow2 dims
    let world = World::new(cfg, WaveParams::default(), a.slots);

    // Probe key: a seeded atom (deterministic — state stays f(seed)).
    let key: Slice = atoms::seeded_atom(&world.fft, cfg, 1);
    // World is re-borrowed mutably below; key is a separate owned Slice.
    let mut world = world;

    let mut per_frame: Vec<f64> = Vec::with_capacity(a.frames); // ms/frame
    let t_start = Instant::now();
    for frame in 0..a.frames {
        let t0 = Instant::now();
        world.tick();
        let _hits = world.store.probe(&key, a.probe_keys);
        if a.render_every > 0 && frame % a.render_every == 0 {
            let _png = world.render_png();
        }
        per_frame.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let total_s = t_start.elapsed().as_secs_f64();

    let mean = per_frame.iter().sum::<f64>() / per_frame.len() as f64;
    let mut sorted = per_frame.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (p50, p99) = (percentile(&sorted, 0.50), percentile(&sorted, 0.99));
    let steps_s = a.frames as f64 / total_s;
    let pass = mean <= FLOOR_MS;

    println!(
        "field-bench W={} H={} slots={} probe-keys={} frames={} render-every={}",
        a.width, a.height, a.slots, a.probe_keys, a.frames, a.render_every
    );
    println!("total: {:.3} ms   ({} frames)", total_s * 1000.0, a.frames);
    println!("ms/frame: mean {:.4}  p50 {:.4}  p99 {:.4}", mean, p50, p99);
    println!("steps/s: {:.1}", steps_s);
    println!("RESULT: {} (floor {:.2} ms)", if pass { "PASS" } else { "FAIL" }, FLOOR_MS);
}
