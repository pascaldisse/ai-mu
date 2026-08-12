//! Task 2 (FIELD CORE WIRING) — headless FieldRuntime demo.
//! Simulates an app render loop driving FieldRuntime at a 120fps target:
//! tick() + probe() every frame (the hot path), excite() strikes sprinkled
//! in, render_png() at display cadence, journal flush() off-cadence only.
//! Ends by proving FieldRuntime::replay() reproduces the live session
//! bit-exactly (ENTROPY.md: state = f(seed) + journal).
//!
//! Run: cargo run -p field --release --example runtime_loop
//! argv (optional): --width --height --slots --frames --render-every
//! defaults match field-bench: 128x128, 4096 slots, 1000 frames.

use field::atoms;
use field::plane::WaveParams;
use field::runtime::FieldRuntime;
use field::FieldConfig;
use std::time::Instant;

const FLOOR_MS: f64 = 1000.0 / 120.0; // 8.33ms — the 120fps floor.

struct Args {
    width: usize,
    height: usize,
    slots: usize,
    frames: usize,
    render_every: usize,
}

impl Default for Args {
    fn default() -> Self {
        Args { width: 128, height: 128, slots: 4096, frames: 1000, render_every: 12 }
    }
}

fn parse_uint(argv: &[String], i: &mut usize, flag: &str, inline: Option<&str>) -> usize {
    let raw = match inline {
        Some(v) => v.to_string(),
        None => {
            *i += 1;
            argv.get(*i).unwrap_or_else(|| panic!("{flag} requires a value")).clone()
        }
    };
    raw.parse::<usize>().unwrap_or_else(|_| panic!("{flag}: invalid integer '{raw}'"))
}

fn parse_args() -> Args {
    let mut a = Args::default();
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
            "--frames" => a.frames = parse_uint(&argv, &mut i, "--frames", inline),
            "--render-every" => a.render_every = parse_uint(&argv, &mut i, "--render-every", inline),
            other => panic!("unknown flag '{other}'"),
        }
        i += 1;
    }
    a
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn main() {
    let a = parse_args();
    let cfg = FieldConfig::new(a.width, a.height); // asserts pow2 dims

    let live_path = std::env::temp_dir().join("field-runtime-demo-live.fldj");
    let resume_path = std::env::temp_dir().join("field-runtime-demo-resume.fldj");
    let _ = std::fs::remove_file(&live_path);
    let _ = std::fs::remove_file(&resume_path);

    let mut rt = FieldRuntime::new(cfg, WaveParams::default(), a.slots, &live_path)
        .expect("FieldRuntime::new");

    // Probe key: a seeded atom (deterministic — state stays f(seed)).
    let key = atoms::seeded_atom(&rt.world.fft, cfg, 1);

    // One strike before the loop starts, to give the wave plane something
    // to propagate (excite() itself is off the timed hot path here).
    rt.excite(cfg.width / 2, cfg.height / 2, 0.3);

    let mut per_frame: Vec<f64> = Vec::with_capacity(a.frames); // ms/frame (tick+probe)
    let mut renders = 0usize;
    let t_start = Instant::now();
    for frame in 0..a.frames {
        let t0 = Instant::now();
        rt.tick(); // hot path: wave_step + RAM-only journal append
        let _hits = rt.probe(&key, 1); // hot path: content-addressed read
        per_frame.push(t0.elapsed().as_secs_f64() * 1000.0);

        // Display cadence — OFF the timed hot-path measurement above.
        if a.render_every > 0 && frame % a.render_every == 0 {
            let _png = rt.render_png();
            renders += 1;
        }
        // A second strike partway through, to exercise excite() mid-run.
        if frame == a.frames / 2 {
            rt.excite(cfg.width / 4, cfg.height / 4, -0.15);
        }
    }
    let total_s = t_start.elapsed().as_secs_f64();

    // Off-cadence journal flush — explicit, never inside the tick loop.
    rt.flush().expect("journal flush");

    let mean = per_frame.iter().sum::<f64>() / per_frame.len() as f64;
    let mut sorted = per_frame.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (p50, p99) = (percentile(&sorted, 0.50), percentile(&sorted, 0.99));
    let steps_s = a.frames as f64 / total_s;
    let pass = mean <= FLOOR_MS;

    println!(
        "field-runtime-loop W={} H={} slots={} frames={} render-every={} renders={}",
        a.width, a.height, a.slots, a.frames, a.render_every, renders
    );
    println!("total: {:.3} ms   ({} frames)", total_s * 1000.0, a.frames);
    println!("ms/frame (tick+probe): mean {:.4}  p50 {:.4}  p99 {:.4}", mean, p50, p99);
    println!("steps/s: {:.1}", steps_s);
    println!("RESULT: {} (floor {:.2} ms)", if pass { "PASS" } else { "FAIL" }, FLOOR_MS);

    // Prove the closed loop: journal replay reproduces the live session
    // bit-exactly (ENTROPY.md: state = f(seed) + journal).
    let live_digest = rt.digest();
    let live_step = rt.world.step_index;
    let replayed =
        FieldRuntime::replay(&live_path, &resume_path).expect("FieldRuntime::replay");
    let replay_ok = replayed.digest() == live_digest && replayed.world.step_index == live_step;
    println!(
        "replay: step_index live={} replay={}  digest live={:#x} replay={:#x}",
        live_step, replayed.world.step_index, live_digest, replayed.digest()
    );
    assert!(replay_ok, "journal replay must reproduce live session bit-exactly");
    println!("REPLAY: PASS (bit-exact live == replay)");

    let _ = std::fs::remove_file(&live_path);
    let _ = std::fs::remove_file(&resume_path);
}
