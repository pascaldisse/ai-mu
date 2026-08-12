//! mu-serve — the WORLD DOOR for Mu (GENESIS-II arc2 段1).
//!
//! Mu must learn from the REAL field (packages/field World: FFT wave plane,
//! N×d store, ultradeterminism), not from a python PDE toy. This bin exposes
//! the world over stdin/stdout as a deterministic request/response stream so
//! any learner (mu/world_env.py) can tick it, poke it, and read its state.
//!
//! Law kept: state = f(seed). No clock, no rand, no threads. Every response
//! carries World::digest() so the caller can gate bit-exact determinism.
//!
//! argv: --width (128) --height (128) --slots (4) --amp (0.08) --ring (40)
//!       --actions (8) --probe (8) --steps (1) --seed (0)
//!
//! stdin (text lines):
//!   RESET <seed>          -> reset world to f(seed), emit frame
//!   ACT <action> [steps]  -> excite ring[action], tick steps, emit frame
//!   STEP [steps]          -> tick only, emit frame
//!   QUIT
//!
//! stdout per frame (raw little-endian, no framing bytes):
//!   u64 digest | u64 step_index | f32 reward | f32[2*d] obs (cur then prev)
//! stderr: human notes only.

use field::plane::{self, WaveParams};
use field::world::World;
use field::{FieldConfig, Slice};
use std::io::{self, BufRead, Read, Write};

struct Args {
    width: usize,
    height: usize,
    slots: usize,
    amp: f32,
    ring: f32,
    actions: usize,
    probe: usize,
    steps: usize,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            width: 128,
            height: 128,
            slots: 4,
            amp: 0.08,
            ring: 40.0,
            actions: 8,
            probe: 8,
            steps: 1,
            seed: 0,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let (flag, inline) = match argv[i].split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (argv[i].clone(), None),
        };
        let mut val = || -> String {
            match &inline {
                Some(v) => v.clone(),
                None => {
                    i += 1;
                    argv.get(i).cloned().unwrap_or_else(|| panic!("missing value for {}", flag))
                }
            }
        };
        match flag.as_str() {
            "--width" => a.width = val().parse().unwrap(),
            "--height" => a.height = val().parse().unwrap(),
            "--slots" => a.slots = val().parse().unwrap(),
            "--amp" => a.amp = val().parse().unwrap(),
            "--ring" => a.ring = val().parse().unwrap(),
            "--actions" => a.actions = val().parse().unwrap(),
            "--probe" => a.probe = val().parse().unwrap(),
            "--steps" => a.steps = val().parse().unwrap(),
            "--seed" => a.seed = val().parse().unwrap(),
            other => panic!("unknown flag {}", other),
        }
        i += 1;
    }
    a
}

/// Ring poke sites — a WORLD FACT declared here, not a learner choice.
fn ring_sites(a: &Args) -> Vec<(usize, usize)> {
    (0..a.actions)
        .map(|k| {
            let th = 2.0 * std::f64::consts::PI * (k as f64) / (a.actions as f64);
            let x = a.width as f64 / 2.0 + a.ring as f64 * th.cos();
            let y = a.height as f64 / 2.0 + a.ring as f64 * th.sin();
            (x as usize % a.width, y as usize % a.height)
        })
        .collect()
}

/// Probe readout = energy in a probe×probe patch co-located with the target
/// site (same immediate-signal rationale as arc1), over the REAL field state.
fn reward(cur: &Slice, prev: &Slice, site: (usize, usize), probe: usize) -> f32 {
    let cfg = cur.cfg;
    let half = (probe / 2) as i64;
    let mut acc = 0.0f64;
    for dy in -half..half {
        for dx in -half..half {
            let px = ((site.0 as i64 + dx) % cfg.width as i64 + cfg.width as i64) % cfg.width as i64;
            let py =
                ((site.1 as i64 + dy) % cfg.height as i64 + cfg.height as i64) % cfg.height as i64;
            let i = py as usize * cfg.width + px as usize;
            acc += (cur.data[i] as f64).powi(2) + (prev.data[i] as f64).powi(2);
        }
    }
    acc as f32
}

fn emit(out: &mut impl Write, w: &World, r: f32) -> io::Result<()> {
    let cur = w.store.read(0);
    let prev = w.store.read(1);
    out.write_all(&w.digest().to_le_bytes())?;
    out.write_all(&w.step_index.to_le_bytes())?;
    out.write_all(&r.to_le_bytes())?;
    for v in cur.data.iter().chain(prev.data.iter()) {
        out.write_all(&v.to_bits().to_le_bytes())?;
    }
    out.flush()
}

/// reset = f(seed) ONLY: one seeded excite at a seed-derived site, no rand.
fn reset(a: &Args, seed: u64) -> World {
    let cfg = FieldConfig::new(a.width, a.height);
    let params = WaveParams { seed, ..WaveParams::default() };
    assert!(params.stable(), "wave params must satisfy the Courant condition");
    let mut w = World::new(cfg, params, a.slots.max(2));
    let m = field::mix64(seed);
    let x = 20 + (m % (a.width as u64 - 40)) as usize;
    let y = 20 + ((m >> 32) % (a.height as u64 - 40)) as usize;
    let mut cur = w.store.read(0);
    let mut prev = w.store.read(1);
    plane::excite(&mut cur, &mut prev, x, y, 0.5, seed);
    w.store.write(0, &cur);
    w.store.write(1, &prev);
    w
}

fn main() -> io::Result<()> {
    let a = parse_args();
    let sites = ring_sites(&a);
    let target = sites[a.actions / 4];
    let mut w = reset(&a, a.seed);

    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    eprintln!(
        "mu-serve: {}x{} d={} actions={} target={:?} amp={} steps={}",
        a.width,
        a.height,
        a.width * a.height,
        a.actions,
        target,
        a.amp,
        a.steps
    );

    for line in stdin.lock().lines() {
        let line = line?;
        let mut it = line.split_whitespace();
        match it.next() {
            Some("RESET") => {
                let seed: u64 = it.next().unwrap_or("0").parse().unwrap();
                w = reset(&a, seed);
                let (c, p) = (w.store.read(0), w.store.read(1));
                let r = reward(&c, &p, target, a.probe);
                emit(&mut out, &w, r)?;
            }
            Some("ACT") => {
                let act: usize = it.next().expect("ACT needs action").parse().unwrap();
                let steps: usize = it.next().map(|s| s.parse().unwrap()).unwrap_or(a.steps);
                let (sx, sy) = sites[act % a.actions];
                let mut cur = w.store.read(0);
                let mut prev = w.store.read(1);
                plane::excite(&mut cur, &mut prev, sx, sy, a.amp, w.params.seed);
                w.store.write(0, &cur);
                w.store.write(1, &prev);
                for _ in 0..steps {
                    w.tick();
                }
                let (c, p) = (w.store.read(0), w.store.read(1));
                let r = reward(&c, &p, target, a.probe);
                emit(&mut out, &w, r)?;
            }
            Some("STEP") => {
                let steps: usize = it.next().map(|s| s.parse().unwrap()).unwrap_or(a.steps);
                for _ in 0..steps {
                    w.tick();
                }
                let (c, p) = (w.store.read(0), w.store.read(1));
                let r = reward(&c, &p, target, a.probe);
                emit(&mut out, &w, r)?;
            }
            Some("QUIT") | None => break,
            Some(other) => {
                eprintln!("mu-serve: unknown command {:?}", other);
            }
        }
    }
    let _ = io::stdin().lock().read(&mut [0u8; 0]);
    Ok(())
}
