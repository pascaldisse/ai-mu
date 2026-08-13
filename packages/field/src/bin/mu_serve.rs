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
//!   VIEW3 <path> [volume] [at <k>]
//!                         -> 3D render (RENDER法: surface/volume only, never a
//!                            heatmap) of the world AT journal position k
//!                            (default: now), reconstructed by replay — no
//!                            frame is ever stored. Then emits a normal frame.
//!   WHEN                  -> 24-byte record: u64 step_index | u64 journal_pos |
//!                            f32 energy | f32 amp_absmax. The observer axis.
//!   PATCH <x> <y> <k>     -> mu-inside atom-1 段1: READ-ONLY local window, no
//!                            mutation, no journal op. u64 digest | u64 step_index
//!                            | f32 0.0 (reward field unused, kept for uniform
//!                            header) | f32[2*k*k] (cur patch then prev patch,
//!                            row-major, torus-wrapped, centered at (x,y) with
//!                            half=k/2 so world_x = (x-half+dx) mod width).
//!   POKEXY <x> <y> [steps] -> mu-inside atom-1 段1: excite the field at an
//!                            ARBITRARY (x,y) (not a declared ring site) and
//!                            tick — the world-side half of the window atom's
//!                            self-poke action. Same journal/emit shape as ACT.
//!   QUIT
//!
//! 4D BLOCK (主令 2026-08-01): the trajectory is a 4D block, held IMPLICITLY.
//! seed + journal ops ARE the compression of every moment; state(t) = replay of
//! the first t ops, a pure function, so "past" costs storage O(ops), not O(t*d).
//! "WHEN" is answerable because the observer sits at a journal POSITION on the
//! entropy axis — time is where you are in the block, not a thing that flowed.
//!
//! stdout per frame (raw little-endian, no framing bytes):
//!   u64 digest | u64 step_index | f32 reward | f32[2*d] obs (cur then prev)
//! stderr: human notes only.

use field::plane::{self, WaveParams};
use field::surface::{digest_rgb, encode_rgb_png, render_rgb, Camera, SurfaceParams};
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
    // THE BLOCK: seed + ops = every moment of this world, compressed.
    let mut cur_seed = a.seed;
    let mut ops: Vec<field::journal::Op> = Vec::new();

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
                cur_seed = seed;
                ops.clear(); // new block: seed alone is the whole past so far
                let (c, p) = (w.store.read(0), w.store.read(1));
                let r = reward(&c, &p, target, a.probe);
                emit(&mut out, &w, r)?;
            }
            Some("ACT") => {
                let act: usize = it.next().expect("ACT needs action").parse().unwrap();
                let steps: usize = it.next().map(|s| s.parse().unwrap()).unwrap_or(a.steps);
                let (sx, sy) = sites[act % a.actions];
                ops.push(field::journal::Op::Excite {
                    x: sx as u32,
                    y: sy as u32,
                    amp_bits: a.amp.to_bits(),
                });
                ops.push(field::journal::Op::Step { count: steps as u32 });
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
            Some("WHEN") => {
                // The observer axis: where this observer sits in the block.
                let cur = w.store.read(0);
                let prev = w.store.read(1);
                let energy = field::plane::energy(&w.params, &cur, &prev) as f32;
                let absmax = cur.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                out.write_all(&w.step_index.to_le_bytes())?;
                out.write_all(&(ops.len() as u64).to_le_bytes())?;
                out.write_all(&energy.to_le_bytes())?;
                out.write_all(&absmax.to_le_bytes())?;
                out.flush()?;
            }
            Some("STEP") => {
                let steps: usize = it.next().map(|s| s.parse().unwrap()).unwrap_or(a.steps);
                ops.push(field::journal::Op::Step { count: steps as u32 });
                for _ in 0..steps {
                    w.tick();
                }
                let (c, p) = (w.store.read(0), w.store.read(1));
                let r = reward(&c, &p, target, a.probe);
                emit(&mut out, &w, r)?;
            }
            Some("VIEW3") => {
                let path = it.next().unwrap_or("mu-view3.png").to_string();
                let rest: Vec<&str> = it.collect();
                let volume = rest.iter().any(|t| *t == "volume");
                let at = rest
                    .iter()
                    .position(|t| *t == "at")
                    .and_then(|i| rest.get(i + 1))
                    .and_then(|v| v.parse::<usize>().ok());
                let cam = Camera::default();
                let p = SurfaceParams { volume, ..SurfaceParams::default() };
                // time travel = replay, never a stored frame (4D block law)
                let viewed = match at {
                    None => w.store.read(0),
                    Some(k) => {
                        let mut past = reset(&a, cur_seed);
                        for op in ops.iter().take(k.min(ops.len())) {
                            past.apply(op);
                        }
                        past.store.read(0)
                    }
                };
                let rgb = render_rgb(&viewed, &cam, &p, 480, 480);
                std::fs::write(&path, encode_rgb_png(&rgb, 480, 480)).expect("write png");
                eprintln!(
                    "mu-serve VIEW3 {} at={:?} journal_pos={} step={} field_digest={:#x} render_digest={:#x}",
                    path,
                    at,
                    ops.len(),
                    w.step_index,
                    w.digest(),
                    digest_rgb(&rgb)
                );
                let (c, p2) = (w.store.read(0), w.store.read(1));
                let r = reward(&c, &p2, target, a.probe);
                emit(&mut out, &w, r)?;
            }
            Some("PATCH") => {
                // mu-inside atom-1 段1: pure read, no mutation, no journal op
                // (4D-block law: only mutating ops belong in `ops`).
                let x: i64 = it.next().expect("PATCH needs x").parse().unwrap();
                let y: i64 = it.next().expect("PATCH needs y").parse().unwrap();
                let k: usize = it.next().expect("PATCH needs k").parse().unwrap();
                let cur = w.store.read(0);
                let prev = w.store.read(1);
                let cfgw = cur.cfg;
                out.write_all(&w.digest().to_le_bytes())?;
                out.write_all(&w.step_index.to_le_bytes())?;
                out.write_all(&0f32.to_le_bytes())?; // reward field unused for a pure read
                let half = (k / 2) as i64;
                for src in [&cur, &prev] {
                    for dy in 0..k as i64 {
                        for dx in 0..k as i64 {
                            let px = ((x + dx - half) % cfgw.width as i64 + cfgw.width as i64)
                                % cfgw.width as i64;
                            let py = ((y + dy - half) % cfgw.height as i64 + cfgw.height as i64)
                                % cfgw.height as i64;
                            let i = py as usize * cfgw.width + px as usize;
                            out.write_all(&src.data[i].to_bits().to_le_bytes())?;
                        }
                    }
                }
                out.flush()?;
            }
            Some("POKEXY") => {
                // mu-inside atom-1 段1: self-poke action — same shape as ACT but
                // at an arbitrary (x,y) instead of a declared ring site.
                let x: usize = it.next().expect("POKEXY needs x").parse().unwrap();
                let y: usize = it.next().expect("POKEXY needs y").parse().unwrap();
                let steps: usize = it.next().map(|s| s.parse().unwrap()).unwrap_or(a.steps);
                ops.push(field::journal::Op::Excite {
                    x: x as u32,
                    y: y as u32,
                    amp_bits: a.amp.to_bits(),
                });
                ops.push(field::journal::Op::Step { count: steps as u32 });
                let mut cur = w.store.read(0);
                let mut prev = w.store.read(1);
                plane::excite(&mut cur, &mut prev, x, y, a.amp, w.params.seed);
                w.store.write(0, &cur);
                w.store.write(1, &prev);
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
