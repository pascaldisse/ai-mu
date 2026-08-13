//! field-view3 — 3D view of the field (RENDER法: surface/volume only).
//! argv: --width/--height (field dims) --out (png path) --px (image side)
//!       --frames --excite-x --excite-y --seed --volume --yaw --pitch --dist --gain
//! Heatmaps are debug-only and NOT emitted here.
use field::plane::WaveParams;
use field::surface::{digest_rgb, encode_rgb_png, render_rgb, Camera, SurfaceParams};
use field::world::World;
use field::FieldConfig;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str, d: f32| -> f32 {
        a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let sget = |k: &str, d: &str| -> String {
        a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned().unwrap_or_else(|| d.to_string())
    };
    let (w, h) = (get("--width", 128.0) as usize, get("--height", 128.0) as usize);
    let px = get("--px", 480.0) as usize;
    let frames = get("--frames", 24.0) as usize;
    let seed = get("--seed", 0.0) as u64;
    let cfg = FieldConfig::new(w, h);
    let mut world = World::new(cfg, WaveParams { seed, ..WaveParams::default() }, 4);
    let (ex, ey) = (get("--excite-x", 44.0) as usize, get("--excite-y", 64.0) as usize);
    world.apply(&field::journal::Op::Excite { x: ex as u32, y: ey as u32, amp_bits: 0.6f32.to_bits() });
    for _ in 0..frames {
        world.tick();
    }
    let cam = Camera {
        yaw: get("--yaw", 0.6),
        pitch: get("--pitch", 0.55),
        dist: get("--dist", 190.0),
        fov: get("--fov", 0.9),
    };
    let p = SurfaceParams {
        gain: get("--gain", 60.0),
        volume: a.iter().any(|x| x == "--volume"),
        ..SurfaceParams::default()
    };
    let rgb = render_rgb(&world.cur(), &cam, &p, px, px);
    let png = encode_rgb_png(&rgb, px, px);
    let out = sget("--out", "field-view3.png");
    std::fs::write(&out, &png).expect("write png");
    println!(
        "field-view3 out={} px={} frames={} mode={} field_digest={:#x} render_digest={:#x}",
        out,
        px,
        frames,
        if p.volume { "volume" } else { "surface" },
        world.digest(),
        digest_rgb(&rgb)
    );
}
