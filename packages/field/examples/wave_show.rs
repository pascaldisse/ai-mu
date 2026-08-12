//! wave_show — 目視用: 撃→波紋→PNG連番 (proof/wave-show/)
use field::fft::Fft2;
use field::plane::{encode_png, excite, wave_step, WaveKernel, WaveParams};
use field::{FieldConfig, Slice};
use std::fs;

fn main() {
    let (w, h) = (64usize, 64usize);
    let cfg = FieldConfig::new(w, h);
    let p = WaveParams { seed: 42, ..WaveParams::default() };
    assert!(p.stable(), "CFL");
    let f = Fft2::new(cfg);
    let k = WaveKernel::new(cfg, &p);

    let mut cur = Slice::zeros(cfg);
    let mut prev = Slice::zeros(cfg);
    excite(&mut cur, &mut prev, w / 2, h / 2, 1.0, p.seed);

    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof/wave-show");
    fs::create_dir_all(&out).unwrap();
    let every = 10usize;
    for step in 0..=120usize {
        if step % every == 0 {
            let png = encode_png(&cur, 0.06);
            fs::write(out.join(format!("wave-{step:03}.png")), png).unwrap();
        }
        let next = wave_step(&f, &k, &cur, &prev);
        prev = cur;
        cur = next;
    }
    println!("wrote frames to {}", out.display());
}
