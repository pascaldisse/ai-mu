//! SURFACE — the field seen as 3D geometry (RENDER法, 2026-08-01).
//!
//! LAW: a field render is ALWAYS 3D — surface or volume. A heatmap is a DEBUG
//! instrument only and may never appear on a product path (`plane::encode_png`
//! is that debug instrument; it is not a renderer).
//!
//! Here the field slice IS a displaced surface: height h(x,y) = amplitude, the
//! wave literally standing up off the plane. Rendered by ray-marching the
//! heightfield (no mesh, no rasterizer, no GPU): per pixel one ray, fixed-step
//! march, bilinear height sample, hit refined by bisection, Lambert + Blinn
//! specular + height-cued albedo. Volume mode marches the same field as
//! emissive density (|amplitude| along the ray) for the same state seen as
//! a cloud instead of a skin.
//!
//! Ultradeterminism (ENTROPY.md): pure f32 arithmetic in a fixed loop order,
//! no rand, no clock, no threads. Same slice + same camera -> same bytes.

use crate::{Scalar, Slice};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Orbit angle around +Z (radians) and elevation (radians).
    pub yaw: f32,
    pub pitch: f32,
    /// Distance from the field centre, in grid cells.
    pub dist: f32,
    /// Vertical field of view (radians).
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { yaw: 0.6, pitch: 0.55, dist: 190.0, fov: 0.9 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceParams {
    /// Amplitude -> height gain, in grid cells per unit amplitude.
    pub gain: f32,
    /// March step, in grid cells.
    pub step: f32,
    /// Max march distance, in grid cells.
    pub far: f32,
    /// Light direction (normalized internally).
    pub light: [f32; 3],
    /// Volume mode: emissive density integration instead of a surface skin.
    pub volume: bool,
    /// Volume absorption per unit |amplitude| per cell.
    pub density: f32,
}

impl Default for SurfaceParams {
    fn default() -> Self {
        SurfaceParams {
            gain: 60.0,
            step: 0.75,
            far: 420.0,
            light: [0.45, -0.6, 0.66],
            volume: false,
            density: 6.0,
        }
    }
}

#[inline]
fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Bilinear height sample, periodic wrap (the field is a torus).
fn height(s: &Slice, gain: f32, x: f32, y: f32) -> f32 {
    let (w, h) = (s.cfg.width as i64, s.cfg.height as i64);
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let wrap = |v: i64, n: i64| ((v % n) + n) % n;
    let (ix0, iy0) = (wrap(x0 as i64, w), wrap(y0 as i64, h));
    let (ix1, iy1) = (wrap(ix0 + 1, w), wrap(iy0 + 1, h));
    let g = |ix: i64, iy: i64| s.data[iy as usize * s.cfg.width + ix as usize];
    let a = g(ix0, iy0) * (1.0 - fx) + g(ix1, iy0) * fx;
    let b = g(ix0, iy1) * (1.0 - fx) + g(ix1, iy1) * fx;
    (a * (1.0 - fy) + b * fy) * gain
}

fn normal(s: &Slice, p: &SurfaceParams, x: f32, y: f32) -> [f32; 3] {
    let e = 0.75f32;
    let hx = height(s, p.gain, x + e, y) - height(s, p.gain, x - e, y);
    let hy = height(s, p.gain, x, y + e) - height(s, p.gain, x, y - e);
    norm3([-hx / (2.0 * e), -hy / (2.0 * e), 1.0])
}

/// Camera basis + origin, in grid-cell world space centred on the field.
fn camera_frame(cam: &Camera, cx: f32, cy: f32) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let eye = [
        cx + cam.dist * cam.pitch.cos() * cam.yaw.cos(),
        cy + cam.dist * cam.pitch.cos() * cam.yaw.sin(),
        cam.dist * cam.pitch.sin(),
    ];
    let target = [cx, cy, 0.0];
    let fwd = norm3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let up = [0.0, 0.0, 1.0];
    let right = norm3([
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0],
    ]);
    let cup = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ];
    (eye, fwd, right, cup)
}

/// Render the slice as a 3D surface (or volume). Returns RGB8, w*h*3 bytes.
pub fn render_rgb(s: &Slice, cam: &Camera, p: &SurfaceParams, w: usize, h: usize) -> Vec<u8> {
    let (cx, cy) = (s.cfg.width as f32 * 0.5, s.cfg.height as f32 * 0.5);
    let (eye, fwd, right, cup) = camera_frame(cam, cx, cy);
    let light = norm3(p.light);
    let aspect = w as f32 / h as f32;
    let tan_half = (cam.fov * 0.5).tan();
    let mut out = vec![0u8; w * h * 3];

    for py in 0..h {
        for px in 0..w {
            let sx = (2.0 * (px as f32 + 0.5) / w as f32 - 1.0) * aspect * tan_half;
            let sy = (1.0 - 2.0 * (py as f32 + 0.5) / h as f32) * tan_half;
            let dir = norm3([
                fwd[0] + right[0] * sx + cup[0] * sy,
                fwd[1] + right[1] * sx + cup[1] * sy,
                fwd[2] + right[2] * sx + cup[2] * sy,
            ]);

            let rgb = if p.volume {
                march_volume(s, &eye, &dir, p)
            } else {
                march_surface(s, &eye, &dir, p, &light)
            };
            let i = (py * w + px) * 3;
            for k in 0..3 {
                out[i + k] = (rgb[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    out
}

fn sky(dir: &[f32; 3]) -> [f32; 3] {
    let t = (dir[2] * 0.5 + 0.5).clamp(0.0, 1.0);
    [0.02 + 0.05 * t, 0.03 + 0.07 * t, 0.05 + 0.14 * t]
}

fn march_surface(s: &Slice, eye: &[f32; 3], dir: &[f32; 3], p: &SurfaceParams, light: &[f32; 3]) -> [f32; 3] {
    let mut t = 0.0f32;
    let mut prev_diff = {
        let pt = [eye[0], eye[1], eye[2]];
        pt[2] - height(s, p.gain, pt[0], pt[1])
    };
    while t < p.far {
        t += p.step;
        let pt = [eye[0] + dir[0] * t, eye[1] + dir[1] * t, eye[2] + dir[2] * t];
        let diff = pt[2] - height(s, p.gain, pt[0], pt[1]);
        if diff <= 0.0 && prev_diff > 0.0 {
            // bisect the crossing for a stable silhouette
            let (mut lo, mut hi) = (t - p.step, t);
            for _ in 0..12 {
                let mid = 0.5 * (lo + hi);
                let q = [eye[0] + dir[0] * mid, eye[1] + dir[1] * mid, eye[2] + dir[2] * mid];
                if q[2] - height(s, p.gain, q[0], q[1]) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let tt = 0.5 * (lo + hi);
            let q = [eye[0] + dir[0] * tt, eye[1] + dir[1] * tt, eye[2] + dir[2] * tt];
            let n = normal(s, p, q[0], q[1]);
            let ndl = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);
            let v = norm3([-dir[0], -dir[1], -dir[2]]);
            let hv = norm3([light[0] + v[0], light[1] + v[1], light[2] + v[2]]);
            let spec = (n[0] * hv[0] + n[1] * hv[1] + n[2] * hv[2]).max(0.0).powf(48.0);
            // albedo cued by SIGNED height (crest vs trough), not a colormap:
            // one hue, luminance carries the geometry.
            let hgt = (q[2] / p.gain.max(1e-6)).clamp(-1.0, 1.0);
            let base = [0.25 + 0.35 * hgt, 0.42 + 0.30 * hgt, 0.62 + 0.20 * hgt];
            let amb = 0.10;
            return [
                base[0] * (amb + 0.9 * ndl) + spec * 0.6,
                base[1] * (amb + 0.9 * ndl) + spec * 0.6,
                base[2] * (amb + 0.9 * ndl) + spec * 0.6,
            ];
        }
        prev_diff = diff;
    }
    sky(dir)
}

fn march_volume(s: &Slice, eye: &[f32; 3], dir: &[f32; 3], p: &SurfaceParams) -> [f32; 3] {
    // The field as a cloud: |amplitude| is density in a slab around z=0.
    let slab = p.gain;
    let mut t = 0.0f32;
    let mut trans = 1.0f32;
    let mut acc = [0.0f32; 3];
    while t < p.far && trans > 0.01 {
        t += p.step;
        let q = [eye[0] + dir[0] * t, eye[1] + dir[1] * t, eye[2] + dir[2] * t];
        if q[2].abs() > slab {
            continue;
        }
        let a = height(s, 1.0, q[0], q[1]);
        // vertical falloff: the wave occupies a slab whose thickness follows |a|
        let hz = (a.abs() * p.gain).max(1e-6);
        if q[2].abs() > hz {
            continue;
        }
        let d = a.abs() * p.density * p.step;
        let em = if a >= 0.0 {
            [0.35 + 0.65 * a.abs(), 0.55, 0.85]
        } else {
            [0.85, 0.45, 0.35 + 0.65 * a.abs()]
        };
        let alpha = 1.0 - (-d).exp();
        for k in 0..3 {
            acc[k] += trans * alpha * em[k];
        }
        trans *= 1.0 - alpha;
    }
    let bg = sky(dir);
    [acc[0] + trans * bg[0], acc[1] + trans * bg[1], acc[2] + trans * bg[2]]
}

/// RGB8 PNG in memory.
pub fn encode_rgb_png(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().expect("png header");
        wr.write_image_data(rgb).expect("png data");
    }
    out
}

/// Bit-exact digest of a render — determinism gate target.
pub fn digest_rgb(rgb: &[u8]) -> u64 {
    crate::fnv1a(rgb)
}

/// Scalar type re-export guard: renders consume the ONE field scalar.
const _: fn(Scalar) -> Scalar = |v| v;
