// V9-WIRE GPU FUSE (task mandate, 2026-07-25) — the v9 eye-test hitgate
// compose, on the GPU. Byte-for-byte port of the CPU reference
// (`rdirect_v9_compose::compose_cpu_reference` / the old
// `NetPresent::v9_hitgate_compose` in main.rs): for every no-hit px
// (depth<=0, only when `hitgate` is set), replace the net's raw demod-log
// output with the bilinear evidence-composite's own demod-log value — SAME
// construction `gather_v9`'s own bilinear read uses
// (rdirect_gather_split.wgsl::low_coord, SAME clamped 2x2 neighbourhood).
// Every hit px (or every px when `hitgate` is off) passes the net's raw
// output through unchanged.
//
// Replaces main.rs's old two-buffer GPU->CPU readback (native AOV +
// low-res split E/D) + CPU per-pixel loop + CPU->GPU re-upload with ONE
// compute dispatch reading both source buffers directly on the GPU. The
// net's own raw output (`net_out`) still arrives via one CPU->GPU write
// (MPSGraph's output readback is unavoidable — see
// `rdirect_unet::run_compiled_forward`'s own doc) but that upload is now
// the ONLY CPU touch in the resolve stage.

struct ComposeU {
  n: u32,
  low_w: u32,
  low_h: u32,
  target_w: u32,
  target_h: u32,
  hitgate: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: ComposeU;
// Net's raw demod-log output, row-major [n,3] f32 (uploaded once/frame from
// the MPSGraph readback — see module doc).
@group(0) @binding(1) var<storage, read> net_out: array<f32>;
// Native-res AOV: 2 vec4/px; [2i+0] = (albedo.xyz, depth).
@group(0) @binding(2) var<storage, read> aov: array<vec4<f32>>;
// Low-res split evidence trace: 2 vec4/low-px; [2j+0] = E (rgb sum, count),
// [2j+1] = D (rgb sum, count) — SAME layout `gather_v9`/`gather_split` read.
@group(0) @binding(3) var<storage, read> accum_ed: array<vec4<f32>>;
// Output: hit-gated net out, row-major [n,3] f32 (feeds `demod`/`evidence`
// exactly like the old CPU-composed buffer did).
@group(0) @binding(4) var<storage, read_write> gated: array<f32>;

const ALBEDO_DEMOD_EPS: f32 = 1.0e-3;
const NO_HIT_SQ: f32 = 1.0e-8;

// rdirect.rs::low_coord — target index -> continuous low-res coordinate.
// SAME as rdirect_gather_split.wgsl's own copy (kept local: this shader has
// no other WGSL include mechanism in this crate's build).
fn low_coord(t: u32, low: u32, tgt: u32) -> f32 {
  return (f32(t) + 0.5) * f32(low) / f32(tgt) - 0.5;
}

// e/ec + d/dc for low-res cell `j` — mirrors the CPU `cell()` closure
// exactly (both E and D counts floor-clamped to >=1 independently).
fn ed_composite(j: u32) -> vec3<f32> {
  let e = accum_ed[2u * j + 0u];
  let d = accum_ed[2u * j + 1u];
  return e.xyz / max(e.w, 1.0) + d.xyz / max(d.w, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn v9_compose(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tw = u.target_w;
  let th = u.target_h;
  if (gid.x >= tw || gid.y >= th) { return; }
  let tx = gid.x;
  let ty = gid.y;
  let px = ty * tw + tx;

  let a = aov[2u * px + 0u];
  let albedo = a.xyz;
  let depth = a.w;

  // hitgate off, or a hit px: net's own raw output stands unchanged (SAME
  // early-continue the CPU reference takes).
  if (u.hitgate == 0u || depth > 0.0) {
    gated[3u * px + 0u] = net_out[3u * px + 0u];
    gated[3u * px + 1u] = net_out[3u * px + 1u];
    gated[3u * px + 2u] = net_out[3u * px + 2u];
    return;
  }

  let lw = u.low_w;
  let lh = u.low_h;
  let fx = low_coord(tx, lw, tw);
  let fy = low_coord(ty, lh, th);
  let x0 = floor(fx);
  let y0 = floor(fy);
  let dx = fx - x0;
  let dy = fy - y0;
  let x0i = min(u32(max(x0, 0.0)), lw - 1u);
  let x1i = min(u32(max(x0 + 1.0, 0.0)), lw - 1u);
  let y0i = min(u32(max(y0, 0.0)), lh - 1u);
  let y1i = min(u32(max(y0 + 1.0, 0.0)), lh - 1u);

  let c00 = ed_composite(y0i * lw + x0i);
  let c10 = ed_composite(y0i * lw + x1i);
  let c01 = ed_composite(y1i * lw + x0i);
  let c11 = ed_composite(y1i * lw + x1i);

  let top = c00 * (1.0 - dx) + c10 * dx;
  let bot = c01 * (1.0 - dx) + c11 * dx;
  let composite = top * (1.0 - dy) + bot * dy;

  let alb_sq = dot(albedo, albedo);
  var divisor = vec3<f32>(1.0, 1.0, 1.0);
  if (alb_sq > NO_HIT_SQ) {
    divisor = albedo + vec3<f32>(ALBEDO_DEMOD_EPS);
  }

  let out = log(max(composite / divisor, vec3<f32>(0.0)) + vec3<f32>(1.0));
  gated[3u * px + 0u] = out.x;
  gated[3u * px + 1u] = out.y;
  gated[3u * px + 2u] = out.z;
}
