// V9-WIRE ATOM (GATHER FOLLOW-UP #2, 2026-07-21) — GPU-side f32->fp16
// pack, so the v9 gather's f32 feature tensor can be written directly into
// the fp16 MTLBuffer `UnetLive::forward_gpu_bridged` reads — no CPU
// readback, no CPU conversion loop. `docs/perf/2026-07-21-v9-wire.md`'s own
// §gather decomposition named these two costs (~7.0ms readback + ~11.9ms
// fp16-glue, dominated by the INPUT side: 12.6M scalar elements at
// 640×480×41) as the remaining thieves after the gather itself moved
// GPU-native; both die together once the destination MTLBuffer is the
// SAME one the pack shader targets (`UnetLive::from_wgpu_queue`'s bridge).
//
// `pack2x16float(vec2<f32>) -> u32` (WGSL spec): e.x lands in the LOW 16
// bits, e.y in the HIGH 16 bits of the result — on a little-endian target
// (Apple Silicon) that is byte-for-byte a tightly-packed `[f16, f16]` pair
// in memory, which is exactly what an `MPSGraphTensorData` of
// `MPSDataType::Float16` expects for two consecutive elements. No
// `enable f16;` extension needed (that would require the `SHADER_F16`
// wgpu device feature, requested at device-creation time — out of reach
// for a bridge built from an ALREADY-CREATED wgpu device/queue, and this
// atom's own "no main.rs changes" scope keeps device creation untouched).
//
// Guarded entirely by `arrayLength` on both bindings — no uniform buffer:
// `dst` is sized by the caller to `ceil(src_len/2)` u32s exactly (the
// dispatch bound), `src`'s own length is read directly for the odd-tail
// case (padded with 0.0 in the unused high half, harmless — the caller's
// `dst` byte length never claims that padding element exists to the
// consumer, per `Fp16Packer::encode`'s own doc).

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

// `dst` is often > 65535*64 elements (e.g. 640x480x41/2 ~= 6.3M pairs) --
// Metal caps a single dispatch dimension at 65535 workgroups, so the caller
// (`Fp16Packer::encode`) dispatches a 2D grid (gx up to 65535, gy the
// remainder) instead of a 1D one. `num_workgroups` (a WGSL builtin, the
// caller's own dispatch_workgroups(gx, gy, 1) values) lets the shader
// recover the flat index with no extra uniform.
@compute @workgroup_size(64)
fn pack_fp16(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let row_threads = nwg.x * 64u;
  let i = gid.y * row_threads + gid.x;
  if (i >= arrayLength(&dst)) { return; }
  let base = i * 2u;
  let n = arrayLength(&src);
  let a = src[base];
  var b: f32 = 0.0;
  if (base + 1u < n) {
    b = src[base + 1u];
  }
  dst[i] = pack2x16float(vec2<f32>(a, b));
}
