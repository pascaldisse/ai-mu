# Q30 decay · Metal direct path · ARM64-only host ABI v1

Product path: hand-written ARM64 assembly only (`metal_bridge.s`). No Rust, no wgpu,
no Swift, no Objective-C, no C source exists on this path; every Objective-C message
is an explicit hand-emitted `objc_msgSend` call. There is **no CPU fallback**: if the
GPU path cannot complete, the entry point reports failure.

## Host entry points (public surface for independent runners)

- `_fl_metal_init(x0 = const char *metallib_path) -> x0 = 0 ok / 1 fail`
  Creates the system default `MTLDevice`, loads `metallib_path`, builds the
  `q30_decay` compute pipeline state and one `MTLCommandQueue`. Must be called once
  before any other entry point. State lives in `_g_dev`, `_g_queue`, `_g_pso`.
- `_fl_metal_print_device()` — writes raw evidence to fd 1:
  `metal device: <[dev name] UTF8String>` and
  `metal families supported: <MTLGPUFamily ints for which supportsFamily: is YES>`.
  Family integers are the raw enum values (Apple1..9 = 1001..1009,
  Common1..3 = 3001..3003, Metal3 = 5001).
- `_fl_q30_decay_metal(x0 = cells, x1 = count, w2 = coeff) -> x0 = saturations`
  Same ABI as `_fl_q30_decay_scalar` / `_fl_q30_decay_neon` (see `../q30/CONTRACT.md`),
  executed on the GPU. Returns `-1` if the GPU path did not complete.
  `count = 0` neither reads nor writes and returns 0.
- `_fl_q30_metal_bench(x0 = cells, x1 = count, w2 = coeff, x3 = reps) -> x0 = 0 ok / 1 fail`
  One upload, `reps` real dispatches with the field buffer kept resident on the GPU
  between them, one readback. Measurement only; no speed claim is made anywhere.
- `_fl_put(x0 = cstr)`, `_fl_putdec(x0 = signed)` — write(2) helpers, no libc formatting.

AAPCS64 applies: x0-x18/v0-v7,v16-v31 scratch, callee-saved registers preserved.

## Lane law (identical to the accepted scalar/NEON path)

`p = (int64)coeff * cell` ; `q = (p + 2^29) arithmetic >> 30` ; `out = sat_i32(q)` ;
result = number of lanes where `q` left signed 32-bit range. Halfway values round
toward positive infinity, including negative products.

## Shader interface (external host contract)

File `decay_q30.metal`, built by `xcrun metal` + `xcrun metallib` into `q30.metallib`.

| slot | binding | meaning |
|------|---------|---------|
| kernel | `q30_decay` | entry point name, matched by `newFunctionWithName:` |
| `buffer(0)` | `device int *cells` | in place, `count` little-endian signed i32 |
| `buffer(1)` | `device atomic_uint *sat_count` | single counter, host zeroes it before dispatch |
| `buffer(2)` | `constant Params&` | `{ int coeff; uint count; }`, 8 bytes, host `setBytes:` |

Dispatch: `dispatchThreads:(count,1,1) threadsPerThreadgroup:(64,1,1)`; the kernel also
guards `gid >= count` itself. The shader uses **no native 64-bit integer**: the 64-bit
product is carried as an explicit `lo` u32 / `hi` i32 pair, so the law holds on any
Metal family regardless of int64 support.

### FieldLang generation point

`decay_q30.metal` is the emission target of the future FieldLang Metal backend
(場語 → MSL). The generator must emit exactly the kernel name, the buffer order and
the lane law above; the host bridge stays untouched when the shader is generated.

## Buffer storage

`newBufferWithBytes:length:options:` / `newBufferWithLength:options:` with options `0`
(= `MTLResourceStorageModeShared`), so host and GPU observe one copy of the field;
results are read back through `[buffer contents]`.

## Command flow (real, checked)

`commandBuffer` → `computeCommandEncoder` → `setComputePipelineState:` →
`setBuffer:offset:atIndex:` ×2 → `setBytes:length:atIndex:` → `dispatchThreads:...` →
`endEncoding` → `commit` → `waitUntilCompleted` → `status` must equal
`MTLCommandBufferStatusCompleted (4)`, otherwise failure is reported.
