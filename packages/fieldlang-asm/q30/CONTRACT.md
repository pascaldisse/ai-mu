# Q30 decay · ARM64-only ABI v1

`_fl_q30_decay_neon(x0=cells, x1=count, w2=coeff) -> x0=saturation_count`.
`_fl_q30_decay_scalar` has exactly the same ABI; it is gate oracle only.

- `cells`: mutable array of `count` signed little-endian `i32`; any byte alignment valid.
- in-place only: input/output are identical array; no other overlapping ranges exist.
- `coeff`: every signed `i32` bit-pattern accepted.
- lane law: `p=(int64_t)coeff * cell`; `q=(p + 2^29) arithmetic>>30`; `out=sat_i32(q)`.
  Thus halfway values round toward positive infinity, including negative products.
- result: number of lanes where `q` was outside signed `i32` range; `count=0` neither reads nor writes.
- AAPCS64: x0-x18/v0-v7,v16-v31 scratch; callee-saved registers preserved.

Product path is hand-written ARM64 assembly. NEON vector body processes four i32 lanes
with `smull`, `smull2`, `sshr`, `sqxtn`, `sqxtn2`; scalar tail covers 0..3 lanes exactly.
No alignment promise is required from callers.
