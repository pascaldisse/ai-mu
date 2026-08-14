// CONTRACT (host<->shader, external interface; see CONTRACT.md §shader):
//   kernel name          : q30_decay
//   buffer(0)            : device int* cells      (in place, count elements)
//   buffer(1)            : device atomic_uint* sat_count (single element, host zeroes it)
//   buffer(2)            : constant Params&       { int coeff; uint count; }
//   dispatch             : dispatchThreads(count,1,1) tpg (64,1,1)
//   lane law             : p = (i64)coeff*cell ; q = (p + 2^29) arith>>30 ; out = sat_i32(q)
//   no native 64-bit int is used: the 64-bit product is carried as explicit lo/hi u32.
// FieldLang generation point: this file is the *emission target* of the future
// FieldLang Metal backend (場語 -> MSL). The generator must emit exactly this
// kernel signature and lane law; nothing else in the host bridge may change.
#include <metal_stdlib>
using namespace metal;

struct Params { int coeff; uint count; };

// signed 32x32 -> 64 product, returned as lo (u32) and hi (i32).
static inline void mul_s32_64(int a, int b, thread uint &lo, thread int &hi) {
    uint ua = uint(a), ub = uint(b);
    uint a0 = ua & 0xffffu, a1 = ua >> 16;
    uint b0 = ub & 0xffffu, b1 = ub >> 16;
    uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);
    lo = (p00 & 0xffffu) | (mid << 16);
    uint hiu = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
    if (a < 0) { hiu -= ub; }
    if (b < 0) { hiu -= ua; }
    hi = int(hiu);
}

kernel void q30_decay(device int *cells [[buffer(0)]],
                      device atomic_uint *sat_count [[buffer(1)]],
                      constant Params &p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) { return; }
    uint lo; int hi;
    mul_s32_64(p.coeff, cells[gid], lo, hi);
    // + 2^29 with explicit carry
    uint bias = 1u << 29;
    uint nlo = lo + bias;
    if (nlo < lo) { hi = int(uint(hi) + 1u); }
    lo = nlo;
    // arithmetic >> 30 of the 64-bit value
    uint q_lo = (lo >> 30) | (uint(hi) << 2);
    int  q_hi = hi >> 30;                    // arithmetic in MSL for signed int
    int  out;
    bool fits = (q_hi == 0 && (q_lo >> 31) == 0u) ||
                (q_hi == -1 && (q_lo >> 31) == 1u);
    if (fits) {
        out = int(q_lo);
    } else {
        out = (q_hi >= 0) ? 0x7fffffff : int(0x80000000u);
        atomic_fetch_add_explicit(sat_count, 1u, memory_order_relaxed);
    }
    cells[gid] = out;
}
