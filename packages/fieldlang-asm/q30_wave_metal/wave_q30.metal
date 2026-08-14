// CONTRACT (host<->shader, external interface; see CONTRACT.md):
//   kernel name   : q30_wave
//   buffer(0)     : device const int* cur    (width*height little-endian i32)
//   buffer(1)     : device const int* prev   (same shape; may alias cur exactly)
//   buffer(2)     : device int* out          (same shape; must not overlap cur/prev)
//   buffer(3)     : device atomic_uint* sat_count (single element, host zeroes it)
//   buffer(4)     : constant Params&         { uint width, height;
//                                              uint c_cur_lo; int c_cur_hi;
//                                              uint c_lap_lo; int c_lap_hi;
//                                              uint c_prev_lo; int c_prev_hi;
//                                              uint count; uint pad; } 40 bytes
//   count         : host-side u64-checked width*height (u32-overflow rejected on the
//                   host before any dispatch). The kernel recomputes n = w*h and
//                   refuses to run any thread unless n == count (product agreement),
//                   then guards gid < count.
//   dispatch      : dispatchThreads(count,1,1) tpg (64,1,1); kernel guards gid too.
//   lane law      : lap = cur[y][x-1]+cur[y][x+1]+cur[y-1][x]+cur[y+1][x] - 4*cur[y][x]
//                   (every index periodic; width==1 or height==1 wrap onto themselves)
//                   q(c,v) = (c*v + 2^29) arith>>30 , computed per coefficient independently
//                   acc = q(c_cur,cur) + q(c_lap,lap) + q(c_prev,prev)   [true 64-bit]
//                   out = sat_i32(acc)  -- saturation applied exactly once, after the sum
//   result        : sat_count = number of cells whose acc left signed 32-bit range
// No native 64-bit integer is used: every 64-bit quantity is an explicit
// (lo u32, hi i32) pair, so the law holds on any Metal family.
// FieldLang generation point: this file is the emission target of the 場語 -> MSL
// backend for the two-dimensional five-point wave; the generator must emit exactly
// this kernel name, buffer order and lane law.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width, height;
    uint c_cur_lo;  int c_cur_hi;
    uint c_lap_lo;  int c_lap_hi;
    uint c_prev_lo; int c_prev_hi;
    uint count;     uint pad;
};

struct i64p { uint lo; int hi; };

static inline i64p from_i32(int v) { i64p r; r.lo = uint(v); r.hi = v >> 31; return r; }

static inline i64p add64(i64p a, i64p b) {
    i64p r;
    r.lo = a.lo + b.lo;
    uint carry = (r.lo < a.lo) ? 1u : 0u;
    r.hi = int(uint(a.hi) + uint(b.hi) + carry);
    return r;
}

static inline i64p neg64(i64p a) {
    i64p r;
    r.lo = ~a.lo + 1u;
    r.hi = int(~uint(a.hi) + ((a.lo == 0u) ? 1u : 0u));
    return r;
}

// unsigned 32x32 -> 64
static inline void umul32(uint a, uint b, thread uint &lo, thread uint &hi) {
    uint a0 = a & 0xffffu, a1 = a >> 16;
    uint b0 = b & 0xffffu, b1 = b >> 16;
    uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);
    lo = (p00 & 0xffffu) | (mid << 16);
    hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
}

// low 64 bits of a full 64x64 product; exact whenever the true product fits in i64.
static inline i64p mul64(i64p a, i64p b) {
    uint lo, hi;
    umul32(a.lo, b.lo, lo, hi);
    hi += a.lo * uint(b.hi);
    hi += uint(a.hi) * b.lo;
    i64p r; r.lo = lo; r.hi = int(hi); return r;
}

// (v + 2^29) arithmetic >> 30, exact on the (lo,hi) pair.
static inline i64p q30_round(i64p v) {
    i64p bias; bias.lo = 1u << 29; bias.hi = 0;
    i64p s = add64(v, bias);
    i64p r;
    r.lo = (s.lo >> 30) | (uint(s.hi) << 2);
    r.hi = s.hi >> 30;
    return r;
}

kernel void q30_wave(device const int *cur     [[buffer(0)]],
                     device const int *prev    [[buffer(1)]],
                     device int *out           [[buffer(2)]],
                     device atomic_uint *sat_count [[buffer(3)]],
                     constant Params &p        [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    uint w = p.width, h = p.height;
    uint n = w * h;
    if (n != p.count) { return; }   // host/kernel dimensions-product agreement
    if (gid >= p.count) { return; }
    uint y = gid / w;
    uint x = gid - y * w;

    uint xm = (x == 0u) ? (w - 1u) : (x - 1u);
    uint xp = (x + 1u == w) ? 0u : (x + 1u);
    uint ym = (y == 0u) ? (h - 1u) : (y - 1u);
    uint yp = (y + 1u == h) ? 0u : (y + 1u);

    int c0 = cur[gid];
    int cl = cur[y * w + xm];
    int cr = cur[y * w + xp];
    int cu = cur[ym * w + x];
    int cd = cur[yp * w + x];

    // lap in true 64-bit: neighbours summed, then 4*centre subtracted.
    i64p lap = add64(add64(from_i32(cl), from_i32(cr)), add64(from_i32(cu), from_i32(cd)));
    i64p c4 = from_i32(c0);
    c4 = add64(c4, c4);
    c4 = add64(c4, c4);
    lap = add64(lap, neg64(c4));

    i64p k_cur;  k_cur.lo  = p.c_cur_lo;  k_cur.hi  = p.c_cur_hi;
    i64p k_lap;  k_lap.lo  = p.c_lap_lo;  k_lap.hi  = p.c_lap_hi;
    i64p k_prev; k_prev.lo = p.c_prev_lo; k_prev.hi = p.c_prev_hi;

    // each coefficient product rounds independently, then the terms are summed.
    i64p acc = q30_round(mul64(k_cur, from_i32(c0)));
    acc = add64(acc, q30_round(mul64(k_lap, lap)));
    acc = add64(acc, q30_round(mul64(k_prev, from_i32(prev[gid]))));

    // exactly one saturation, after the sum.
    bool fits = (acc.hi == 0  && (acc.lo >> 31) == 0u) ||
                (acc.hi == -1 && (acc.lo >> 31) == 1u);
    int res;
    if (fits) {
        res = int(acc.lo);
    } else {
        res = (acc.hi >= 0) ? 0x7fffffff : int(0x80000000u);
        atomic_fetch_add_explicit(sat_count, 1u, memory_order_relaxed);
    }
    out[gid] = res;
}
