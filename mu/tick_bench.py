"""mu/tick_bench.py — measure ms/tick and dispatch count of the fused
interactive path (repr -> pred -> dyn) via mx.compile, on THIS machine.

Interactive tick = one 0-sim amortized step (field-recon §3 option (a)):
  s      = repr(obs)              # encode current field slice
  p, v   = pred(s)                # amortized policy/value, no search
  s2, r  = dyn(s, action_onehot)  # advance world-model latent + predicted reward
All three calls wrapped in ONE mx.compile'd function -> traced once, replayed
every tick (no Python-side graph rebuild, no per-call recompilation as long as
shapes are static).

HARD CONSTRAINT (task spec): interactive tick <=4-8 dispatches, fused;
interactive forward < 8.33ms (120fps floor, FIELD.md).

MEASUREMENT METHOD:
- ms/tick: REAL wall-clock, synchronized per call (mx.eval each tick) over
  many iterations after a warmup (compile) call. This is a genuine measurement
  on this machine, not a budget estimate.
- dispatch count: mlx has no public CLI-parseable Metal-dispatch counter.
  mx.metal.start_capture()/stop_capture() writes a .gputrace bundle but it is
  a binary Xcode-GPU-debugger format with no headless parser (checked: `file`
  shows opaque binary/zlib blobs, plutil/xctrace do not decode it) -> a true
  per-tick GPU dispatch count is UNVERIFIED at v0 (would need Xcode Instruments
  GUI). What IS reported below is a STRUCTURAL ESTIMATE: the count of distinct
  matmul/conv primitive ops in the compiled graph (repr: conv+conv+linear=3,
  dyn: linear+linear+linear=3, pred: linear+linear+linear=3 -> 9 total). This
  is an upper bound on real GPU kernel dispatches (mx.compile may fuse some
  elementwise ops into adjacent matmul kernels, never fewer matmul kernels
  than distinct weight matrices). Marked ESTIMATE, not measured.
"""
import time
import statistics

import mlx.core as mx

from nets import MuNet, GRID, CHANNELS, action_onehot

# structural op count per net (matmul/conv primitives only; relu/concat/reshape
# are elementwise/free and fuse into the producer or consumer kernel)
STRUCTURAL_OPS = {
    "repr (conv+conv+linear)": 3,
    "dyn (trunk+combined_head)": 2,
    "pred (trunk+combined_head)": 2,
}


def build_tick_fn(net):
    def tick(obs, a_onehot):
        s = net.repr(obs)
        policy_logits, value = net.pred(s)
        s2, r = net.dyn(s, a_onehot)
        return s2, r, policy_logits, value

    return mx.compile(tick)


def bench(iters=2000, warmup=50):
    net = MuNet()
    tick_fn = build_tick_fn(net)

    obs = mx.zeros((1, GRID, GRID, CHANNELS))
    a_onehot = action_onehot(mx.array([0], dtype=mx.uint32))

    # warmup: first call triggers mx.compile trace; discard from timing
    for _ in range(warmup):
        out = tick_fn(obs, a_onehot)
        mx.eval(out)

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = tick_fn(obs, a_onehot)
        mx.eval(out)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    mean_ms = sum(times_ms) / n
    median_ms = times_ms[n // 2]
    p95_ms = times_ms[int(n * 0.95)]
    p99_ms = times_ms[int(n * 0.99)]
    min_ms = times_ms[0]
    max_ms = times_ms[-1]

    total_structural_dispatch = sum(STRUCTURAL_OPS.values())

    print(f"mlx device: {mx.default_device()}")
    print(f"iters={iters} (after {warmup} warmup calls, discarded)")
    print(f"ms/tick: mean={mean_ms:.4f} median={median_ms:.4f} "
          f"p95={p95_ms:.4f} p99={p99_ms:.4f} min={min_ms:.4f} max={max_ms:.4f}")
    print(f"120fps budget = 8.3333 ms/frame")
    print(f"MEASURED mean < 8.33ms budget: {mean_ms < 8.3333}")
    print(f"MEASURED p99  < 8.33ms budget: {p99_ms < 8.3333}")
    print()
    print("dispatch count (STRUCTURAL ESTIMATE, not a measured GPU trace — see module docstring):")
    for name, n_ops in STRUCTURAL_OPS.items():
        print(f"  {name}: {n_ops}")
    print(f"  TOTAL structural matmul/conv ops per tick: {total_structural_dispatch}")
    print(f"  hard constraint 4-8 dispatches: "
          f"{'PASS' if total_structural_dispatch <= 8 else 'OVER by ' + str(total_structural_dispatch - 8)} "
          f"(estimate; real GPU dispatch count UNVERIFIED, no headless Metal profiler available)")

    return {
        "mean_ms": mean_ms, "median_ms": median_ms, "p95_ms": p95_ms,
        "p99_ms": p99_ms, "min_ms": min_ms, "max_ms": max_ms,
        "structural_dispatch_estimate": total_structural_dispatch,
    }


if __name__ == "__main__":
    bench()
