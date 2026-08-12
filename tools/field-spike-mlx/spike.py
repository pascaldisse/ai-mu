#!/usr/bin/env python3
"""
W4 METAL/MLX SPIKE — measurement only, no Rust.
Measures MLX-on-Metal cost of the field-engine per-tick primitives on this
machine (Apple M1 Pro) against the 120fps budget (8.33ms/frame), per
adversary verdict §4 + docs/research/2026-08-01-field-recon.md §2.

Primitives:
  (1) bind  = fft2 -> complex elementwise multiply -> ifft2, fp32, at 128^2 and 512^2
  (2) probe = matmul key(1xd) @ store(Nxd).T, N in {4096, 16384}, d=16384
  (3) fused tick = bind(128^2) + bind(512^2) + probe(N=16384) under mx.compile

Each timed block: 20 warmup iters (not counted) + 200 measured iters.
Reports mean/p50/p99 ms + PASS/FAIL vs 8.33ms budget.
Also runs a numpy CPU baseline for (1) and (2) for comparison.

Deterministic: mx.random.seed(0), np.random.seed(0).

Outputs:
  tools/field-spike-mlx/results.json   (raw numbers)
  docs/research/2026-08-01-field-metal-spike.md (report, written separately
    by the caller after reading results.json — this script only writes JSON
    + prints a table to stdout).
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

BUDGET_MS = 1000.0 / 120.0  # 8.33ms

WARMUP = 20
ITERS = 200


def now_ms():
    return time.perf_counter() * 1000.0


def stats_ms(samples_ms):
    arr = np.array(sorted(samples_ms))
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def verdict(mean_ms):
    return "PASS" if mean_ms <= BUDGET_MS else "FAIL"


# ---------------------------------------------------------------------------
# MLX primitives
# ---------------------------------------------------------------------------

def mlx_bind_fn_factory(mx, n):
    a = mx.random.normal((n, n)).astype(mx.float32)
    b = mx.random.normal((n, n)).astype(mx.float32)

    def bind():
        fa = mx.fft.fft2(a.astype(mx.complex64))
        fb = mx.fft.fft2(b.astype(mx.complex64))
        prod = fa * fb
        out = mx.fft.ifft2(prod)
        mx.eval(out)
        return out

    return bind


def mlx_probe_fn_factory(mx, n_rows, d):
    key = mx.random.normal((1, d)).astype(mx.float32)
    store = mx.random.normal((n_rows, d)).astype(mx.float32)

    def probe():
        out = key @ store.T
        mx.eval(out)
        return out

    return probe


def time_mlx(mx, fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    samples = []
    for _ in range(iters):
        t0 = now_ms()
        fn()
        mx.synchronize()
        t1 = now_ms()
        samples.append(t1 - t0)
    return stats_ms(samples)


# ---------------------------------------------------------------------------
# numpy CPU baselines
# ---------------------------------------------------------------------------

def np_bind_fn_factory(n):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)

    def bind():
        fa = np.fft.fft2(a)
        fb = np.fft.fft2(b)
        prod = fa * fb
        out = np.fft.ifft2(prod)
        return out

    return bind


def np_probe_fn_factory(n_rows, d):
    rng = np.random.default_rng(0)
    key = rng.standard_normal((1, d)).astype(np.float32)
    store = rng.standard_normal((n_rows, d)).astype(np.float32)

    def probe():
        return key @ store.T

    return probe


def time_np(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = now_ms()
        fn()
        t1 = now_ms()
        samples.append(t1 - t0)
    return stats_ms(samples)


# ---------------------------------------------------------------------------
# Device / env info
# ---------------------------------------------------------------------------

def device_info():
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        info["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip()
    except Exception as e:
        info["cpu_brand"] = f"ERROR: {e}"
    try:
        info["mem_bytes"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        )
    except Exception as e:
        info["mem_bytes"] = f"ERROR: {e}"
    try:
        info["macos_version"] = subprocess.check_output(
            ["sw_vers", "-productVersion"]
        ).decode().strip()
    except Exception as e:
        info["macos_version"] = f"ERROR: {e}"
    return info


def main():
    results = {"budget_ms": BUDGET_MS, "warmup": WARMUP, "iters": ITERS}
    results["device"] = device_info()

    np.random.seed(0)

    try:
        import mlx.core as mx
    except Exception as e:
        results["mlx_import_error"] = repr(e)
        with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("MLX IMPORT FAILED:", repr(e))
        print("UNVERIFIED: mlx not importable on this machine")
        sys.exit(1)

    mx.random.seed(0)
    results["mlx_version"] = getattr(mx, "__version__", "unknown")
    try:
        import mlx
        results["mlx_package_version"] = getattr(mlx, "__version__", "unknown")
    except Exception:
        pass
    results["numpy_version"] = np.__version__

    print("=" * 78)
    print("W4 METAL/MLX SPIKE — device:", results["device"]["cpu_brand"])
    print("macOS:", results["device"]["macos_version"], "| python:", sys.version.split()[0])
    print("mlx:", results["mlx_version"], "| numpy:", results["numpy_version"])
    print("budget:", f"{BUDGET_MS:.4f} ms/frame (120fps)")
    print("=" * 78)

    results["tests"] = {}

    # (1) bind at 128^2 and 512^2 -- MLX
    for n in (128, 512):
        key = f"bind_mlx_{n}x{n}"
        print(f"\n[MLX] bind fft2/mul/ifft2 @ {n}x{n} fp32 ...")
        fn = mlx_bind_fn_factory(mx, n)
        st = time_mlx(mx, fn)
        st["verdict"] = verdict(st["mean_ms"])
        results["tests"][key] = st
        print(f"  mean={st['mean_ms']:.4f}ms p50={st['p50_ms']:.4f}ms "
              f"p99={st['p99_ms']:.4f}ms -> {st['verdict']}")

    # (1b) numpy CPU baseline for bind
    for n in (128, 512):
        key = f"bind_numpy_{n}x{n}"
        print(f"\n[numpy CPU] bind fft2/mul/ifft2 @ {n}x{n} fp32 ...")
        fn = np_bind_fn_factory(n)
        st = time_np(fn)
        st["verdict"] = verdict(st["mean_ms"])
        results["tests"][key] = st
        print(f"  mean={st['mean_ms']:.4f}ms p50={st['p50_ms']:.4f}ms "
              f"p99={st['p99_ms']:.4f}ms -> {st['verdict']}")

    # (2) probe matmul -- MLX
    d = 16384
    for n_rows in (4096, 16384):
        key = f"probe_mlx_N{n_rows}_d{d}"
        print(f"\n[MLX] probe key(1x{d}) @ store({n_rows}x{d}).T ...")
        fn = mlx_probe_fn_factory(mx, n_rows, d)
        st = time_mlx(mx, fn)
        st["verdict"] = verdict(st["mean_ms"])
        results["tests"][key] = st
        print(f"  mean={st['mean_ms']:.4f}ms p50={st['p50_ms']:.4f}ms "
              f"p99={st['p99_ms']:.4f}ms -> {st['verdict']}")

    # (2b) numpy CPU baseline for probe
    for n_rows in (4096, 16384):
        key = f"probe_numpy_N{n_rows}_d{d}"
        print(f"\n[numpy CPU] probe key(1x{d}) @ store({n_rows}x{d}).T ...")
        fn = np_probe_fn_factory(n_rows, d)
        st = time_np(fn)
        st["verdict"] = verdict(st["mean_ms"])
        results["tests"][key] = st
        print(f"  mean={st['mean_ms']:.4f}ms p50={st['p50_ms']:.4f}ms "
              f"p99={st['p99_ms']:.4f}ms -> {st['verdict']}")

    # (3) fused tick under mx.compile: bind(128) + bind(512) + probe(N=16384)
    print(f"\n[MLX] fused tick (compiled): bind@128 + bind@512 + probe(N=16384,d={d}) ...")
    a128 = mx.random.normal((128, 128)).astype(mx.float32)
    b128 = mx.random.normal((128, 128)).astype(mx.float32)
    a512 = mx.random.normal((512, 512)).astype(mx.float32)
    b512 = mx.random.normal((512, 512)).astype(mx.float32)
    key1 = mx.random.normal((1, d)).astype(mx.float32)
    store1 = mx.random.normal((16384, d)).astype(mx.float32)

    def fused_tick_impl(a128, b128, a512, b512, key1, store1):
        fa1 = mx.fft.fft2(a128.astype(mx.complex64))
        fb1 = mx.fft.fft2(b128.astype(mx.complex64))
        out1 = mx.fft.ifft2(fa1 * fb1)

        fa2 = mx.fft.fft2(a512.astype(mx.complex64))
        fb2 = mx.fft.fft2(b512.astype(mx.complex64))
        out2 = mx.fft.ifft2(fa2 * fb2)

        out3 = key1 @ store1.T
        return out1, out2, out3

    fused_tick_compiled = mx.compile(fused_tick_impl)

    def fused_tick():
        out = fused_tick_compiled(a128, b128, a512, b512, key1, store1)
        mx.eval(out)
        return out

    st = time_mlx(mx, fused_tick)
    st["verdict"] = verdict(st["mean_ms"])
    st["headroom_factor"] = BUDGET_MS / st["mean_ms"] if st["mean_ms"] > 0 else None
    results["tests"]["fused_tick_mlx_compiled"] = st
    print(f"  mean={st['mean_ms']:.4f}ms p50={st['p50_ms']:.4f}ms "
          f"p99={st['p99_ms']:.4f}ms -> {st['verdict']} "
          f"(headroom={st.get('headroom_factor'):.2f}x)"
          if st.get("headroom_factor") is not None else "")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("Results written to", out_path)
    fused = results["tests"]["fused_tick_mlx_compiled"]
    print(f"FUSED TICK VERDICT: {fused['verdict']} "
          f"(mean {fused['mean_ms']:.4f}ms vs budget {BUDGET_MS:.4f}ms, "
          f"headroom {fused.get('headroom_factor', 0):.2f}x)")
    print("=" * 78)


if __name__ == "__main__":
    main()
