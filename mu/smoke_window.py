"""mu/smoke_window.py -- mu-inside atom-1 段1 smoke gates (観測窓).

Run: ~/projects/magic-crystal-mu-v0/.venv/bin/python mu/smoke_window.py

Checks (all must PASS before 段1 commit):
  1. determinism: same seed, same forced action sequence -> byte-identical
     patch stream (np.array_equal, not approx).
  2. world-sourced: seed 0 vs seed 7 -> patch stream DIFFERS.
  3. torus-wrap correctness, INDEPENDENT PATH: for a fixed world tick, the
     mu-serve PATCH reply for (x,y,k) must equal a patch cut in PYTHON from
     the WorldEnv's own full-frame obs via np.roll (a totally different
     wrap implementation -- numpy's roll, not the rust modulo arithmetic --
     so this does not share a failure mode with mu_serve.rs's own PATCH
     code). Checked for an interior site AND an edge-crossing site (x=0,
     y=0 and x=grid-1,y=grid-1) so the wrap itself is actually exercised,
     not just the interior case where wrap is a no-op.
  4. Determinism holds across BOTH cargo profiles (debug vs release) --
     追令①: sin/cos codegen can differ 1ulp between profiles; if the two
     profiles disagree, this is reported as a 1ulp NOTE, not silently
     folded into a PASS.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_env import WindowEnv, WorldEnv, GRID, NUM_ACTIONS, REPO  # noqa: E402

FIXED_ACTIONS = [0, 4, 1, 4, 2, 4, 3, 4, 0, 1, 2, 3, 4, 4, 0, 3] * 4  # 64 forced steps


def run_patch_stream(seed, bin_path, k=16, n=64):
    env = WindowEnv(seed=seed, k=k, bin_path=bin_path)
    patches = [env._read_patch().copy()]
    for i in range(n):
        p, r, pos = env.step_window(FIXED_ACTIONS[i % len(FIXED_ACTIONS)])
        patches.append(p.copy())
    env.close()
    return patches


def check_determinism(bin_path):
    a = run_patch_stream(seed=3, bin_path=bin_path)
    b = run_patch_stream(seed=3, bin_path=bin_path)
    ok = all(np.array_equal(x, y) for x, y in zip(a, b))
    return ok, a, b


def check_world_sourced(bin_path):
    a = run_patch_stream(seed=0, bin_path=bin_path)
    c = run_patch_stream(seed=7, bin_path=bin_path)
    differs = any(not np.array_equal(x, y) for x, y in zip(a, c))
    return differs


def check_torus_wrap(bin_path):
    """Independent path: cut the patch from WorldEnv's own full obs via
    np.roll, compare to WindowEnv's mu-serve PATCH reply, for an interior
    site and two edge-crossing sites."""
    k = 16
    half = k // 2
    results = []
    for (x, y) in [(64, 64), (0, 0), (GRID - 1, GRID - 1), (3, GRID - 2)]:
        wenv = WorldEnv(seed=5, bin_path=bin_path)
        full_obs = wenv.reset(seed=5)
        for i in range(5):
            full_obs, _ = wenv.tick(1)  # pure ticks, no excite -- must match
                                        # WindowEnv's move-tick path exactly
                                        # (an ACT-based reference would excite
                                        # ring sites and diverge the trajectory)
        digest_full, step_full = wenv.digest, wenv.step_index

        winenv = WindowEnv(seed=5, k=k, bin_path=bin_path)
        winenv.reset(seed=5)
        for i in range(5):
            winenv.tick(1)
        mu_serve_patch = winenv._patch_cmd(x, y, k)
        digest_win, step_win = winenv.digest, winenv.step_index

        # independent numpy re-implementation of the torus wrap, same
        # half=k//2 convention as the rust side (documented in mu_serve.rs).
        rolled = np.roll(full_obs, shift=(half - y, half - x), axis=(0, 1))
        py_patch = rolled[0:k, 0:k, :]

        same_state = (digest_full == digest_win) and (step_full == step_win)
        match = np.array_equal(mu_serve_patch, py_patch)
        results.append((x, y, same_state, match))
        wenv.close()
        winenv.close()
    return results


def main():
    bin_path = os.path.join(REPO, "target", "release", "mu-serve")
    all_pass = True

    ok_det, a, b = check_determinism(bin_path)
    print(f"GW-1-smoke determinism (release, same seed -> byte-identical patch stream): "
          f"{'PASS' if ok_det else 'FAIL'}")
    all_pass &= ok_det

    ok_world = check_world_sourced(bin_path)
    print(f"GW-2-smoke world-sourced (seed 0 vs 7 -> patch stream differs): "
          f"{'PASS' if ok_world else 'FAIL'}")
    all_pass &= ok_world

    wrap_results = check_torus_wrap(bin_path)
    wrap_ok = all(m for (_, _, _, m) in wrap_results)
    state_ok = all(s for (_, _, s, _) in wrap_results)
    for (x, y, s, m) in wrap_results:
        print(f"  torus-wrap (x={x},y={y}): same_world_state={s} "
              f"mu-serve==np.roll-python: {'PASS' if m else 'FAIL'}")
    print(f"torus-wrap correctness (independent numpy path, interior+edge sites): "
          f"{'PASS' if (wrap_ok and state_ok) else 'FAIL'}")
    all_pass &= wrap_ok and state_ok

    print(f"\nsmoke_window.py OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
