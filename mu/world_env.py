"""mu/world_env.py — GENESIS-II arc2 段1b: Mu's env IS the real field.

Replaces mu/field_env.py (a python PDE toy) as the observation source. This
class speaks to `mu-serve` (packages/field/src/bin/mu_serve.rs) over a pipe:
the world is the rust FFT wave plane + N×d store, the same substrate the
engine runs at 120fps — not a re-implementation of it in numpy.

Interface is drop-in with FieldEnv: reset(seed) -> obs, step(action) -> (obs, r).

obs = (H, W, 2) float32 = [cur, prev]. That pair IS the complete plane state
(second-order wave: next = f(cur, prev)) — no synthetic third channel is
invented here. Reward + poke sites are declared WORLD-side (rust); this file
decides nothing about the world, it only carries bytes.

Godseed law: every byte comes from f(seed). Each frame also carries the
world's own digest + step_index, so a caller can gate bit-exact determinism
against the engine's own determinism gates, not against a python replica.
"""
import os
import struct
import subprocess

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN = os.path.join(REPO, "target", "release", "mu-serve")

GRID = 128
NUM_ACTIONS = 8
CHANNELS = 2  # cur, prev — the whole plane state, nothing invented


class WorldEnv:
    """The real field as a gym-shaped env. One subprocess = one world."""

    def __init__(self, seed=0, grid=GRID, actions=NUM_ACTIONS, steps=1,
                 amp=0.08, probe=8, bin_path=DEFAULT_BIN):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"{bin_path} missing — build it: cargo build --release -p field --bin mu-serve"
            )
        self.grid, self.actions, self.d = grid, actions, grid * grid
        self.frame_bytes = 8 + 8 + 4 + CHANNELS * self.d * 4
        self.proc = subprocess.Popen(
            [bin_path, "--width", str(grid), "--height", str(grid),
             "--actions", str(actions), "--steps", str(steps),
             "--amp", str(amp), "--probe", str(probe), "--seed", str(seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.digest = 0
        self.step_index = 0
        self.reset(seed)

    # ------------------------------------------------------------------ io
    def _cmd(self, line):
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()
        buf = self.proc.stdout.read(self.frame_bytes)
        if buf is None or len(buf) != self.frame_bytes:
            raise RuntimeError(f"mu-serve short frame: {0 if buf is None else len(buf)}")
        self.digest, self.step_index = struct.unpack_from("<QQ", buf, 0)
        reward = struct.unpack_from("<f", buf, 16)[0]
        planes = np.frombuffer(buf, dtype="<f4", count=CHANNELS * self.d, offset=20)
        obs = planes.reshape(CHANNELS, self.grid, self.grid).transpose(1, 2, 0)
        return np.ascontiguousarray(obs, dtype=np.float32), float(reward)

    # ----------------------------------------------------------------- api
    def reset(self, seed=0):
        obs, _ = self._cmd(f"RESET {seed}")
        return obs

    def step(self, action: int):
        return self._cmd(f"ACT {int(action) % self.actions}")

    def tick(self, steps=1):
        return self._cmd(f"STEP {steps}")

    def close(self):
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    env = WorldEnv(seed=0)
    obs = env.reset(seed=0)
    print("obs", obs.shape, obs.dtype, "channels = [cur, prev] (real field plane state)")
    for i in range(5):
        obs, r = env.step(i % NUM_ACTIONS)
        print(f"step {i}: reward={r:.6f} absmax={np.abs(obs).max():.6f} "
              f"digest={env.digest:#x} step_index={env.step_index}")

    # G2/G4 smoke: same seed -> bit-identical stream; different seed -> differs.
    def roll(seed):
        e = WorldEnv(seed=seed)
        e.reset(seed=seed)
        acc = []
        for i in range(5):
            o, r = e.step(i % NUM_ACTIONS)
            acc.append((o.tobytes(), r, e.digest))
        e.close()
        return acc

    a, b, c = roll(0), roll(0), roll(1)
    print("determinism same-seed bit-identical:", a == b)
    print("world-sourced (seed change moves obs):", a != c)
