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

    # ---------------------------------------------------- 4D block / observer
    def view3(self, path, volume=False, at=None):
        """3D render (RENDER法: surface/volume, never a heatmap) of this world —
        `at=k` reconstructs journal position k by REPLAY (4D block law: the
        past is recomputed from seed+ops, never stored as frames)."""
        cmd = f"VIEW3 {path}" + (" volume" if volume else "") + (f" at {int(at)}" if at is not None else "")
        return self._cmd(cmd)

    def when(self):
        """The observer axis: where this observer sits in the block.
        -> dict(step_index, journal_pos, energy, absmax). 'When' is a POSITION."""
        self.proc.stdin.write(b"WHEN\n")
        self.proc.stdin.flush()
        buf = self.proc.stdout.read(24)
        si, jp, en, am = struct.unpack("<QQff", buf)
        return {"step_index": si, "journal_pos": jp, "energy": en, "absmax": am}

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


class WindowEnv(WorldEnv):
    """mu-inside atom-1 段1 「観測窓」 — local KxK observation window + self-position.

    Design question 1 (patch source: mu-serve PATCH vs python-side slice of
    the full frame) — DECIDED: mu-serve (rust), not python.
      reason: WorldEnv already carries the torus-wrap arithmetic for `excite`
      (plane::excite, rust) and for the existing probe `reward()` (rust). A
      python-side re-slice would be a SECOND, independent implementation of
      the same wrap logic in a different language -- exactly the kind of
      duplication that drifts (index-off-by-one, or the two float paths
      disagreeing under different libm/codegen -- the 1ulp risk named in the
      追令). One source of wrap truth in rust closes that.
      cost paid: touches mu_serve.rs (the WORLD DOOR file, explicitly not
      frozen -- journal.rs/world.rs/store.rs are frozen and untouched here)
      + one extra IPC round-trip for the `poke` action (POKEXY replies with
      the FULL frame like ACT does, for protocol uniformity with RESET/ACT/
      STEP; the window is then read by a SEPARATE `PATCH` call). For k=16 that
      full-frame reply is 128*128*2*4=131072 bytes vs the patch's own
      16*16*2*4=2048 bytes -- 64x more than the useful payload on poke ticks
      only (move ticks pay nothing extra: STEP's frame is already discarded
      by tick(), then ONE PATCH read). Accepted: correctness (one wrap impl)
      over bytes-on-the-wire; 追令⑧ forces raw f32 bytes on every leg anyway
      (PATCH's own reply is raw bytes, never text/json), so there is no
      float-round-trip cost either way.

    Design question 2 (action space: replace vs coexist with the existing
    8-ring-poke actions) — DECIDED: REPLACE, for THIS class's action
    interface only. WorldEnv/mu-serve's ACT (0..7, ring sites) is untouched
    byte-for-byte and stays available to every other caller (online.py,
    gate_inside.py) -- nothing about it changed. WindowEnv simply does not
    expose it; it defines its OWN 5-action space instead:
      reason: ring sites are WORLD-DECLARED external targets (MU-INSIDE.md
      already names this pattern "a task handed in from outside" for the
      reward case; the same critique applies to actions -- poking a ring site
      chosen by the world, not by mu's own position, is not a self-located
      act). More concretely for THIS atom: a ring-poke action whose site
      falls OUTSIDE the KxK window is invisible to the observation -- the
      (patch_t, action) -> patch_t+1 regression (段2) would then contain
      actions with no supervised effect inside their own input/target pair,
      pure label noise for exactly the cases where mu can't see what it did.
      Self-located actions (move north/south/east/west, poke-here) keep
      every action's effect reachable inside the window it is trained on.
    """

    MOVE_STEP = 4  # design constant (torus grid units per move action), not a
                   # predicted simulation value -- 非硬碼 law is about not
                   # hardcoding OUTCOMES, not about banning design parameters.
    ACTIONS = ("up", "down", "left", "right", "poke")
    NUM_WINDOW_ACTIONS = len(ACTIONS)
    _DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

    def __init__(self, k=16, move_step=MOVE_STEP, **kwargs):
        self.k = k
        self.move_step = move_step
        self.pos = [0, 0]
        super().__init__(**kwargs)

    # -------------------------------------------------------------- io
    def _patch_cmd(self, x, y, k):
        header = 20  # 8 (digest) + 8 (step_index) + 4 (reward, unused=0.0)
        n = 2 * k * k
        self.proc.stdin.write(f"PATCH {int(x)} {int(y)} {int(k)}\n".encode())
        self.proc.stdin.flush()
        buf = self.proc.stdout.read(header + n * 4)
        if buf is None or len(buf) != header + n * 4:
            raise RuntimeError(f"mu-serve short PATCH frame: {0 if buf is None else len(buf)}")
        self.digest, self.step_index = struct.unpack_from("<QQ", buf, 0)
        arr = np.frombuffer(buf, dtype="<f4", count=n, offset=header)
        patch = arr.reshape(CHANNELS, k, k).transpose(1, 2, 0)
        return np.ascontiguousarray(patch, dtype=np.float32)

    def _read_patch(self):
        return self._patch_cmd(self.pos[0], self.pos[1], self.k)

    # ------------------------------------------------------------- api
    def reset(self, seed=0):
        super().reset(seed=seed)  # advances/seeds the real field (world-sourced)
        gh = self.grid // 2
        self.pos = [gh, gh]  # deterministic center start -- a design constant,
                              # not a value predicted from the simulation
        return self._read_patch()

    def step_window(self, action: int):
        """-> (patch (k,k,2) f32, reward f32, pos (x,y)). One world tick per call."""
        action = int(action) % self.NUM_WINDOW_ACTIONS
        name = self.ACTIONS[action]
        if name == "poke":
            x, y = self.pos
            _, reward = self._cmd(f"POKEXY {x} {y}")  # world tick + excite at self
        else:
            dx, dy = self._DELTA[name]
            self.pos[0] = (self.pos[0] + dx * self.move_step) % self.grid
            self.pos[1] = (self.pos[1] + dy * self.move_step) % self.grid
            _, reward = self.tick(1)  # world tick, no excite (STEP)
        patch = self._read_patch()
        return patch, reward, (self.pos[0], self.pos[1])


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
