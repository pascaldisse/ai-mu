"""mu/field_env.py — richer multi-channel field env, replaces the toy
single-channel DampedWaveEnv (train.py) per GENESIS-II arc1 spec ①.

3 channels = heat (H) / flow (F) / pressure (P), matching GENESIS.md's
"phenomena = channel dynamics (fire/water/storm = field species interacting)"
model: three coupled scalar fields on one 128x128 periodic grid, each its own
damped-wave/diffusion process, cross-coupled so the channels genuinely
interact (not three independent copies of the old toy env):

  dH/dt = diff_h * lap(H)          - damp_h * H  + k1 * F        (flow convects heat)
  dF/dt = c_f^2   * lap(F)          - damp_f * F  + k2*(H-mean(H)) - k3*lap(P)
                                                     (heat buoyancy drives flow;
                                                      pressure gradient opposes it)
  dP/dt = c_p^2   * lap(P)          - damp_p * P  + k4 * lap(F)  (flow curvature sources pressure)

All couplings use lap()/mean() (isotropic, no directional bias) to keep the
update seed-deterministic and axis-symmetric. Explicit Euler, dt small enough
to stay stable over long online runs (no episodic resets needed for arc1 ②:
this env is meant to be ticked forever, unlike the old 20-step-episodic toy).

Godseed law (AI-MU.md / ENTROPY.md): state = f(seed). reset(seed) is the only
place randomness enters (initial hot-spot placement); step() is pure/deterministic.

Actions: 8 fixed ring "poke" locations (unchanged layout from the old env) —
each poke injects an impulse into BOTH heat and flow at that location (a
thermal+kinetic event), so an action now has a genuine multi-channel effect.
Reward = probe readout energy = sum of squares over ALL 3 channels in an 8x8
patch co-located with one designated poke location (target_action), same
"immediate single-step per-action signal" rationale as the old env (needed for
depth<=1 planning / one-step value estimation to have real gradient), clipped
to [0, 5].
"""
import numpy as np

GRID = 128
NUM_ACTIONS = 8
CHANNELS = 3  # heat, flow, pressure


class FieldEnv:
    """Seeded 128x128x3 periodic multi-channel field. state = f(seed) (godseed law)."""

    def __init__(
        self,
        seed=0,
        dt=0.1,
        diff_h=0.2, damp_h=0.05,
        c_f=0.5, damp_f=0.1,
        c_p=0.5, damp_p=0.1,
        k1=0.05, k2=0.05, k3=0.1, k4=0.1,
        poke_strength=0.08,  # tuned so probe reward ramps gradually over a
                             # ~30-step episode instead of saturating at the
                             # clip within ~9 steps (measured: strength=0.6 ->
                             # 76% of steps at clip ceiling, near-constant
                             # reward, poor RL signal; strength=0.08 -> 1.3% at
                             # clip, std=1.3 over a 0-5 range, real per-step
                             # variation for the critic to fit)
        reward_clip=5.0,
    ):
        self.dt = dt
        self.diff_h, self.damp_h = diff_h, damp_h
        self.c_f, self.damp_f = c_f, damp_f
        self.c_p, self.damp_p = c_p, damp_p
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.poke_strength = poke_strength
        self.reward_clip = reward_clip

        self._xx, self._yy = np.meshgrid(np.arange(GRID), np.arange(GRID))
        self.poke_locs = [
            (
                GRID // 2 + int(40 * np.cos(2 * np.pi * i / NUM_ACTIONS)),
                GRID // 2 + int(40 * np.sin(2 * np.pi * i / NUM_ACTIONS)),
            )
            for i in range(NUM_ACTIONS)
        ]
        self._impulses = [
            self.poke_strength * np.exp(-((self._xx - lx) ** 2 + (self._yy - ly) ** 2) / 8.0).astype(np.float32)
            for lx, ly in self.poke_locs
        ]

        # probe co-located with ONE designated poke location (immediate per-action
        # signal, same rationale as the old DampedWaveEnv — see mu/BENCH.md corpses).
        self.target_action = NUM_ACTIONS // 4  # =2 for NUM_ACTIONS=8
        target_lx, target_ly = self.poke_locs[self.target_action]
        self.probe_lo_x, self.probe_hi_x = target_ly - 4, target_ly + 4
        self.probe_lo_y, self.probe_hi_y = target_lx - 4, target_lx + 4

        self.reset(seed)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.H = np.zeros((GRID, GRID), dtype=np.float32)
        self.F = np.zeros((GRID, GRID), dtype=np.float32)
        self.P = np.zeros((GRID, GRID), dtype=np.float32)
        x0, y0 = rng.integers(20, GRID - 20, size=2)
        self.H += 0.5 * np.exp(-((self._xx - x0) ** 2 + (self._yy - y0) ** 2) / 50.0).astype(np.float32)
        return self._obs()

    @staticmethod
    def _lap(f):
        return np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f

    def step(self, action: int):
        imp = self._impulses[action]
        self.H = self.H + imp
        self.F = self.F + imp

        lap_H, lap_F, lap_P = self._lap(self.H), self._lap(self.F), self._lap(self.P)

        dH = self.diff_h * lap_H - self.damp_h * self.H + self.k1 * self.F
        dF = self.c_f ** 2 * lap_F - self.damp_f * self.F + self.k2 * (self.H - self.H.mean()) - self.k3 * lap_P
        dP = self.c_p ** 2 * lap_P - self.damp_p * self.P + self.k4 * lap_F

        self.H = self.H + self.dt * dH
        self.F = self.F + self.dt * dF
        self.P = self.P + self.dt * dP

        pH = self.H[self.probe_lo_x:self.probe_hi_x, self.probe_lo_y:self.probe_hi_y]
        pF = self.F[self.probe_lo_x:self.probe_hi_x, self.probe_lo_y:self.probe_hi_y]
        pP = self.P[self.probe_lo_x:self.probe_hi_x, self.probe_lo_y:self.probe_hi_y]
        reward = float(np.clip(np.sum(pH ** 2) + np.sum(pF ** 2) + np.sum(pP ** 2), 0.0, self.reward_clip))
        return self._obs(), reward

    def _obs(self):
        return np.stack([self.H, self.F, self.P], axis=-1).astype(np.float32)  # (128,128,3)


if __name__ == "__main__":
    env = FieldEnv(seed=0)
    obs = env.reset(seed=0)
    print("obs", obs.shape, obs.dtype, "channels order = [heat, flow, pressure]")
    for i in range(5):
        obs, r = env.step(i % NUM_ACTIONS)
        print(f"step {i}: reward={r:.5f} H_absmax={np.abs(env.H).max():.4f} "
              f"F_absmax={np.abs(env.F).max():.4f} P_absmax={np.abs(env.P).max():.4f}")

    # determinism smoke check: reset+replay same action sequence twice, bit-identical
    env2 = FieldEnv(seed=0)
    obs2 = env2.reset(seed=0)
    for i in range(5):
        obs2, r2 = env2.step(i % NUM_ACTIONS)
    env3 = FieldEnv(seed=0)
    obs3 = env3.reset(seed=0)
    for i in range(5):
        obs3, r3 = env3.step(i % NUM_ACTIONS)
    print("determinism (obs2==obs3 bit-identical):", bool(np.array_equal(obs2, obs3)))
