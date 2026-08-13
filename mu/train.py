"""mu/train.py — self-play + train loop on a toy seeded damped-wave env.

Toy env (DampedWaveEnv): 128x128 periodic 5-point-stencil damped wave
(matches the "plane" atom in FIELD.md / adversary review §1 — same shape as
the field slice the nets consume, no separate renderer). Godseed-compatible:
state = f(seed) (reset() is fully seeded, no hidden randomness — AI-MU.md
"superdeterminism/godseed" + ENTROPY.md ultradeterminism law).

Actions: 8 fixed "poke" locations (ring around center) — agent chooses where
to inject a velocity impulse. Reward = probe readout energy (sum of u^2 in a
fixed 8x8 patch at the field center) — literally "probe readout energy" per
the task spec.

Self-play: at each step, mu.mcts.plan() (batched, sims=16, depth=1, ONE
dynamics call) gives a policy_target/value_target used as MuZero-style
training labels; action is sampled from that target (epsilon-greedy mixed
with uniform random for exploration). Transition stored in a replay buffer.

Losses (per MuZero, adapted to the decoder for the explicit slice-prediction
requirement):
  loss_slice  = MSE(decoder(dyn(repr(obs),a)), next_obs)   [predict-next-field-slice]
  loss_reward = MSE(dyn_reward, true_reward)
  loss_policy = CE(pred_policy_logits, planned policy_target)
  loss_value  = MSE(pred_value, planned value_target)
  total = sum of the four (equal weight, v0)

Run: python train.py [--steps N] [--seed S]
Prints one line per step: step, total loss, and the four components.
"""
import argparse
import random
import time
from collections import deque

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from nets import MuNet, GRID, NUM_ACTIONS, action_onehot
from mcts import plan


class DampedWaveEnv:
    """Seeded 128x128 periodic damped wave. state = f(seed) (godseed law)."""

    def __init__(self, seed=0, dt=0.2, c=1.0, damping=0.15, poke_strength=0.6, reward_clip=5.0):
        self.dt, self.c, self.damping, self.poke_strength = dt, c, damping, poke_strength
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
        # probe co-located with ONE designated poke location (TARGET_ACTION) so
        # the reward difference between actions is an IMMEDIATE, single-step
        # effect (direct energy injection under the probe) rather than a
        # resonance that only appears after many steps of sustained pumping --
        # needed so a depth=1 (one dynamics call) plan() has real per-action
        # signal to distill into policy_target. Still literally "probe readout
        # energy" of the seeded damped wave (FIELD.md plane atom).
        self.target_action = NUM_ACTIONS // 4  # =2 for NUM_ACTIONS=8
        target_lx, target_ly = self.poke_locs[self.target_action]
        self.probe_lo_x, self.probe_hi_x = target_ly - 4, target_ly + 4  # row bounds
        self.probe_lo_y, self.probe_hi_y = target_lx - 4, target_lx + 4  # col bounds
        self.reset(seed)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.u = np.zeros((GRID, GRID), dtype=np.float32)
        self.v = np.zeros((GRID, GRID), dtype=np.float32)
        x0, y0 = rng.integers(20, GRID - 20, size=2)
        self.u += 0.5 * np.exp(-((self._xx - x0) ** 2 + (self._yy - y0) ** 2) / 50.0).astype(np.float32)
        return self._obs()

    def _lap(self, f):
        return np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f

    def step(self, action: int):
        self.v = self.v + self._impulses[action]
        a = self.c ** 2 * self._lap(self.u) - self.damping * self.v
        self.v = self.v + self.dt * a
        self.u = self.u + self.dt * self.v
        patch = self.u[self.probe_lo_x:self.probe_hi_x, self.probe_lo_y:self.probe_hi_y]
        reward = float(np.clip(np.sum(patch ** 2), 0.0, self.reward_clip))
        return self._obs(), reward

    def _obs(self):
        return self.u.reshape(GRID, GRID, 1).astype(np.float32)


def loss_fn(model, obs, next_obs, action, reward, ptgt, vtgt):
    s = model.repr(obs)
    s2, r_pred = model.dyn(s, action_onehot(action))
    policy_logits, value_pred = model.pred(s)
    decoded_next = model.decoder(s2)

    loss_slice = mx.mean((decoded_next - next_obs) ** 2)
    loss_reward = mx.mean((r_pred.reshape(-1) - reward) ** 2)
    logp = nn.log_softmax(policy_logits, axis=-1)
    loss_policy = -mx.mean(mx.sum(ptgt * logp, axis=-1))
    loss_value = mx.mean((value_pred.reshape(-1) - vtgt) ** 2)
    total = loss_slice + loss_reward + loss_policy + loss_value
    return total, (loss_slice, loss_reward, loss_policy, loss_value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--buffer-cap", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=0.2, help="uniform-random exploration mix")
    ap.add_argument("--episode-len", type=int, default=20, help="steps per episode before reset (godseed: state=f(seed))")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    env = DampedWaveEnv(seed=args.seed)
    net = MuNet()
    opt = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(net, loss_fn)

    buffer = deque(maxlen=args.buffer_cap)
    episode_idx = 0
    obs_np = env.reset(seed=args.seed + episode_idx)

    t_start = time.perf_counter()
    log_lines = []
    for step in range(args.steps):
        if step > 0 and step % args.episode_len == 0:
            episode_idx += 1
            obs_np = env.reset(seed=args.seed + episode_idx)  # episodic reset, state=f(seed)
        obs_mx = mx.array(obs_np)[None, ...]
        out = plan(net, obs_mx, sims=args.sims, depth=1)
        policy_target = np.array(out["policy_target"].tolist(), dtype=np.float32)
        value_target = float(out["value_target"].item())

        eps_now = args.eps * max(0.0, 1.0 - step / args.steps)  # linear decay -> less noise late in training
        if random.random() < eps_now:
            action = random.randrange(NUM_ACTIONS)
        else:
            action = int(np.random.choice(NUM_ACTIONS, p=policy_target / policy_target.sum()))

        next_obs_np, reward = env.step(action)
        buffer.append((obs_np, action, reward, next_obs_np, policy_target, value_target))
        obs_np = next_obs_np

        if len(buffer) >= args.batch:
            batch = random.sample(buffer, args.batch)
            obs_b = mx.array(np.stack([b[0] for b in batch]))
            action_b = mx.array(np.array([b[1] for b in batch], dtype=np.uint32))
            reward_b = mx.array(np.array([b[2] for b in batch], dtype=np.float32))
            next_obs_b = mx.array(np.stack([b[3] for b in batch]))
            ptgt_b = mx.array(np.stack([b[4] for b in batch]))
            vtgt_b = mx.array(np.array([b[5] for b in batch], dtype=np.float32))

            (total, (l_slice, l_r, l_p, l_v)), grads = loss_and_grad(
                net, obs_b, next_obs_b, action_b, reward_b, ptgt_b, vtgt_b
            )
            opt.update(net, grads)
            mx.eval(net.parameters(), total)

            line = (f"step={step:4d} loss_total={total.item():.5f} "
                    f"slice={l_slice.item():.5f} reward={l_r.item():.5f} "
                    f"policy={l_p.item():.5f} value={l_v.item():.5f}")
            log_lines.append(line)
            if step % 10 == 0 or step == args.steps - 1:
                print(line)

    t_end = time.perf_counter()
    print(f"# {len(log_lines)} train steps in {t_end - t_start:.2f}s "
          f"({(t_end - t_start) / max(1, len(log_lines)) * 1000:.2f} ms/train-step incl. plan())")

    # dump full log for BENCH.md
    with open("train_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("full log -> mu/train_log.txt")


if __name__ == "__main__":
    main()
