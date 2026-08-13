"""mu/online.py — GENESIS-II arc1 ②: online/continual learning loop.

Mu ticks INSIDE the world sim and learns AS IT RUNS — no offline episodes, no
self-play-THEN-train phases. Every world tick does, in order:

  1. amortized_policy(net, obs)  — ONE fused mx.compile'd forward (repr->pred),
     same "interactive tick" path as tick_bench.py. This picks the action a
     real-time agent would use (no MCTS/planning — too expensive for a 120fps
     hot loop; see mu/BENCH.md corpses for why MCTS was dropped from the
     online path).
  2. epsilon-greedy sample action from softmax(policy_logits).
  3. env.step(action) -> next_obs, reward   (field_env.py, numpy, off-graph).
  4. push (obs,action,reward,next_obs) into a small FIXED-size rolling window
     (maxlen=BUFFER_CAP, recency-biased, no episode boundaries respected) and
     immediately (same tick, no separate phase) sample a fixed-size mini-batch
     from it for the gradient step — still "online"/no-offline-episodes: the
     buffer is a variance-reduction device for single-tick REINFORCE-style
     gradients (batch=1 was measured too noisy for a real decreasing loss
     trend — see Corpses below), NOT a collect-then-train split. Fixed batch
     size (padding with repeats of the newest transition while the buffer is
     still filling) keeps ONE constant compiled shape for the whole run, so
     ms/tick measurement never includes a mid-run recompile event.
  5. ONE mx.compile'd train step (forward + backward + optimizer.update, state
     captured via mx.compile(inputs=state, outputs=state) per MLX's documented
     compiled-training-loop pattern) — this IS the "learns as it runs" step,
     applied every tick, no offline phase.

Loss (single-sample online actor-critic + world-model, replaces MCTS-distillation
since planning is off the hot-path here):
  loss_slice  = MSE(decoder(dyn(repr(obs),a)), next_obs)      [world model target]
  loss_reward = MSE(dyn_reward, true_reward)                  [world model target]
  td_target   = reward + discount * value(dyn(repr(obs),a))    [MuZero-consistent: bootstrap
                                                                 through the LEARNED dynamics'
                                                                 predicted next latent, not by
                                                                 re-encoding next_obs -- saves a
                                                                 whole extra repr() forward/backward
                                                                 per tick; stop_gradient, no target net @ v0]
  advantage   = stop_gradient(td_target - value(repr(obs)))
  loss_policy = -mean(advantage * logp(action_taken))          [1-step actor-critic]
  loss_value  = MSE(value(repr(obs)), stop_gradient(td_target))
  total = sum of the four (equal weight, v0, same convention as train.py)

Corpses (kept, not code):
- 死 batch=1 single-sample online SGD, no buffer: reason — loss_total's
  second-half average was HIGHER than its first-half average over a 400-step
  run (measured: 0.62 -> 1.06, an INCREASING trend) due to unbounded-variance
  single-sample REINFORCE/TD gradients occasionally spiking loss_total into
  the 15-55 range. Fixed: small rolling-window mini-batch (below), still
  fully online/per-tick, not an offline phase.
- 死 reward = sum-of-squares over an 8x8x3-channel probe patch with clip=5.0,
  poke_strength=0.6 (unchanged from the old 1-channel env): reason — measured
  76-97% of steps saturated at the reward clip ceiling within ~9 ticks of an
  episode, giving a near-constant reward signal (poor value-function training
  target). Fixed: poke_strength=0.08 (measured 1.3% clip-saturation, std=1.3
  over the 0-5 range, real per-step variation) — see field_env.py.
- 死 naive replay-buffer batching applied to the REINFORCE/TD policy+value loss
  too (not just slice/reward): reason — measured loss_total exploding to
  +122/-97/+58 by step ~300-380 (off-policy policy-gradient without importance
  correction, amplified by mixing up-to-200-tick-old actions/logp against the
  CURRENT, already-shifted policy). Fixed: policy_loss/value_loss masked to
  ONLY the fresh index-0 (current-tick, genuinely on-policy) sample in the
  batch; slice/reward (supervised regression, no policy-staleness issue) still
  use the full batch.
- 死 batch_size=8/16 for the replay-buffered world-model loss: reason —
  measured train_step_fn cost scales ~linearly with batch (B=1: 3.0ms, B=4:
  4.75ms, B=8: 9.93ms, B=16: 19.96ms -- Conv2d over 128x128x3 images is NOT
  free per extra batch row on this GPU, unlike the tiny latent-only MLP heads).
  B=8 alone blew the 8.33ms tick budget. Fixed: batch_size=4 (measured
  well under budget, see BENCH.md).

arc2 段2 (2026-08-01): env swapped FieldEnv (python PDE toy) -> WorldEnv (the
real rust field World over mu-serve). obs is now (128,128,2)=[cur,prev], the
plane's complete state; reward + poke sites are declared world-side. The
learning loop itself is unchanged — that is the point: the same online
continual learner, now fed by the engine instead of a replica of it.

Run: python online.py [--steps N] [--seed S] [--check-determinism]
"""
import argparse
import random
import time
from collections import deque

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from nets import MuNet, GRID, CHANNELS, NUM_ACTIONS, action_onehot
from mcts import amortized_policy
from world_env import WorldEnv  # arc2 段2: the env IS the rust field (mu-serve)

DISCOUNT = 0.9
ADV_CLIP = 1.0        # arc2 段3: bound the actor-critic step (see loss_fn)
ENTROPY_BETA = 0.01   # arc2 段3: keep exploration alive without eps hacks
REWARD_RMS_EPS = 1e-3 # arc2 段3: running-RMS reward scaling (real field rewards
                      # are unclipped and range 0..3.5; raw scale drove the
                      # value head and the advantage apart)


def make_online_step(net, opt):
    """Returns (select_action_fn, train_step_fn, state) — both mx.compile'd,
    state = [net.state, opt.state] captured/updated per MLX's compiled-training
    pattern (mx.compile(fn, inputs=state, outputs=state))."""

    state = [net.state, opt.state]

    def _select(obs):
        policy_logits, value, s = amortized_policy(net, obs)
        return policy_logits, value

    # arc2 段3 BUG (inherited from arc1): compiled WITHOUT inputs=state, the
    # acting policy was FROZEN at trace-time weights — mx.compile captures
    # arrays by value, so every learning update was invisible to action
    # selection. Proof: two 900-step runs with different loss functions
    # produced a bit-identical action stream. inputs=state fixes it.
    select_action_fn = mx.compile(_select, inputs=state)

    def loss_fn(net, obs, next_obs, action, reward, onpolicy_mask):
        """onpolicy_mask: (B,1), 1.0 at index 0 (the fresh current-tick transition),
        0.0 elsewhere (older replay-buffer samples). slice/reward are supervised
        regression targets -> safe to train on the WHOLE batch (off-policy is fine
        for a world-model target). policy/value are REINFORCE/TD -- off-policy
        without importance-sampling correction diverges catastrophically (measured
        corpse below) -> masked to the single on-policy sample only."""
        s = net.repr(obs)
        policy_logits, value = net.pred(s)
        a_onehot = action_onehot(action)
        s2_pred, r_pred = net.dyn(s, a_onehot)
        decoded_next = net.decoder(s2_pred)

        # bootstrap value target through the WORLD MODEL's own predicted next
        # latent (s2_pred), not by re-encoding the real next_obs -- saves a full
        # extra repr() forward+backward per tick vs. the target-encoding approach.
        _, value_next = net.pred(s2_pred)
        value_next = mx.stop_gradient(value_next)

        td_target = reward.reshape(-1, 1) + DISCOUNT * value_next
        advantage = mx.stop_gradient(td_target - value)

        logp_all = nn.log_softmax(policy_logits, axis=-1)
        logp_taken = mx.sum(logp_all * a_onehot, axis=-1, keepdims=True)
        n_onpolicy = mx.sum(onpolicy_mask)
        # arc2 段3: advantage CLIP + entropy bonus. Unbounded advantage x logp
        # diverged on the real field (measured: policy loss -> -148 by step 900,
        # policy stayed uniform); arc1's toy env hid it behind a reward clip.
        advantage = mx.clip(advantage, -ADV_CLIP, ADV_CLIP)
        entropy = -mx.sum(mx.exp(logp_all) * logp_all, axis=-1, keepdims=True)
        loss_policy = -mx.sum(onpolicy_mask * (advantage * logp_taken + ENTROPY_BETA * entropy)) / n_onpolicy
        loss_value = mx.sum(onpolicy_mask * (value - mx.stop_gradient(td_target)) ** 2) / n_onpolicy
        loss_slice = mx.mean((decoded_next - next_obs) ** 2)
        loss_reward = mx.mean((r_pred.reshape(-1, 1) - reward.reshape(-1, 1)) ** 2)

        total = loss_slice + loss_reward + loss_policy + loss_value
        return total, (loss_slice, loss_reward, loss_policy, loss_value)

    loss_and_grad_fn = nn.value_and_grad(net, loss_fn)

    def _step(obs, next_obs, action, reward, onpolicy_mask):
        (total, comps), grads = loss_and_grad_fn(net, obs, next_obs, action, reward, onpolicy_mask)
        opt.update(net, grads)
        return total, comps

    train_step_fn = mx.compile(_step, inputs=state, outputs=state)

    return select_action_fn, train_step_fn, state


def run(steps=250, seed=0, lr=1e-3, eps=0.2, log_path=None, quiet=False, episode_len=30,
        batch_size=4, buffer_cap=200):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    env = WorldEnv(seed=seed)
    net = MuNet()
    opt = optim.Adam(learning_rate=lr)
    select_action_fn, train_step_fn, state = make_online_step(net, opt)
    buffer = deque(maxlen=buffer_cap)

    onpolicy_mask_mx = mx.array(np.array([[1.0]] + [[0.0]] * (batch_size - 1), dtype=np.float32))  # (B,1), fixed

    def make_batch(obs_np, action, reward, next_obs_np):
        """Push newest transition, build a FIXED-size batch with the CURRENT
        transition always at index 0 (on-policy anchor) + (batch_size-1) older
        samples from the rolling buffer (pad with repeats while buffer is still
        filling) -- same shape from tick 0 -> no recompiles."""
        buffer.append((obs_np, action, reward, next_obs_np))
        rest = random.sample(buffer, batch_size - 1) if len(buffer) - 1 >= batch_size - 1 else \
            (list(buffer) + [buffer[-1]] * (batch_size - 1 - len(buffer)))[:batch_size - 1]
        batch = [buffer[-1]] + rest
        obs_b = mx.array(np.stack([b[0] for b in batch]))
        next_obs_b = mx.array(np.stack([b[3] for b in batch]))
        action_b = mx.array(np.array([b[1] for b in batch], dtype=np.uint32))
        reward_b = mx.array(np.array([b[2] for b in batch], dtype=np.float32))
        return obs_b, next_obs_b, action_b, reward_b

    # periodic env resets (godseed-seeded, same rationale as train.py: "bounds
    # field growth" -- FieldEnv is stable-but-drifting over thousands of
    # ticks (see field_env.py stability test), so an unbounded run makes the
    # slice-prediction MSE trend non-stationary (target magnitude keeps
    # growing) regardless of learning quality. This resets the ENVIRONMENT
    # only -- the LEARNING loop stays online/per-tick throughout, no offline
    # collect-then-train phases (arc1 ②'s actual requirement).
    episode_idx = 0
    rms_acc, rms_n = 0.0, 0
    obs_np = env.reset(seed=seed)
    log_lines = []
    tick_ms = []

    # warmup: first call to each mx.compile'd fn triggers tracing (~1-2s one-time
    # cost) -- run it once here, discarded, so the timed loop below measures the
    # STEADY-STATE per-tick cost (the real 120fps-budget question), same warmup
    # convention as tick_bench.py.
    _warm_obs = mx.array(obs_np)[None, ...]
    _pl, _v = select_action_fn(_warm_obs)
    mx.eval(_pl, _v)
    _ = int(mx.random.categorical(_pl[0]).item())  # warm mx.random.categorical's own one-time cost too
    _next_obs_np, _reward = env.step(0)
    _wobs_b, _wnext_b, _wact_b, _wrew_b = make_batch(obs_np, 0, _reward, _next_obs_np)
    _wtotal, _wcomps = train_step_fn(_wobs_b, _wnext_b, _wact_b, _wrew_b, onpolicy_mask_mx)
    mx.eval(state, _wtotal)
    buffer.clear()  # undo the warmup step's effect on the buffer before the real loop
    env.reset(seed=seed)  # undo the warmup step's effect on env state before the real loop
    obs_np = env.reset(seed=seed)

    t_wall_start = time.perf_counter()
    for step in range(steps):
        if step > 0 and step % episode_len == 0:
            episode_idx += 1
            obs_np = env.reset(seed=seed + episode_idx)  # env-only reset, godseed; learning is unbroken/per-tick
        t0 = time.perf_counter()

        obs_mx = mx.array(obs_np)[None, ...]
        policy_logits, value = select_action_fn(obs_mx)

        if random.random() < eps:
            action = random.randrange(NUM_ACTIONS)
        else:
            # sample entirely on-device (mx.random.categorical), one .item()
            # sync at the end -- avoids a numpy softmax/tolist CPU round-trip
            # per tick (was the 2nd-largest per-tick cost after train_step).
            action = int(mx.random.categorical(policy_logits[0]).item())

        next_obs_np, reward_raw = env.step(action)  # real field tick, off-graph
        # running-RMS reward scaling: deterministic (pure function of the seeded
        # trajectory), no clip ceiling, keeps the critic target O(1).
        rms_acc = rms_acc + reward_raw * reward_raw
        rms_n = rms_n + 1
        reward = reward_raw / ((rms_acc / rms_n) ** 0.5 + REWARD_RMS_EPS)

        obs_b, next_obs_b, action_b, reward_b = make_batch(obs_np, action, reward, next_obs_np)
        total, (l_slice, l_r, l_p, l_v) = train_step_fn(obs_b, next_obs_b, action_b, reward_b, onpolicy_mask_mx)
        mx.eval(state, total)  # force sync -> real wall-clock incl. learn step

        t1 = time.perf_counter()
        tick_ms.append((t1 - t0) * 1000.0)

        obs_np = next_obs_np

        line = (f"step={step:4d} tick_ms={tick_ms[-1]:.4f} loss_total={total.item():.5f} "
                f"slice={l_slice.item():.5f} reward={l_r.item():.5f} "
                f"policy={l_p.item():.5f} value={l_v.item():.5f} action={action} r={reward_raw:.5f}")
        log_lines.append(line)
        if not quiet and (step % 20 == 0 or step == steps - 1):
            print(line)

    t_wall_end = time.perf_counter()

    tick_ms_sorted = sorted(tick_ms)
    n = len(tick_ms_sorted)
    mean_ms = sum(tick_ms_sorted) / n
    p95_ms = tick_ms_sorted[int(n * 0.95)]
    p99_ms = tick_ms_sorted[min(n - 1, int(n * 0.99))]

    summary = {
        "steps": n,
        "mean_ms": mean_ms, "p95_ms": p95_ms, "p99_ms": p99_ms,
        "max_ms": tick_ms_sorted[-1], "min_ms": tick_ms_sorted[0],
        "wall_s": t_wall_end - t_wall_start,
    }

    if not quiet:
        print(f"# {n} online ticks (env+action+learn) in {summary['wall_s']:.2f}s")
        print(f"ms/tick (incl. learn-step): mean={mean_ms:.4f} p95={p95_ms:.4f} "
              f"p99={p99_ms:.4f} min={summary['min_ms']:.4f} max={summary['max_ms']:.4f}")
        print(f"120fps budget = 8.3333 ms/tick")
        print(f"MEASURED mean << 8.33ms budget: {mean_ms < 8.3333} (margin {8.3333/mean_ms:.1f}x)")
        print(f"MEASURED p99  << 8.33ms budget: {p99_ms < 8.3333} (margin {8.3333/p99_ms:.1f}x)")

    if log_path:
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines) + "\n")

    return log_lines, summary, obs_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--log", type=str, default="online_log.txt")
    ap.add_argument("--episode-len", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--buffer-cap", type=int, default=200)
    ap.add_argument("--check-determinism", action="store_true")
    args = ap.parse_args()

    if args.check_determinism:
        kw = dict(episode_len=args.episode_len, batch_size=args.batch_size, buffer_cap=args.buffer_cap)
        lines_a, _, obs_a = run(steps=80, seed=args.seed, lr=args.lr, eps=args.eps, log_path=None, quiet=True, **kw)
        lines_b, _, obs_b = run(steps=80, seed=args.seed, lr=args.lr, eps=args.eps, log_path=None, quiet=True, **kw)
        obs_identical = bool(np.array_equal(obs_a, obs_b))
        # strip the wall-clock tick_ms field (expected to differ run-to-run) before
        # comparing -- everything else (loss components, action taken) must match
        # bit-for-bit for a truly deterministic seed->trajectory claim.
        import re as _re
        strip_ms = lambda l: _re.sub(r"tick_ms=[\d.]+ ", "", l)
        lines_a_s = [strip_ms(l) for l in lines_a]
        lines_b_s = [strip_ms(l) for l in lines_b]
        loss_identical = lines_a_s == lines_b_s
        print(f"determinism (same seed -> same final obs, bit-identical): {obs_identical}")
        print(f"determinism (same seed -> same full 80-step log incl. losses/actions, tick_ms excluded): {loss_identical}")
        if not loss_identical:
            for i, (la, lb) in enumerate(zip(lines_a_s, lines_b_s)):
                if la != lb:
                    print(f"  first divergence at step {i}:\n    A: {la}\n    B: {lb}")
                    break
        return

    run(steps=args.steps, seed=args.seed, lr=args.lr, eps=args.eps, log_path=args.log,
        episode_len=args.episode_len, batch_size=args.batch_size, buffer_cap=args.buffer_cap)


if __name__ == "__main__":
    main()
