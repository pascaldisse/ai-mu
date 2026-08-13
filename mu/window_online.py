"""mu/window_online.py -- mu-inside atom-1 段2: online (patch_t, action) ->
patch_{t+1} prediction, error learning, minimal closed loop.

No RL, no reward, no policy/value nets -- the atom brief is explicit: "予測し
誤差で学ぶ閉ループ" (predict, learn from error, closed loop). Every world
tick does, in order: 1) choose an action (off-graph, weight-independent --
see below), 2) env.step_window(action) -> next patch (real field, mu-serve),
3) ONE mx.compile'd forward+backward+optimizer.update on the single fresh
(patch_t, action, patch_{t+1}) transition (batch=1 -- the brief literally
says "1 tick 毎に前向き+逆伝播+更新", no buffering/batching asked for; this
is also a plain supervised regression, not the REINFORCE/TD combination
mu/online.py's corpses found batch=1 too noisy for -- MSE gradient variance
is far lower than policy-gradient variance, no buffer needed here).

stop_gradient law (MU-INSIDE.md/追令): applies to REWARD fed into a policy
graph, so the optimizer cannot shortcut its own objective. This atom has no
reward and no policy graph -- err IS the supervised loss target, and must
receive gradient directly (that is what "learn from error" means for a
predictor). Nothing to stop_gradient here; noted explicitly so this omission
reads as a decision, not a miss.

Action stream is DELIBERATELY independent of net weights (python `random.
Random(seed)`, never touching mx.random) so the (patch,action,next_patch)
trajectory for a given --seed is IDENTICAL across --predictor choices and
across --lr choices -- required for GW-3's controlled comparisons (b/d/e) to
actually hold everything but the one varied factor constant.

mx.compile pattern: state=[net.state, opt.state], inputs=state, outputs=state
(the arc2段3 frozen-policy bug this guards against was a compiled ACTING
function silently seeing stale weights -- inapplicable here since there is no
compiled acting function at all, action choice is off-graph -- but the
TRAINING step still needs inputs=state/outputs=state or the optimizer/weight
updates would not persist between compiled calls; both wired correctly and
warmup-checked below).

No separate warmup pre-step (unlike mu/online.py): that pattern spends one
real gradient step outside the reported step count, which needs a
params/opt-state snapshot+restore to stay honest -- extra machinery for a
"minimal" loop. Instead step 0 legitimately IS the first compiled call (its
ms/tick includes the one-time mx.compile trace cost); perf stats below
report step 0 separately and compute mean/p95/p99 over steps[1:], documented,
not hidden.

Run: python window_online.py [--steps N] [--seed S] [--lr LR] [--k K]
     [--predictor {window,const}] [--fixed-actions] [--log PATH]
"""
import argparse
import random
import time

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from world_env import WindowEnv
from window_nets import (WindowPredictor, ConstPredictor, NormWindowPredictor,
                         action_onehot, NUM_WINDOW_ACTIONS, K_DEFAULT)
from window_eval import eval_err

PREDICTORS = ("window", "const", "norm")

# 段2 measured (200-step probe, seed=1, predictor=window): local KxK patch
# energy ranges 1.8e-18 (wavefront hasn't reached the patch yet -- 3/200
# ticks) .. 5.1e-3, median 6.5e-4, p5 4.6e-5. Unlike the full 128x128 field
# (online.py's err_rel, never near-zero -- averaged over 16384 cells), a
# 16x16 window legitimately sees near-empty patches early in a run/far from
# the seed excite site, and raw err=err/energy blew up to 3.0e9 there (a
# division artifact, not a model-quality signal). WINDOW_ENERGY_FLOOR floors
# the denominator ~1 decade below the measured p5, well above the near-zero
# outliers (<1e-9), so err_rel stays a meaningful diagnostic instead of being
# dominated by a few empty-patch ticks. This is a DIAGNOSTIC floor only (no
# reward/policy here to exploit by shouting -- the online.py ENERGY_FLOOR
# risk was specific to a reward an actor could inflate; nothing here selects
# actions to influence its own denominator).
WINDOW_ENERGY_FLOOR = 1e-5


def make_net(predictor, k):
    if predictor == "window":
        return WindowPredictor(k=k)
    if predictor == "norm":
        return NormWindowPredictor(k=k)
    return ConstPredictor(k=k)


def make_step(net, opt):
    state = [net.state, opt.state]

    def loss_fn(net, patch, next_patch, a_onehot):
        pred = net(patch, a_onehot)
        err_raw = mx.mean((pred - next_patch) ** 2)
        energy = mx.mean(next_patch ** 2)
        denom = mx.maximum(energy, WINDOW_ENERGY_FLOOR)
        err_rel = err_raw / denom
        # atom-2 段1: TRAIN on the per-sample scale-free error (err/own energy),
        # not raw MSE -- otherwise loud ticks own the gradient and quiet ticks
        # (same real dynamics, smaller amplitude) contribute ~nothing. Reported
        # err stays the raw MSE so logs remain comparable with atom-1.
        b = patch.shape[0]
        se = mx.mean(((pred - next_patch) ** 2).reshape(b, -1), axis=-1)
        sc = mx.mean((next_patch ** 2).reshape(b, -1), axis=-1) + \
             mx.mean((patch ** 2).reshape(b, -1), axis=-1)
        err = mx.mean(se / mx.maximum(sc, 1e-12))
        persist_err = mx.mean((patch - next_patch) ** 2)
        persist_rel = persist_err / denom
        return err, (err_raw, err_rel, persist_err, persist_rel, energy)

    loss_and_grad_fn = optim_value_and_grad(net, loss_fn)

    def _step(patch, next_patch, a_onehot):
        (err, aux), grads = loss_and_grad_fn(net, patch, next_patch, a_onehot)
        opt.update(net, grads)
        return err, aux

    step_fn = mx.compile(_step, inputs=state, outputs=state)
    return step_fn, state


def optim_value_and_grad(net, loss_fn):
    import mlx.nn as nn
    return nn.value_and_grad(net, loss_fn)


def choose_action(step, act_rng, fixed_actions):
    if fixed_actions:
        return step % NUM_WINDOW_ACTIONS  # deterministic round-robin, seed-independent
    return act_rng.randrange(NUM_WINDOW_ACTIONS)


def run(steps=250, seed=0, lr=1e-3, k=K_DEFAULT, predictor="window",
        fixed_actions=False, log_path=None, quiet=False,
        eval_set=None, eval_every=0, replay=0, batch=8, action_seed=None):
    assert predictor in PREDICTORS, predictor
    mx.random.seed(seed)   # net init only
    np.random.seed(seed)   # unused directly but kept for safety/consistency

    env = WindowEnv(seed=seed, k=k)
    net = make_net(predictor, k)
    opt = optim.Adam(learning_rate=lr)
    step_fn, state = make_step(net, opt)

    act_rng = random.Random(seed if action_seed is None else action_seed)  # action stream: independent of mx.random,
                                    # independent of predictor/lr -- same
                                    # trajectory guaranteed across GW-3 A/Bs

    patch_np = env.reset(seed=seed)

    eval_hist = []  # (step, err, err_rel, persist_err, persist_rel)
    if eval_set is not None and eval_every:
        eval_hist.append((0,) + eval_err(net, eval_set))
    err_hist, err_rel_hist = [], []
    persist_err_hist, persist_rel_hist = [], []
    energy_hist = []
    action_hist = []
    tick_ms = []
    log_lines = []

    # atom-2 段2 replay (optional, default OFF = atom-1 behaviour unchanged):
    # bounded ring of the most recent `replay` transitions; one minibatch of
    # `batch` per tick (always including the fresh transition), drawn from a
    # python RNG seeded off `seed` -- off-graph, weight-independent, so the
    # GW-3 A/B controls still vary exactly one factor.
    buf_p, buf_n, buf_a = [], [], []
    buf_rng = random.Random(seed + 7919)

    t_wall_start = time.perf_counter()
    for step in range(steps):
        action = choose_action(step, act_rng, fixed_actions)
        next_patch_np, reward_world, pos = env.step_window(action)

        t0 = time.perf_counter()
        if replay:
            buf_p.append(patch_np); buf_n.append(next_patch_np); buf_a.append(action)
            if len(buf_p) > replay:
                del buf_p[0]; del buf_n[0]; del buf_a[0]
            idx = [len(buf_p) - 1] + [buf_rng.randrange(len(buf_p))
                                      for _ in range(batch - 1)]
            patch_mx = mx.array(np.stack([buf_p[i] for i in idx]))
            next_patch_mx = mx.array(np.stack([buf_n[i] for i in idx]))
            a_onehot = action_onehot(mx.array([buf_a[i] for i in idx]))
        else:
            patch_mx = mx.array(patch_np)[None, ...]
            next_patch_mx = mx.array(next_patch_np)[None, ...]
            a_onehot = action_onehot(mx.array([action]))
        err, (err_v, err_rel_v, persist_err_v, persist_rel_v, energy_v) = step_fn(
            patch_mx, next_patch_mx, a_onehot)
        mx.eval(state, err)
        t1 = time.perf_counter()
        tick_ms.append((t1 - t0) * 1000.0)

        e = float(err_v.item())
        er = float(err_rel_v.item())
        pe = float(persist_err_v.item())
        pr = float(persist_rel_v.item())
        en = float(energy_v.item())
        err_hist.append(e); err_rel_hist.append(er)
        persist_err_hist.append(pe); persist_rel_hist.append(pr)
        energy_hist.append(en); action_hist.append(action)

        line = (f"step={step:4d} tick_ms={tick_ms[-1]:.4f} predictor={predictor} "
                f"action={action} pos={pos[0]},{pos[1]} err={e:.6f} err_rel={er:.6f} "
                f"persist_err={pe:.6f} persist_rel={pr:.6f} energy={en:.6f}")
        log_lines.append(line)
        if not quiet and (step % 50 == 0 or step == steps - 1):
            print(line)

        patch_np = next_patch_np
        if eval_set is not None and eval_every and (step + 1) % eval_every == 0:
            eval_hist.append((step + 1,) + eval_err(net, eval_set))

    t_wall_end = time.perf_counter()
    env.close()

    tick_ms_rest = sorted(tick_ms[1:]) if len(tick_ms) > 1 else sorted(tick_ms)
    n = len(tick_ms_rest)
    mean_ms = sum(tick_ms_rest) / n if n else float("nan")
    p95_ms = tick_ms_rest[int(n * 0.95)] if n else float("nan")
    p99_ms = tick_ms_rest[min(n - 1, int(n * 0.99))] if n else float("nan")

    summary = {
        "steps": steps, "seed": seed, "lr": lr, "k": k, "predictor": predictor,
        "fixed_actions": fixed_actions,
        "step0_ms": tick_ms[0] if tick_ms else float("nan"),
        "mean_ms": mean_ms, "p95_ms": p95_ms, "p99_ms": p99_ms,
        "max_ms": max(tick_ms_rest) if tick_ms_rest else float("nan"),
        "min_ms": min(tick_ms_rest) if tick_ms_rest else float("nan"),
        "wall_s": t_wall_end - t_wall_start,
        "err_hist": err_hist, "err_rel_hist": err_rel_hist,
        "persist_err_hist": persist_err_hist, "persist_rel_hist": persist_rel_hist,
        "energy_hist": energy_hist, "action_hist": action_hist,
        "eval_hist": eval_hist, "replay": replay, "batch": batch,
    }

    if not quiet:
        print(f"# {steps} online window ticks in {summary['wall_s']:.2f}s "
              f"(step0={summary['step0_ms']:.3f}ms incl. one-time mx.compile trace, "
              f"excluded from stats below)")
        print(f"ms/tick (steps 1..N-1): mean={mean_ms:.4f} p95={p95_ms:.4f} "
              f"p99={p99_ms:.4f} min={summary['min_ms']:.4f} max={summary['max_ms']:.4f}")
        print(f"120fps budget = 8.3333 ms/tick; mean<<budget: {mean_ms < 8.3333} "
              f"(margin {8.3333 / mean_ms:.1f}x)" if mean_ms == mean_ms else "")

    if log_path:
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines) + "\n")

    return log_lines, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--predictor", type=str, default="window", choices=list(PREDICTORS))
    ap.add_argument("--fixed-actions", action="store_true")
    ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--log", type=str, default="window_online_log.txt")
    ap.add_argument("--check-determinism", action="store_true")
    args = ap.parse_args()

    if args.check_determinism:
        kw = dict(lr=args.lr, k=args.k, predictor=args.predictor, fixed_actions=args.fixed_actions)
        lines_a, _ = run(steps=60, seed=args.seed, log_path=None, quiet=True, **kw)
        lines_b, _ = run(steps=60, seed=args.seed, log_path=None, quiet=True, **kw)
        import re as _re
        strip_ms = lambda l: _re.sub(r"tick_ms=[\d.]+ ", "", l)
        la = [strip_ms(l) for l in lines_a]
        lb = [strip_ms(l) for l in lines_b]
        identical = la == lb
        print(f"determinism (same seed -> same 60-step log, tick_ms excluded): {identical}")
        if not identical:
            for i, (x, y) in enumerate(zip(la, lb)):
                if x != y:
                    print(f"  first divergence at step {i}:\n    A: {x}\n    B: {y}")
                    break
        return

    run(steps=args.steps, seed=args.seed, lr=args.lr, k=args.k, predictor=args.predictor,
        fixed_actions=args.fixed_actions, log_path=args.log,
        replay=args.replay, batch=args.batch)


if __name__ == "__main__":
    main()
