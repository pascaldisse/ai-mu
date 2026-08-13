"""mu/window_eval.py -- atom-2 段1: STATIONARY evaluation harness for the
(patch, action) -> patch' predictor.

Why this exists (atom-1 corpse): GW-3(b) failed because the reported metric
was the ONLINE err_rel stream. That stream is non-stationary for reasons that
have nothing to do with weights -- early ticks see near-empty patches (the
wavefront has not arrived), the energy denominator moves by decades, and the
action/position trajectory wanders. Hence even lr=0 (frozen weights) showed a
0.41 "improvement" ratio: the WORLD got easier, not the model better.

Fix: a FIXED held-out set of transitions, collected once from a DIFFERENT
seed's trajectory, evaluated with a forward pass only (no grad, no optimizer)
at fixed step checkpoints. Then:
  * lr=0 gives literally identical numbers at every checkpoint (flat, exactly)
  * any decrease is attributable to weights alone (data held constant)
  * the persistence null is computed on the SAME frozen transitions

metric: err_rel = mean_i MSE(pred_i, next_i) / max(mean_i MSE(0, next_i),
FLOOR) -- one aggregate ratio over the whole set (NOT a mean of per-sample
ratios, which would be dominated by the near-empty patches, atom-1's known
division artifact).
"""
import numpy as np
import mlx.core as mx

from world_env import WindowEnv
from window_nets import action_onehot, NUM_WINDOW_ACTIONS

EVAL_FLOOR = 1e-9  # aggregate denominator floor; the aggregate energy over a
                   # whole held-out set is never near-zero (measured), so this
                   # is a divide-by-zero guard, not a shaping constant.


def collect_eval_set(seed, k=16, n=64, action_seed=None):
    """Fixed held-out transitions from a trajectory the trainer never sees."""
    import random
    rng = random.Random(seed if action_seed is None else action_seed)
    env = WindowEnv(seed=seed, k=k)
    patch = env.reset(seed=seed)
    P, A, N = [], [], []
    for _ in range(n):
        a = rng.randrange(NUM_WINDOW_ACTIONS)
        nxt, _r, _pos = env.step_window(a)
        P.append(patch); A.append(a); N.append(nxt)
        patch = nxt
    env.close()
    return (mx.array(np.stack(P)), mx.array(np.array(A)), mx.array(np.stack(N)))


def eval_err(net, eval_set):
    """-> (err, err_rel, persist_err, persist_rel). Forward only."""
    P, A, N = eval_set
    pred = net(P, action_onehot(A))
    err = mx.mean((pred - N) ** 2)
    energy = mx.maximum(mx.mean(N ** 2), EVAL_FLOOR)
    persist = mx.mean((P - N) ** 2)
    mx.eval(err, energy, persist)
    return (float(err.item()), float((err / energy).item()),
            float(persist.item()), float((persist / energy).item()))
