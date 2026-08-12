"""mu/mcts.py — batched planning over the MuZero nets. NO python per-node loop.

Two modes, per the field-recon 120Hz sketch (docs/research/2026-08-01-field-recon.md
§3, option a/b):

  amortized_policy(net, obs)
      0-sim: ONE fused forward (repr -> pred). This IS the interactive-tick
      path — no search, policy/value straight from the net. Used by
      tick_bench.py for the <8.33ms budget.

  plan(net, obs, sims=16, depth=1)
      8-32 sims batched into dynamics call(s), batch dim = SIMS, not nodes.
      default depth=1 -> exactly ONE dynamics call (root expand: sims parallel
      action samples, one batched dyn+pred pass, pick best by value). depth>1
      is supported as a batched multi-step rollout (still batch=sims per step,
      a small fixed python loop over DEPTH, never over nodes/sims) for
      training-time targets in train.py; interactive path always uses depth=1
      or amortized_policy.

Root-action aggregation (visit-count analog) is a fixed python loop over
NUM_ACTIONS (8 — a constant, not a per-node/per-sim loop).
"""
import mlx.core as mx
import mlx.nn as nn

from nets import NUM_ACTIONS, action_onehot


def amortized_policy(net, obs):
    """ONE fused forward pass: repr(obs) -> pred(s). No search.
    Returns (policy_logits, value, latent_s)."""
    s = net.repr(obs)
    policy_logits, value = net.pred(s)
    return policy_logits, value, s


def plan(net, obs, sims: int = 16, depth: int = 1, discount: float = 0.9, temp: float = 0.5):
    """Batched planning. batch dim = sims. depth is a small fixed python loop
    (default 1 -> exactly one dynamics call), never a per-node loop.

    Root branching is a DETERMINISTIC uniform tile over NUM_ACTIONS (sims must
    be a multiple of NUM_ACTIONS: 8/16/24/32 all qualify) — every action gets
    an equal number of batched rollouts, so the resulting action_values reflect
    genuine value differences (from the learned dyn/pred nets), not sampling
    noise from an as-yet-untrained prior. policy_target is then softmax(Q/temp)
    over those action_values — a one-step lookahead distillation target (NOT a
    raw visit count, since allocation is uniform by construction).

    Returns dict: policy_target (NUM_ACTIONS,), value_target (scalar),
    root_value (scalar, amortized pred value, no search), best_action (int).
    """
    assert obs.shape[0] == 1, "plan() roots a single observation; batch=sims is internal"
    assert sims % NUM_ACTIONS == 0, "sims must be a multiple of NUM_ACTIONS for uniform root tiling"
    s0 = net.repr(obs)                       # (1, LATENT_DIM)
    policy0_logits, value0 = net.pred(s0)     # (1,NUM_ACTIONS), (1,1)

    # expand root to `sims` parallel branches — batch dim = sims, uniform per action
    reps = sims // NUM_ACTIONS
    s = mx.repeat(s0, sims, axis=0)                                  # (sims, LATENT_DIM)
    root_action = mx.repeat(mx.arange(NUM_ACTIONS, dtype=mx.uint32), reps)  # (sims,)
    action = root_action
    returns = mx.zeros((sims,))
    disc = 1.0

    for _ in range(depth):                              # fixed small loop over DEPTH, not nodes
        a_onehot = action_onehot(action)                 # (sims, NUM_ACTIONS)
        s_next, r = net.dyn(s, a_onehot)                 # ONE dynamics call, batch=sims
        returns = returns + disc * r.reshape(-1)
        disc = disc * discount
        s = s_next
        if depth > 1:
            next_logits, _ = net.pred(s)
            next_probs = mx.softmax(next_logits, axis=-1)
            action = mx.random.categorical(mx.log(next_probs + 1e-8))

    _, leaf_value = net.pred(s)                          # bootstrap leaf value, batch=sims
    returns = returns + disc * leaf_value.reshape(-1)     # (sims,)

    # aggregate per root action: uniform allocation (reps each) -> plain mean per
    # action, a single reshape+mean (batched, no per-node python loop).
    action_values = mx.mean(returns.reshape(NUM_ACTIONS, reps), axis=-1)  # (NUM_ACTIONS,)
    policy_target = mx.softmax(action_values / temp, axis=-1)
    value_target = mx.sum(policy_target * action_values)
    best_action = int(mx.argmax(action_values).item())

    return {
        "policy_target": policy_target,       # (NUM_ACTIONS,) softmax(Q/temp) distillation target
        "value_target": value_target,         # scalar, planned value
        "action_values": action_values,       # (NUM_ACTIONS,)
        "root_value_amortized": value0.reshape(()),  # scalar, 0-sim value (no search)
        "best_action": best_action,
    }


if __name__ == "__main__":
    import time
    from nets import MuNet, GRID

    net = MuNet()
    obs = mx.random.normal((1, GRID, GRID, 1))

    pl, v, s = amortized_policy(net, obs)
    print("amortized: policy", pl.shape, "value", v.shape)

    for sims in (8, 16, 32):
        t0 = time.perf_counter()
        out = plan(net, obs, sims=sims, depth=1)
        mx.eval(out["policy_target"], out["value_target"])
        t1 = time.perf_counter()
        print(f"plan sims={sims} depth=1: {(t1-t0)*1000:.3f}ms "
              f"policy_target={out['policy_target'].tolist()} "
              f"value_target={out['value_target'].item():.4f} "
              f"best_action={out['best_action']}")
