"""mu/gate_inside.py — gates for 内側学習 (mu-inside): does mu learn the WORLD
from inside, rather than the one task the world hands it?

Gate law here (inherited, HANDOFF-MU-WORLD.md): gate on BEHAVIOUR, not loss.
A loss trend could not see the frozen-policy bug; an action histogram could.

GI-1 determinism  : same seed -> bit-identical log (tick_ms excluded) for the
                    intrinsic modes too (the reward now depends on the model's
                    own error => a new way to become nondeterministic).
GI-2 world-sourced: change the seed -> the err/reward stream MUST move. Guards
                    against an intrinsic reward that is a self-contained
                    python fiction detached from the field.
GI-3 behaviour    : world-reward vs intrinsic-reward action histograms must
                    DIFFER, and intrinsic coverage entropy must be HIGHER
                    (task-seeking collapses onto the target site; world-seeking
                    should not). Plus: model error must still fall.
GI-4 perf         : ms/tick vs the 8.33ms 120fps budget. Marked UNVERIFIED
                    when machine load > 2.0 (control fact: arc1's committed
                    "2-4ms" re-measured 21ms under load).

Run: python mu/gate_inside.py [--steps 200]
"""
import argparse
import math
import os
import re

import numpy as np
import mlx.core as mx

import online
from nets import MuNet, NUM_ACTIONS, action_onehot
from world_env import WorldEnv

HELDOUT_SEED = 123  # never trained on


def heldout_err_rel(net, residual=True, seed=HELDOUT_SEED, ticks=200):
    """GI-5 (audit R1 homework ①): score a net on a FIXED forced-action rollout
    in an unseen world, learning OFF. Removes both confounds at once:
      - policy confound (a trained policy may simply visit easier states)
      - denominator confound (both nets see the IDENTICAL frames, so the
        err_rel denominator is bit-identical between them -- mu cannot move it)
    """
    ev = WorldEnv(seed=seed)
    o = ev.reset(seed=seed)
    rels = []
    for t in range(ticks):
        a = t % NUM_ACTIONS
        no, _ = ev.step(a)
        s = net.repr(mx.array(o)[None, ...])
        s2, _ = net.dyn(s, action_onehot(mx.array([a], dtype=mx.uint32)))
        dec = net.decoder(s2)
        if residual:
            dec = mx.array(o)[None, ...] + dec
        tgt = mx.array(no)[None, ...]
        rel = mx.mean((dec - tgt) ** 2) / (mx.mean(tgt ** 2) + 1e-12)
        mx.eval(rel)
        rels.append(float(rel.item()))
        o = no
    ev.close()
    return np.array(rels)


def frozen_twin(init_params):
    net = MuNet()
    net.update(init_params)
    mx.eval(net.parameters())
    return net

STRIP = lambda l: re.sub(r"tick_ms=[\d.]+ ", "", l)


def hist(lines):
    h = [0] * 8
    for l in lines:
        h[int(re.search(r"action=(\d)", l).group(1))] += 1
    return h


def entropy(h):
    n = sum(h)
    return -sum((c / n) * math.log(c / n) for c in h if c) if n else 0.0


def halves(xs):
    m = len(xs) // 2
    return float(np.mean(xs[:m])), float(np.mean(xs[m:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "GATE-INSIDE.txt"))
    args = ap.parse_args()
    S, SEED = args.steps, args.seed
    rows = []

    def emit(s):
        print(s)
        rows.append(s)

    load1 = os.getloadavg()[0]
    emit(f"# machine load1={load1:.2f} steps={S} seed={SEED}")

    runs = {}
    for mode in ("world", "surprise", "surprise_rel", "progress"):
        lines, summ, _ = online.run(steps=S, seed=SEED, eps=0.1, log_path=None, quiet=True,
                                    reward_mode=mode)
        runs[mode] = (lines, summ)
        emit(f"[{mode}] hist={hist(lines)} H={entropy(hist(lines)):.4f} "
             f"err {halves(summ['err_hist'])[0]:.6f}->{halves(summ['err_hist'])[1]:.6f} "
             f"err_rel {halves(summ['rel_hist'])[0]:.6f}->{halves(summ['rel_hist'])[1]:.6f} "
             f"mean_ms={summ['mean_ms']:.3f} p99_ms={summ['p99_ms']:.3f}")

    # GI-1 determinism (intrinsic modes: reward depends on the model itself)
    a, _, _ = online.run(steps=60, seed=SEED, eps=0.1, log_path=None, quiet=True, reward_mode="progress")
    b, _, _ = online.run(steps=60, seed=SEED, eps=0.1, log_path=None, quiet=True, reward_mode="progress")
    gi1 = [STRIP(l) for l in a] == [STRIP(l) for l in b]
    emit(f"GI-1 determinism (progress, 60 steps, bit-identical log): {'PASS' if gi1 else 'FAIL'}")

    # GI-2 world-sourced: another seed must move the err stream
    _, s_other, _ = online.run(steps=S, seed=SEED + 7, eps=0.1, log_path=None, quiet=True,
                               reward_mode="surprise")
    e0 = np.array(runs["surprise"][1]["err_hist"])
    e1 = np.array(s_other["err_hist"])
    gi2 = not np.array_equal(e0, e1)
    emit(f"GI-2 world-sourced (seed {SEED} vs {SEED+7} err streams differ): {'PASS' if gi2 else 'FAIL'} "
         f"(mean {e0.mean():.6f} vs {e1.mean():.6f})")

    # GI-3 behaviour: intrinsic != task-seeking, and error still falls
    hw, hs, hp = hist(runs["world"][0]), hist(runs["surprise"][0]), hist(runs["progress"][0])
    diff = hw != hs or hw != hp
    Hw, Hs, Hp = entropy(hw), entropy(hs), entropy(hp)
    cover = Hs > Hw or Hp > Hw
    # model quality = SCALE-FREE error (see online.py loss_fn): absolute MSE
    # rises simply because mu makes the field louder, so it cannot decide
    # "is the model getting better". err_rel can.
    rel = {m: halves(runs[m][1]["rel_hist"]) for m in runs}
    falls = all(v[1] < v[0] for v in rel.values())
    gi3 = diff and cover and falls
    emit(f"GI-3 behaviour: hist_differs={diff} coverage_H world={Hw:.4f} surprise={Hs:.4f} "
         f"progress={Hp:.4f} (intrinsic>task={cover}) err_rel_falls={falls} "
         f"{ {m: f'{v[0]:.5f}->{v[1]:.5f}' for m, v in rel.items()} } -> {'PASS' if gi3 else 'FAIL'}")

    # GI-4 perf
    worst = max(runs[m][1]["mean_ms"] for m in runs)
    worst99 = max(runs[m][1]["p99_ms"] for m in runs)
    verdict = "PASS" if (worst < 8.3333 and worst99 < 8.3333) else "FAIL"
    if load1 > 2.0:
        verdict = f"UNVERIFIED (load1={load1:.2f} > 2.0; numbers are load-contaminated) [{verdict} as measured]"
    emit(f"GI-4 perf: worst mean_ms={worst:.3f} worst p99_ms={worst99:.3f} vs 8.3333 -> {verdict}")

    # ---- GI-5 held-out fixed trajectory: trained vs frozen-init, learning OFF
    ho = {}
    for mode in runs:
        s = runs[mode][1]
        ho[mode] = heldout_err_rel(s["net"], residual=s["residual"])
    frozen = heldout_err_rel(frozen_twin(runs["progress"][1]["init_params"]),
                             residual=runs["progress"][1]["residual"])
    fh = halves(frozen)
    ratios = {m: float(ho[m].mean() / frozen.mean()) for m in ho}
    gi5 = all(r <= 0.95 for r in ratios.values())
    flat = abs(fh[1] / fh[0] - 1.0) <= 0.10
    emit(f"GI-5 control-line flatness: frozen halves {fh[0]:.4f}->{fh[1]:.4f} "
         f"flat_within_10pct={flat} (if False the control DRIFTS -- only the "
         f"same-frames trained/frozen ratio is admissible evidence, not the trend)")
    emit(f"GI-5 held-out (seed {HELDOUT_SEED}, 200 forced actions, learning OFF): "
         f"frozen-init err_rel={frozen.mean():.4f} halves {fh[0]:.4f}->{fh[1]:.4f} (control line) | "
         + " ".join(f"{m}={ho[m].mean():.4f}(x{ratios[m]:.3f})" for m in ho)
         + f" -> {'PASS' if gi5 else 'FAIL'}")

    # ---- GI-6 shout-exploit: does the reward drive mu to inflate the field?
    eg = {m: halves(runs[m][1]["energy_hist"]) for m in runs}
    grow = {m: (v[1] / v[0] - 1.0) * 100 for m, v in eg.items()}
    gi6 = grow["surprise_rel"] <= grow["surprise"]
    emit("GI-6 shout-exploit (field energy growth 1st->2nd half %): "
         + " ".join(f"{m}={grow[m]:+.1f}" for m in grow)
         + f" | countermeasure surprise_rel <= surprise: {'PASS' if gi6 else 'FAIL'}")

    # ---- GI-7 beats the persistence null model ("next frame = this frame")
    # MARGIN, not sign. With residual decoding the net can pass a sign-only test
    # by emitting ~zero delta = reproducing the null model exactly (measured:
    # margin 0.0001 = 0.1%). Beating persistence by a rounding error is not a
    # world model. Threshold: 5% below the null.
    PERSIST_MARGIN = 0.95
    pers = {m: float(np.mean(runs[m][1]["persist_hist"][-50:])) for m in runs}
    last = {m: float(np.mean(runs[m][1]["rel_hist"][-50:])) for m in runs}
    frac = {m: last[m] / pers[m] for m in runs}
    gi7 = all(frac[m] <= PERSIST_MARGIN for m in runs)
    emit("GI-7 vs persistence null (last 50 ticks, err_rel / |obs-next|² ratio, need <=0.95): "
         + " ".join(f"{m}={last[m]:.4f}/{pers[m]:.4f}=x{frac[m]:.4f}" for m in runs)
         + f" -> {'PASS' if gi7 else 'FAIL'}")

    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
