"""mu/gate_window.py -- mu-inside atom-1 段3: GATE-WINDOW.txt.

Run: ~/projects/magic-crystal-mu-v0/.venv/bin/python mu/gate_window.py > mu/GATE-WINDOW.txt

GW-1 determinism, GW-2 world-sourced, GW-3 learning-real (5 controls,
追令② adds (e) const-predictor baseline -- ALL FIVE required, any one
failing forbids an "学習実在" claim), GW-4 perf (UNVERIFIED, not PASS/FAIL,
if `uptime` load1 > 2.0 -- 追令①/既往: a committed "2-4ms" was 21ms under
load, never trust a load-contaminated number as PASS/FAIL).

追令① determinism scope: GW-1 also compares release vs debug mu-serve
binaries (sin/cos codegen can 1ulp-diverge between profiles) and reports any
divergence as a 1ulp NOTE, not folded into PASS.
"""
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_env import REPO  # noqa: E402
from window_online import run, make_step  # noqa: E402
from window_eval import collect_eval_set  # noqa: E402
from window_nets import WindowPredictor, action_onehot, K_DEFAULT  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

SEED = 4          # GW-3 canonical seed (explored during 段2 smoke -- not
                   # cherry-picked post-hoc: same seed used throughout 段2's
                   # design-sanity checks, reused here for continuity)
N_GW3 = 300        # atom-1 online-stream controls (kept, for GW-1/2 continuity)
# atom-2: GW-3 is re-measured on a STATIONARY held-out set (window_eval.py).
# corpse: atom-1's GW-3 read the ONLINE err_rel stream, whose first/second-half
# ratio moves for world reasons alone (empty early patches, energy decades) --
# that is why the lr=0 control (b) showed 0.41 and FAILED flatness. Data held
# constant + forward-only eval => (b) is exactly 1.0000 by construction and any
# remaining decrease is attributable to weights alone.
N_TRAIN = 2000     # training ticks per GW-3 condition
EVAL_EVERY = 500
EVAL_N = 64        # held-out transitions
EVAL_SEED = 1004   # a world seed the trainer (SEED=4) never runs on
PREDICTOR = "norm"  # atom-2 scale-normalised residual net
REPLAY, BATCH, LR = 2048, 16, 3e-4
ACT_SEED_A, ACT_SEED_B = 99, 7  # two independent action streams for GW-3(d)
FLAT_TOL = 0.10    # "flat" band for the lr=0 control -- same convention this
                   # worktree's own MU-INSIDE.md/GATE-INSIDE.txt already used
                   # (GI-5 "flat_within_10pct"); reused for comparability
BEAT_RATIO = 0.95  # "meaningfully below" a baseline -- same convention as
                   # this worktree's GI-7 ("vs persistence null ... need <=0.95")


def strip_ms(line):
    return re.sub(r"tick_ms=[\d.]+ ", "", line)


def _tree_maxdiff(a, b):
    """max|a-b| over a nested params tree -- plain python recursion + mx.max,
    a DIFFERENT code path than the loss/err computation (no shared failure
    mode with the thing being verified, per 追令2①)."""
    if isinstance(a, dict):
        return max((_tree_maxdiff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        return max((_tree_maxdiff(x, y) for x, y in zip(a, b)), default=0.0)
    return float(mx.max(mx.abs(a - b)).item())


def gw1b_weight_change():
    """追令2①: assert compiled weights ACTUALLY move after one train step --
    independent of the loss/err trend (which shares the computation path and
    could mask a frozen-weight bug, per the arc2段3 precedent -- structurally
    inapplicable here since window_online has no compiled ACTING fn, but the
    TRAIN step itself still needs a direct, non-loss-based check). Same
    synthetic (patch,next_patch,action) input for both lr conditions so only
    lr differs."""
    out = {}
    for lr, label in [(1e-3, "lr=1e-3"), (0.0, "lr=0")]:
        mx.random.seed(SEED)
        net = WindowPredictor(k=K_DEFAULT)
        opt = optim.Adam(learning_rate=lr)
        step_fn, state = make_step(net, opt)
        before = tree_map(lambda a: mx.array(a), net.parameters())
        mx.eval(before)
        mx.random.seed(123)
        patch = mx.random.normal((1, K_DEFAULT, K_DEFAULT, 2))
        next_patch = mx.random.normal((1, K_DEFAULT, K_DEFAULT, 2))
        a_onehot = action_onehot(mx.array([0]))
        step_fn(patch, next_patch, a_onehot)
        mx.eval(state)
        maxdiff = _tree_maxdiff(before, net.parameters())
        out[label] = maxdiff
    lr_moves = out["lr=1e-3"] > 0.0
    lr0_frozen = out["lr=0"] == 0.0
    return out, lr_moves, lr0_frozen


def gw1_determinism():
    lines_a, _ = run(steps=60, seed=SEED, predictor="window", quiet=True, log_path=None)
    lines_b, _ = run(steps=60, seed=SEED, predictor="window", quiet=True, log_path=None)
    sa, sb = [strip_ms(l) for l in lines_a], [strip_ms(l) for l in lines_b]
    same_seed_ok = sa == sb

    rel_bin = os.path.join(REPO, "target", "release", "mu-serve")
    dbg_bin = os.path.join(REPO, "target", "debug", "mu-serve")
    profile_note = "debug binary not built, profile comparison skipped"
    profile_identical = None
    if os.path.exists(dbg_bin):
        from world_env import WindowEnv
        FIXED = [0, 4, 1, 4, 2, 4, 3, 4, 0, 1, 2, 3, 4, 4, 0, 3] * 4

        def patch_stream(bin_path):
            e = WindowEnv(seed=SEED, k=16, bin_path=bin_path)
            ps = [e._read_patch().copy()]
            for i in range(60):
                p, r, pos = e.step_window(FIXED[i % len(FIXED)])
                ps.append(p.copy())
            e.close()
            return ps

        rel_p = patch_stream(rel_bin)
        dbg_p = patch_stream(dbg_bin)
        profile_identical = all(np.array_equal(x, y) for x, y in zip(rel_p, dbg_p))
        if not profile_identical:
            maxdiff = max(float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
                           for x, y in zip(rel_p, dbg_p))
            profile_note = f"release vs debug DIVERGE, max_abs_diff={maxdiff:.3e} (1ulp-class note, not PASS/FAIL)"
        else:
            profile_note = "release vs debug byte-identical, 60-step forced-action patch stream"

    return same_seed_ok, profile_identical, profile_note


def gw2_world_sourced():
    _, sa = run(steps=200, seed=0, predictor="window", fixed_actions=True, quiet=True, log_path=None)
    _, sb = run(steps=200, seed=7, predictor="window", fixed_actions=True, quiet=True, log_path=None)
    # fixed_actions=True -> the ACTION stream is identical for both seeds;
    # any difference in err/err_rel is then attributable ONLY to the world
    # (env seed), not to a different random action sequence.
    ea, eb = np.array(sa["err_hist"]), np.array(sb["err_hist"])
    differs = not np.array_equal(ea, eb)
    return differs, float(ea.mean()), float(eb.mean())


def _train(eval_set, lr=LR, fixed_actions=False, predictor=PREDICTOR, steps=N_TRAIN,
           action_seed=None):
    return run(steps=steps, seed=SEED, lr=lr, predictor=predictor,
               fixed_actions=fixed_actions, action_seed=action_seed,
               quiet=True, log_path=None,
               eval_set=eval_set, eval_every=EVAL_EVERY, replay=REPLAY, batch=BATCH)[1]


def gw3_controls():
    eval_set = collect_eval_set(seed=EVAL_SEED, k=K_DEFAULT, n=EVAL_N)

    # (a) trained: held-out err_rel at step 0 vs at the end. Same frozen data
    #     at both checkpoints -> only the weights differ.
    s_main = _train(eval_set)
    ev = s_main["eval_hist"]
    a_first, a_last = ev[0][2], ev[-1][2]
    a_pass = a_last < a_first

    # (b) lr=0 control: identical everything except the learning rate.
    s_frozen = _train(eval_set, lr=0.0)
    evf = s_frozen["eval_hist"]
    b_first, b_last = evf[0][2], evf[-1][2]
    b_ratio = b_last / b_first if b_first else float("inf")
    b_pass = abs(b_ratio - 1.0) <= FLAT_TOL

    # (c) persistence null on the SAME held-out transitions ("next = this").
    #     Independent of any training run; read from the eval set itself.
    c_persist = ev[0][4]
    c_ratio = a_last / c_persist if c_persist else float("inf")
    c_pass = c_ratio <= BEAT_RATIO

    # (d) trajectory-independence: the decrease must not be a property of ONE
    #     action stream. Two INDEPENDENT action-RNG seeds (world seed fixed),
    #     both must fall.
    #     corpse (f2e7ef05, kept): (d) was `fixed_actions=True` = round-robin
    #     up,down,left,right,poke. That 5-cycle has ZERO net displacement, so
    #     the observation window never leaves its start locale -- measured by
    #     an independent path (position set over 500 ticks, no net involved):
    #     3 distinct positions vs 128 for a random action stream. The trained
    #     net then sees ~one locale and is evaluated on a held-out random-walk
    #     trajectory: err_rel 0.6798 -> 0.7022, FAIL. That is a train/eval
    #     DISTRIBUTION result (degenerate probe), not evidence about learning;
    #     replacing a degenerate probe is legitimate, hiding it is not -- hence
    #     it is still run and reported below as a NOTE.
    s_d1 = _train(eval_set, action_seed=ACT_SEED_A)
    s_d2 = _train(eval_set, action_seed=ACT_SEED_B)
    d1_first, d1_last = s_d1["eval_hist"][0][2], s_d1["eval_hist"][-1][2]
    d2_first, d2_last = s_d2["eval_hist"][0][2], s_d2["eval_hist"][-1][2]
    d_first, d_last = d1_first, d1_last
    d_pass = (d1_last < d1_first) and (d2_last < d2_first)
    s_rr = _train(eval_set, fixed_actions=True)
    rr_first, rr_last = s_rr["eval_hist"][0][2], s_rr["eval_hist"][-1][2]

    # (e) const-predictor null (input-blind), trained identically.
    s_const = _train(eval_set, predictor="const")
    eve = s_const["eval_hist"]
    e_first, e_last = eve[0][2], eve[-1][2]
    e_ratio = a_last / e_last if e_last else float("inf")
    e_pass = e_ratio <= BEAT_RATIO

    controls = {
        "a": dict(first=a_first, second=a_last, curve=[(s, r) for s, _e, r, _pe, _pr in ev],
                   pass_=a_pass),
        "b": dict(first=b_first, second=b_last, ratio=b_ratio, flat_tol=FLAT_TOL, pass_=b_pass),
        "c": dict(trained_second=a_last, persist_first=c_persist, persist_second=c_persist,
                   ratio=c_ratio, beat_ratio=BEAT_RATIO, pass_=c_pass),
        "d": dict(first=d_first, second=d_last, pass_=d_pass,
                   a_seed=ACT_SEED_A, b_seed=ACT_SEED_B,
                   d2_first=d2_first, d2_last=d2_last,
                   rr_first=rr_first, rr_last=rr_last),
        "e": dict(trained_second=a_last, const_first=e_first, const_second=e_last,
                   ratio=e_ratio, beat_ratio=BEAT_RATIO, pass_=e_pass),
    }
    all_pass = all(c["pass_"] for c in controls.values())
    return controls, all_pass, s_main


def gw4_perf(s_main):
    load1 = None
    try:
        out = subprocess.check_output(["uptime"]).decode()
        m = re.search(r"load averages?:\s*([\d.]+)", out)
        if m:
            load1 = float(m.group(1))
    except Exception:
        pass
    budget = 8.3333
    verdict = "UNVERIFIED"
    reason = ""
    if load1 is not None and load1 > 2.0:
        reason = f"load1={load1} > 2.0, numbers load-contaminated -- 既往: committed 2-4ms measured 21ms under load"
    else:
        verdict = "PASS" if s_main["mean_ms"] < budget and s_main["p99_ms"] < budget else "FAIL"
        reason = "measured idle (load1<=2.0)"
    return load1, budget, verdict, reason


def main():
    print(f"# mu-inside atom-2 gate_window.py -- seed={SEED} N_GW3={N_GW3} "
          f"GW-3: held-out eval set seed={EVAL_SEED} n={EVAL_N}, train={N_TRAIN} ticks, "
          f"predictor={PREDICTOR} lr={LR} replay={REPLAY} batch={BATCH}")

    same_seed_ok, profile_identical, profile_note = gw1_determinism()
    gw1_verdict = "PASS" if same_seed_ok else "FAIL"
    print(f"GW-1 determinism (seed={SEED}, 60 steps, tick_ms-excluded log bit-identical): "
          f"{gw1_verdict}")
    print(f"GW-1-profile (release vs debug mu-serve, 追令①): {profile_note}")

    wdiffs, lr_moves, lr0_frozen = gw1b_weight_change()
    gw1b_verdict = "PASS" if (lr_moves and lr0_frozen) else "FAIL"
    print(f"GW-1b weight-update mechanism (追令2①, direct param read, independent of "
          f"loss trend): lr=1e-3 max|Δw|={wdiffs['lr=1e-3']:.3e} (moves: "
          f"{'PASS' if lr_moves else 'FAIL'}) · lr=0 max|Δw|={wdiffs['lr=0']:.3e} "
          f"(frozen: {'PASS' if lr0_frozen else 'FAIL'}) -> {gw1b_verdict}")

    differs, ea_mean, eb_mean = gw2_world_sourced()
    gw2_verdict = "PASS" if differs else "FAIL"
    print(f"GW-2 world-sourced (seed 0 vs 7, fixed_actions=True to isolate world "
          f"from action-stream randomness, 200 steps, err streams differ): {gw2_verdict} "
          f"(mean err seed0={ea_mean:.6e} seed7={eb_mean:.6e})")

    controls, gw3_all_pass, s_main = gw3_controls()
    c = controls
    print(f"GW-3 learning-real, 5 controls required (追令②), any FAIL forbids "
          f"\"学習実在\" claim:")
    print(f"  (a) held-out err_rel step0={c['a']['first']:.6f} final={c['a']['second']:.6f} "
          f"curve={[(s, round(r, 4)) for s, r in c['a']['curve']]} "
          f"decrease={'PASS' if c['a']['pass_'] else 'FAIL'}")
    print(f"  (b) lr=0 control held-out err_rel step0={c['b']['first']:.6f} "
          f"final={c['b']['second']:.6f} ratio={c['b']['ratio']:.4f} "
          f"flat(|ratio-1|<={FLAT_TOL}): {'PASS' if c['b']['pass_'] else 'FAIL'}")
    print(f"  (c) persistence baseline (same held-out set): trained final={c['c']['trained_second']:.6f} "
          f"persist={c['c']['persist_first']:.6f} "
          f"ratio(trained/persist)={c['c']['ratio']:.4f} beat<={BEAT_RATIO}: "
          f"{'PASS' if c['c']['pass_'] else 'FAIL'}")
    print(f"  (d) trajectory-independence, 2 independent action streams: "
          f"act_seed={c['d']['a_seed']} {c['d']['first']:.6f}->{c['d']['second']:.6f} · "
          f"act_seed={c['d']['b_seed']} {c['d']['d2_first']:.6f}->{c['d']['d2_last']:.6f} "
          f"both-decrease: {'PASS' if c['d']['pass_'] else 'FAIL'}")
    print(f"  (d-NOTE, not gating) round-robin 5-cycle probe (corpse of f2e7ef05): "
          f"{c['d']['rr_first']:.6f}->{c['d']['rr_last']:.6f} — degenerate: that cycle has zero "
          f"net displacement, 3 distinct window positions over 500 ticks vs 128 for a random "
          f"action stream (measured without the net) ⇒ train/eval distribution artifact, "
          f"reported, not counted")
    print(f"  (e) 追令② const-predictor null (input-blind, same training): trained final={c['e']['trained_second']:.6f} "
          f"const step0={c['e']['const_first']:.6f} const final={c['e']['const_second']:.6f} "
          f"ratio(trained/const)={c['e']['ratio']:.4f} beat<={BEAT_RATIO}: "
          f"{'PASS' if c['e']['pass_'] else 'FAIL'}")
    gw3_verdict = "PASS" if gw3_all_pass else "FAIL"
    print(f"GW-3 OVERALL: {gw3_verdict}" + ("" if gw3_all_pass else
          " -- \u5b66\u7fd2\u5b9f\u5728\u4e3b\u5f35\u7981\u6b62 (learning-exists claim FORBIDDEN, per law: any one control failing blocks the claim)"))

    load1, budget, gw4_verdict, gw4_reason = gw4_perf(s_main)
    print(f"GW-4 perf: step0_ms={s_main['step0_ms']:.4f} mean_ms={s_main['mean_ms']:.4f} "
          f"p95_ms={s_main['p95_ms']:.4f} p99_ms={s_main['p99_ms']:.4f} vs budget={budget} "
          f"load1={load1} -> {gw4_verdict} ({gw4_reason})")

    print(f"\n# summary: GW-1={gw1_verdict} GW-1b={gw1b_verdict} GW-2={gw2_verdict} "
          f"GW-3={gw3_verdict} GW-4={gw4_verdict}")


if __name__ == "__main__":
    main()
