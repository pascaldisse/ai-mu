"""mu/gate_window_r2.py -- mu-inside atom-2r: GW-3 re-try with ONE model
strengthening move + the stationary held-out measurement the corpse designed.

Move (ONE, 定): scale-invariance package = NormWindowPredictor (per-sample
RMS norm, 512x2 hidden) trained on scale-free error. Replay buffer exists in
window_online.py (corpse) but is SEALED here: replay=0 everywhere.

Measurement fix (window_eval.py, corpse rationale): the online err_rel
stream is non-stationary for weight-independent reasons (empty early patches,
denominator drifts by decades) -- that is WHY online GW-3(b) fails for any
model. Held-out fixed transitions (seed 11, never trained on) evaluated at
checkpoints attribute any decrease to weights alone.

GW-3R controls (same laws: any FAIL forbids 学習実在 claim):
  (a) held-out err_rel decreases step0 -> step300         [norm lr=1e-3]
  (b) lr=0: held-out err_rel EXACTLY flat at all ckpts    [norm lr=0]
  (c) trained beats persistence null on the SAME held-out set, ratio<=0.95
  (d) old WindowPredictor same protocol (attribution of the move)
  (e) trained beats const-predictor null on held-out, ratio<=0.95

Run: ~/projects/magic-crystal-mu-v0/.venv/bin/python mu/gate_window_r2.py > mu/GATE-WINDOW-R2.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from window_online import run  # noqa: E402
from window_eval import collect_eval_set, eval_err  # noqa: E402

SEED = 4          # training trajectory seed -- same as gate_window.py GW-3
EVAL_SEED = 11    # held-out world seed, never trained on
N = 300
EVAL_EVERY = 75
BEAT = 0.95       # same convention as GI-7 / GW-3


def ckpt_table(tag, eval_hist):
    lines = [f"  {tag} ckpt err_rel: " + " ".join(
        f"@{s}={er:.6f}" for (s, e, er, pe, pr) in eval_hist)]
    return "\n".join(lines)


def main():
    print(f"# mu-inside atom-2r gate_window_r2.py -- train seed={SEED} "
          f"held-out seed={EVAL_SEED} N={N} eval_every={EVAL_EVERY} replay=0(sealed)")
    eval_set = collect_eval_set(EVAL_SEED, k=16, n=64)
    P, A, Nx = eval_set
    print(f"# held-out set: {P.shape[0]} transitions, k=16, "
          f"persist_rel(set)={eval_err.__doc__ and ''}", end="")
    # persistence on the held-out set (weight-free, computed once)
    import mlx.core as mx
    persist = float(mx.mean((P - Nx) ** 2).item())
    energy = float(mx.mean(Nx ** 2).item())
    persist_rel = persist / max(energy, 1e-9)
    print(f"persist_err={persist:.6e} energy={energy:.6e} persist_rel={persist_rel:.6f}")

    common = dict(steps=N, seed=SEED, quiet=True, log_path=None,
                  eval_set=eval_set, eval_every=EVAL_EVERY, replay=0)

    _, s_norm = run(lr=1e-3, predictor="norm", **common)
    _, s_frozen = run(lr=0.0, predictor="norm", **common)
    _, s_old = run(lr=1e-3, predictor="window", **common)
    _, s_const = run(lr=1e-3, predictor="const", **common)

    eh = s_norm["eval_hist"]
    first_rel, final_rel = eh[0][2], eh[-1][2]
    a_pass = final_rel < first_rel
    print(ckpt_table("(a) norm lr=1e-3", eh))
    print(f"  (a) held-out err_rel step0={first_rel:.6f} step{N}={final_rel:.6f} "
          f"decrease: {'PASS' if a_pass else 'FAIL'}")

    ef = s_frozen["eval_hist"]
    rels = [er for (s, e, er, pe, pr) in ef]
    b_pass = all(r == rels[0] for r in rels)
    print(ckpt_table("(b) norm lr=0  ", ef))
    print(f"  (b) lr=0 held-out EXACTLY flat: {'PASS' if b_pass else 'FAIL'}")

    c_ratio = final_rel / persist_rel if persist_rel else float("inf")
    c_pass = c_ratio <= BEAT
    print(f"  (c) trained final={final_rel:.6f} vs persistence null={persist_rel:.6f} "
          f"ratio={c_ratio:.4f} beat<={BEAT}: {'PASS' if c_pass else 'FAIL'}")

    eo = s_old["eval_hist"]
    print(ckpt_table("(d) old window ", eo))
    print(f"  (d) attribution: old-model final={eo[-1][2]:.6f} vs norm final={final_rel:.6f} "
          f"(norm better: {final_rel < eo[-1][2]})")

    ec = s_const["eval_hist"]
    e_ratio = final_rel / ec[-1][2] if ec[-1][2] else float("inf")
    e_pass = e_ratio <= BEAT
    print(ckpt_table("(e) const null ", ec))
    print(f"  (e) trained final={final_rel:.6f} vs const final={ec[-1][2]:.6f} "
          f"ratio={e_ratio:.4f} beat<={BEAT}: {'PASS' if e_pass else 'FAIL'}")

    all_pass = a_pass and b_pass and c_pass and e_pass
    print(f"GW-3R OVERALL: {'PASS' if all_pass else 'FAIL'}"
          + ("" if all_pass else " -- 学習実在主張禁止 stays in force"))
    print("# NOTE: online GW-3 (gate_window.py) verdict is NOT superseded by this file;"
          "\n#       claim requires green here AND an idle-machine confirmation run.")


if __name__ == "__main__":
    main()
