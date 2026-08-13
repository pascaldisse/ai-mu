# HANDOFF — lane `mu-world` (Mu × the real field), 2026-08-01, @opus5

worktree `~/projects/mc-mu-world` · branch `mu-world` = field-unified@a4c9f1b6 ⊕ mu-v0@e0d3a6cb
venv: `~/projects/magic-crystal-mu-v0/.venv/bin/python` · build: `cargo build --release -p field`

## SHA chain
- `6787d2a4` arc2 段1a `packages/field/src/bin/mu_serve.rs` — world door
- `30401377` arc2 段1b `mu/world_env.py` — bridge
- `4d8781c8` arc2 段2 learner on the real field (G3 FAIL)
- `64ef041d` arc2 段3 frozen-policy FIX + adv clip + entropy + RMS reward (G3 PASS)
- `f77df7b6` arc3 段1 `packages/field/src/surface.rs` + bin `field-view3` (RENDER法)
- `76d3763c` arc3 段2 4D block: `VIEW3 [at k]` by replay, `WHEN` observer axis
- `77f3dafc` arc3 段3 MCTS verdict = 否 (numbers in `mu/BENCH.md`)

## Laws in force here
- Mu learns from the ENGINE's field, never a python replica. `mu/field_env.py` = retired toy.
- RENDER法: field render = 3D surface/volume ALWAYS. heatmap (`plane::encode_png`) = debug only, never a product path.
- 4D block: trajectory = seed + journal ops (implicit). Past = replay, never stored frames. "When" = journal position on the entropy axis (`WHEN`).
- Own impl only (no Google/TF/PyTorch; MLX allowed). Ultradeterminism: state = f(seed).
- Gate on BEHAVIOUR, not loss: an action-histogram trend. (Loss trend could not see the frozen-policy bug.)

## Gates
green: G2 determinism (bit-identical 80-step log+obs) · G3 learning (act2 29→111/150, reward 0.16→1.01, value loss 0.68→0.11) · G4 world-sourced · G5 render determinism · G6 4D-block replay ≡ live
open: **G1 ms/tick vs 8.33ms — UNVERIFIED, machine load 8→76 all session.** Rerun idle: `python mu/online.py --steps 300 --eps 0.1`. Control fact: arc1's committed "2.0-4.1ms PASS" measured 21.1ms in the same session ⇒ that number is load-dependent, not a baseline.

## The bug worth remembering
`mx.compile(fn)` captures arrays BY VALUE. Without `inputs=state` the acting policy is frozen at trace-time weights — arc1 shipped that. Proof used: two runs with DIFFERENT loss functions emitted a bit-identical action stream.

## Next (in order)
1. G1 on an idle machine; train step (7.3ms, conv over 128×128×2) is the only real cost — select 0.98 / env+IPC 1.11 / batch 0.10.
2. MCTS repair: replace the REINFORCE policy term with cross-entropy distillation of `plan()`'s `policy_target`, then re-run the A/B (`--plan-sims 16` vs 0, 600 ticks, act2 histogram + raw reward).
3. Speak-back: `WHEN` is the "when" answer; needs a "where/what" companion (probe over the store) before any dialogue arc.
4. Render on the tick path is unbenchmarked (480² CPU ray-march, currently offline-only).
