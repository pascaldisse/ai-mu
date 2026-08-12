# FIELD capacity/SNR characterization — W8, 2026-08-01

Branch `field/capacity` (worktree `~/projects/magic-crystal-field-capacity`, base
field-unified @816c4c5e + docs-only tip). Files: `packages/field/tests/capacity.rs`
+ this doc. atoms.rs is todo!() in this tree → test-local atom helper, spec from
src/atoms.rs doc comment. 真=measured (this run) · 記=derived/analytic · 假=assumed.

---

## Setup
- cfg 64×64 → d = W·H = 4096. Ops: REAL `ops::{bundle, similarity}`.
- Atom helper: unit-magnitude random-phase spectrum, phase(kx,ky) = f(seed, idx)
  via `mix64` ONLY; conjugate symmetry S[-k] = conj(S[k]); self-conjugate bins
  (DC + Nyquist combos) = ±1 from seed → |S[k]| ≡ 1 → Parseval ||atom|| = 1
  (1/d-normalized inverse), measured within 1e-3.
- Materialization: naive_idft2 formula is O(d²) ≈ 16.7M trig pairs PER ATOM; the
  mandated pipeline (256 members + 5×256 fresh distractors ≈ 1536 atoms) would be
  hours. Helper materializes via test-local separable cached-twiddle idft2 —
  ALGEBRAICALLY IDENTICAL math (same sign, same 1/d, same f64 acc order);
  pinned by `fast_idft_equiv_naive`: |fast − lib naive_idft2| < 1e-6 on 4 seeds.
- Seeds: members = 0..256 (nested first-K); distractors fresh per K =
  1_000_000 + K·1000 + i. No rand crate. Deterministic to the bit.

## Measured table @ d = 4096 (256 fresh distractors per K)
| K | member μ | mean\|dist\| μ | margin | accuracy | min member | max dist |
|---|----------|---------------|--------|----------|------------|----------|
| 1   | 1.000000 | 0.013100 | 0.986900 | 100.0% | 1.000000 | 0.044967 |
| 4   | 0.496594 | 0.012951 | 0.483644 | 100.0% | 0.487358 | 0.044284 |
| 16  | 0.252563 | 0.012101 | 0.240462 | 100.0% | 0.227766 | 0.043344 |
| 64  | 0.126587 | 0.013503 | 0.113084 | 100.0% | 0.088709 | 0.037229 |
| 256 | 0.061475 | 0.012763 | 0.048711 |  82.0% | 0.022789 | 0.046952 |

## sqrt(d/K) fit
- member μ ≈ 1.004/√K (fit over K∈{1..256}; μ·√K = 1.000, 0.993, 1.010, 1.013,
  0.984 — within 1.5% of 1/√K, ≪ factor 2) 真.
- distractor noise floor FLAT in K: mean|dist| ≈ 0.8·(1/√d) = 0.0125, std =
  1/√d = 0.0156 for ALL K 真. 記: sim(bundle, dist) = Σ_K dot(a_j, dist)/(√K·1),
  each dot ~ N(0, 1/d) → std = √(K/d)/√K = 1/√d — d-only, K-independent.
- margin = 1.004/√K − 0.8/√d: shrinks ~1/√K until it hits the d-floor 真.
- retrieval cliff 記: member max vs distractor max ≈ √(2 ln N)/√d → K* ≈ d/(2 ln N).
  Measured: 100% at K≤64, 82% at K=256 → cliff in (64, 369) at d=4096, N=256 ✓.
- Adversary §2 verdict CONFIRMED in-substrate: one bundle degrades ~1/√K; the
  N×d store is the fix, not bigger bundles.

## Extrapolation d = 16384 (128²)
Same model (member term d-independent, floor = 1/√d):
- member μ ≈ 1/√K unchanged; noise std = 1/128 = 0.0078, mean|dist| ≈ 0.0063.
- max over N=256 candidates ≈ 0.026 → K ≤ 256 at ≈100% with 2.4× margin;
  cliff K* ≈ 16384/11.1 ≈ 1476.
- 假 (scale assumption): cross-term orthogonality holds at 128² as at 64² —
  same mix64 phase construction, purely d-scaled.

## Store slot budget rec (N×d resident matrix)
- Rule: keep K ≤ d/22 per slot (cliff K* = d/11, halved for margin) → d=4096:
  ≤ 186, use K ≤ 64 (measured 100%, 6× headroom) · d=16384: ≤ 745, use K ≤ 256.
- World capacity ≈ N·d/22 items at ~100% retrieval over a 256-candidate probe;
  beyond that, split the world across slots (the store IS the fix).
- Probe-set growth N: capacity per slot shrinks ~1/√(2 ln N) — logarithmic, cheap
  to scale the store.

## Gates (run TODAY, no todo-deps)
`cargo build -p field` → green (1 pre-existing dead_code warn, lib).
`cargo test -p field --test capacity` → 3 passed (atom_helper_sanity,
fast_idft_equiv_naive, capacity_snr_at_4096), 13.5s. Full paste in report.

## UNVERIFIED
None. (atom helper is test-local by decree — atoms.rs itself stays W1's;
its integration path = the equiv test pinning the fast path to lib naive_idft2.)
