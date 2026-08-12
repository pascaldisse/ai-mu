# ADVERSARY REPORT — v9-wire merge · 2026-07-28

VERDICT: HOLDS

Subject: merge commit `a0fde0b18165d6083a41993cbbd8198e7988df1a`
— subject `Merge branch 'v9-wire': live gather/compose/async/ordeal work`,
current `main` HEAD.
Reviewer: @opus5 (adversary lane). Nothing re-merged, nothing pushed.

Topology confirmed by direct read, not by claim:

| ref | sha | subject |
|---|---|---|
| `a0fde0b1` | `a0fde0b18165d608…` | Merge branch 'v9-wire': live gather/compose/async/ordeal work |
| `^1` | `f3ecf70c53e4458a…` | Merge branch 'v9-body': integrator NEE, mirror-threshold, reference-tool work |
| `^2` | `1530cf0a3f776a9a…` | v9-wire STACK 2: ordeal probe-equals-live compose (input clamp + hitgate) finished + run on v9y-best (FAIL, honest) + REAL-OR-BLACK law verified both ways offscreen 8431 |
| merge-base(`^1`,`^2`) | `b0f0ab5b16c03475…` | — |

## CONCORDANCE

### Lane A — v9-wire landed intact (no dropped hunks)

`git diff --numstat a0fde0b1^2 a0fde0b1 -- <file>` is EMPTY for every named
v9-wire artifact, i.e. the merge tree's copy is byte-identical to the
v9-wire branch tip. Blob-sha equality (`git rev-parse ^2:path` vs
`rev-parse merge:path`) confirms independently:

| file | P2 blob | merge blob | |
|---|---|---|---|
| `src/rdirect_gather.rs` | `0af768a79f` | `0af768a79f` | IDENTICAL |
| `src/rdirect_v9_compose.rs` | `9fd77ce366` | `9fd77ce366` | IDENTICAL |
| `src/rdirect_v9_compose.wgsl` | `ba22cfec44` | `ba22cfec44` | IDENTICAL |
| `src/main.rs` | `53f57ef0e6` | `53f57ef0e6` | IDENTICAL |
| `tests/rdirect_gather_v9_ordeal.rs` | `2f2d70e43f` | `2f2d70e43f` | IDENTICAL |
| `tests/rdirect_v9_compose_parity.rs` | `91a23fc477` | `91a23fc477` | IDENTICAL |

Also zero-diff vs the branch tip: `src/rdirect_pack16.wgsl`,
`src/rdirect_gather_split.wgsl`, `src/rdirect.rs`, `src/lib.rs`,
`examples/rdirect_frame_wall.rs`, `examples/real_image_ordeal.rs`,
`examples/v9_async_trace_parity.rs`, `examples/v9_ed_composite_parity.rs`,
`docs/perf/2026-07-21-v9-wire.md`.

Content spot-check (the work is wired, not merely present):
`src/lib.rs:16` `pub mod rdirect_gather;` and `:20` `pub mod
rdirect_v9_compose;` register both modules; `main.rs:42/54` import
`rdirect_gather::{…}` and `rdirect_v9_compose::{compose_cpu_reference,
V9ComposePass}`; the async path exists as fields + methods
(`async_trace` `main.rs:1395`, `v9_async` `:1476`,
`resolve_frame_v9_async`, `UnetLive::submit_gpu_bridged_async`) and the
GPU fuse `GAIA_V9_COMPOSE=gpu` is read once at construction
(`main.rs:2050`). Both new test files carry 2 `#[test]` fns each
(`rdirect_v9_compose_parity.rs` `run_case(hitgate: bool)` — the
input-clamp/hitgate ordeal shape the branch-tip subject describes).

### Lane B — v9-body (parent1) NOT reverted

`integrator.rs` and `integrator.wgsl` blobs at `a0fde0b1` are IDENTICAL to
their blobs at `a0fde0b1^1` — v9-wire never touched them
(`git diff --numstat b0f0ab5b a0fde0b1^2 -- src/integrator.rs
src/integrator.wgsl` is empty), so there was nothing to clobber.

Requested dials, grepped at the merge tree (not the branch):
- `GAIA_EMISSIVE_NEE` — 3 occurrences, both at `^1` and at merge;
  live at `integrator.rs:392` `surface: [width, height, 1,
  env_flag("GAIA_EMISSIVE_NEE", false) as u32]`.
- `GAIA_NEE_CLAMP` — 2 occurrences, both at `^1` and at merge; live at
  `integrator.rs:412` `temporal_flags: [0, 512, 0,
  env_f32("GAIA_NEE_CLAMP", 0.0).to_bits()]`.
- mirror threshold present: `integrator.rs:343-344` "STAGE 2 (mirror
  threshold): roughness at/below this takes the delta-mirror branch
  instead of stochastic GGX" (`GAIA_MIRROR_ROUGHNESS_MAX`, documented at
  `:21`).

All 8 v9-body trainers (`examples/rdirect_train_v9{d,e,f,g,h,i,j,k}.rs`)
are present in `git ls-tree a0fde0b1`. `git diff --numstat a0fde0b1^1
a0fde0b1` contains ZERO pure-deletion entries — the merge is additive
relative to parent1.

### Union proof — the one file BOTH lanes touched

`src/rdirect_unet.rs` is the only shared-edit file, so it is the only
place a real drop could hide. Line accounting closes exactly:

```
base -> P1 (v9-body):  147   0
base -> P2 (v9-wire):  934  28
P1   -> merge:         934  28     == base->P2  (v9-wire delta applied whole)
P2   -> merge:         147   0     == base->P1  (v9-body delta applied whole)
```

A dropped hunk on either side would shrink one of the bottom two rows.
Neither shrank. The `P2 -> merge` delta is v9-body's
`UnetWeights::validate_shapes_against` structural self-check (+147/-0,
`rdirect_unet.rs:1656+`) — present in the merge tree.

File-SET equality closes the same argument tree-wide:
`{files where merge != P2}` == `{files v9-body changed base->P1}` and
`{files where merge != P1}` == `{files v9-wire changed base->P2}`, with
exactly two names falling out of each set:
`data/rdirect-weights-v9c.{bin,provenance.json}`. Resolved, not waved:
those two paths are ABSENT at the merge-base (`git ls-tree b0f0ab5b` →
0 matches) and were added on BOTH branches with IDENTICAL content
(bin `19e56307f10f`, provenance `91f55524e0b7` at P1, P2 AND merge).
Same bytes from both sides = nothing to drop. Not a defect.

The large deletion block in `git diff a0fde0b1^1 a0fde0b1^2` (trainers,
weights, `integrator.rs -450`, `integrator.wgsl -344`) is a branch-point
artifact, not a v9-wire removal: v9-wire forked before v9-body's work, and
`git diff --diff-filter=D --name-only b0f0ab5b a0fde0b1^2 --
packages/scrying-glass/examples/` is empty — v9-wire deleted nothing.

**CONCORDANCE:** both lanes are present in one tree, and the two proofs are
independent — blob-sha identity would survive a wrong line count, and the
numstat union identity would survive a wrong blob comparison. Both agree,
and the build in §BUILD would break on a truncated `rdirect_unet.rs` (the
one merged-by-both file) regardless of what the diffs said.

## DOCTRINE-CONCORDANCE

Grepped, not assumed. `doctrine_concordance` literal count per touched
trainer:

```
rdirect_train_v9d.rs  1      rdirect_train_v9h.rs  0
rdirect_train_v9e.rs  1      rdirect_train_v9i.rs  0
rdirect_train_v9f.rs  0      rdirect_train_v9j.rs  0
rdirect_train_v9g.rs  0      rdirect_train_v9k.rs  0
```

v9d (`:972`) and v9e (`:837`) carry the explicit field verbatim:
`"STAGE 2 of the v9-body lane: v8d's TIER1(estimator-init: NOT ported, no
analytic estimator-init exists for a conv net yet — FRESH He-init instead,
disclosed)+TIER2(K-averaged noise2noise, K={k_draws}, teacher=validator-
only)+TIER3(structure: moving-camera history w/ REAL reprojection +
curved-mirror/low-roughness pose + EMA history-source, kept as structural
mechanisms, NOT the training signal)"`.

v9f–v9k carry NO `doctrine_concordance` key, but do carry the same
TIER-language provenance fields verbatim — `"init": "FRESH He-init
(UnetWeights::new_random) — no conv-net analytic estimator-init exists
yet, disclosed gap vs v8d's TIER1"` and `"loss": "whole-image MSE(net(step
features), mean of K={k_draws} independent draw radiances) — teacher NEVER
in the loss, validator only (v8d TIER2 doctrine, ported)"`
(v9f `:1000-1001`, v9g `:1078-1079`, v9h `:1097-1098`, v9k `:3264-3265`)
— and each header declares byte-identical inheritance of every unlisted
mechanism from its base (v9f←v9e, v9g←v9f, v9h←v9g, v9i←v9h, v9j←v9i,
v9k←v9j), each round changing ONE variable (winsorize default, CURE 4
gradient, dose 4.0, CURE 5 no-hit bypass, CURE 5 instrumentation, bar-res
crop training). So the doctrine chain is inherited by declaration, not
restated per file.

Mapped onto NEURAL.md §TRAINING DOCTRINE's own names (1 STRUCTURE,
2 EQUATIONS AS SIGNAL, 3 DATA = COMPUTE):
- **STRUCTURE** — NOT implemented, disclosed in-file (no analytic
  estimator-init exists for a conv net; fresh He-init instead). The
  moving-camera history / real reprojection / EMA history-source
  mechanisms the trainers label "TIER3(structure…)" are structural
  scaffolding kept OUT of the training signal by their own words.
- **EQUATIONS AS SIGNAL** — implemented, and it is the training signal:
  K-averaged noise2noise (`GAIA_V9_K`, `DRAW_B_SEED_BASE: u32 = 0xB222`
  "v8d parity — noise2noise label draws"), teacher demoted to
  validator-only, teacher NEVER in the loss. This is the tier NEURAL.md
  names noise2noise under explicitly.
- **DATA = COMPUTE** — implemented in substance: every sample is
  engine-generated per seed (pose-diversity pool, `draws_per_epoch: 3`,
  and in v9k 128×72 crops of true 640×480 frusta via
  `IntegratorUniform::build_cropped`). No external corpus anywhere.

Adversary note (numbering collision, recorded honestly): the trainers'
internal `TIER3` label denotes *structure*, which is NEURAL.md's tier **1**.
Their TIER-numbering is v8d-lineage vocabulary, not NEURAL.md's ordering.
Cited above by NAME to avoid propagating the collision. This is a docs
hygiene wart, not a doctrine violation — the substance of tiers 2 and 3
is present and tier 1's absence is disclosed at both provenance and
header level.

DOCTRINE-CONCORDANCE: the touched trainers (`rdirect_train_v9{d,e,f,g,h,i,j,k}.rs`) implement EQUATIONS-AS-SIGNAL (K-averaged noise2noise, teacher=validator-only, teacher never in the loss) + DATA=COMPUTE (all samples engine-generated per seed; v9k trains on bar-res crops), with STRUCTURE deliberately NOT implemented and disclosed in-file ("no analytic estimator-init exists for a conv net yet — FRESH He-init instead, disclosed") — verified by direct grep of `doctrine_concordance` in v9d/v9e and of the inherited `init`/`loss` provenance fields in v9f–v9k.

## BUILD

Independent re-run at `a0fde0b1` == current `main` HEAD (worktree clean of
tracked modifications; only untracked scratch present). All under
`nice -n19`, `-j2`.

`cargo build --release -p scrying-glass -j2` → **EXIT=0**, 3 warnings
(the known pre-existing lib/bin set):

```
22 | use scrying_glass::bvh::{Bvh, BvhParams, DEFAULT_DEGRADE_RATIO, DynamicSplice, RefitParams};
   |                                          ^^^^^^^^^^^^^^^^^^^^^
   = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: fields `temporal_enabled`, `temporal_params`, `t_bind`, `t_parity`, `t_prev`, and `last_view` are never read
    --> packages/scrying-glass/src/main.rs:3291:5
warning: method `view_key` is never used
    --> packages/scrying-glass/src/main.rs:3986:8

warning: `scrying-glass` (bin "scrying-glass") generated 3 warnings (run `cargo fix --bin "scrying-glass" -p scrying-glass` to apply 1 suggestion)
    Finished `release` profile [optimized] target(s) in 54.31s
```

`cargo build --release --examples -p scrying-glass -j2` → **EXIT=0**,
`grep -c '^error'` = **0**:

```
warning: unused variable: `uni_target`
  --> packages/scrying-glass/examples/v9_async_trace_parity.rs:86:9
warning: `scrying-glass` (example "v9_async_trace_parity") generated 2 warnings
warning: unused import: `local_max_3x3`
  --> packages/scrying-glass/examples/mirror_autopsy.rs:29:78
warning: `scrying-glass` (example "mirror_autopsy") generated 2 warnings
    Finished `release` profile [optimized] target(s) in 1.05s
```

CACHE HONESTY: that examples run finished in 1.05s — fully warm cargo
cache, so on its own it is a freshness assertion, not a compile. Two
anti-cache checks were therefore forced:

- `touch` on `v9_ed_composite_parity.rs`, `v9_async_trace_parity.rs`,
  `rdirect_frame_wall.rs` then rebuild those three → **EXIT=0, errors=0**,
  `Finished release profile [optimized] target(s) in 32.41s` (real work).
- `cargo test --release -p scrying-glass --test rdirect_v9_compose_parity
  --test rdirect_gather_v9_ordeal --no-run -j2` → **EXIT=0, errors=0**,
  `Finished … in 1m 13s`, producing
  `Executable tests/rdirect_gather_v9_ordeal.rs (…-e8419af42633289a)` and
  `Executable tests/rdirect_v9_compose_parity.rs (…-caedc20f00f00ce7)`.
  Both new v9-wire test targets compile and link at the merge HEAD.

Warnings are pre-existing and non-blocking; zero errors anywhere.

## PUSH BLOCKER (charter, not content)

Recorded because an adversary that hides it is decoration: an independent
run of the gate itself REFUSES this merge —

```
$ bash tools/wilde-jagd-gate.sh a0fde0b1
WILDE JAGD REFUSED — merge a0fde0b1 lacks a real adversary ARTIFACT.
Required trailer: Adversary-Report: <path committed IN this merge>
The file must exist in the merge tree and contain "VERDICT: HOLDS" + a CONCORDANCE section.
WILDE JAGD REFUSED — merge a0fde0b1 lacks required trailers.
Required: Adversary: <agent> HOLDS
Required: Concordance: checked
GATE EXIT=1
```

`git log -1 --format=%B a0fde0b1 | git interpret-trailers --parse` returns
EMPTY — the merge commit carries no `Adversary:`, no `Concordance:
checked`, no `Adversary-Report:` trailer. The gate resolves the report via
`git show "${commit}:${report_path}"`, i.e. inside the MERGE's OWN tree, so
a report committed after the merge (this file) can never satisfy it. This
merge is not pushable as-is; making it pushable requires re-creating the
merge commit with the trailers and this report in its tree, which is
outside this review's mandate (no re-merge, no push, no history rewrite).

This does not change the verdict: the merge's CONTENT is sound. It marks
the merge as GATE-INCOMPLETE for push purposes.

## SCOPE / GAP

Honest limits of this review:
- Verified: merge integrity (blob + numstat union + file-set equality),
  v9-body dial survival, compile/link of lib, all examples, and both new
  test targets, doctrine provenance by grep, gate re-run.
- NOT verified here: the v9-wire ordeals/parity tests were compiled, not
  EXECUTED (GPU ordeals; the branch tip's own subject already records the
  probe-equals-live ordeal outcome as "FAIL, honest" on v9y-best — that is
  the lane's own recorded finding about the NET, not about this merge's
  integrity, and this report makes no claim either way about it).
- No pixel claim is made: no render was captured for this review.
