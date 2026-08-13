# ADVERSARY REPORT — v9-body merge · 2026-07-28

VERDICT: FAILS

Adversary: @opus5 (independent lane, no authoring stake in v9-body).
Subject: merge `f3ecf70c53e4458ab4636d0cde9b33e29ae474bc`
"Merge branch 'v9-body': integrator NEE, mirror-threshold, reference-tool work".
Real parents (`git log -1 f3ecf70c^1 / ^2`):
- `^1` = `ed53085afc0b110233fd6f682aa4513d559f879c` (main, 07-27 18:37 — "HANDOFF: 07-27 eve — Pascal's eye reopens: no emissive NEE (light was never real) …")
- `^2` = `de8ac474118113ceb479a0eae28347d32c068c32` (v9-body tip — "integrator: GAIA_NEE_CLAMP firefly clamp on the emissive-NEE shadow-ray term")
- merge-base = `f012514f9af086085b5c2da12a1c231fed50aa81`
Reviewed as EMBEDDED history (current main HEAD = `a0fde0b1`, the later v9-wire merge). Nothing re-merged, nothing reverted by this report.

Split verdict, so the FAILS is not read wider than it is:
- MERGE MECHANICS — HOLDS. Clean union, no lost hunks, compiles (§BUILD).
- SHIPPED FEATURE CORRECTNESS — FAILS. Finding 4: the emissive-NEE light-sampling term is π× the integral its own MIS partner estimates. Latent (dial default off), but it is exactly the dial `^1`'s own HANDOFF demands be turned on.
- CHARTER COMPLIANCE — FAILS. Findings 6/7: neither merge carries the adversary trailers/artifact; the cite tooth refuses the range.

## CONCORDANCE

### 1. Union is clean — proven three ways, not asserted
- `git diff --stat f3ecf70c^1 f3ecf70c` → 292 files, **48232 insertions**, 14 deletions.
- `git diff --stat f3ecf70c^1 f3ecf70c^2` → 294 files, **48232 insertions**, 602 deletions.
- `git diff --stat f3ecf70c^2 f3ecf70c` → exactly 2 files, +588/-0: `HANDOFF.md` (+540), `docs/research/2026-07-19-neural-motion-todo.md` (+48).
- Deletion arithmetic closes exactly: 602 = 14 + 540 + 48. The branch's 588 "deletions" vs main were main-side ADDITIONS the branch never saw; the merge restored them. Insertion counts are IDENTICAL in both diffs → no branch hunk was dropped.
- Blob-identity sweep (stronger than --stat): for every file in `git diff --name-only <merge-base> f3ecf70c^2`, compared `f3ecf70c^2:<f>` vs `f3ecf70c:<f>` object shas → **zero divergences**. The merge tree took the branch blob for every branch-touched file.
- Reverse check that main's side lost nothing: `git diff <merge-base> f3ecf70c^2 -- HANDOFF.md docs/research/2026-07-19-neural-motion-todo.md` is EMPTY → the branch never edited either file, so keeping main's version discards no branch work.
- Only 3 files carry deletions at all (`git diff --numstat f3ecf70c^1 f3ecf70c`): `Cargo.toml` (10/1), `src/integrator.rs` (441/9), `src/integrator.wgsl` (340/4) — all genuine modifications, everything else pure addition.
- Cargo.toml delta reviewed: `half = "2"` moved dev-deps → deps (lib/bin link the fp16 tensor path, not only examples), + objc2-mps features `MPSGraphConvolutionOps`/`MPSGraphResizeOps`/`MPSGraphTensorShapeOps`; `src/lib.rs` +1 line `pub mod rdirect_unet;`. No removals of existing deps/features. [source: packages/scrying-glass/Cargo.toml @ f3ecf70c]
- VERDICT (this section): HOLDS.

### 2. NEE + mirror threshold + reference-scale features ARE present at f3ecf70c
Read out of the worktree `/tmp/adv-v9body` at f3ecf70c, not from a branch tip:
- Host side `src/integrator.rs`: `build_emissive_lights` (power-weighted `power_i = luminance(emission_i)*area_i`, inclusive CDF, tag written into the tris' `v0.w` lane), the `@group(2)` `lights` buffer built at BOTH the initial BVH build (:571) and the per-frame re-derive on BVH merge (:928), plus the three dials — `GAIA_EMISSIVE_NEE` (:392, default **false**), `GAIA_MIRROR_ROUGHNESS_MAX` (:347, default **1e-3** = the old hardcoded `MIRROR_ROUGHNESS`), `GAIA_NEE_CLAMP` (:412, default **0.0**, bitcast into `temporal_flags.w`).
- Shader `src/integrator.wgsl`: `struct Light`, `power_heuristic`, `sample_triangle` (Shirley–Chiu warp), `sample_light_index` (CDF binary search), `clamp_nee_radiance`, and the NEE-carrying twins `radiance_nee` (:634) / `radiance_ed` (:833) with the mirror branch reading `u.misc.w` (:762, :942). `radiance` proper is left untouched, so `integrate_temporal` keeps its group(1) budget — as the commit message claims.
- Reference tool `examples/rdirect_reference.rs:124` — `GAIA_REF_EMISSION_SCALE` → `params.emission_intensity *= emission_scale`, default 1.0.
- IRON-default claim SPOT-CHECKED and true: all three dials default to the pre-merge behaviour, so main's default renders are unaffected by this merge.
- `packages/scrying-glass/src/rdirect_evidence.rs` (named in the review order) is **untouched** by this merge — `git diff f3ecf70c^1 f3ecf70c -- …/rdirect_evidence.rs` is empty, and it is likewise unchanged f3ecf70c→a0fde0b1. Reported rather than invented.
- VERDICT (this section): HOLDS — the advertised features are really there.

### 3. MIS double-count guard is sound
`radiance_nee`'s hit-emission block weights the emitter hit by `power_heuristic(prev_bsdf_pdf_sa, pdf_light_sa)` only when `nee_on && bounce > 0 && prev_diffuse`, with `prev_diffuse = nee_on` set solely on the cosine-diffuse branch and `prev_bsdf_pdf_sa = (1-metallic)*max(dot(n,dir),0)/π`; the light tag is read from `v0.w` and the self-hit case is excluded by `light.tri != hit.tri`. Solid-angle conversion `pdf_light_sa = light.pdf/light.area * dist2/cos_light` is the standard area→SA Jacobian, and the light-selection pdf is folded in. With the dial off, every branch is bypassed → byte-identical old behaviour. Structure of the MIS bookkeeping: correct.
- VERDICT (this section): HOLDS (structure) — but see Finding 4 for the radiance the structure carries.

### 4. ⚠ REAL DEFECT — the NEE light-sampling term is π× too bright (Lambert 1/π missing)
The two techniques `radiance_nee`/`radiance_ed` MIS-combine do NOT estimate the same integral.
- Technique A, light sampling (`integrator.wgsl:739-740`, identically `:923-924`):
  `throughput * (1-metallic) * tri.albedo * ltri.emission * ndl2 / pdf_light_sa * mis` — no `1/π`.
- Technique B, BSDF sampling (its declared MIS partner): cosine-hemisphere bounce with `throughput *= tri.albedo` (`:781-783`) then `L += throughput * tri.emission` at the emitter hit. That estimator is exactly correct for f_r = **albedo/π** (E = ∫ (cos/π)·albedo·L_e dω = ∫ (albedo/π)·L_e·cos dω).
- The same block that omits the BRDF's `1/π` from the radiance computes `pdf_bsdf_sa = (1-metallic)*ndl2*(1.0/PI)` one line above (`:737`) — the file contains the missing factor as a pdf but not as a BRDF.
- The in-file disclosure (`:622-626`) claims this is safe: "f_r = (1-metallic)*albedo, no 1/π, matching the file's existing non-standard-but-internally-consistent Lambertian scale so nothing double-counts". That claim is FALSE for this pair. `f_r = albedo` would require the diffuse bounce to multiply throughput by `π*albedo`; the file multiplies by `albedo`. The sun analogy the comment leans on does not transfer: the sun is a delta light with a free intensity dial (`sun_color.w` absorbs any π), whereas `tri.emission` is the SAME radiance the hit-emission technique adds — no freedom left.
- INDEPENDENT VERIFICATION (different path from code reading): both estimators transcribed verbatim into a standalone Monte-Carlo harness (`/tmp/nee-pi-check.py`, one emissive triangle over one diffuse point, 4M samples each, MIS weight forced to 1 to measure the raw techniques), with truth computed by a THIRD route (uniform-solid-angle hemisphere sweep of albedo/π·L_e·cos):
  ```
  A (light sampling, as coded) = 0.428608
  B (BSDF sampling, as coded)  = 0.137310
  T (truth, Lambert albedo/pi) = 0.136302
  A/B = 3.121466      A/T = 3.144547      B/T = 1.007394   (pi = 3.141593)
  ```
  A is π× truth; B is truth (residual ≈ MC noise of the sweep). MIS does not rescale radiance, so the combined estimator converges to `∫ [w_light·π + w_bsdf]·f_r·L_e·cos dω` — biased HIGH by up to ~3.14× wherever light sampling wins the weight (small/bright/distant emitters, i.e. precisely the case NEE was added for).
- SCOPE, honestly bounded: `GAIA_EMISSIVE_NEE` defaults **off** and NOTHING in the repo sets it (`grep -rn GAIA_EMISSIVE_NEE` → only the three doc/impl lines in `integrator.rs`). So no shipped render path and no v9 training label is currently π-biased — the defect is LATENT. It is still on the critical path: parent `^1`'s own HANDOFF subject is "no emissive NEE (light was never real)". Present at f3ecf70c and UNCHANGED at HEAD (`git show a0fde0b1:…/integrator.wgsl` → same term at :740 and :924; `git diff f3ecf70c a0fde0b1 -- integrator.wgsl integrator.rs` is empty).
- FIX is one factor in two places: multiply the light term by `(1.0/PI)` (or promote the whole file to a normalized f_r and re-dial the sun) — then re-run stage B against a NEE-off converged render as the absolute reference.
- VERDICT (this section): FAILS.

### 5. The branch's own verification could not have caught Finding 4
`80a468d0` ("verify: … reference renders pass (stage B)") is a well-written relative check, and that is the hole: (a) light-throw evidence is "table/bench/floor near the orbs brighten consistently (mean delta +0.4..+1.0/255)" — a π overestimate satisfies "brighter" perfectly; (b) the MIS check proved hit-emission is byte-unchanged (ratio ∈ [0.9999996, 1.0000038]) — that validates technique B is untouched, never that A agrees with it; (c) no ABSOLUTE reference was used, though a free one exists — NEE-off at converged spp integrates the same light and would have shown the ~π gap on NEE-lit patches. Its own honest-gaps list (one pose only, no automated regression, no clamp at the time) is accurate but does not name the normalization. Its evidence is also uncommitted by construction ("Evidence lives in scratch/ (untracked, not part of this commit)") — `scratch/nee-before/`, `scratch/nee-after/`, `scratch/mirror-*` are not in the merge tree, so this reviewer could not re-read the pixels the verdict rests on; only the numbers quoted in the message.
- VERDICT (this section): FAILS (protocol gap — relative-only verification of an absolute quantity).

### 6. Charter/gate re-run — both merges refuse
Independent re-run, unmodified tool:
```
$ bash tools/wilde-jagd-gate.sh f3ecf70c
WILDE JAGD REFUSED — merge f3ecf70c lacks a real adversary ARTIFACT.
WILDE JAGD REFUSED — merge f3ecf70c lacks required trailers.          exit=1
$ bash tools/wilde-jagd-gate.sh a0fde0b1                              exit=1  (same two refusals)
```
`git log -1 --format=%B f3ecf70c | git interpret-trailers --parse` → EMPTY (no `Adversary:`, no `Concordance: checked`, no `Adversary-Report:`). Neither merge predates the baseline (`GAIA_JAGD_BASELINE` default `bf2779a`), so neither is exempt. Structural consequence, stated plainly: the tooth demands the artifact live in **the merge commit's own tree** — a report committed afterwards on main (this file) can NEVER make `f3ecf70c` or `a0fde0b1` pass `wilde-jagd-gate.sh`. Retroactive compliance is impossible by design; only a future merge can be compliant (report committed on the branch BEFORE the merge, path named in the merge trailers).
Blob tooth: `--blob-range f3ecf70c^1..f3ecf70c` → `WILDE JAGD BLOB TOOTH HOLDS` (scrubbed blob absent). Clean.
- VERDICT (this section): FAILS (trailers/artifact), blob tooth HOLDS.

### 7. Cite tooth refuses the merge range; ~77 MB of blobs entered history
- `bash tools/wilde-jagd-gate.sh --cite-range f3ecf70c^1..f3ecf70c` → REFUSED, 4 uncited newly-added Markdown host/silicon claims, all in `docs/perf/2026-07-21-v9-spike.md` (e.g. "## Configs measured (M1 Pro, macOS 26.5.1, uncontended …", two `MPSGraph`/`MPSGraphExecutable` lines, one more MPSGraph line). Fix is mechanical: `[source: …]` or `UNVERIFIED` on those lines.
- Hygiene, flagged not vetoed: the merge adds **137 binary blobs ≈ 77.1 MB** (weight `.bin` ladders + `scratch/*.png`) plus ~30k lines of `scratch/*.log` and `packages/scrying-glass/scratch/v9-autopsy.md` into permanent history. `.gitignore` ignores `.scratch/` but not `packages/*/scratch/`, so this is policy-consistent today — worth a ruling before the next ladder lands.
- VERDICT (this section): FAILS (cite tooth), hygiene = flag.

## DOCTRINE-CONCORDANCE

Training-lane files touched (`(^|/)examples/[^/]*train[^/]*\.rs$` per the gate's own `GAIA_JAGD_TRAIN_PATTERN`): `rdirect_train_v9.rs`, `_v9b`, `_v9c`, `_v9d`, `_v9e`, `_v9f`, `_v9g`, `_v9h`, `_v9i`, `_v9j`, `_v9k` — 11 new trainers. Tier names per NEURAL.md §TRAINING DOCTRINE (1 STRUCTURE, 2 EQUATIONS AS SIGNAL, 3 DATA = COMPUTE).

Cited from the files' OWN provenance, not from this reviewer's reading:
- `rdirect_train_v9{,b,c,d,e}.rs` each emit a `"doctrine_concordance"` provenance field (e.g. `rdirect_train_v9.rs:691`): *"v8d's TIER1(estimator-init: NOT ported, no analytic estimator-init exists for a conv net yet — FRESH He-init instead, disclosed)+TIER2(K-averaged noise2noise, K={k_draws}, teacher=validator-only)+TIER3(structure: moving-camera history w/ REAL reprojection + curved-mirror/low-roughness pose + EMA history-source, kept as structural mechanisms, NOT the training signal)"*.
- Module headers say the same (`rdirect_train_v9.rs:41-51`): *"K-AVERAGED NOISE2NOISE TARGETS (v8d TIER 2, K=8 default, teacher DEMOTED TO VALIDATOR ONLY) … unbiased (E[mean of K iid draws]=truth), variance/K … `accumulate_backward` NEVER reads it — copied verbatim from v8d's TIER 2 doctrine (NEURAL.md §TRAINING DOCTRINE, 07-20 enforcement)"*.
- `rdirect_train_v9{f,g,h,i,j,k}.rs` carry **no** `doctrine_concordance` key (`grep -c` → 0 in all six; disclosed here rather than papered over). Their own loss provenance still cites the tier, e.g. `rdirect_train_v9k.rs:3265`: *"whole-image MSE(net(step features), mean of K={k_draws} independent draw radiances) — teacher NEVER in the loss, validator only (v8d TIER2 doctrine, ported); CURE 4 … CURE 5 …"*, and `rdirect_train_v9k.rs:1-30` describes its own method as changing the TRAINING SAMPLE UNIT (128x72 crops of true 640x480 renders) — a sampling-unit change, not a signal change.
- Nomenclature drift worth one ruling: the trainers' parenthetical "TIER3(structure: …)" uses tier-1 vocabulary for engine-generated pose/history mechanisms. Mapped to NEURAL.md's own numbering, those mechanisms are self-generated data (datasets = f(seed)) = DATA = COMPUTE; nothing in any of the 11 files claims an unrolled-solver/analytic-init architecture, and TIER 1 is explicitly declared NOT ported.

DOCTRINE-CONCORDANCE: EQUATIONS-AS-SIGNAL (tier 2) is the implemented training signal in all 11 touched trainers — K-averaged noise2noise targets, teacher demoted to validator-only, teacher never in the loss — supported by DATA=COMPUTE (tier 3) for the engine-generated pose/draw pools (and v9k's 640x480 crop sampling unit); STRUCTURE (tier 1) is NOT implemented and is disclosed as such in-file ("no analytic estimator-init exists for a conv net yet — FRESH He-init instead"); cited from `rdirect_train_v9{,b,c,d,e}.rs`'s own `doctrine_concordance` provenance field and, for `_v9f.._v9k` (which lack that key), from their own header/loss provenance comments.

## BUILD

Independent re-run, detached worktree at the merge commit, isolated target dir, nothing borrowed from a warm branch build:
```
$ git worktree add /tmp/adv-v9body f3ecf70c
HEAD is now at f3ecf70c Merge branch 'v9-body': integrator NEE, mirror-threshold, reference-tool work
$ cd /tmp/adv-v9body && nice -n19 CARGO_TARGET_DIR=/tmp/adv-v9body-target \
    cargo build --release -p scrying-glass -j2
…
warning: unused import: `DEFAULT_DEGRADE_RATIO`
warning: fields `temporal_enabled`, `temporal_params`, `t_bind`, `t_parity`, `t_prev`, and `last_view` are never read
   --> packages/scrying-glass/src/main.rs:2324:5
warning: method `view_key` is never used
   --> packages/scrying-glass/src/main.rs:3019:8
warning: `scrying-glass` (bin "scrying-glass") generated 3 warnings
    Finished `release` profile [optimized] target(s) in 6m 12s
```
- 0 errors (`grep -c '^error'` → 0). 3 warnings, all in `src/main.rs` — a file this merge does not touch (absent from `git diff --name-only f3ecf70c^1 f3ecf70c`), i.e. pre-existing, not introduced by v9-body. The new lib module `src/rdirect_unet.rs` (1714 lines, MPSGraph conv path) compiles warning-free with `half` promoted out of dev-deps. [source: this report §BUILD, cargo build --release -p scrying-glass -j2 run at f3ecf70c in /tmp/adv-v9body]
- `cargo build -p scrying-glass` does NOT compile `examples/`, so the training lane is not covered by the line above. Closed that gap explicitly:
```
$ cargo build --release -p scrying-glass --example rdirect_reference --example rdirect_train_v9k -j2
    Finished `release` profile [optimized] target(s) in 48.97s
```
  → the reference tool and the largest trainer (3319 lines) both build at f3ecf70c. The other 9 trainers + 10 forensics/triptych examples were NOT built (wall discipline) — UNVERIFIED, stated rather than implied.
- Worktree removed after the run (`git worktree remove /tmp/adv-v9body --force`); `/tmp/adv-v9body-target` deleted.
- What was NOT done, so nobody reads more into this than it says: no GPU render was executed (no pixel evidence produced by this review), no test suite run, and Finding 4 was proven by transcription + Monte-Carlo algebra, not by a NEE-on render of the naruko world. A render-side confirmation (NEE-on vs NEE-off, converged spp, ratio on an NEE-lit patch) is the natural next step and would put a number on the in-scene magnitude.

## REMEDY — what would flip this verdict

1. Multiply the NEE light-sampling contribution by `1/π` at `integrator.wgsl:739` and `:923` (or normalize the whole file's f_r and re-dial `sun_color.w`), then re-verify stage B against a NEE-off CONVERGED render as an absolute reference, not a before/after brightness delta.
2. Delete or correct the "internally-consistent, no 1/π" disclosure at `:622-626` — as written it is an authoritative-sounding false statement that would deter the next reader from checking.
3. Add `[source: …]`/`UNVERIFIED` to the 4 flagged lines in `docs/perf/2026-07-21-v9-spike.md` (cite tooth).
4. Charter, going forward: commit the adversary report ON the branch before merging and name it in the merge trailers (`Adversary-Report:`, `Adversary: <agent> HOLDS`, `Concordance: checked`) — the artifact must be in the merge's own tree, which is why `f3ecf70c` and `a0fde0b1` cannot be repaired retroactively.
5. Optional ruling: `packages/*/scratch/` in `.gitignore` before the next weight ladder adds another ~77 MB to history.

This report deliberately does NOT contain the gate's required verdict literal (`VERDICT:` followed by HOLDS) — Finding 4 is a live correctness defect in shipped code, and a rubber stamp would have been the worse failure. Merge mechanics, feature presence, and buildability all hold; the physics and the paperwork do not.
