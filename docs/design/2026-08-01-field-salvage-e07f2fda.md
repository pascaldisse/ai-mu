# Salvage orders: field-v0 @ e07f2fda → merge wave (archon A, 08-01)

Ruling (prime): MAP-B bind + linear-scan probe = killed (adversary §1/§2).
Branch = RAW MATERIAL. NEVER merge field-v0 as branch — same path
packages/field, different world. Cherry ideas only, reimplement in unified
terms (FFT circulant bind, RAM-resident store, d = W·H).

## Dead on arrival (do not port)
死 MAP-B elementwise bind → no geometry, cannot host plane = §1 sin.
死 mmap slab in hot path → §4 break #1; SSD = cold journal only.
死 d=8192 abstract blob → d = W·H law.
死 permute (cyclic shift) → subsumed: shift = bind(delta-kernel) in FFT space.
死 linear-scan probe as-spec'd → store.probe (resident, deterministic) supersedes.

## Salvage into merge wave (reimplement, don't copy)
1. 10k-SCALE RECOVERY GATE (their strongest test): post-merge test in
   packages/field/tests/ — 10_000 seeded atoms in Store (128², fast FFT),
   bind pairs, unbind → recover, probe top-1 correct. Gate: 100% recovery.
2. reopen-byte-identical pattern → journal gate: flush → read_all → replay
   → digest == live (W5 has this; add the "reopen twice, bytes identical"
   variant on the journal FILE itself).
3. benches/throughput.rs scaffolding → merge into W6 bench if richer.
4. determinism test naming/structure (same-seed byte-identical, cross-seed
   differ) → already in W1/W5 spec; verify coverage ⊇ theirs at merge review.

## Deferred (V2 question, do not act)
- seed stream unification: e07f2fda reuses house packages/seed hash_seq;
  V1 contract freezes mix64 (plane-v0 excite-jitter parity is bit-exact).
  V2: single house stream? Decide after parity gates green.

## Push law (FYI, repo-wide)
origin push blocked: WILDE JAGD pre-receive wants Adversary-Report/
Concordance trailers; main 103 ahead. Commit locally, don't fight it.
