# FIELD V1 contract — frozen skeleton (archon A/nyari, 2026-08-01)

Binding: adversary verdict docs/research/2026-08-01-field-adversary.md.
THE SPEC = module docs in packages/field/src/*.rs (frozen signatures).
This file = ownership + gates only. Base branch: field-unified.

## Ownership (no overlaps; lib.rs/ops.rs/store.rs/Cargo.toml FROZEN, archon's)
- W1 core:    fft.rs internals (radix-2) · atoms.rs · tests/core.rs
- W2 plane:   plane.rs internals · tests/plane_parity.rs (ref: plane-v0 0587c8d3)
- W3 journal: journal.rs internals · tests/store_journal.rs
- W4 spike:   tools/field-spike-mlx/* · docs/research/2026-08-01-field-metal-spike.md
- W5 gates:   tests/determinism.rs
- W6 world:   world.rs internals · src/bin/bench.rs
- W7 docs:    FIELD.md amendments · HANDOFF.md log entry
- W8 capacity: tests/capacity.rs · docs/research/2026-08-01-field-capacity.md

## Laws
- worktrees only, branch field/<name> from field-unified. main untouched.
- signatures frozen: replace todo!() bodies + add owned files ONLY.
- ultradeterminism: no rand crate, no clock in state, f32 persisted as bits.
- no mmap/flush/alloc-heavy in tick hot path. SSD = cold journal only.
- commit every stage (timeouts eat rooms, not commits).

## Gates
- cargo build -p field && cargo build -p field --tests  → green, always.
- cargo test -p field --test <own file>                 → green where deps
  allow; runtime blocked by others' todo!() = report UNVERIFIED-until-merge.
- bun: n/a (no TS touched) — state in report.

## Merge
Wave 2: octopus merge field/* → field-unified → full cargo test -p field +
bench + full-size parity. Then report up.
