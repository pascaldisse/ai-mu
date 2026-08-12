//! field-bench — 120fps floor gate. OWNED BY W6.
//! argv (all optional, defaults in parens — IRON: params not hardcodes):
//!   --width (128) --height (128) --slots (4096) --probe-keys (1)
//!   --frames (1000) --render-every (12)
//! Loop per frame: world.tick() + store.probe(key) + render every Nth.
//! Report: ms/frame mean/p50/p99, steps/s, PASS/FAIL vs 8.33ms (120fps).
//! Timing via Instant = harness only; SIM state stays f(seed).
fn main() {
    todo!("W6: bench per module doc")
}
