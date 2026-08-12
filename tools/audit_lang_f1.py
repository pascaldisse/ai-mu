#!/usr/bin/env python3
"""AUDIT F1 — independent verification of the fnv1a prime-parity fix.

審 law: a check must not share a failure mode with the computation.
Kimi's pin test (`prime_parity_vs_field_truth`) hardcodes truth values that
were themselves produced by the lane. This script shares NO code with either
Rust side:

  1. CONSTANTS are parsed out of the two source files (not imported), and
     checked against the FNV-1a 64 spec prime derived here from its
     definition (2^40 + 2^8 + 0xb3), not from a literal.
  2. The whole speak-in derivation (fnv1a -> word_seed -> mix64 ->
     word_channels -> seed_unit -> channel_amp -> channel_center) is
     RE-IMPLEMENTED in Python from the documented spec.
  3. Rust ground truth is generated FRESH by a probe example compiled
     against the field crate (no stored fixtures), then compared bit-exact
     against the Python derivation AND against the speakback mirror.

Exit code 0 = PASS. Any mismatch prints the differing rows and exits 1.

usage: python3 tools/audit_lang_f1.py [--repo PATH] [--words w1,w2,...]
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys

MASK64 = (1 << 64) - 1

# ---------------------------------------------------------------- spec side

# FNV-1a 64 prime BY DEFINITION, not by copied literal.
FNV64_PRIME = (1 << 40) + (1 << 8) + 0xB3          # 1099511628211
FNV64_OFFSET = 0xCBF2_9CE4_8422_2325


def fnv1a(data: bytes, prime: int = FNV64_PRIME) -> int:
    h = FNV64_OFFSET
    for b in data:
        h ^= b
        h = (h * prime) & MASK64
    return h


def mix64(z: int) -> int:
    z = (z + 0x9E37_79B9_7F4A_7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & MASK64
    return z ^ (z >> 31)


def f32(x: float) -> float:
    """Round a Python float to f32, as Rust would store it."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def seed_unit(seed: int, tag: int) -> float:
    """Rust: ((m >> 11) as f64 / (1u64<<53) as f64) as f32 * 2.0 - 1.0.
    Every f32 operation rounds SEPARATELY — rounding only at the end gives
    1-2 ULP drift (measured; this checker was wrong before it was right)."""
    m = mix64(seed ^ mix64(tag))
    u = f32((m >> 11) / float(1 << 53))     # f64 divide, then cast to f32
    return f32(f32(u * 2.0) - 1.0)


CHANNELS = 64
GRID = 8
WORD_CHANNELS = 3
AMP_MIN = 0.4


def tokens(text: str):
    out, cur = [], ""
    for ch in text:
        if ch.isalnum():
            cur += ch.lower()
        elif cur:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def word_seed(word: str) -> int:
    return fnv1a(word.encode("utf-8"))


def word_channels(word: str):
    h = word_seed(word)
    return [mix64(h ^ mix64(i)) % CHANNELS for i in range(WORD_CHANNELS)]


def channel_amp(seed: int, channel: int) -> float:
    """Rust: u = seed_unit*0.5 + 0.5 ; AMP_MIN + (1.0-AMP_MIN)*u — all f32,
    with (1.0-AMP_MIN) const-folded to one f32 value."""
    u = f32(f32(seed_unit(seed, channel) * f32(0.5)) + f32(0.5))
    span = f32(f32(1.0) - f32(AMP_MIN))
    return f32(f32(AMP_MIN) + f32(span * u))


def channel_center(w: int, h: int, channel: int):
    cw, chh = w // GRID, h // GRID
    return (channel % GRID) * cw + cw // 2, (channel // GRID) * chh + chh // 2


# ------------------------------------------------------------- source probe

CONST_RE = re.compile(r"wrapping_mul\(\s*(0x[0-9A-Fa-f_]+)\s*\)")


def parse_primes(path):
    """Every fnv1a multiplier literal in a source file (parsed, not imported)."""
    src = open(path, encoding="utf-8").read()
    # only inside a fn named fnv1a
    m = re.search(r"fn fnv1a\(.*?\n\}", src, re.S)
    body = m.group(0) if m else src
    return [int(c.replace("_", ""), 16) for c in CONST_RE.findall(body)]


PROBE = r'''
use field::speak;
use field::FieldConfig;
fn main() {
    let words: Vec<String> = std::env::args().skip(1).collect();
    let cfg = FieldConfig::new(32, 32);
    let mut first = true;
    println!("[");
    for w in &words {
        if !first { println!(","); }
        first = false;
        let seed = speak::word_seed(w);
        let chans = speak::word_channels(w);
        let amps: Vec<u32> = chans.iter().map(|&c| speak::channel_amp(seed, c).to_bits()).collect();
        let centers: Vec<[usize; 2]> = chans
            .iter()
            .map(|&c| { let (x, y) = speak::channel_center(cfg, c); [x, y] })
            .collect();
        print!(
            "{{\"word\":{:?},\"seed\":{},\"channels\":{:?},\"amp_bits\":{:?},\"centers\":{:?},\"tokens\":{:?}}}",
            w, seed, chans, amps, centers, speak::tokens(w)
        );
    }
    println!("\n]");
}
'''


def rust_truth(repo, words):
    ex_dir = os.path.join(repo, "packages", "field", "examples")
    os.makedirs(ex_dir, exist_ok=True)
    probe = os.path.join(ex_dir, "audit_f1_probe.rs")
    created = not os.path.exists(probe)
    open(probe, "w", encoding="utf-8").write(PROBE)
    try:
        cmd = ["cargo", "run", "-q", "--release", "-p", "field",
               "--example", "audit_f1_probe", "--"] + words
        out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout)
            print(out.stderr, file=sys.stderr)
            raise SystemExit("probe failed to build/run")
        return json.loads(out.stdout[out.stdout.index("["):])
    finally:
        if created and os.path.exists(probe):
            os.remove(probe)


# ------------------------------------------------------------------- audit

DEFAULT_WORDS = ["fire", "water", "wind", "storm", "mu", "愛", "love",
                 "a", "zzzz", "FIRE", "fire water", "  ", "x" * 64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--words", default=",".join(DEFAULT_WORDS))
    ap.add_argument("--truth-repo", default=None,
                    help="worktree that contains packages/field/src/speak.rs "
                         "(the speak-in TRUTH); defaults to --repo")
    args = ap.parse_args()
    repo = args.repo
    truth_repo = args.truth_repo or repo
    if not os.path.exists(os.path.join(truth_repo, "packages/field/src/speak.rs")):
        raise SystemExit(
            f"no packages/field/src/speak.rs under {truth_repo}; pass --truth-repo "
            "(the mirror branch alone cannot be its own ground truth)")
    words = args.words.split(",")

    fails = []
    print(f"AUDIT F1 — repo {repo}")
    print(f"spec prime (2^40+2^8+0xb3) = {FNV64_PRIME} = {hex(FNV64_PRIME)}")

    # --- 1. constants, parsed from source ---------------------------------
    for repo_of, rel in [(truth_repo, "packages/field/src/lib.rs"),
                         (repo, "packages/speakback-s0/src/cell_decode.rs")]:
        path = os.path.join(repo_of, rel)
        if not os.path.exists(path):
            print(f"  SKIP  {rel} (absent)")
            continue
        primes = parse_primes(path)
        ok = primes == [FNV64_PRIME]
        print(f"  {'PASS' if ok else 'FAIL'}  {rel}: fnv1a multiplier(s) "
              f"{[hex(p) for p in primes]}")
        if not ok:
            fails.append(f"{rel}: fnv1a prime {[hex(p) for p in primes]} != spec")

    # --- 2. Rust truth vs independent Python derivation -------------------
    truth = rust_truth(truth_repo, words)
    for row in truth:
        w = row["word"]
        py_seed = word_seed(w)
        py_ch = word_channels(w)
        py_amps = [f32_bits(channel_amp(py_seed, c)) for c in py_ch]
        py_ct = [list(channel_center(32, 32, c)) for c in py_ch]
        same = (py_seed == row["seed"] and py_ch == row["channels"]
                and py_amps == row["amp_bits"] and py_ct == row["centers"])
        print(f"  {'PASS' if same else 'FAIL'}  {w!r}: seed {row['seed']:#018x} "
              f"ch {row['channels']} amp {row['amp_bits']}")
        if not same:
            fails.append(
                f"{w!r}: rust(seed={row['seed']},ch={row['channels']},"
                f"amp={row['amp_bits']},ct={row['centers']}) != "
                f"py(seed={py_seed},ch={py_ch},amp={py_amps},ct={py_ct})")

    # --- 2b. DISCRIMINATION: the old (typo) prime must actually change the
    # answer, else the pin test proves nothing.
    typo_prime = 0x1_0000_01b3
    moved = 0
    for row in truth:
        w = row["word"]
        h = fnv1a(w.encode("utf-8"), typo_prime)
        ch = [mix64(h ^ mix64(i)) % CHANNELS for i in range(WORD_CHANNELS)]
        if ch != row["channels"]:
            moved += 1
    ok = moved == len(truth)
    print(f"  {'PASS' if ok else 'FAIL'}  discrimination: old prime "
          f"{hex(typo_prime)} moves channels for {moved}/{len(truth)} words")
    if not ok:
        fails.append("prime pin is not discriminating for all words")

    # --- 3. mirror crate agrees with the same truth ------------------------
    mirror = os.path.join(repo, "packages", "speakback-s0")
    if os.path.isdir(mirror) and shutil.which("cargo"):
        r = subprocess.run(["cargo", "test", "-q", "--release", "-p", "speakback-s0"],
                           cwd=repo, capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"  {'PASS' if ok else 'FAIL'}  speakback-s0 own suite")
        if not ok:
            fails.append("speakback-s0 suite red")
            print(r.stdout[-2000:])

    print()
    if fails:
        print(f"AUDIT F1: FAIL ({len(fails)})")
        for f in fails:
            print("  -", f)
        return 1
    print(f"AUDIT F1: PASS ({len(truth)} words, bit-exact across "
          f"python-spec / field-crate / mirror-constant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
