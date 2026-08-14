#!/usr/bin/env python3
"""Mutation teeth: each Q30 MSL corruption must make the independent GPU gate RED."""
from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
runner = root / "q30_metal_runner"
source = root / "q30_metal_runner.swift"
fixture = root / "q30_metal_vectors.bin"
text = source.read_text()
start, end = text.index('let defaultSource = #"""') + len('let defaultSource = #"""'), text.index('"""#\n\nstruct Reader')
msl = text[start:end]
teeth = {
    "shift": (">> 30", ">> 29"),
    "rounding-plus-2^29": ("+ (1L << 29)", "+ 0L"),
    "saturate": ("cells[tid] = -2147483648", "cells[tid] = 0"),
    "thread-index": ("if (tid >= count) return;", "if (tid >= count - 1) return;"),
}
if not runner.exists() or not fixture.exists():
    raise SystemExit("RED: build runner and fixture first")
for name, (old, new) in teeth.items():
    if msl.count(old) != 1:
        raise SystemExit(f"RED: tooth anchor ambiguous: {name}")
    mutant = root / f"q30_mutant_{name}.metal"
    mutant.write_text(msl.replace(old, new))
    run = subprocess.run([str(runner), "--fixture", str(fixture), "--source", str(mutant)], text=True, capture_output=True)
    mutant.unlink()
    if run.returncode == 0:
        sys.stdout.write(run.stdout)
        raise SystemExit(f"RED: mutation survived: {name}")
    print(f"mutation={name} result=RED-required-observed")
print("q30-metal-mutations=green teeth=4")
