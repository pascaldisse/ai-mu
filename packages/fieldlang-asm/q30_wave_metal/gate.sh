#!/bin/sh
set -eu
cd "$(dirname "$0")"
V=../q30_wave/wave_vectors.bin
D=b826a11494d9e988ad90cc2db93aceceb77229ae741e028a2a785339751b493e
echo "$D  $V" | shasum -a 256 -c -
# Forbidden source scan: no Rust/C/Swift/Python source on the product path.
if find . -maxdepth 1 -type f \( -name '*.rs' -o -name '*.c' -o -name '*.swift' -o -name '*.py' \) | grep -q .; then exit 1; fi
if grep -nEi 'rust|cargo|rustc|wgpu|swiftc|python' ./*.s build.sh; then exit 1; fi
trap 'rm -f *.o *.air wave_metal_runner q30_wave.metallib gate-base.log gate-edge.log wv.edge; for f in wave_q30.metal metal_bridge.s; do test -f "$f.save" && mv "$f.save" "$f"; done' EXIT HUP INT TERM
./build.sh
# live GPU corpus: 138 vectors, scalar=NEON=Metal, offsets 0/4/28, alias, reps,
# per-dispatch saturation reset, count0/negative/overflow host checks.
./wave_metal_runner q30_wave.metallib "$V" > gate-base.log
tail -1 gate-base.log
grep -q 'metal device: ' gate-base.log
grep -q 'wave_metal_runner: 138 Q30WAVE2 scalar=neon=metal ok' gate-base.log
S4=$(grep -c 'metal command status: 4' gate-base.log)
EN=$(grep -c 'metal command error: nil' gate-base.log)
test "$S4" -eq 690
test "$EN" -eq 690
printf 'gpu evidence: status4=%s errornil=%s (138 cases x 5 dispatches)\n' "$S4" "$EN"
# runner independent-decode teeth: corrupted fixtures must be rejected.
cp "$V" wv.magic; printf 'X' | dd of=wv.magic bs=1 seek=0 conv=notrunc >/dev/null 2>&1
if ./wave_metal_runner q30_wave.metallib wv.magic >/dev/null 2>&1; then rm -f wv.magic; exit 1; fi
rm -f wv.magic; printf 'mutation fixture-magic rejected=ok\n'
cp "$V" wv.trail; printf 'ZZZZ' >> wv.trail
if ./wave_metal_runner q30_wave.metallib wv.trail >/dev/null 2>&1; then rm -f wv.trail; exit 1; fi
rm -f wv.trail; printf 'mutation fixture-trailing rejected=ok\n'
SZ=$(wc -c < "$V")
dd if="$V" of=wv.trunc bs=1 count=$((SZ-17)) >/dev/null 2>&1
if ./wave_metal_runner q30_wave.metallib wv.trunc >/dev/null 2>&1; then rm -f wv.trunc; exit 1; fi
rm -f wv.trunc; printf 'mutation fixture-truncated rejected=ok\n'
# beyond-fixture edge corpus (1x1 / w=1 / h=1 / saturation / 64x64 / 65x33 tail, c_cur=±2^31):
# deterministic shell generator, live GPU, scalar=NEON=Metal exact.
../q30_wave/gen_wave_vectors.sh wv.edge edge
./wave_metal_runner q30_wave.metallib wv.edge edge > gate-edge.log
grep -q 'wave_metal_runner: 6 edge Q30WAVE2 scalar=neon=metal ok' gate-edge.log
test "$(grep -c 'metal command status: 4' gate-edge.log)" -eq 30
printf 'edge6 gpu run=ok (30 dispatches status4)\n'
printf '\377' | dd of=wv.edge bs=1 seek=9 conv=notrunc >/dev/null 2>&1
if ./wave_metal_runner q30_wave.metallib wv.edge edge >/dev/null 2>&1; then rm -f wv.edge; exit 1; fi
rm -f wv.edge; printf 'mutation edge-corruption rejected=ok\n'
# mut <name> <file> <perl-expr>: apply, rebuild, the live GPU run must go RED.
mut() { n=$1; f=$2; e=$3; cp "$f" "$f.save"; perl -0pe "$e" "$f.save" > "$f"; cmp -s "$f" "$f.save" && { echo "mutation $n: no-op" >&2; exit 1; }; if ./build.sh >/dev/null 2>&1 && ./wave_metal_runner q30_wave.metallib "$V" >/dev/null 2>&1; then echo "mutation $n SURVIVED" >&2; mv "$f.save" "$f"; ./build.sh >/dev/null; exit 1; fi; mv "$f.save" "$f"; printf 'mutation %s rejected=ok\n' "$n"; }
# --- shader teeth (12), each a distinct law clause, each RED on the real GPU ---
mut rounding-half        wave_q30.metal 'if(!$d&&s/bias\.lo = 1u << 29/bias.lo = 1u << 28/){$d=1}'
mut q30-scale            wave_q30.metal 'if(!$d&&s/\(s\.lo >> 30\)/(s.lo >> 29)/){$d=1}'
mut sat-window           wave_q30.metal 'if(!$d&&s/\(acc\.lo >> 31\) == 0u/(acc.lo >> 31) == 1u/){$d=1}'
mut sat-sign             wave_q30.metal 'if(!$d&&s/res = \(acc\.hi >= 0\)/res = (acc.hi < 0)/){$d=1}'
mut lap-centre-4x        wave_q30.metal 'if(!$d&&s/c4 = add64\(c4, c4\);\n    c4 = add64\(c4, c4\);/c4 = add64(c4, c4);/){$d=1}'
mut periodic-x           wave_q30.metal 'if(!$d&&s/\(x == 0u\) \? \(w - 1u\)/(x == 0u) ? 0u/){$d=1}'
mut periodic-y           wave_q30.metal 'if(!$d&&s/\(y == 0u\) \? \(h - 1u\)/(y == 0u) ? 0u/){$d=1}'
mut buffer-slot          wave_q30.metal 'if(!$d&&s/cur     \[\[buffer\(0\)\]\]/cur     [[buffer(1)]]/ && s/prev    \[\[buffer\(1\)\]\]/prev    [[buffer(0)]]/){$d=1}'
mut mul64-hi-term        wave_q30.metal 'if(!$d&&s/hi \+= a\.lo \* uint\(b\.hi\);/hi += 0u * uint(b.hi);/){$d=1}'
mut sat-count-double     wave_q30.metal 'if(!$d&&s/atomic_fetch_add_explicit\(sat_count, 1u/atomic_fetch_add_explicit(sat_count, 2u/){$d=1}'
mut params-lo-hi         wave_q30.metal 'if(!$d&&s/k_cur\.hi  = p\.c_cur_hi;/k_cur.hi  = 0;/){$d=1}'
mut round-independence   wave_q30.metal 'if(!$d&&s/add64\(acc, q30_round\(mul64\(k_lap, lap\)\)\)/add64(acc, q30_round(add64(mul64(k_lap, lap), from_i32(1))))/){$d=1}'
mut count-agreement      wave_q30.metal 'if(!$d&&s/if \(n != p\.count\) \{ return; \}/if (n == p.count) { return; }/){$d=1}'
# --- host bridge teeth (9), hand-assembly encoder/ABI clauses, each RED live ---
mut host-out-buffer-index metal_bridge.s 'if(!$d&&s/mov x2, x21\n    mov x3, x23\n    mov x4, #2/mov x2, x21\n    mov x3, x23\n    mov x4, #5/){$d=1}'
mut host-binding-swap     metal_bridge.s 'if(!$d&&s/mov x2, x19\n    mov x3, x23\n    mov x4, #0/mov x2, x20\n    mov x3, x23\n    mov x4, #0/){$d=1}'
mut host-params-length    metal_bridge.s 'if(!$d&&s/mov x3, #40\n    mov x4, #4/mov x3, #8\n    mov x4, #4/){$d=1}'
mut host-count-mismatch   metal_bridge.s 'if(!$d&&s/str w21, \[sp, #128\]            \/\/ host-side u64-checked count into Params/add w16, w21, #1\n    str w16, [sp, #128]/){$d=1}'
mut host-mtlsize-abi      metal_bridge.s 'if(!$d&&s/str x24, \[sp, #96\]/sub x16, x24, #1\n    str x16, [sp, #96]/){$d=1}'
mut host-count0           metal_bridge.s 'if(!$d&&s/Lwm_zero:\n    mov x0, #0/Lwm_zero:\n    mov x0, #1/){$d=1}'
mut host-sat-reset        metal_bridge.s 'if(!$d&&s/str wzr, \[x9\]                  \/\/ sat reset per dispatch/nop/){$d=1}'
mut host-status4          metal_bridge.s 'if(!$d&&s/cmp x0, #4\n    b\.ne Lwd_fail/cmp x0, #3\n    b.ne Lwd_fail/){$d=1}'
mut host-nserror          metal_bridge.s 'if(!$d&&s/bl _objc_msgSend\n    cbnz x0, Lwd_fail\n    adrp x0, Lwm_status4\@PAGE/bl _objc_msgSend\n    cbz x0, Lwd_fail\n    adrp x0, Lwm_status4\@PAGE/){$d=1}'
mut host-encoder-offset   metal_bridge.s 'if(!$d&&s/mov x2, x19\n    mov x3, x23\n    mov x4, #0/mov x2, x19\n    mov x3, #0\n    mov x4, #0/){$d=1}'
./build.sh >/dev/null
./wave_metal_runner q30_wave.metallib "$V" > gate-base.log
grep -q 'wave_metal_runner: 138 Q30WAVE2 scalar=neon=metal ok' gate-base.log
printf '%s\n' 'q30_wave_metal gate: frozen138 GPU scalar=neon=metal offsets alias reps custody guard, shader-teeth=13 host-teeth=10, all red=ok'
