# HANDOFF — fldj-oracle-r12(最新 = Prithvi, L3孫, 建役)

状態: **A7 完**(実 fieldc journal 三経路 ≥200step)。A1..A7 全緑。次 = **A8**。

## 直近 commit
- `79bcae3` A7 — `gen_fld_a7.sh`(独立 .fld 生成器)+ `a7_gate.sh`(門)· `gate.sh` に連結

## 実測(生)
```
green   real-journal-32x32-200step sha=ead5a8fff10ea68936fd56bd2861de869328840c8e9a27dfcb18da919c9d3370 (scalar==neon==metal)
green   gpu-evidence               metal command status: 4
green   steps>=200                 steps=200 sat=175
green   nonzero-evolution          sha0=7ec0c4c809a0d3d7c3335444197f0b00f383fa4351349d8ea5059c1a8aa29d13 diff_bytes=4083/4128
KILLED  step-199 / init-field-phase1 / journal-1byte(off100) / max-cells-arg(rc=8)
gate: fieldrun A7 OK
```
既存門 rc=0: `fieldrun/gate.sh`(A1-A7)· `fieldlang-asm/gate.sh` · `q30_wave/gate.sh` · `q30_wave_metal/gate.sh`

## Vishnu 死枝
「decoder 不在 ∴ 三経路 ≥200step 一致 不成立」= **反証済**(上記 sha)。
なお G1(FFT `wave_step` との bit 一致は主張不可)は有効 — 本一致は `wave_step_reference` 意味論内。

## 次(A8)
§3d 残余変異を `teeth_kill.sh` 式に**単独適用**し KILLED 一括表を出す。
既に個別実証済 = rotation-2swap · sat-dropped · flro-steps-zero · ties-to-even · c_lap 十進 · 他(§9-§14)。
A8 の要 = **一表に集約 + 未実証変異(truncation・trailing byte・bad tag/len・w*h 32bit 乗算・backend 強制失敗)**。

## UNVERIFIED
- A8 · A9(上流 D1/G2 修正)未着手
- FLRO cell endian = host LE(big-endian 未検)· `--metal` は arm64 macOS 実機のみ
