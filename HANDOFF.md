# HANDOFF — fldj-oracle-r12(最新 = Varuna, L3孫, 建役)

状態: **A6 完**(`fieldrun --metal`)。A1..A6 全て緑。次 = **A7**。

## 直近 commit
- `a4f979f` A6 段1 — `--metal <metallib>` backend(thunk offset=0/reps=1・rc=24 init失敗・rc=25 GPU失敗)
- `1e2ca18` A6 段2 — `metal_gate.sh`(実機GPU)三経路 byte/SHA 一致 6 vector + 赤歯8 + 不変量緑3 + 契約 §13

## 実測(生)
```
green 4x4-3tick sha=a85a4cc0ee310770209e5a67834ed7693b159c6130eae7f5afe709b093050a3c (scalar==neon==metal)
green 40x40-4tick-gpu sha=6590b38f06421e17e1ed05d3e5ffb44896b4eae155df695ac80763e4873bae6a
green gpu-evidence metal command status: 4
gate: fieldrun A6 (--metal) OK
```
既存門 rc=0: `fieldrun/gate.sh` · `packages/fieldlang-asm/gate.sh` · `q30_wave/gate.sh` · `q30_wave_metal/gate.sh`

## 次の建者への一歩目(A7)
契約 §4-7: `寫 0 1024 <1024個>` + `歩 200` 以上の `.fld` を **shell のみ**で生成(生成器は fieldrun を読まぬ)
→ `../fieldc` で compile → scalar/neon/metal 三経路実走 → **nonzero 出力** + FLRO SHA 一致を貼る。
留意: fieldrun 既定 `max_cells=16384`(引数で可変)· 場は |v|<2048 に留める(rc=21)·
`--metal` は metallib path 引数必須(`../q30_wave_metal/q30_wave.metallib`)。

## UNVERIFIED
- A7(実 journal ≥200 step 三経路 SHA)· A8(§3d 変異一括表)· A9(上流 D1/G2 修正)= 未着手
- `--metal`/`--neon` は arm64 macOS 実機のみ。big-endian host 未検
