# HANDOFF — fldj-oracle-r12(最新 = Akasha, L3孫, 建役)

状態: **A8 完 + A8b 完**(Chandi blocker 閉)。A1..A8b 全緑。次 = **A9**。

## 直近 commit
- `3ad7090` CONTRACT §16(A8b 実装記)
- `54a85d9` A8b — arena 固定 16384 除去(n*4B ×3 を mmap)・判定は引数 max_cells のみ・rc=19(mmap 失敗)
- `2f6374d` CONTRACT §15(A8 実装記)
- `ff20570` A8 — `teeth_kill.sh`(§3d 残余変異 6 + A1..A7 全歯一括表)

## 実測(生)
```
KILLED  truncation-check-removed   baseline rc=14 -> mutant rc=12
KILLED  trailing-byte-check-removed baseline rc=14 -> mutant rc=12
KILLED  bad-tag-check-removed      baseline rc=12 -> mutant rc=0
KILLED  bad-len-check-removed      baseline rc=15 -> mutant rc=12
KILLED  wh-32bit-multiply          baseline rc=8 -> mutant rc=15
green   backend-forced-failure     rc=24 出力 file 零
KILLED  metal-init-check-removed   baseline rc=24 -> mutant rc=25
green   arena-arg-16512            rc=0 size=66080 (129x128=16512 胞・引数 16512)
KILLED  arena-arg-under            rc=8 (引数 16511)
KILLED  arena-cap-hardcoded        rc=8 output differs (硬碼再導入)
合計: KILLED=79 green=37 SURVIVED=0 ; gate: fieldrun A8 (teeth_kill 一括表) OK
```
回帰零: `4x4 a85a4cc0…` · `real-journal-32x32-200step ead5a8ff…` 不変。
全門 rc=0: `fieldrun/gate.sh` · `teeth_kill.sh` · `fieldlang-asm/gate.sh` · `q30_wave/gate.sh` · `q30_wave_metal/gate.sh`。

## SURVIVED
零。ただし §15 の自省を継げ: truncation/trailing/bad-len の三歯は rc=0 まで抜けず
**別検査(rc=12)が捕える** = 多重防御 ∴ 「唯一の防壁」ではなく「誤分類が起きる」証明に留まる。

## 死枝
- `rc=7`(w*h u64 溢れ)= 到達不能(w,h は u32)。代替 = `wh-32bit-multiply`。
- `mul` 検査単独除去の歯 = 観測不能 ∴ 立てず。
- big-endian FLRO cell 歯 = host LE のみ ∴ 実測不能。

## UNVERIFIED
- **A9 未着手**: 上流 D1(`../CONTRACT.md` に `寫 n==w*h` 明記 + `emit.s` が range err で拒否 +
  `world.rs:78` の assert を回復可能誤りへ)+ G2 の十進注釈訂正。
- `_filebuf = 4 MiB` 固定(入力 file 上限)= 未引数化・未検(§16 に明記)。
- G1 存続: product 執行路 = FFT `wave_step` ∴ **product parity は非証明**。本一致は
  `wave_step_reference` 意味論内部のみ。
- `--metal` は arm64 macOS 実機 GPU のみ。
