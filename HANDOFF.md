# HANDOFF — Tulasi(建・L3孫) → 次lane

枝 `fldj-oracle-r12` · worktree `$HOME/worktrees/fldj-oracle-r12` · 頭 `cb9f828`(前任 Ganga `c7fd0d2`)

## 本lane commit(2段)
- `3687b21` §17 長走 parity 基準を**測定前に凍結**(RMSREL 主 / MEDREL·P95REL 副 · 閾 1e-3 · 分類律)
- `cb9f828` §17b 実測 + `longrun_parity.sh`(shell) + `gen_fld_a7.sh` に EBASE/ESPAN 引数化(既定不変 sha `b83c386a…`)

## 結論
- 非飽和 200step: `RMSREL=1.101e-4 ≤ 1e-3` **PASS** · 冪則 `6.60e-6·N^0.538` = **緩慢蓄積(ランダムウォーク)**、指数発散に非ず。
- **既定 A7 振幅は 1step で飽和**(sat=47→175)∴ 相対 O(1) 乖離。三経路 byte 一致は之を検知せぬ。
- 実用則: sat=0 を保つ限り 200step で 1.1e-4、外挿で ~1.1e4 step まで 1e-3 未満(**EXTRAPOLATED 未実測**)。

## 門(raw)
```
gate.sh/coef_gate.sh/q20_gate.sh/neon_gate.sh/a7_gate.sh 全 rc=0 · ../gate.sh rc=0
teeth_kill: KILLED=84 green=40 SURVIVED=0
metal_gate.sh: gate: fieldrun A6 (--metal) OK
golden 不変: A7 63f868bc673b… / 4x4 33074917e723…
```

## 死枝
- §15 の最悪上界 E(N) を長走に流用 = 死(G=2.84 発散 ∴ 判定不能)→ 統計量へ。
- §17 の毎tick率 `r` による分類律 = **欠陥自認**(r=1.0085 が両閾の隙間)。冪指数 p で判別せよ。§17b に骸記載。
- `od` の `*` 圧縮 = 偽差 ∴ `od -v` 必須(Ganga 継承)。

## 未験 / 残務
- **UNVERIFIED**: N>200 は未実測(外挿のみ)· 個別胞最悪誤差は統計量で覆わぬ。
- **G1 存続**: product = `plane.rs:79-90` FFT ∴ product parity 非証明。
- 次候補: 飽和境界の数値決定(振幅 対 sat 発生 step)· FFT 経路 対 reference の f32 同士 parity。
- root/main 非接触 · /tmp 非使用 · 新規 Rust/C/Swift/Python 零。
