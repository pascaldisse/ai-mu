# HANDOFF — Ganga(建・L3孫) → 次lane

枝 `fldj-oracle-r12` · worktree `$HOME/worktrees/fldj-oracle-r12` · 頭 `fa1c221`(前任 Ashwin `9044da3`)

## 本lane commit(3段)
- `0dbd0c1` §15 parity 許容誤差を **測定前に凍結**(E(N)=u·ΣG^i, u=2^-20, G=2.8404 · 飽和走は対象外)
- `4296522` §15b parity **実測 PASS** + `replay_fldj.rs` に `FLDJ_REF=1` 分岐(既存file最小改・新規file零)
- `fa1c221` A9-5: `max_slots` 第6引数化(硬碼 1024 除去) · `../gate.sh` fieldc 不在 → 明示 rc=3 · §16

## parity 実測(生)
```
t1 cells=4    1tick  MAXABS=4.563481e-07 BOUND=9.537000e-07 MAXREL=2.453352e-07 PASS
t2 cells=16   3tick  MAXABS=9.769963e-14 BOUND=1.406000e-05 MAXREL=2.840499e-14 PASS(一様場=lap0 特異)
t4 cells=64   3tick  MAXABS=1.907348e-06 BOUND=1.406000e-05 MAXREL=2.960336e-07 PASS
t5 cells=256  6tick  MAXABS=5.722046e-06 BOUND=2.000000e-04 MAXREL=6.712411e-07 PASS
t6 cells=1024 10tick MAXABS=1.049042e-05 BOUND=3.360000e-03 MAXREL=1.209630e-06 PASS
```
全走 sat=0。傾向 = 誤差は G^N より遥かに遅く増(量子化雑音蓄積)。
再現: `FLDJ_REF=1 target/debug/examples/replay_fldj x.fldj` 対 `od -An -v -td4 -j32 x.flro`(**`-v` 必須**、`*` 圧縮=偽差)。

## 門(raw)
```
teeth_kill.sh   合計: KILLED=84 green=40 SURVIVED=0
gate.sh coef_gate.sh q20_gate.sh neon_gate.sh   全 rc=0
metal_gate.sh   gate: fieldrun A6 (--metal) OK(実機GPU)
../gate.sh rc=0 · FIELDC=./nope ../gate.sh rc=3(SKIP-ENV)
golden 不変: A7 63f868bc673b… / 4x4 33074917e723…
```

## 死枝
- 期待 rc=11(slots-hi)= 誤 → 真 rc=10(`fieldrun.s:393`)。歯を 10 へ訂正。
- `od` の `*` 行圧縮で偽の巨大差(3.44)を一度観測 = 検が算と別の失敗様式で入った例 → `-v` で解消。

## 未験 / 残務
- **UNVERIFIED**: 長走(A7 200tick)parity = 未測(E(200) 発散 ∴ 主張せぬ)。
- **G1 存続**: product 執行路 `plane.rs:79-90` = FFT 分光 ∴ bit 一致は原理上主張し得ぬ。§15b は
  `wave_step_reference`(5点 stencil)意味論への一致のみを示す。
- 次の一歩候補: G1 に対し FFT 経路 対 reference の f32 同士 parity を上流側で測る(fieldrun 圏外)。
- root/main 非接触 · /tmp 非使用 · 新規 Rust/C/Swift/Python 零。
