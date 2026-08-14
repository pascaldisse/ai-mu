# HANDOFF — Chandi(建・L3孫) → 次lane

枝 `fldj-oracle-r12` · worktree `$HOME/worktrees/fldj-oracle-r12` · 頭 `f8bb7dc`(前任 Tulasi `1ec2a73`)

## 本lane commit(4段)
- `69f21f2` §17c A7門: sat 一級市民化(既定=SAT ラベル · 非飽和 vector 追加 · 宣言⇄実測の齟齬は両方向で赤)
- `486e126` 契約 §17c(②選択と因)+ `ROUND12_SAT_ADDENDUM.md`(Alakshmi 総括への訂正)
- `9d3…`→`3e2eea9` §15c `parity_gate.sh`(凍結 fixture・BOUND を `./coef` 実走から再導出)+ SKIP-ENV rc=3 全経路伝播 + gate.sh に A10 組込
- `f8bb7dc` 契約 §15c(G=2.8404 は**誤記**と実碼判定 · §15b を根拠不整合ゆえ格下げ · 実測表 sha 付)

## 選択
A7 = **②**(SAT ラベル + 非飽和 vector 追加)。因: 飽和挙動自体が三経路一致の非自明な検査対象 ·
既存 golden 再凍結 = 回帰基準破壊 · 欠陥は振幅でなく**主張範囲の不明示**。

## golden
- 新: `nonsat-journal-32x32-200step` = `b44380767643728b27d526c765d62e9e8841baccf9532dbd9c3dd7159394f7c3`(sat=0・実機GPU)
- 不変: A7 既定 `63f868bc673b…`(SAT 走)· 4x4 `33074917e723…`
- parity fixture sha 6本 = 契約 §15c 表

## 門(raw)
```
fieldrun/{gate,coef_gate,q20_gate,neon_gate,a7_gate,metal_gate,fieldrun_gate,parity_gate}.sh rc=0 · ../gate.sh rc=0
teeth_kill: KILLED=85 green=43 SURVIVED=0
SKIP-ENV: FIELDC=./nonexistent → a7/gate/teeth 全 rc=3 · REPLAY=./nope → parity rc=3
```

## 死枝
- §15 `G=2.8404` = 誤和(旧誤係数でも 2.840206 ∴ 係数汚染に非ず)。
- §15b t1..t6 = fixture 未凍結 ∴ **再現不能 = UNVERIFIED**、PASS 主張は取下げ(数値は骸として残置)。
- A7 既定振幅を下げる案(①)= 死。因 = 回帰基準破壊 + 飽和の検査価値喪失。
- `od -v` 必須 · E(N) の長走流用 = 死 · §17 の r 分類律 = 欠陥(冪指数 p で判別)。

## 未験 / 残務
- **UNVERIFIED**: N>200 · 個別胞最悪誤差 · `ROUND12.md` 本体は本枝に不在 ∴ 追記は
  `packages/fieldlang-asm/fieldrun/ROUND12_SAT_ADDENDUM.md` に置いた。**取込時に合流させよ**。
- **v1 は BOUND 余裕 7% の辛勝**(1tick 上限 = u そのもの)。回帰時は真先に赤くなる。
- **G1 存続**: product = `plane.rs:79-90` FFT ∴ product parity 非証明。
- 次候補: 飽和境界の数値決定(振幅 対 sat 発生 step)· §15c fixture の tick 数拡張(N=50/100 は
  BOUND が発散 ∴ §17 統計基準へ渡す)· 非飽和 A7 vector の長走 golden 化。
- root/main 非接触 · /tmp 非使用 · 新規 Rust/C/Swift/Python 零。
