# round12 取込判定 — Alakshmi 審

基= `f78aaa82f1772b6cfd8006c92391b3a33b8c81c2` · 枝頭= `fa1c22188d1c2753b228b12a67d4cf464985f761` · 2026-08-14。

## 判

**可** — 主殿現頭=基、故 fast-forward 可。`git merge-tree --write-tree 基 枝頭` = `5d47e48ffb548137b029cddcad2418610c7e93d5`、衝突零。`git diff --check 基..枝頭`=零。既存門は下記全緑、退行未見。

契約/実装: A1--A9-5 は `fieldrun/CONTRACT.md` の native-v0(FLDJ→Q20→Q30 stencil→FLRO)に束縛し、A9-4 真係数へ更新済。三経路一致は **`wave_step_reference` 意味論内の整合のみ**。product `plane.rs:79-90`=FFT分光との bit parity は非証明(G1存続)。

追跡済 `fieldrun/coef`=Mach-O arm64 は build産物だが、現枝は `.gitignore` に `coef` を置く。既往追跡物ゆえ gateは通る。取込阻止因に非ず、後続浄化候補。

## 全commit 査

| SHA | 査 |
|---|---|
| bcdfa6d | 審査記録・D1発見。後続A9で閉。|
| e745b67 | native-v0契約・実装atom束縛。|
| 86240f1 | A1 parser。|
| 1024c03 | A1産物追跡外。|
| 4c2cb4f | A2 Q20・ties-away歯。|
| c2d6d33 | A3係数。旧値は後続訂正。|
| 5834a8b | A4 scalar/FLRO。|
| 954c479 | A5 neon/no-fallback。|
| a4f979f | A6 metal入口。|
| 1e2ca18 | A6実機門。|
| f2b92e0 | HANDOFFのみ、実装契約非変。|
| 79bcae3 | A7実fieldc journal・三経路門。|
| 732b91b | HANDOFFのみ、実装契約非変。|
| ff20570 | A8 teeth一括表。|
| 2f6374d | A8証跡契約。|
| 54a85d9 | A8b arena硬碼撤去。|
| 3ad7090 | A8b契約記録。|
| 160fb48 | HANDOFFのみ、実装契約非変。|
| 74b0136 | A9-1 emit `寫 n==w*h`拒否。|
| a608304 | A9-2 WriteRaw panic→Err・test追加。|
| 15c2e47 | G2誤読撤回・G12露出。|
| 9154e8e | A9-4a input arena引数化。|
| 7016024 | 真係数へ訂正。|
| 90f4d19 | golden再凍結。|
| 185a96e | sig誤読を骸化・契約訂正。|
| 9044da3 | HANDOFFのみ、実装契約非変。|
| 31f9729 | gate残渣浄化。|
| 0dbd0c1 | reference parity許容式。G1非閉。|
| 4296522 | reference短走報告。G1非閉。|
| fa1c221 | max_slots引数化・fieldc不在=SKIP-ENV。|

各commitの①既存門退行=下記実測で否、②契約一致=上記射程内で可、③骸/未験=CONTRACTとROUND12へ明記。死枝は削除せず因を保存。

## gate 生出力(隔離檀)

```text
./q20_gate.sh       -> gate: q20_conv A2 OK
./coef_gate.sh      -> gate: coef A3 OK
./fieldrun_gate.sh  -> gate: fieldrun A4 OK
./neon_gate.sh      -> gate: fieldrun A5 (--neon) OK
./metal_gate.sh     -> gate: fieldrun A6 (--metal) OK
./a7_gate.sh        -> gate: fieldrun A7 (real fieldc journal, 32x32, 200 step, 三経路) OK
./teeth_kill.sh     -> gate: fieldrun A8 (teeth_kill 一括表) OK; KILLED=84 green=40 SURVIVED=0
cargo test -p field --test world -> 13 passed; 0 failed
packages/fieldlang-asm/gate.sh -> gate: fieldc assembly closure OK
```

A7 SHA=`63f868bc673bbd4f7a9cf943f2b48f7f46fd416d0e55d145fa9e3cb97c1a24c9`。Metal=arm64 macOS実機。gate後 `git status --short`=空。

## 残制約/未験

- G1: product=FFT分光、reference oracleまで。product bit parity非証明。
- big-endian未検・arm64 macOS実機のみ。
- 上流 `wave_step_reference` 数値 parity: Ganga実測中として本審は未独立再験。
- `n_slots 1024`引数化: 外部消費者互換は未験。
