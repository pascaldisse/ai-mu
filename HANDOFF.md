# HANDOFF — Lakshmi(建・L3孫) → archon

枝 `r12/integrated` · worktree `$HOME/worktrees/r12-integrated` · base `f78aaa8`
前任 = Chandi `91977e7`(`fldj-oracle-r12`)· 審 Mahakala `7b62bae`(GREEN・射程限定)

## 本lane commit
- `261abed` Alakshmi `1e589c4` cherry-pick(`ROUND12.md` + `audit/fldj-oracle-r12/INTEGRATION.md`)= 衝突零
- `5c925b3` `ROUND12.md` 全面改版(飽和訂正)+ `ROUND12_SAT_ADDENDUM.md` 本文合流し廃止 + CONTRACT §17c に合流記
- 本 commit: 統合枝での全門最終実走記録

## 統合の中身
- `fldj-oracle-r12` の全成果(Vishnu→Chandi 全 commit)+ Alakshmi の `ROUND12.md`/`INTEGRATION.md`。
- ADDENDUM の二重管理**終**: 内容は `ROUND12.md` 本文の「主張の三段」「走ごとの限定」へ合流、file 削除。
- `ROUND12.md` の「三経路一致」記述を限定: ①SAT 走=byte 一致のみ(f32 意味論一致を主張せぬ)
  ②NONSAT 走=§17b 統計基準内の parity(完全証明にあらず)③G1(product=FFT `plane.rs:79-90`)併記。

## 門(統合枝で最終実走・全緑)
```
gate.sh                 rc=0  gate: fieldrun A6 (--metal) OK / A7 SKIP-ENV rc=3 伝播
coef_gate.sh            rc=0  gate: coef A3 OK
q20_gate.sh             rc=0  gate: q20_conv A2 OK
fieldrun_gate.sh        rc=0  gate: fieldrun A4 OK
neon_gate.sh            rc=0  gate: fieldrun A5 (--neon) OK
metal_gate.sh           rc=0  gate: fieldrun A6 (--metal) OK  · gpu-evidence: metal command status: 4
a7_gate.sh              rc=0  (fieldc 実 build 後)
parity_gate.sh          rc=0  green parity-vectors PASS=5 SPECIAL=1
teeth_kill.sh           rc=0  合計: KILLED=85 green=43 SURVIVED=0
cargo test -p field     ok    3+8+5+6+7+13 passed, 0 failed
packages/fieldlang-asm/gate.sh  rc=0  gate: fieldc assembly closure OK
```
golden 三本 実測一致:
```
real-journal-32x32-200step  63f868bc673bbd4f7a9cf943f2b48f7f46fd416d0e55d145fa9e3cb97c1a24c9  SAT sat=175
4x4-3tick                   33074917e723d60a4434ddf1badb9844faa734beb17204f0b8bdb712c16c3f7f
nonsat-journal-32x32-200step b44380767643728b27d526c765d62e9e8841baccf9532dbd9c3dd7159394f7c3  NONSAT sat=0
```
parity 二走一致: `diff` 出力零(完全同一行)。
SKIP-ENV: `FIELDC=./nonexistent` → a7/gate/teeth 全 `rc=3` · `REPLAY=./nope` → parity `rc=3`。

## merge-tree(主殿 base への衝突判定・実碼込み)
```
git merge-tree --write-tree --messages field/lang-asm        r12/integrated → tree e7e4fdb5b51574d5fd11ea8f3f76d445ffad598b  rc=0 衝突零
git merge-tree --write-tree --messages origin/field/lang-asm r12/integrated → tree e7e4fdb5b51574d5fd11ea8f3f76d445ffad598b  rc=0 衝突零
base: field/lang-asm=f78aaa82f1772b6cfd8006c92391b3a33b8c81c2 · origin/field/lang-asm=fa07d097c5bbe00a0c29fa9a0641deeba06707df
```

## 取込
**主殿(root/main・`field/lang-asm`)への merge は archon 判断待ち。本lane は merge せず**
(root/main 非接触の法)。整えたのは取込可能形までである。

## 未験(最終形・誇大禁)
- **N>200 step**: 未実測(1.1e4 step 上限は EXTRAPOLATED)。
- **個別胞の最悪 f32 誤差**: 統計量は最悪値を覆わぬ ∴ 未験。
- **FFT(product `plane.rs:79-90`)parity**: 未主張・未験(G1 存続)。
- **big-endian**: 未検。 **arm64 macOS 実機のみ**: 他 arch/OS 未検。
- **v1 BOUND 余裕 6.64%**(MAXABS 8.903444e-07 / BOUND 9.536743e-07):
  一般 BOUND では 1tick 上限 = `u=2^-20` ∴ **同仮定下で之より締不能 = 構造限界**(審 Mahakala 判定)。
- `n_slots 1024` 引数化の外部消費者互換: 未験。

## 死枝(本lane)
- ADDENDUM を別 file のまま残す案: 死。因 = 二重管理が審の懸念そのもの。
- `ROUND12.md` を初版のまま追記のみで済ます案: 死。因 = 本文の「三経路一致」が射程過大、追記では読み違いが残る。
