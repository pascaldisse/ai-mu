# round12 総括(最終・飽和訂正済)

基=`f78aaa8`・枝=`r12/integrated`(`fldj-oracle-r12` 全成果 + Alakshmi `1e589c4` 合流)。

**改版記**: 初版(Alakshmi `1e589c4`)は §17b の飽和発見**以前**に書かれた ∴ 「三経路一致」の射程を
広く取り過ぎていた。本版が訂正版。旧 `ROUND12_SAT_ADDENDUM.md`(Chandi)は本文へ合流させ廃した
(二重管理終)。原典 = CONTRACT.md §17b/§17c/§15c。

## 主張の三段(之が総括の骨)

1. **三経路 byte 一致**(scalar/neon/metal・実機 GPU)= 飽和走・非飽和走の**双方**で成立(§17c)。
   之が意味するのは「三経路が**同一の Q20 実装意味論**(飽和込み)を持つ」事のみ。
2. **f32 `wave_step_reference` 意味論との一致** = **非飽和走のみ**・**統計的**
   (§17b(2): `RMSREL(200)=1.101e-4 ≤ 1e-3`)。bit 一致に非ず。個別胞の最悪誤差は覆わず
   ∴ 完全証明にあらず。
3. **product 経路(FFT `plane.rs:79-90`)との parity = 未主張(G1 存続)**。
   reference oracle 一致までが射程、product bit parity は**非証明**。

## 走ごとの限定(誇大禁)

- A7 既定 vector `63f868bc673bbd4f7a9cf943f2b48f7f46fd416d0e55d145fa9e3cb97c1a24c9`
  = **SAT 走(sat=175)**。主張し得るのは **byte 一致のみ**。f32 意味論一致を**主張せぬ**
  (§17b(1): 飽和下 RMSREL 0.29→0.56 = 相対 O(1) 乖離)。
- 4x4-3tick golden `33074917e723d60a4434ddf1badb9844faa734beb17204f0b8bdb712c16c3f7f`
  = 三経路 byte 一致の回帰基準。f32 意味論主張は §15c の parity 門が別途担う。
- A7 非飽和 vector `b44380767643728b27d526c765d62e9e8841baccf9532dbd9c3dd7159394f7c3`
  (sat=0・32x32・200step)= **f32 意味論 parity を §17b の統計基準内で主張し得る唯一の A7 走**。
- 実用則: **sat=0 を保つ限り** 200step で相対誤差 1.1e-4 級。真の上限は誤差蓄積でなく**飽和**
  (振幅が Q20 域 `|v|<2^11` を出た瞬間に破綻)。~1.1e4 step 外挿は **EXTRAPOLATED 未実測**。

## 達成

- A1: FLDJ v1 decoder。
- A2: f32bits→Q20(ties-away/reject)。
- A3: canonical header→Q30係数。
- A4: scalar 実行体・三buffer回転・FLRO。
- A5: neon、fallback零。
- A6: metal、実機GPU門。
- A7: 実 fieldc journal 32x32x200step、scalar=neon=metal FLRO byte一致(SAT 走 `63f868bc…`)+
  非飽和 vector `b443807676…`(§17c)。
- A8: teeth_kill一括表。最終 `KILLED=85 green=43 SURVIVED=0`。
- A8b: arena/max_bytes/max_slots 硬碼撤去。
- A9-1: emit `寫 n!=w*h`拒否。A9-2: 上流 WriteRaw assert→回復可能Err。
- A9-4: 係数訂正・golden再凍結。真 `sig=16760439`。
- A9-5/§15/§15c: 上流 reference parity を**測定前**に凍結 → 凍結 fixture 実測(PASS=5 SPECIAL=1)。
  `G` 誤記(2.8404)を実碼判定で訂正 → `G=2.840199988335`。§15b は根拠不整合ゆえ**格下げ**。
- §17/§17b: 長走 parity 基準を測定前凍結 → 実測。冪則 `RMSREL=6.60e-6·N^0.538`(緩慢蓄積)。
- §17c: A7 門で `sat` を一級市民化(`SAT|NONSAT` 事前宣言・食い違いは両方向で赤)。
- SKIP-ENV: fieldc/metallib 不在は `rc=3` に統一(環境依存の偽赤終)。

## 死枝

- Saraswati `89f2eee`: 実体無、contract-only。
- Vishnu初期 REJECT: FLDJ橋不在観測。後のA7が native-v0橋で反証。
- 偽歯: Metal `reps=2`・`offset=4,28`。因=bridge ABI不変量として成立、欠陥変異に非ず。
- `sig=16759927`共有誤読: 二重独立検算が同一誤入力を共有、同時に外れた。真sigへ訂正。
- bc 負除算: 零方向截断、算術shift検算に不適。
- merge-tree 二重誤り(建 `e7e4fdb5…`=中間段 `5c925b3` の tree を頭と誤記 · 審 `95a1874…`=統合枝を含まぬ `merge-tree f78aaa8 origin/field/lang-asm` で「不一致」判定): 両方死。根=**何と何を比べたか(base SHA と tip SHA)を記さなかった事**。真値 = `git merge-tree --write-tree field/lang-asm db8c13f35450ee26647bd0081bab039f26d7d4cd` -> `6c2b407eec07ca97b692aa4c58a6e94b02dfeb39` rc=0(三 base 全一致)。
- `../fieldc`: 環境依存偽赤。fieldc不在は `rc=3(SKIP-ENV)`。
- §17c ①案(既定振幅を下げて golden 再凍結): 棄却。因=飽和一致の検査価値喪失+回帰基準破壊。
- §17 分類律の毎tick率 `r`: 冪則を一定率で表そうとした誤り。判別は冪指数 `p` で行う(自認・骸)。
- 初版 ROUND12.md の「三経路一致= reference 意味論内の整合」: 射程過大 ∴ 本版で訂正。
- Marut lane timeout。Nirrti初番: 檀repo誤認。

## 存続制約

- **G1**: product 執行路=FFT分光(`plane.rs:79-90`)∴ product との **bit parity 非証明**。
- big-endian 未検。arm64 macOS 実機のみ。

## 未験(最終形)

- **N>200 step**: 未実測。1.1e4 step の実用上限は **EXTRAPOLATED**。
- **個別胞の最悪 f32 誤差**: 統計量(RMSREL/MEDREL/P95REL)は最悪値を覆わぬ ∴ 未験。
- **FFT(product)経路 parity**: 未主張・未験(G1)。
- **big-endian**: 未検。
- **arm64 macOS 実機のみ**: 他 arch/OS 未検。
- **v1 の BOUND 余裕 6.64%**(`MAXABS=8.903444e-07` 対 `BOUND=9.536743e-07`):
  1tick の上限は `u=2^-20` そのもの ∴ **一般 BOUND では同仮定下で之より締められぬ = 構造限界**
  (審 Mahakala 判定)。辛勝と読むべし、余裕大と読むな。
- `n_slots 1024` 引数化: 外部消費者互換 未験。

## 門(最終実走は HANDOFF.md に生出力)

`q20/coef/fieldrun/neon/metal/a7/teeth_kill/parity`=緑 · `teeth_kill KILLED=85 green=43 SURVIVED=0` ·
`cargo test -p field`=緑 · `fieldlang-asm/gate.sh`=緑 · SKIP-ENV `rc=3`。

## 取込

**主殿(root/main)への merge は archon 判断待ち**。本枝は取込可能形まで整えたのみ、merge せず。
