# round12 総括

基=`f78aaa8`・枝=`fa1c221`。三経路一致=**`wave_step_reference` 意味論内の整合**のみ。product `plane.rs:79-90`=FFT分光との bit parity は主張せず。

## 達成

- A1: FLDJ v1 decoder。
- A2: f32bits→Q20(ties-away/reject)。
- A3: canonical header→Q30係数。
- A4: scalar 実行体・三buffer回転・FLRO。
- A5: neon、fallback零。
- A6: metal、実機GPU門。
- A7: 実 fieldc journal 32x32x200step、scalar=neon=metal FLRO byte一致。SHA=`63f868bc673bbd4f7a9cf943f2b48f7f46fd416d0e55d145fa9e3cb97c1a24c9`。
- A8: teeth_kill一括表。`KILLED=84 green=40 SURVIVED=0`。
- A8b: arena/max_bytes/max_slots 硬碼撤去。
- A9-1: emit `寫 n!=w*h`拒否。A9-2: 上流 WriteRaw assert→回復可能Err。
- A9-4: 係数訂正・golden再凍結。真 `sig=16760439`。

## 死枝

- Saraswati `89f2eee`: 実体無、contract-only。
- Vishnu初期 REJECT: FLDJ橋不在観測。後のA7が native-v0橋で反証。
- 偽歯: Metal `reps=2`・`offset=4,28`。因=bridge ABI不変量として成立、欠陥変異に非ず。
- `sig=16759927`共有誤読: 二重独立検算が同一誤入力を共有、同時に外れた。真sigへ訂正。
- bc 負除算: 零方向截断、算術shift検算に不適。
- `../fieldc`: 環境依存偽赤。fieldc不在は `rc=3(SKIP-ENV)`。
- Marut lane timeout。
- Nirrti初番: 檀repo誤認。

## 存続制約

- G1: product執行路=FFT分光(`plane.rs:79-90`)。reference oracle一致までが射程、product bit parity非証明。
- big-endian未検。
- arm64 macOS実機のみ。

## 未験

- 上流 `wave_step_reference` 数値 parity: Ganga実測中、本締めでは未独立再験。
- `n_slots 1024`引数化: 外部消費者互換未験。

## 門

`q20/coef/fieldrun/neon/metal/a7/teeth_kill`=緑。`cargo test -p field --test world`=13緑。`fieldc gate`=緑。
