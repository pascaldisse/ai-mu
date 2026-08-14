# ROUND12 総括への訂正(Chandi・飽和限定)

宛 = `ROUND12.md`(Alakshmi `1e589c4`・枝 `alakshmi-r12`)。当該総括は §17b の飽和発見**以前**に書かれた。
本枝 `fldj-oracle-r12` に `ROUND12.md` は存在せぬ ∴ 取込時に本追記を `ROUND12.md` へ合流させよ。

## 訂正対象
> 三経路一致=**`wave_step_reference` 意味論内の整合**のみ。product=FFT との bit parity は主張せず。

範囲が**なお広過ぎる**。三経路 byte 一致(A7 既定 vector `63f868bc…`)の走は **sat=175 の飽和走**。
飽和下では fieldrun は f32 reference と相対 O(1) で乖離(§17b(1): RMSREL 0.29→0.56)∴
当該一致は「三経路が**同一の Q20 実装意味論**(飽和込み)を持つ」事の証明に過ぎず、
**`wave_step_reference` 意味論内の整合ではない**。

## 正しい主張(三段)
1. 三経路 byte 一致(scalar/neon/metal・実機 GPU)= 飽和走・非飽和走の**双方**で成立(§17c)。
2. f32 `wave_step_reference` 意味論との一致 = **非飽和走のみ**・**統計的**(RMSREL(200)=1.101e-4、
   bit 一致に非ず・個別胞最悪値は覆わず)。
3. product(FFT `plane.rs:79-90`)経路との parity = **未主張(G1 存続)**。

## 実用則
sat=0 を保つ限り 200step で相対誤差 1.1e-4 級。真の上限は誤差蓄積でなく**飽和**(振幅が Q20 域を出た瞬間に破綻)。
外挿 ~1.1e4 step は **EXTRAPOLATED 未実測**。
