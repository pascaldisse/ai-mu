# HANDOFF — Ashwin(建・L3孫) → 次lane

枝 `fldj-oracle-r12` · worktree `$HOME/worktrees/fldj-oracle-r12` · 頭 `185a96e`

## 本lane commit(4段)
- `9154e8e` A9-4a Marut 骸救出: fieldrun 入力buffer mmap+`max_bytes` 第5引数化・静的 `_filebuf` 廃止
- `7016024` A9-4b 係数訂正(G12 閉): c_cur/c_prev 真値・coef_gate WANT 更新・赤歯を骸値へ再照準
- `90f4d19` A9-4c golden 再凍結: 全門 SHA 更新
- `185a96e` A9-4d CONTRACT §3b/§3c/§5 導出訂正・旧値骸化

## 確定係数(三者独立一致: Ashwin bit分解 / Rudra / L2 実f32)
```
sig(damping 0x3F7FBE77) = 16760439   (旧 16759927 = 誤讀 = 死)
m_dd  = 13408351   dd = .09989999979734420776
c_cur = 2040216832 = 0x799B3D00      (旧 2040220160 = 死)
c_prev= -966475008 = -0x399B3D00     (旧 -966478272 = 死)
c_lap =   10737419 = 0x00A3D70B      (不変・damping非依存)
```
丸め分岐 raw: `n/2^24=13408351 rem 6707531 < half 8388608 ∴切捨` ·
`a=255027105 /16=15939194 rem 1 <8 ∴切捨` · `b=120809377 /8=15101172 rem 1 <4 ∴切捨` ·
`s=180143990463529 /2^24=10737418 rem 9395241 > half ∴切上`。
Rudra commit `eee015fab9321038a5e838e84d25237444acfc06` = 本lane `7016024` と **coef.s byte 同値**
∴ cherry-pick 不要(空 commit になる)。取込済と看做せ。

## 新 golden(実測・再凍結)
```
A7 32x32x200 3経路   63f868bc673bbd4f7a9cf943f2b48f7f46fd416d0e55d145fa9e3cb97c1a24c9
4x4-3tick            33074917e723d60a4434ddf1badb9844faa734beb17204f0b8bdb712c16c3f7f
3c-2x2-one-tick      fa6cbcebf5db39f6f2d2cae9114d9ee60f96ad025009a6770e96d7387ede0a35
8x8-3tick-vecpath    99771ec140caede7098ddb9ff1cc36f05ee0a9cb8ea418cbd4df9bfdc7163152
9x9-3tick-vecedge    70a04aa34c5c3f64d55a3c43215c5c896f11a13a776ccae92e60a19a9b230afb
saturating-vector    ec09887b045dd627e746c0b7aaa3fc830e6368d41bd973c88e8287e22e38b744  (不変=飽和支配)
§3c cells            1950456 20972 20972 0   (旧 1950460 = 骸)
```
骸(誤定数時代): `ead5a8fff10e…` `a85a4cc0ee31…` `c34a1425…` `39343e40…` `169ff90f…`。

## 門(raw)
`teeth_kill.sh` 全走: **合計: KILLED=81 green=38 SURVIVED=0**
`gate.sh / coef_gate.sh / q20_gate.sh / fieldrun_gate.sh / neon_gate.sh` = 全 rc=0
`metal_gate.sh` = `gate: fieldrun A6 (--metal) OK`(実機GPU・command status 4)
A7 三経路 byte 一致 = scalar==neon==metal(実機)。

## 未験 / 残務
- **UNVERIFIED**: 上流 `wave_step_reference` との数値 parity は **未実測**(時間切れ)。
  係数が真値になった今も **G1 は存続**(product `plane.rs:79-90` = FFT分光 ∴ 原理上 bit 一致し得ぬ)。
  次lane の一歩目 = reference oracle との比較実測 → 一致せぬ差は契約に因つきで残す。**詐称禁**。
- A9 残務: `n_slots 1024` 引数化 · `gate.sh` の `../fieldc` 環境依存偽赤の明示 rc 化。
  (`_filebuf 4MiB` 引数化 = 完了 · 内部固定の引数化対象外理由 = fieldrun.s .bss 節に明記済)
- root/main 非接触 · /tmp 非使用 · 新規 Rust/C/Swift/Python 零。
