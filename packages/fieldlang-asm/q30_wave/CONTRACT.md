# Q30 二次元五点波

ABI:

`_fl_q30_wave_scalar(x0=cfg, x1=cur, x2=prev, x3=out) -> x0=saturations`

`_fl_q30_wave_neon` 同ABI。`cfg` = `i32 width, i32 height, i64 c_cur, i64 c_lap, i64 c_prev`。
AArch64自然配置: offsets `0,4,8,16,24`。width,height は正、`width*height` は addressable `i32` 胞数。

各 `(x,y)`:

`lap = cur[y][x-1] + cur[y][x+1] + cur[y-1][x] + cur[y+1][x] - 4*cur[y][x]`。

各添字=周期 wrap。`lap`・三項積・acc= signed 64-bit。各係数積= `(coefficient * value + 2^29) >> 30`。acc は一回のみ signed i32 飽和、飽和胞数を返す。

alias 方針（許す input alias を明記）:

- `cur` と `prev` は相互 alias 可。**同一基址** `cur == prev` を含む。
- `out` は `cur`/`prev` と一切重複禁止（partial overlap も禁止）。read-before-write stencil 故。
- `out` alias は **不許可**。故に入力 custody は alias 有無に依らず常に厳格比較。

入力不変性: 呼出は `cur`・`prev` の1バイトも改変せぬ。書込は `out[0 .. width*height)` のみ。
この範囲外（前後いずれも）への書込を禁ず。`out` alias を許さぬ故、期待変化する入力 case は存在せぬ。

非整列: `cfg` 以外の `cur`/`prev`/`out` は 4byte 自然整列を要求せぬ。門は +1B 非整列で全経路を再走する。

範囲: `|lap| <= 2^34-4`; `|c_lap| <= 2^29`（境界含）。lap積を含む各積と和=i64 安全域。`c_cur=2^31`（dd=0）を含むため係数=i64。

実装: `wave_scalar.s`=胞毎scalar、`wave_neon.s`=内部4胞vector(2d二半)+端胞scalar経路。
両者ともx18不触・x19-x28不触。`wave_neon.s`は外部scalar実装へ委譲せぬ(b/bl 零)。
係数i64は `c = (c>>1) + (c-(c>>1))` の二半 smull で積む。lap i64は `lap = (lap>>4)*16 + (lap&15)` へ分割。
門: `./gate.sh` — 凍結133 vectors digest厳密·変異12赤·disasm NEON証明·非整列(+1B)·`cur==prev` alias live 呼出·
入力 custody 独立複写比較(aligned/+1B/alias 各 scalar・NEON 呼出毎)·`out` 前後 0xA5 guard·
live callee-saved sentinel(x19-x28/LR/SP, 8x8 で NEON vector 経路を含む)。

変異teeth(各々 red 必須):
`rounding-half` `q-scale` `lane-order` `sat-negative` `vec-boundary` `scalar-fallback`(x19) `x18-tooth`
`unaligned-tooth`(cur を16B丸め — 整列入力は恒等、+1B経路のみ赤)
`alias-tooth`(`cur==prev` 時のみ prev 狂わす)
`out-guard-tooth`(one-past-end 書込 — 出力本体は正しく guard のみ検出)
`input-custody-tooth` / `input-custody-scalar-tooth`(`Ldone` 到達後に `cur` 破壊 — 出力・飽和数は正しく custody のみ検出)。
