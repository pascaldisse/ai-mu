# Q30 二次元五点波

ABI:

`_fl_q30_wave_scalar(x0=cfg, x1=cur, x2=prev, x3=out) -> x0=saturations`

`_fl_q30_wave_neon` 同ABI。`cfg` = `i32 width, i32 height, i64 c_cur, i64 c_lap, i64 c_prev`。
AArch64自然配置: offsets `0,4,8,16,24`。width,height は正、`width*height` は addressable `i32` 胞数。

各 `(x,y)`:

`lap = cur[y][x-1] + cur[y][x+1] + cur[y-1][x] + cur[y+1][x] - 4*cur[y][x]`。

各添字=周期 wrap。`lap`・三項積・acc= signed 64-bit。各係数積= `(coefficient * value + 2^29) >> 30`。acc は一回のみ signed i32 飽和、飽和胞数を返す。

alias: `cur` と `prev` は相互 alias 可。`out` は `cur`/`prev` と一切重複禁止（partial overlapも禁止）。read-before-write stencil 故。

範囲: `|lap| <= 2^34-4`; `|c_lap| <= 2^29`（境界含）。lap積を含む各積と和=i64 安全域。`c_cur=2^31`（dd=0）を含むため係数=i64。

実装: `wave_scalar.s`=胞毎scalar、`wave_neon.s`=内部4胞vector(2d二半)+端胞scalar経路。
両者ともx18不触・x19-x28不触。`wave_neon.s`は外部scalar実装へ委譲せぬ(b/bl 零)。
係数i64は `c = (c>>1) + (c-(c>>1))` の二半 smull で積む。lap i64は `lap = (lap>>4)*16 + (lap&15)` へ分割。
門: `./gate.sh` — 凍結133 vectors digest厳密·変異5赤·disasm NEON証明·非整列(+1B)·cur/prev alias live。
