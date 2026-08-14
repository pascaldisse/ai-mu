# Q30 二次元五点波 · Metal direct path (MSL) · host contract

Product path = `wave_q30.metal` + `metal_bridge.s`(手書ARM64 host bridge) + `wave_metal_runner.s`(assembly-only runner) + `build.sh`/`gate.sh`。禁外来語source は product closure に無し(門が走査)。

## Shader interface (external host contract)

| slot | binding | meaning |
|---|---|---|
| kernel | `q30_wave` | entry name |
| `buffer(0)` | `device const int *cur` | `width*height` little-endian i32 |
| `buffer(1)` | `device const int *prev` | 同形。`cur` と同一基址 alias 可 |
| `buffer(2)` | `device int *out` | 同形。`cur`/`prev` と重複禁 |
| `buffer(3)` | `device atomic_uint *sat_count` | 単一。host が事前に零化 |
| `buffer(4)` | `constant Params&` | 40 bytes、下記 |

`Params` = `uint width, height; uint c_cur_lo; int c_cur_hi; uint c_lap_lo; int c_lap_hi;
uint c_prev_lo; int c_prev_hi; uint count; uint pad;` — 係数 i64 は lo/hi 対で渡す
(shader は native 64-bit 整数を用いぬ)。`count` = **host 側 u64 checked** `width*height`
(`metal_bridge.s`: 負寸=拒否(-1)、`umulh` 相当 `lsr #32` 検査で u32 上限超=拒否(-1)、
`width==0||height==0` = 無読無書で 0)。

Kernel 側(実装済): `n = w*h` を再算し `n != p.count` なら**全thread 即return**(dimensions
product 一致検査)、然る後 `gid >= p.count` を守る。両検査の非空虚は門の
`count-agreement`(shader逆極性=赤)と `host-count-mismatch`(host が count+1 を書く=赤)が採点。

Dispatch: `dispatchThreads(count,1,1)` tpg `(64,1,1)`、MTLSize は **間接 aggregate ABI**
(x2/x3=pointer)で渡す。Encoder offset 非零可(4B 倍数): bridge は buffers 0/1/2 に byte offset
を適用、runner は `0/4/28` を全走。storage = shared(options 0)。`sat_count` は **dispatch 毎に**
host が零化(反復 dispatch は resource を再生成せず再走)。command status `4` 検査 + `error` nil
検査、証跡を fd1 へ raw 出力。NSError は library load 失敗時に `localizedDescription` を出力。

Host entry(`metal_bridge.s`):
`_fl_wave_metal_init(x0=metallib path)->0/1` ·
`_fl_q30_wave_metal(x0=cfg{i32 w,i32 h,i64 c_cur,c_lap,c_prev; offsets 0,4,8,16,24}, x1=cur,
x2=prev, x3=out, x4=byte_offset, x5=reps)->saturations / -1(GPU未完、CPU fallback 無し)`。

## Lane law（受理済 scalar/NEON と同一）

`lap = cur[y][x-1]+cur[y][x+1]+cur[y-1][x]+cur[y+1][x] - 4*cur[y][x]`、全添字=周期。
`w==1` / `h==1` は自身へ wrap。`lap` は true i64(lo/hi 対)で積む。
`q(c,v) = (c*v + 2^29) arith>>30` を **係数毎に独立** に取り、
`acc = q(c_cur,cur) + q(c_lap,lap) + q(c_prev,prev)`。
飽和は **和の後に一度だけ** i32 へ。`sat_count` = 範囲外胞数。

入力不変: `cur`/`prev` は 1 byte も改変せぬ(`device const`)。書込は `out[0..w*h)` のみ。

## 要求門（bridge/runner atom で実装）


- 凍結 `../q30_wave/wave_vectors.bin` digest 厳密 → **138** vectors(Q30WAVE2 wire, i64係数 `c_cur=±2^31`・`c_lap=±2^29`・lap極値を含む)を audit runner が**独立 decode**
- decode は各record + 全file の remaining-length を厳密検証(truncated/trailing/malformed=赤)、`width*height` は **u64 checked**(zero/negative/arena上限=拒否)
- 各 vector: fixture の want/sat(=受理済 scalar/NEON 産) と Int64 参照が一致することを先に検し、
  然る後 GPU 出力を byte/sat 厳密比較 → scalar=NEON=Metal
- encoder offset `0/4/28` bytes・`cur==prev` alias 呼出・out 前後 0xA5 guard・入力 custody 全byte比較
- fixture 外 edge 6 case(1x1・w=1・h=1・飽和・64x64・65x33 tail、係数 ±2^31)—
  `gen_wave_vectors.sh <out> edge`(shell-only)が生成、runner edge mode(6 record 厳密)が GPU 実走
- command status `4` / error `nil` を明示出力
門実装: `./gate.sh` — 凍結138 digest → runner 独立厳密decode(残長/trailing/u64 dims) →
実機 GPU 138×5 dispatch(offsets 0/4/28・alias cur==prev・reps=2 sat-reset・count0/負寸/u32超)
scalar=NEON=Metal byte+sat 厳密 → fixture 3 teeth + shader 13 teeth + host 10 teeth 全赤。

- 変異 teeth(各々 actual GPU RED、`gate.sh` の mut() が逆弱化して赤を実測):
  shader 13: `rounding-half` `q30-scale` `sat-window` `sat-sign` `lap-centre-4x` `periodic-x`
  `periodic-y` `buffer-slot` `mul64-hi-term` `sat-count-double` `params-lo-hi`
  `round-independence` `count-agreement`
  host 10: `host-out-buffer-index` `host-binding-swap` `host-params-length` `host-mtlsize-abi`
  `host-count0` `host-sat-reset` `host-status4` `host-nserror` `host-encoder-offset`
  `host-count-mismatch`

custody は runner が cur/prev 独立複写を各実装呼出後に全byte比較(bridge は入力を
MTLBuffer へ複写する為 host 入力は不変、比較が之を採点)。out 周囲 0xA5 guard 32B×2。
