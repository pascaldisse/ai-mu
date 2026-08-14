# Q30 二次元五点波 · Metal direct path (MSL) · host contract

Product path = `wave_q30.metal` + hand-written ARM64 host bridge + assembly runner。
本commit は MSL/外部ABI evidence のみ。bridge/runner/gate は未収録、故に受理不能。
Rust/C/Swift/Python source は product closure に禁。

## Shader interface (external host contract)

| slot | binding | meaning |
|---|---|---|
| kernel | `q30_wave` | entry name |
| `buffer(0)` | `device const int *cur` | `width*height` little-endian i32 |
| `buffer(1)` | `device const int *prev` | 同形。`cur` と同一基址 alias 可 |
| `buffer(2)` | `device int *out` | 同形。`cur`/`prev` と重複禁 |
| `buffer(3)` | `device atomic_uint *sat_count` | 単一。host が事前に零化 |
| `buffer(4)` | `constant Params&` | 32 bytes、下記 |

`Params` = `uint width, height; uint c_cur_lo; int c_cur_hi; uint c_lap_lo; int c_lap_hi;
uint c_prev_lo; int c_prev_hi;` — 係数 i64 は lo/hi 対で渡す(shader は native 64-bit 整数を用いぬ)。

Dispatch: `dispatchThreads(width*height,1,1)` tpg `(64,1,1)`。kernel は `n = w*h` を **u32** で算し `gid >= n` を守るのみ
(現 `wave_q30.metal` 87-89 行 = 實態)。`Params` に `count` field は **無い**。

> **UNVERIFIED / 未実装(round11 自攻で摘出)**: 本 dir には kernel と本書のみ有り、host bridge も runner も gate も **存在せぬ**。
> 故 `width*height` の u64 checked は **host 側の未履行要求**であり、實装された事実ではない。
> kernel の `n = w*h` は u32 wrap する ∴ host が u64 checked(u32 上限超=拒否)を果たさぬ限り
> `gid` guard は空になり得る。この一行は **要求**であり記述ではない。
Encoder offset 非零 可(4B 倍数、Metal 規約)。門は byte offset `0/4/28` を全走。

## Lane law（受理済 scalar/NEON と同一）

`lap = cur[y][x-1]+cur[y][x+1]+cur[y-1][x]+cur[y+1][x] - 4*cur[y][x]`、全添字=周期。
`w==1` / `h==1` は自身へ wrap。`lap` は true i64(lo/hi 対)で積む。
`q(c,v) = (c*v + 2^29) arith>>30` を **係数毎に独立** に取り、
`acc = q(c_cur,cur) + q(c_lap,lap) + q(c_prev,prev)`。
飽和は **和の後に一度だけ** i32 へ。`sat_count` = 範囲外胞数。

入力不変: `cur`/`prev` は 1 byte も改変せぬ(`device const`)。書込は `out[0..w*h)` のみ。

## 要求門（bridge/runner atom で実装）

(以下すべて **未実装 = UNVERIFIED**。實行された門は q30_wave(CPU)側のみ。)

- 凍結 `../q30_wave/wave_vectors.bin` digest 厳密 → **138** vectors(Q30WAVE2 wire, i64係数 `c_cur=±2^31`・`c_lap=±2^29`・lap極値を含む)を audit runner が**独立 decode**
- decode は各record + 全file の remaining-length を厳密検証(truncated/trailing/malformed=赤)、`width*height` は **u64 checked**(zero/negative/arena上限=拒否)
- 各 vector: fixture の want/sat(=受理済 scalar/NEON 産) と Int64 参照が一致することを先に検し、
  然る後 GPU 出力を byte/sat 厳密比較 → scalar=NEON=Metal
- encoder offset `0/4/28` bytes・`cur==prev` alias 呼出・out 前後 0xA5 guard・入力 custody 全byte比較
- fixture 外 edge 6 case(1x1・w=1・h=1・飽和・64x64・65x33 tail、係数 ±2^31)
- command status `4` / error `nil` を明示出力
- 変異 teeth 12、各々 actual GPU RED 必須:
  `rounding-half` `q30-scale` `sat-window` `sat-sign` `lap-centre-4x` `periodic-x` `periodic-y`
  `buffer-slot` `mul64-hi-term` `round-independence` `out-one-past-end`
  `out-guard-only`(出力本体は正しく、one-past-end への一書きのみ — guard だけが捕らえる非空證)

未閉: assembly host bridge(次atom)。custody teeth は `device const` 故 shader 側から破れず、
host bridge atom にて採点する。
