# Q30 二次元五点波 · Q30WAVE2 fixture

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


凍結wire=`Q30WAVE2\0`: record係数=LE i64 `c_cur,c_lap,c_prev`。V1のLE i32係数wireは本runner対象外（ABI i64を表せぬ）。V2 corpus=旧133+境界5; `c_cur=±2^31`・`c_lap=±2^29`・lap極値・項順序を含む。

## 段1 (Vishnu atom7-final): 残長・checked dims

`wave_runner` は今、file end を保持し `Lneed` にて **各 record 前・各 field 前** に残長を厳密要求する。
`width*height` は `umulh`+`mul` の **u64 checked**、`0`/負/arena上限超=拒否。全record後の cursor は
file end と **完全一致** せねばならぬ(trailing garbage=赤)。

kill証明(實機 M1 Pro、base 8a1d244 の runner と本runnerの差):
```
--- base runner vs trailing:   wave_runner: 138 Q30WAVE2 ok   base_rc=0   ← SURVIVE(Kali blocker 1)
--- new  runner vs trailing:   wave_runner: failure           new_rc=1    ← RED
```
門teeth: fixture-trailing / fixture-truncated / fixture-dims-overflow / fixture-zero-dim / fixture-record-truncated。

## round11 自攻(Vishnu 自作を他人として攻む)の結果

kill 3・survive 3。修正済:
1. **arena頂 OOB(真の欠陥)**: `n<=0x4000` を許すが out 基址=+64・非整列経路=+1・上位guard=+65+bytes+32
   \∴ 実必要 65633 > 65536 arena。n>=16360 で guard を越境破壊していた。作業arenaを 65792 へ拡張。
   歯: `arena-top` (n=16384 緑 / n=16385 赤、count緩和 probe 上で)。
2. **段1 teeth の空虚**: trailing/truncated/dims/zero/record-truncated の5歯は
   末端一致検査 + `cases==138` に吸収され、`Lneed` と u64 checked mul を **殺していなかった**
   (Lneed を `ret` のみに弱めても門は緑だった)。単独露出歯を追加:
   `remaining-length tooth`(need_probe=末端検査緩和) / `u64-checked dims tooth`(32bit mul へ戻すと **SIGSEGV**)
   / `arena-full short-read tooth`(bounds_probe)。
3. **Metal CONTRACT の嘆願**: `q30_wave_metal` に host bridge も gate も無い \∴ 記述を要求(UNVERIFIED)へ訂正。
   `.bak.s`(作業残骸 496行)を製品木より削除。

survive(攻めたが破れず):
- 負 w/h・`w=1,h=2^30`・`w=INT_MIN`・65536x65536: すべて拒否。
- cursor==file end・w/h 欠落・名長のみ・9byte file: すべて拒否。
- x16/x17 規約: `bl Lneed` は _main 本体3箇所のみ、x30 は _main で未使用、
  x9/x11/x12/x13 は Lneed を跨いで生存し Lneed は触れぬ \∴ 規約正当。

死枝: `cbz x13, Lfail`(product==0) は **到達不能** — `w>=1 && h>=1 && w,h<=2^31` \∴ product>=1。
帯として残すが、これは歯を持たぬ(死枝と明記)。
