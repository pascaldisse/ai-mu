# atom8 独立審査 其一 — FLDJ semantics oracle (Vishnu, L3, base f78aaa8)

全記述=實碼 path:line。推測零。未測=UNVERIFIED。

## 1. FLDJ v1 wire(凍結記述, path:line)

生産者=`packages/fieldlang-asm/emit.s`(手ARM64) · 消費者=`packages/field/src/journal.rs`。

header 48B, LE — `packages/field/src/journal.rs:37` `HEADER_LEN=48`, 書込 `:41-55`, 讀込 `:178-195`:

| off | size | field | 碼 |
|---|---|---|---|
| 0 | 4 | magic `0x4A44_4C46` = "FLDJ" | journal.rs:35, :44 |
| 4 | 4 | version u32 = 1 | journal.rs:36, :45 |
| 8 | 4 | w u32 (`cfg.width`) | journal.rs:46 |
| 12 | 4 | h u32 (`cfg.height`) | journal.rs:47 |
| 16 | 4 | c f32-BITS | journal.rs:48 |
| 20 | 4 | dt f32-BITS | journal.rs:49 |
| 24 | 4 | damping f32-BITS | journal.rs:50 |
| 28 | 4 | dx f32-BITS | journal.rs:51 |
| 32 | 8 | seed u64 | journal.rs:52 |
| 40 | 4 | range f32-BITS | journal.rs:53 |
| 44 | 4 | n_slots u32 | journal.rs:54 |

ops = tag u8 + payload LE(`op_bytes` journal.rs:58-99, 逆 `:198-241`):

1 SeedAtom{slot u32, seed u64} :61-65 · 2 Excite{x,y,amp_bits u32} :66-71 ·
3 Step{count u32} :72-75 · 4 Bind{dst,a,b u32} :76-81 ·
5 Bundle{dst u32, len u32, srcs u32×len} :82-89 · 6 WriteRaw{slot u32, len u32, bits u32×len} :90-97。

読側健全性: bad magic=InvalidData :181 · bad version=:185 · 未知tag=InvalidData :233-238 ·
vec長は残長で拒否 `check_len` :124-133(`len*4 > remaining`→UnexpectedEof) · 截断=Cursor read_exact UnexpectedEof :102。

實測 wire(本lane生成、fieldc 實走):
```
$ ./fieldc audit/fldj-oracle-r12/oracle.fld audit/fldj-oracle-r12/oracle.fldj
00000000: 464c 444a 0100 0000 2000 0000 2000 0000  FLDJ.... ... ...
00000010: 0000 803f cdcc cc3d 77be 7f3f 0000 803f  ...?...=w..?...?
00000020: 0903 0000 0000 0000 0000 803f 0800 0000  ...........?....
00000030: 0102 0000 0039 3000 0000 0000 0002 1000  .....90.........
00000040: 0000 1000 0000 0000 803f 03c8 0000 0003  .........?......
00000050: 4000 0000 0403 0000 0002 0000 0000 0000  @...............
00000060: 0005 0400 0000 0200 0000 0200 0000 0300  ................
00000070: 0000 0605 0000 0004 0000 0000 0080 3f00  ..............?.
00000080: 0000 0000 0080 bf00 0000 40              ..........@
sha256=9a97e9c425bd0cfb9aaaa8ee49f46a38e241ea34d4be19ac491eaaed217be7fe
```
header の c/dt/damping/dx/range = CONTRACT.md:36-38 の driver 既定値と byte 一致(實測)。
seed=0x309 = 777 = `界` 第4引数の override(CONTRACT.md:45-47 の 界-override 條、實測確認)。

## 2. slot semantics(實装行で確定)

執行門 = `packages/field/src/world.rs:35` `World::apply` 唯一(world.rs:5-6 の法)。

- **Step**: `world.rs:57-61` — `for _ in 0..count { self.tick() }`。**slot引数を持たぬ**。
  `tick` world.rs:91-98 = 固定 slot 0/1 のみ: `cur=read(0)`, `prev=read(1)`,
  `next=wave_step(fft,kernel,cur,prev)`, `write(1,cur)`, `write(0,next)`, `step_index+=1`。
  ∴ Step の作用域 = 予約 slot 0(cur)/1(prev) に固定、journal に slot 情報は無い(wire も持たぬ:tag3=count のみ)。
- **WriteRaw**: `world.rs:76-84` — `assert_eq!(data_bits.len(), self.cfg.d())` の後、
  `store.row_mut(slot)` へ `f32::from_bits(*bits)` を要素毎複写。
  ∴ payload = **f32 bit pattern の生列**、長さは **必ず d = w*h**、slot 上限検査は `row_mut` 任せ。
- Excite: world.rs:44-56 — slot 0/1 を read→`plane::excite`→write back。amp = `f32::from_bits(amp_bits)`。

## 3. floatbits→Q cell 量子化 = **存在せぬ**(實測)

`rg "from_bits|f32|float" packages/fieldlang-asm/q30_wave/*.s q30_wave_metal/*.metal emit.s` = **出力零**。
∴ FLDJ の f32-bit payload を Q30 固定小数 cell へ落とす碼は木中に無い。
q30 系の量子化は係数側のみ: `q(c,v) = (c*v + 2^29) arith>>30`(q30_wave/CONTRACT.md、
q30_wave_metal/CONTRACT.md「Lane law」)、入力 cell は最初から LE i32。
**故に「floatbits→Q cell 量子化を実装行で確定」= 実装不在ゆえ確定不能 → 要求は native-v0 contract の欠落**。

## 4. 舊 Rust semantics との差異 → **reject(未対応)**

| 項 | Rust `field`(FLDJ 執行系) | asm q30 系 |
|---|---|---|
| 数域 | f32 | i32 cell + i64 係数 Q30 |
| step 法 | 周波数域: `plane.rs:79-90` `wave_step` = FFT前進→k1_spec乗→逆FFT + `a_prev*prev` | 空間5点 stencil(q30_wave/CONTRACT.md) |
| 5点 stencil | 参照 oracle としてのみ存在 `plane.rs:93-113` `wave_step_reference` | 之が product |
| 入力形式 | `.fldj`(FLDJ magic) | `Q30WAVE2\0` fixture(wave_runner.s:44-52 で magic 厳密照合) |
| 係数 | c,dt,damping,dx,range(f32 bits) | c_cur,c_lap,c_prev(i64) |

∴ **FLDJ→scalar/neon/metal の意味論橋は実装されていない**。
FLDJ params(c/dt/damping/dx) から Q30 係数三つ組への写像を規定した碼も文書も無い(rg 実測)。
明示 **native-v0 contract 要求**: 「FLDJ header params → (c_cur,c_lap,c_prev) i64 Q30 の凍結写像」
＋「WriteRaw f32-bits → i32 cell 量子化規則(丸め・飽和)」＋「Step count → runner 反復の意味」。
之無き限り「三経路が FLDJ を同一に実行する」主張は **reject**。

## 5. 三経路 實走(實測、誇張禁)

### 5a. FLDJ journal を三経路へ投入 → 全経路 **拒否**(実測)
```
$ ./q30_wave/wave_runner audit/fldj-oracle-r12/oracle.fldj
wave_runner: failure                                   rc=1   (scalar+neon runner)
$ ./q30_wave_metal/wave_metal_runner q30_wave.metallib audit/fldj-oracle-r12/oracle.fldj
metal device: Apple M1 Pro
wave_metal_runner: failure                             rc=1
```
理由=magic 照合(wave_runner.s:44-52 `Q30WAVE2\0`)。∴ **FLDJ 一本での三経路比較は実行不能**。

### 5b. 三経路一致が實在する範囲 = Q30WAVE2 fixture のみ(實測、實機 M1 Pro)
```
$ ./q30_wave_metal/wave_metal_runner q30_wave.metallib ../q30_wave/wave_vectors.bin
metal command status: 4
metal command error: nil
wave_metal_runner: 138 Q30WAVE2 scalar=neon=metal ok (offsets 0/4/28, alias, reps, count0/neg/overflow)
rc=0
```
```
$ ./q30_wave/gate.sh   (末尾)
mutation input-custody-scalar-tooth rejected=ok
q30_wave gate: frozen138-Q30WAVE2 scalar+neon aligned+unaligned alias ABI custody guard mutations=ok
```
但し fixture record = **単一 step の stencil 適用**であり、≥200 step の時間発展ではない(record 構造 wave_runner.s:56-80 実測)。
∴ **「≥200 steps・三経路 digest 一致」は本 base では未達 = UNVERIFIED/実行不能**。

### 5c. ≥200 step の nonzero evolution は Rust 経路のみで實測
```
$ ./fieldc oracle_nowr.fld oracle_nowr.fldj   # 界32 32 8 777 / 種 / 撃 / 歩200 / 歩64 / 縛 / 束
sha256=e320a34587f04f225a0b1f8bffd407c13f28c28aabc98952dd82c94bb0cec6a9
$ cargo run -q --example replay_fldj -p field -- oracle_0step.fldj oracle_nowr.fldj
audit/fldj-oracle-r12/oracle_0step.fldj digest=a4d3e542dfbcad0c
audit/fldj-oracle-r12/oracle_nowr.fldj  digest=bd2f6c3cde34096b
```
264 step(200+64)適用で digest 変化 ∴ evolution nonzero(實測)。asm 側の対応値は存在せぬ。

## 6. 欠陥(實測、本審査で発見)

**D1 — fieldc が replay で panic する journal を出す(生産者/消費者 契約不整合)**
`寫 5 4 ...`(len=4)を fieldc は rc=0 で受理・出力するが:
```
$ cargo run -q --example replay_fldj -p field -- audit/fldj-oracle-r12/oracle.fldj
thread 'main' panicked at packages/field/src/world.rs:78:17:
assertion `left == right` failed: WriteRaw payload must be d bits
  left: 4  right: 1024
```
CONTRACT.md:16 の `寫 slot n b1..bn` に n==w*h 制約が書かれておらず、emit.s も検査せぬ(rc=0 実測)。
消費側は `assert!`(panic)であり `io::Error` でもない。
要求: (a) CONTRACT に `n == w*h` を明記 + emit.s が syntax/range err で拒否、或は
(b) `World::apply` が WriteRaw 長不一致を回復可能な誤りとして扱う。**現状=無防備な panic 面**。

## 死枝
- 「fieldc journal 一本で三経路実走比較」→ 死。因=runner が Q30WAVE2 magic 厳密、FLDJ decoder 無し(5a 実測)。新 decoder 執筆は任務禁令(新規碼書くな)。
- 「floatbits→Q cell 量子化行の特定」→ 死。因=該当実装が木中に不在(rg 零、§3)。
- 「≥200 step の scalar=neon=metal digest 一致貼付」→ 死。因=fixture が単一 step record、時間発展 runner 不在。誑らず reject と記す。

## UNVERIFIED
- FLDJ params → Q30 i64 係数 写像(規定不在)
- WriteRaw f32-bits → i32 Q cell 量子化(実装不在)
- 三経路 ≥200 step 時間発展一致(実行経路不在)
- q30_wave_metal gate.sh 全走(本lane未実行 — runner 直走 138 ok のみ實測)
- fieldc の 寫 長検査(D1)修正後の挙動

## VERDICT
**REJECT** — 「FLDJ semantics oracle が三経路(scalar/neon/metal)で成立」は本 base f78aaa8 では **成立せぬ**。
成立するのは Q30WAVE2 fixture 単一 step 上の三経路一致(實測 green)のみ。
FLDJ wire 自体は §1 の通り凍結記述可能で生産者/消費者一致(D1 の 寫 長を除く)。
先行条件 = native-v0 contract(§4 の三写像)＋ D1 修正。
