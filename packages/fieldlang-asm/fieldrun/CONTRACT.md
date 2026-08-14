# fieldrun V0 CONTRACT — 訂正版 (oracle: Vishnu, base Saraswati 89f2eee)

状態: **contract-only**。実行可能 binary 不在(確定)。本書 = 次の建者が曖昧無しに実装できる粒度。
本書中の数値は §5 の算出経路で **独立に手算**(新規 Rust/C/Swift/Python 零)。
**実行時 PASS の主張は本書に一切無い** — §7 の全項目 = UNVERIFIED。

## 0. 上流実装の位置(path:line 固定)

- FLDJ wire 生産者 = `../emit.s` / `../driver.s`(手ARM64)、消費者参照実装 = `../../field/src/journal.rs`
  (header 48B: `:37,:41-55,:178-195` · op tag/payload: `:58-99` / `:198-241`)
- FLDJ 執行意味論 = `../../field/src/world.rs:35` `World::apply`(唯一の変更門)
- 物理法 = `../../field/src/plane.rs:93-113` `wave_step_reference`(5点stencil)
- Q30 backend 律 = `../q30_wave/CONTRACT.md` · `../q30_wave_metal/CONTRACT.md`

## 1. Saraswati 契約 対 実装 の齟齬(一件ずつ)

**G1 (誤帰属)** 「The physical FLDJ formula is Rust's `plane.rs` law」→ product 執行路は
`plane.rs:79-90` `wave_step` = **FFT 分光**(`k1_spec` 乗算)であり 5点 stencil ではない。
引かれた式は `plane.rs:93-113` `wave_step_reference`(= parity oracle)。
**訂正**: fieldrun V0 は `wave_step_reference` の意味論に整合する、と明記せよ。
`wave_step`(FFT)との bit 一致は **主張してはならぬ**(浮動小数の別経路)。fieldrun は
「FLDJ runtime parity」ではなく「FLDJ **reference-law** subset runtime」である。
※係数の符号は一致: `plane.rs:99` `a_prev = -(1.0-dd)` = `dd-1` ✓。

**G2 【撤回・A9 Marut 実測】** 旧文：「`0x3F7FBE77` の実値 = `0.99896949529647827148`、
f32 最近傍 of 0.999 = `0x3F7FC077`(512 ulp 差)、上流注釈 `(0.999)` は誤」= **偽**。
因 = §5 の f32 decode で frac を誤讀(`0x7FBE77` = **8371831**、8371319 非)。
真値: `0x3F7FBE77` = `0.99900001287460327148` = f32 最近傍 of 0.999 ∴ 上流注釈は **正**。
(`0x3F7FC077` = `0.99903053045272827148`、512 ulp 上だが 0.999 の最近傍ではない。)
∴ 上流への要求は「十進の訂正」ではなく **「bits が法・十進は描写に過ぎず」の明記** のみ
(A9 で澀、`../CONTRACT.md` Defaults 節)。但し本誤読は **本 lane の係数に流入した** → G12。

**G12 (係数汚染・A9 Marut 実測・未閉)** `coef.s:52-57` の固定定数は G2 誤読十進から導かれた。
実測(f32 意味論、bits から直接): `dd = damping*dt = 0.09989999979734420776`、
`(2-dd)*2^30 = 2040216832`・`(dd-1)*2^30 = -966475008`・`k*2^30 = 10737419`。
実装値 = `c_cur=2040220160`・`c_prev=-966478272`・`c_lap=10737419` ∴ **c_cur が +3328 Q30 単位
(相対 1.63e-6)、c_prev が -3264 ずれる**(c_lap のみ一致)。このずれは上記 0.998969… 経路を
再現する(誤十進 → c_cur=2040220160・c_prev=-966478272 一致)∴ 原因同定は確定。
帰結: 三経路(scalar/NEON/Metal)は互いに byte 一致するが、**上流 `wave_step_reference` とは
原理上一致し得ぬ** ∴ G1(product parity 非証明)は単なる未証ではなく **既知の不一致**。
**【A9-4 Ashwin/Rudra: 閉じた】** L2 判断 = 「係数を真値へ直し golden 再凍結」を採る。
実施: `c_cur 2040220160→2040216832` · `c_prev -966478272→-966475008`(`c_lap` 不変)。
三者独立導出一致(Ashwin bit分解 · Rudra · L2 実 f32 `dd bits=0x3dcc985f`)。
骸(誤定数時代の golden、想定内に壊れた): `4x4-3tick a85a4cc0ee31…` · `A7 32x32x200 ead5a8fff10e…` ·
`3c-2x2 c34a1425…` · `8x8 39343e40…` · `9x9 169ff90f…` · §3c cells `1950460 …`。
新凍結値 = §10/§12/§13 の表を見よ。`saturating-vector ec09887b…` は不変(飽和支配ゆえ係数非依存)= 独立な裏付け。
なお **G1 は存続**: 係数が真値になっても product 執行路 `plane.rs:79-90` は FFT 分光 ∴ bit 一致は
原理上主張し得ぬ。射程 = `wave_step_reference`(5点 stencil)の意味論まで。
本 lane 実測 hexdump(oracle.fldj off 0x18)= `77be 7f3f` = 0x3F7FBE77 ✓。

**G3 (Q20/Q30 の scale 関係が未記載)** cell = Q20、係数 = Q30。q30 律
`q(c,v)=(c*v+2^29)>>30` は c を**無次元 Q30**として扱う ∴ v の固定小数 scale は kernel に
不可視で、Q20 in → Q20 out。矛盾は無いが、契約に明記無き為 実装者が Q20↔Q30 の
再scale を挿入する誤りを招く。**訂正**: 「kernel は cell scale に無関知。Q20 cell を
そのまま i32 として渡し、出力も Q20。再scale 禁」と明記。

**G4 (f32→Q20 の非有限/範囲外が未定義)** 「integer bit decomposition and round-to-nearest,
ties away from zero」のみ。NaN/±Inf/|v|≥2048(Q20 で i32 溢れ)の扱いが無い。**穴**。§2(a) で確定。

**G5 (Step の「swaps the slot pointers exactly」が実装と不一致)**
`world.rs:91-98` は 2者 swap ではない: `next = f(cur,prev)`(第三の buffer)→ `write(1,cur)` →
`write(0,next)`。加えて `../q30_wave/CONTRACT.md` は **`out` が `cur`/`prev` と重なる事を禁ずる**
(read-before-write stencil)∴ **第三 buffer は必須**。契約は第三 buffer に言及せぬ。**Blocker級曖昧**。§2(d) で確定。

**G6 (FLRO の未定義点)** どの slot を吐くか未記載 · `steps` の定義(累計 Step count か tick 数か)未記載 ·
saturation 数を運ばぬ(三経路とも `sat` を返すのに parity 検査から落ちる)· cell endianness は明記済 ✓。§2(c) で確定。

**G7 (header 検査範囲が未分割)** 「All physical fields ... must equal canonical bit patterns」だが
w/h/n_slots/seed の扱いが無い。w,h は `界` で自由(u64 checked product 必須、
`../q30_wave/wave_runner.s` の既知の穴を踏まぬ事)、n_slots は ≥2 要求、seed は
SeedAtom 不受理ゆえ無視、version==1・magic 厳密。§2(e)。

**G8 (backend ABI 名が未記載)** 呼ぶべき記号が無い。実在は:
`_fl_q30_wave_scalar` / `_fl_q30_wave_neon`(x0=cfg,x1=cur,x2=prev,x3=out → x0=sat)、
`_fl_wave_metal_init(x0=metallib path)->0/1` / `_fl_q30_wave_metal(x0=cfg,x1,x2,x3,x4=byte_offset,x5=reps)->sat/-1`。
`cfg` = `{i32 w, i32 h, i64 c_cur, c_lap, c_prev}` offsets `0,4,8,16,24`。**訂正**: 契約に明記。
Metal は CPU fallback 無し(`-1`)∴ `--metal` で `-1` = 非零 exit。

**G9 (上流 D1 との関係)** `../fieldc` は `寫 slot n ...` の `n != w*h` を **検査せず rc=0 で出す**
(本 lane 実測: replay が `world.rs:78` で panic)。fieldrun が `len == width*height` を
拒否要件に持つのは正しく、**D1 の消費側緩和にはならぬ**(生産側の穴は別途要修正)。

**G10 (証拠要件 2 の内部矛盾)** 「full-plane `寫` を含む .fld」を要求するが、`../CONTRACT.md` の
`寫` は十進 u64 引数列 ∴ 32x32 で 1024 個の十進数(fieldc の 16MiB buf 内、可)。
一方 V0 は `SeedAtom`/`Excite` を **loud reject** する ∴ 初期場は `寫` のみで作る。
契約は之を要求と読めるが明示されていない。**訂正**: 「V0 の初期場投入は `寫` 一手のみ」と明記。

**G11 (lap 範囲の継承要件)** `../q30_wave/CONTRACT.md` は `|lap| <= 2^34-4`・`|c_lap| <= 2^29` を
**前提条件**として課す。Q20 cell が i32 全域を取り得る故 |lap| 最大 = `4*(2^31-1)+4*2^31 = 2^34-4`
= 境界丁度(適合)。`c_lap` は §3 の値 `10737419 <= 2^29` ✓。**訂正**: 之を契約に検査項として書け。

## 2. 拘束決定(確定案)

### (a) 決定論的 f32-bits → Q20
入力 = u32 bit pattern `b`。整数演算のみ、host 浮動小数 禁。
```
s = b>>31 ; e = (b>>23)&0xFF ; f = b&0x7FFFFF
e==0xFF                     -> REJECT (exit!=0)   ; NaN・±Inf
e==0 && f==0                -> 0                  ; ±0 は共に +0
e==0                        -> subnormal: sig=f      , E = -126-23
else                        -> sig=f+2^23          , E = e-127-23
value = (-1)^s * sig * 2^E
q_exact = value * 2^20  ==>  整数部/小数部は shift のみで得る:
  S = E+20
  S >= 0 : mag = sig << S              (小数部無し、丸め不要)
  S <  0 : mag = (sig + (1 << (-S-1))) >> (-S)     ; round-half-AWAY-from-zero
           (絶対値上で行う故 half-away = 加算後 shift で厳密)
           -S >= 64 の時は mag = 0(sig < 2^24 ∴ 情報は全て丸め落ち)
q = s ? -mag : mag
mag > 2^31          -> REJECT (exit!=0)         ; 上溢れ = 黙って飽和せぬ
s==0 && mag == 2^31 -> REJECT                   ; +2^31 は i32 に無い
s==1 && mag == 2^31 -> q = -2^31                ; 丁度表現可
```
∴ 表現可能域 = `[-2048.0, +2048.0)` の f32。範囲外・非有限 = **loud reject**(飽和禁)。
理由: WriteRaw は初期条件であり、静かな飽和は parity 主張を汚す。飽和は kernel 内部
(acc の i32 一回飽和)にのみ許す。

### (b) canonical header → Q30 係数(定数 + 導出)
検査: `c==0x3F800000 && dt==0x3DCCCCCD && damping==0x3F7FBE77 && dx==0x3F800000 && range==0x3F800000`
(byte 一致、数値比較禁)。之を満たす時に限り、係数は **固定定数**:
```
c_cur  = 2040220160        // = round((2-dd)*2^30)
c_lap  =   10737419        // = round(k*2^30),  k=(c*dt/dx)^2
c_prev = -966478272        // = round((dd-1)*2^30)
```
導出律(実装は上の定数を **即値で** 持て。実行時 f32 算 禁):
`dd = fl32(damping*dt)` · `k = fl32(fl32(fl32(c*dt)/dx)^2)` · 各段 f32 最近傍偶数丸め
(= `plane.rs:60-61,97-99` と同じ順序)。§5 に手算全経路。
`c=dx=1.0` ゆえ `courant = dt` は厳密。
不変式(実装が assert せよ): `|c_lap| <= 2^29` ✓ · `|c_cur| <= 2^31` ✓ · `|c_prev| < 2^30` ✓。

### (c) FLRO wire(v0 確定)
```
off  size  field
0    4     magic = "FLRO"  (0x4F52_4C46 LE u32)
4    4     version u32 = 0
8    4     width  u32          ; FLDJ header と一致
12   4     height u32
16   8     steps  u64          ; 適用した Step の count 総和 = tick 総数
24   8     sat    u64          ; 全 tick 累計の飽和胞数(backend 戻り値の和)
32   4*w*h i32 LE Q20 cells    ; **最終 slot0 (cur)** の内容
```
header = 32B。`out.bin` 長 = `32 + 4*w*h` 厳密。三経路の byte/SHA-256 一致要求は
本 wire 全体(steps・sat を含む)に掛かる ∴ 飽和数の差異も赤に落ちる。
※Saraswati 原案(24B・sat 無し・slot 未指定)からの変更 = G6 の穴埋め。

### (d) WriteRaw slots0/1 + Step swap 意味論
三 buffer 必須(`out` の alias が q30 ABI で禁ぜられる為):
```
状態 = (P_cur, P_prev, P_scratch)  三つの互いに素な w*h i32 領域
初期 = 全域 0
WriteRaw{slot=0} -> P_cur   へ Q20 変換結果を書く
WriteRaw{slot=1} -> P_prev  へ同様
WriteRaw{slot>=2}-> REJECT
Step{count}: count 回、各回:
    sat += backend(cfg, cur=P_cur, prev=P_prev, out=P_scratch)
    (P_cur, P_prev, P_scratch) <- (P_scratch, P_cur, P_prev)      // 三者回転
    steps += 1
```
之が `world.rs:91-98`(`prev <- 旧cur`, `cur <- next`)と厳密に同値。
「2者 swap」は **誤**(G5)。回転は Step の内側で毎 tick 起き、Step 境界を跨いで永続する。
`count==0` = 無作用(合法)。`P_cur == P_prev` の alias は起こさぬ(独立領域)。
Metal 経路の `reps` 引数は **1 固定**(`reps>1` は同一入力の再走であり時間発展ではない)。

### (e) header 検査の分割(G7 確定)
`magic==FLDJ` · `version==1` · 物理5値 = (b)の bit 一致 · `w>=1 && h>=1` かつ
`w*h` を **u64 checked**(`umulh` 非零 = 拒否)· `w*h <= 実装 arena 上限`(定数、硬碼禁=既定値+引数)·
`n_slots >= 2`(slot0/1 を使う為)· `seed` = 無視(SeedAtom 不受理)。
op stream: 未知 tag・vec 長が残長超過・末端不一致(trailing byte)= 全て非零 exit。

## 3. 試験ベクタ(独立手算、§5 に経路)

### 3a. f32-bits → Q20(§2a の律)
| f32 bits | 値 | Q20 期待 | 根拠 |
|---|---|---|---|
| `0x00000000` | +0 | `0` | e==0,f==0 |
| `0x80000000` | -0 | `0` | 同上(-0→+0) |
| `0x3F800000` | 1.0 | `1048576` | 2^20 |
| `0xBF800000` | -1.0 | `-1048576` | |
| `0x3F000000` | 0.5 | `524288` | |
| `0x3DCCCCCD` | 0.1(f32) | `104858` | 13421773/2^27*2^20 = 13421773/128 = 104857.6015625 → away → 104858 |
| `0xBDCCCCCD` | -0.1 | `-104858` | 対称 |
| `0x34000000` | 2^-21 | `1` | q=0.5 丁度 → ties **away** → 1(round-half-even なら 0 = 歯) |
| `0xB4000000` | -2^-21 | `-1` | |
| `0x33FFFFFF` | 2^-21 の直下 | `0` | q<0.5 |
| `0x34000001` | 2^-21 の直上 | `1` | q>0.5 |
| `0x447FFFFF` | 1023.99994 | `1073741760` | sig=8388607, S=-3+20-... §5c |
| `0xC5000000` | -2048.0 | `-2147483648` | 丁度 i32 min |
| `0x45000000` | +2048.0 | REJECT | mag==2^31、正で表現不可 |
| `0x7F800000` | +Inf | REJECT | e==0xFF |
| `0x7FC00000` | NaN | REJECT | e==0xFF |
| `0x00000001` | 最小 subnormal | `0` | S 極小 |

### 3b. 係数(§2b)
canonical header → `c_cur=2040216832` `c_lap=10737419` `c_prev=-966475008`。
骸(A9-4): 旧 `c_cur=2040220160`・`c_prev=-966478272` = **誤定数時代の値**(因 = damping frac 誤讀 §5)。
非 canonical header(例 dt を `0x3E000000`=0.25 に変えた FLDJ)→ **REJECT**(係数を算出せぬ)。

### 3c. 一 tick の完全手検査(2x2、周期 wrap)
`w=h=2`、`cur = [1048576, 0, 0, 0]`(= 1.0,0,0,0 Q20)、`prev = [0,0,0,0]`。
`w==2` ゆえ x-1 と x+1 は同一胞へ wrap(`../q30_wave/CONTRACT.md` の周期律、`w==1`/`h==1` は自身へ wrap)。
```
i=0: lap = cur[1]+cur[1]+cur[2]+cur[2] - 4*cur[0] = 0+0+0+0 - 4194304 = -4194304
     q(c_cur,cur0)  = (2040216832*1048576 + 2^29) >> 30 = 1992399
     q(c_lap,lap)   = (10737419*(-4194304) + 2^29) >> 30 = -41943   (§5d)
     q(c_prev,0)    = (0 + 2^29) >> 30 = 0
     acc = 1992399 - 41943 + 0 = 1950456      (骸: 誤定数時代 1950460)
i=1: lap = cur[0]+cur[0]+cur[3]+cur[3] - 4*cur[1] = 2097152
     q(c_lap,lap) = 20972 ; 他 0 ∴ acc = 20972
i=2: 同 i=1 -> 20972
i=3: lap = cur[2]+cur[2]+cur[1]+cur[1] - 0 = 0 -> acc = 0
out = [1950456, 20972, 20972, 0], sat = 0    (骸: 誤定数時代 1950460 …)
```
(`>>` = 算術右シフト。負値の丸めは shift 定義に従う = `../q30_wave` 律と同一。)
之は scalar/NEON/Metal 三経路とも同値でなければならぬ。

### 3d. 赤変異(必須、各々単独で赤)
`ties-to-even`(3a の `0x34000000` が 0 に落ちる)· `c_lap 十進起こし`(damping=0.999 を使うと
`c_cur=2040219776`≠、G2 が刺さる)· `2者swap`(§2d を swap に弱めると 2 tick 目から乖離)·
`FLRO sat 欄削除` · `WriteRaw 飽和(reject の代りに clamp)` · truncation · trailing byte ·
bad tag · bad op length · `w*h` 32bit 乗算(u64 checked 撤去)· backend 強制失敗(`--metal` で init 失敗)。

## 4. 実装 atom 列(順序付き・各 atom 一歩)

1. **A1** `fldj_parse.s`: header 48B 厳密 decode + §2e 全検査。op stream を走査のみ(実行せぬ)、
   末端一致・残長・未知tag・vec長で非零 exit。門 = 手作 FLDJ 断片(shell 生成)で赤4種。
2. **A2** `q20_conv.s`: §2a を整数のみで実装 + §3a の 17 vector を自己試験。ties-away の歯必須。
3. **A3** `coef.s`: §2b の canonical bit 一致検査 → 即値3定数。非 canonical で reject。歯 = dt 改変。
4. **A4** `fieldrun_scalar`: A1+A2+A3 + §2d の三 buffer 回転 + `_fl_q30_wave_scalar` 呼出 + FLRO 書出。
   門 = §3c の 2x2 一 tick を byte 一致で採点。
5. **A5** `--neon`: `_fl_q30_wave_neon` へ切替のみ。門 = scalar と FLRO byte/SHA 一致。
6. **A6** `--metal`: `_fl_wave_metal_init` + `_fl_q30_wave_metal`(reps=1、offset=0)。`-1` = 非零 exit。
   門 = 三経路 SHA-256 一致(実機 GPU)。
7. **A7** 実 journal: `寫 0 1024 <1024個>` + `歩 200` 以上を含む `.fld` を **shell 生成**(生成器は
   fieldrun を読まぬ = 独立)→ `../fieldc` で compile → 三経路実走 → nonzero 出力 + SHA 一致を貼る。
8. **A8** teeth: §3d の全変異を `teeth_kill.sh` 式に単独適用 → 全 KILLED を実測表で出す。
9. **A9** 上流 D1 修正(別 atom・別担当可): `../CONTRACT.md` に `寫 n==w*h` 明記 + `emit.s` が
   range err で拒否 + `world.rs:78` の assert を回復可能誤りへ。加えて G2 の十進注釈訂正。

## 5. 算出経路(独立検算、bc + 手算のみ — 新規 Rust/C/Swift/Python 零)

f32 decode: `0x3F7FBE77` → e=0x7E=126, f=0x7FBE77=**8371831**, sig=2^23+f=**16760439**,
値=`16760439/2^24` = `0.99900001287460327148`。
【A9 訂正】旧記 `f=8371319 ∴ 16759927/2^24 = .99896949529647827148` = **誤**(frac 誤讀、§1 G2 参照)。
下の bc 行の damping 系の数値は全てこの誤値に乗っている ∴ 判読注意(G12)。
`0x3DCCCCCD` → e=0x7B=123, f=0x4CCCCD=5033165, sig=13421773, 値=`13421773/2^27`。
```
$ bc <<< 'scale=20; 16760439/2^24'      -> .99900001287460327148   (damping 実値 = f32最近傍 of 0.999)
$ bc <<< 'scale=20; 13421773/2^27'      -> .10000000149011611938   (dt 実値)
```
骸(A9-4): 旧 sig `16759927`(→ .99896949529647827148)= **誤讀**。0x7FBE77 = 8371831、
8371319 に非ず(512 の讀み落ち)。Vishnu §5 と Surya の「独立検算」は **同一の誤入力を共有**
∴ 一致は無意味だった(失敗様式共有)。真 sig = `8371831 + 2^23 = 16760439`。

`dd = fl32(damping*dt)`: 厳密積 = `16760439*13421773/2^51`。dd≈0.09990 ∈ [2^-4,2^-3)
∴ f32 は `m/2^27`, m∈[2^23,2^24)。`m = round_even(16760439*13421773 / 2^24)`:
```
$ bc: n=16760439*13421773            -> 224954807638347   (48 bit)
      n/2^24 = 13408351  rem 6707531 ;  half = 8388608  ->  rem < half ∴ 切捨
      m_dd = 13408351                  (dd = 13408351/2^27 = .09989999979734420776)
```
`2-dd ∈ [1,2)` ∴ `m2/2^23`。`(2-dd)*2^23 = 2^24 - m_dd/16`:
```
$ bc: a = 2^28 - m_dd = 255027105 ; a/16 = 15939194 rem 1 ; half=8 -> 切捨
      m2 = 15939194 ;  c_cur = m2 * 2^7 = 2040216832        (厳密、丸め無し)
```
`1-dd ∈ [0.5,1)` ∴ `m3/2^24`。`(1-dd)*2^24 = 2^24 - m_dd/8`:
```
$ bc: b = 2^27 - m_dd = 120809377 ; b/8 = 15101172 rem 1 ; half=4 -> 切捨
      m3 = 15101172 ;  c_prev = -(m3 * 2^6) = -966475008    (厳密)
```
`k = fl32(dt^2)`(c=dx=1.0 ゆえ courant=dt 厳密)。`k≈0.01 ∈ [2^-7,2^-6)` ∴ `m4/2^30`:
```
$ bc: s = 13421773^2 = 180143990463529 ; s/2^24 = 10737418 rem 9395241 ; half=8388608 -> 切上
      m4 = 10737419 ;  c_lap = m4 = 10737419                (厳密)
```
**独立検算(十進の別経路 = awk 倍精度、上の整数経路と失敗様式を共有せぬ・A9-4 実走)**:
```
dd            = .09989999979734420776
fl32(2-dd)*2^30 = 2040216832.0000   (厳密積 (2-dd)*2^30 = 2040216840 → f32 丸めで 2040216832)
fl32(dd-1)*2^30 = -966475008.0000   (厳密 -966475016 → f32 丸め)
k_exact*2^30    = 10737418.560000   → round = 10737419 = c_lap
```
∴ 三定数とも整数経路と一致(Rudra 独立導出・L2 第三経路 f32 実演算 `dd bits=0x3dcc985f` とも三者一致)。
~~G2 の 512 ulp：`0.999 - 0.99896949529 = 3.0505e-5` / ulp 比 511.8~~ 【A9 死】
因 = 入力定数が誤(frac 誤讀)。両経路が同じ誤値を共有していた ∴ 「別経路一致」は役に立たず
(= 失敗様式共有の実例。検は算と同じ穴を持ってはならぬ)。
真値経路: `0x3F7FBE77` は 0.999 の f32 最近傍そのもの ∴ ulp 差 = **0**。
§3c の量子化(bc 実走、本 lane 実測):
```
$ bc: cc=2040216832; cl=10737419; v=1048576
      (cc*v+2^29)/2^30                  -> 1992399
      cl*(-4194304)+2^29 = -45035462590464 ; floor(/2^30) -> -41943   (算術shift=floor)
      (cl*2097152+2^29)/2^30            -> 20972
```
**独立検算(除算の別経路)**: `c_cur*2^20/2^30 = c_cur/2^10 = 2040216832/1024 = 1992399.25`
∴ 半加算後 floor = `1992399` ✓ · `c_lap*2^22/2^30 = 10737419/256 = 41943.043` →
`floor(-41943.043+0.5) = -41943` ✓ · `10737419/512 = 20971.52` → `floor(20972.02) = 20972` ✓。
**之は算術の実測であり、実装の実走ではない** — 実装側の一致は A4 の門が最初の実測点。

## 6. 判定

契約は **健全に到達可能** だが Saraswati 原文のままでは **G1・G4・G5・G6・G7・G8 が blocker**
(誤帰属・未定義・実装不能な swap 記述)。本書 = 之を全て埋めた **訂正契約**。
実行可能 binary は依然不在 ∴ 「FLDJ runtime parity」は現時点 **未成立**、§4 の A1..A8 が要る。

## 7. UNVERIFIED(実行時 PASS の詐称禁)

- fieldrun binary 一切不在 — §2/§3 の全律は **未実走**
- §3a の 17 vector · §3b の非canonical reject — **未実測**(手算のみ)
- §3c の 2x2 期待値は bc で算出済(§5)だが **実装での実走は未了**
- 三経路 FLRO SHA 一致 · ≥200 step の nonzero 出力 — **未実行**
- §3d の変異が実際に赤に落ちるか — **未実測**
- §5 の係数三定数は手算(bc)で **算出済・実測照合未了**(実装との突合は A3 の門が最初)
- G2 の上流十進注釈誤りは **本書では未修正**(A9 で上流 `../CONTRACT.md` を直す)

## 8. A1 実装記(Brahma)

`fldj_parse.s`(手ARM64)+ `build.sh` + `gen_fldj.sh`(shell のみ生成器)+ `gate.sh`。
用: `fldj_parse <path.fldj> [max_cells=16384] [max_slots=1024]`(硬碼禁 = 既定 + 引数)。
rc 表(検査毎に別経路): 2 header短 · 3 magic · 4 version · 5 w==0 · 6 h==0 · 7 w*h u64溢
· 8 w*h>上限 · 9 n_slots<2 · 10 n_slots>上限 · 11 物理5値 非canonical · 12 未知tag
· 13 WriteRaw slot>=2 · 14 op切断/末尾余剰 · 15 WriteRaw len != w*h · 16 open/read · 17 用法。
seed = 無視(SeedAtom 不受理 ∴ 拒否経路無し)。
rc=7 は防御的: u32×u32 < 2^64 故 現 wire では到達不能(**UNVERIFIED**、歯無し)。
32bit 乗算 wrap の歯は `65536x65536 -> rc=8` で立てた(u64 なら 2^32 > 上限)。
D1 の穴(`n != w*h`)は消費側で rc=15 として閉じた(生産側修正は A9 のまま)。
op 実行・Q20 変換・係数導出は本 atom の外(A2/A3/A4)。

## 9. A2 実装記(Agni)

`q20_conv.s`(手ARM64・整数演算のみ、FP/SIMD レジスタ零 = `q20_gate.sh` が走査で強制)+
`q20_gate.sh`(shell のみ)。用: `q20_conv <hexbits>` → stdout 十進 Q20 一行。
rc: 0 成功 · 20 非有限(NaN/±Inf)· 21 範囲外(|mag|>2^31 或 +2^31)· 17 用法/hex 不正。
飽和禁 = 拒否は loud。公開記号 `_q20_from_bits(x0=bits) -> x0=q, x1=err` は A4 が呼ぶ。

**§3a の erratum(実測、律との矛盾)**: `0x34000000` の e フィールド = 0x68 = 104
∴ 値 = `2^-23`(`2^-21` ではない)、`q = 2^-3 = 0.125` → §2a の律で **0**、表の期待値 `1` は誤。
`0x33FFFFFF`/`0x34000001` も同じ 2 段ずれ。§2a の律を法とし、§3a が意図した
「q=0.5 丁度 = ties 境界」は `0x35000000`(=2^-21)である。門は §3a 全 17 行を
律の値で走らせ、加えて真の tie 四行(`0x35000000`/`0xB5000000`/`0x34FFFFFF`/`0x35000001`)と
`0xFF800000`(-Inf)・`0xC5000001`(|mag|>2^31)を足す。**§2a の律自体は無変更**。

赤歯(全 KILLED 実測、各々単独で赤): `ties-to-even`(0x35000000→0)·
`truncate`(0.1→104857)· `clamp-no-reject`(0xC5000001 が rc=0)·
`plus-2p31-allowed`(0x45000000 が rc=0)· `reject-boundary-tight`(-2048.0 を誤拒否)。
UNVERIFIED: A3 以降(係数・三buffer回転・FLRO・三経路 SHA)は未着手。

## 10. A3 実装記(Surya)

`coef.s`(手ARM64・整数のみ、FP レジスタ/FP 命令 零 = `coef_gate.sh` が二重走査で強制)+
`coef_gate.sh`(shell のみ)。用: `coef <c> <dt> <damping> <dx> <range>`(各 hex bits)
→ stdout `c_cur=2040216832 c_lap=10737419 c_prev=-966475008`。
rc: 0 成功 · 22 非 canonical(loud reject)· 23 不変式 `|c_lap|>2^29` · 17 用法/hex 不正。
公開記号 `_coef_from_header(x0=5*u32 LE, x1=3*i64 出力域) -> x0=rc`(A4 が呼ぶ)。
検査 = 5値の **byte 一致**のみ。実行時 f32 算 零・係数は即値。

**独立検算(bc 実走、§5 と別入力・同律の再導出)**:
```
16760439*13421773 = 224954807638347 ; /2^24 = 13408351 rem 6707531 < 8388608 ∴ 切捨 → m_dd=13408351
2^28-13408351 = 255027105 ; /16 = 15939194 rem 1 < 8 ∴ 切捨 → 15939194*2^7 = 2040216832  = c_cur ✓
2^27-13408351 = 120809377 ; /8  = 15101172 rem 1 < 4 ∴ 切捨 → 15101172*2^6 =  966475008  = -c_prev ✓
13421773^2 = 180143990463529 ; /2^24 = 10737418 rem 9395241 > 8388608 ∴ 切上 → 10737419 = c_lap ✓
```
丸め分岐は全て rem 対 half の比較で確定、境界 tie 無し。
骸(A9-4): 旧「Surya 独立検算 ∴ Vishnu §5 と一致」= **無効**。両者とも入力 sig を `16759927` と
誤讀していた ∴ 同一の穴を通った。検は算と失敗様式を共有してはならぬ、の実例として残す。
不変式: `10737419 <= 2^29 = 536870912` ✓(実装で実測検査、歯 `lap-over-2p29` が rc=23 を実証)。

**§3a erratum の独立検証(Agni 主張の再確認)**: `0x34000000` の e フィールド =
`(0x34000000>>23)&0xFF = 104` ∴ E = 104-127 = -23、仮数 = 2^23 ∴ 値 = `2^-23`。
§3a の「2^-21」は誤、Agni の訂正が **正**。真の 2^-21 = `0x35000000`(e=106)も実測一致。
∴ §3a の当該 3 行(`0x34000000`/`0x33FFFFFF`/`0x34000001`)は bit literal 側の誤記であり、
§2a の律は無変更。

赤歯(全 KILLED 実測): `c_cur-decimal-damping`(十進 0.999 起こし → 2040219776)·
`c_lap-truncate` · `c_prev-sign` · `dt-check-removed` · `damping-check-removed` ·
`range-check-removed` · `invariant-removed` · `lap-over-2p29`(rc=23 実証)。
UNVERIFIED: A4 以降(三 buffer 回転・`_fl_q30_wave_scalar` 呼出・FLRO 書出・三経路 SHA)未着手。

## 11. A4 実装記(Soma)

`fieldrun.s`(手ARM64・整数のみ、FP/SIMD レジスタ・FP 命令 零 = `fieldrun_gate.sh` が二重走査で強制)
+ `fieldrun_gate.sh`(shell のみ)+ `gen_fldj.sh` に `PAYLOAD`/`FILL`(payload bit 列指定)追加。
用: `fieldrun <in.fldj> <out.flro> [max_cells=16384]`(硬碼禁 = 既定 + 引数)。**arena は §16 で mmap 化**(A4 当時の「arena 実寸 16384 胞」= 第二の固定上限 = 誤、Chandi が blocker として摘出)。
連結: `q20_conv.s`/`coef.s` を `sed` で `_main` 改名した lib 版 + `../q30_wave/wave_scalar.s`
(記号衝突回避、新規 Rust/C/Swift/Python 零)。

**実走実測(§3c 一致)**: 2x2・`cur=[1.0,0,0,0]`・`prev=0`・`歩 1` →
```
FLRO header: 46 4c 52 4f 00 00 00 00 02 00 00 00 02 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
cells      : 1950460 20972 20972 0     size=48B (32 + 4*4)
```
= §3c 期待値と **byte 一致**(捏造無し、初回実走で一致)。§7 の「§3c 実装未実走」は **解消**。

三 buffer 回転(§2d)を x25=cur/x26=prev/x27=scratch で実装、`_fl_q30_wave_scalar` は x19-x28 を
保存する故 Step 境界を跨いで永続。`out` は常に第三領域 ∴ alias 無し。
sat = 全 tick 累計(`add x19,x19,x0`)、steps = tick 総数。
飽和の実測: 全胞 `0x44FFFFFF`(2047.99..)2x2 一tick → `sat=4`。
独立検算(bc): `(2040220160*2147483520+2^29)>>30 = 4080440077 > 2^31-1` ∴ 4 胞飽和 ✓。

rc: 0 · 2/3/4/5/6/7/8/9/10(header、A1 と同表)· 12 未知tag · 13 slot>=2 · 14 op切断 ·
15 len!=w*h · 16 open/read · 17 用法 · 18 出力書込失敗 · 20/21(payload 非有限/範囲外、A2 を
loud 伝播)· 22/23(非canonical/不変式、A3 を loud 伝播)。
`_open` の可変引数 mode は **stack 渡し**(arm64 macOS ABI)— レジスタ渡しは 000 権限の
ファイルを作る(本 lane 実測の死枝)。

赤歯(全 KILLED 実測): `rotation-2swap`(回転を cur⇄scratch の 2 者 swap へ弱化 → 4x4 3tick で乖離)·
`sat-dropped` · `flro-steps-zero` · `flro-sat-zero` · `flro-magic-be`(endian 反転)· `out-alias-cur`
(out=cur → q30 ABI 違反、出力乖離)· `payload-nan`/`payload-inf`(rc=20)· `payload-over-2048`(rc=21)·
`magic`/`version`/`dt-noncanonical`/`damping-decimal`/`writeraw-slot2`/`writeraw-len-ne`/
`nslots-lt2`/`unknown-tag`。

UNVERIFIED: A5(`--neon`)· A6(`--metal`)· A7(実 journal ≥200 step の三経路 SHA 一致)·
A8(§3d 残余変異の一括表)· A9(上流 D1/G2 修正)は **未着手**。
FLRO cell の endian は host LE をそのまま書く ∴ big-endian host での歯は **未検**(現行 arm64 のみ)。

## 12. A5 実装記(Vayu)

`fieldrun --neon`(前置旗のみ): 用 = `fieldrun [--neon] <in.fldj> <out.flro> [max_cells=16384]`。
既定 = scalar。backend は局所 slot `[sp,#128]` の**函数ポインタ**(`_fl_q30_wave_scalar` 或
`_fl_q30_wave_neon`)へ `blr` で分岐。**no-fallback**: `--neon` は neon 記号を直に指す ∴
記号欠落 = **link 不成立**(実行体を作らぬ)。黙って scalar へ落ちる経路は**存在せぬ**
(歯 `neon-symbol-gone` rc=1 · 歯 `default-is-scalar` = 壊れた neon を連結しても旗無しは scalar
出力に byte 一致し `--neon` は乖離)。`build.sh` に `wave_neon.o` を連結(手ARM64のみ)。

**FP/SIMD 走査の範囲(除外理由)**: `fieldrun_gate.sh`/`neon_gate.sh` の FP/SIMD 走査は
**`fieldrun.s` に限る**。NEON 命令は `../q30_wave/wave_neon.s`(既存・凍結門下)に閉じ、
fieldrun 側は記号呼出のみで vector レジスタを一切触れぬ ∴ 走査対象から backend 実装 file を
除外する。之が緩和ではない証拠 = `fieldrun.s` の走査は無変更で緑、かつ neon 経路の正しさは
scalar との FLRO byte 一致で採点される(下記門)。

**門 `neon_gate.sh`(shell のみ)実測**: scalar と `--neon` の FLRO **byte/SHA-256 一致**:
```
3c-2x2-one-tick   fa6cbcebf5db39f6f2d2cae9114d9ee60f96ad025009a6770e96d7387ede0a35
4x4-3tick         33074917e723d60a4434ddf1badb9844faa734beb17204f0b8bdb712c16c3f7f   (A9-4 再凍結値・旧 a85a4cc0… は誤定数時代の骸)
saturating-vector ec09887b045dd627e746c0b7aaa3fc830e6368d41bd973c88e8287e22e38b744
8x8-3tick-vecpath 99771ec140caede7098ddb9ff1cc36f05ee0a9cb8ea418cbd4df9bfdc7163152
9x9-3tick-vecedge 70a04aa34c5c3f64d55a3c43215c5c896f11a13a776ccae92e60a19a9b230afb
```
2x2/4x4 は w<6 ∴ neon の**端胞 scalar 経路のみ**を踏む。4 胞 vector 経路を実際に踏ませる為に
非一様場(胞毎相異)の 8x8(vector 内部)と 9x9(vector 境界 `x+4==w` が露出する幅)を追加。

赤歯(全 KILLED 実測、各々単独): `neon-symbol-gone`(link rc=1)· `neon-round-half`(丸め半 2^29→2^28)·
`neon-tail-drop`(`sqxtn2` 削除 = 上位2 lane 落ち)· `neon-lane-cross`(`ext #8`→`#4`)·
`neon-bound-off-by-1`(vector 条件 `x+5<=w`→`x+4<=w`、**9x9 でのみ露出** = 8x8 では survive した
死枝の因)· `neon-sat-drop`(vector 半の飽和計上落ち)。

UNVERIFIED: A6(`--metal`)· A7(実 journal ≥200 step の三経路 SHA 一致)· A8 · A9 は未着手。
`--neon` は arm64 macOS 実機のみ実測。

## 13. A6 実装記(Varuna)

`fieldrun --metal <metallib>`(第三 backend)。用 =
`fieldrun [--neon | --metal <metallib path>] <in.fldj> <out.flro> [max_cells=16384]`。
**metallib path = 引数(硬碼零** — 門が `fieldrun.s` に `metallib` 文字列が無い事を走査)。
旗解釈 → `_fl_wave_metal_init(x0=path)`(≠0 = **rc=24 loud**)→ backend 函数ポインタ `[sp,#128]` に
局所 thunk `_fr_metal_call` を据える。thunk = `x4=0(offset)` `x5=1(reps)` 固定で
`_fl_q30_wave_metal` を呼び、**戻 -1 = rc=25**(CPU fallback **無し**、黙落経路は構造的に不在)。
連結 = `../q30_wave_metal/metal_bridge.o` + `-lobjc -framework Metal -framework Foundation`。
`fieldrun.s` は整数のみ(FP/SIMD 走査 無変更で緑・GPU 碼は `wave_q30.metal` に閉じる)。

**門 `metal_gate.sh`(shell のみ・実機 GPU 実走)**: 三経路 FLRO **byte/SHA-256 一致**:
```
3c-2x2-one-tick   fa6cbcebf5db39f6f2d2cae9114d9ee60f96ad025009a6770e96d7387ede0a35
4x4-3tick         33074917e723d60a4434ddf1badb9844faa734beb17204f0b8bdb712c16c3f7f  (A9-4 再凍結値・旧 a85a4cc0… は誤定数時代の骸)
saturating-vector ec09887b045dd627e746c0b7aaa3fc830e6368d41bd973c88e8287e22e38b744
8x8-3tick-vecpath 99771ec140caede7098ddb9ff1cc36f05ee0a9cb8ea418cbd4df9bfdc7163152
9x9-3tick-vecedge 70a04aa34c5c3f64d55a3c43215c5c896f11a13a776ccae92e60a19a9b230afb
40x40-4tick-gpu   6590b38f06421e17e1ed05d3e5ffb44896b4eae155df695ac80763e4873bae6a  (A6 追加=1600胞 非一様、GPU thread 多数)
```
GPU 実走証跡 = bridge の fd1 出力 `metal command status: 4` を門が実測 grep。

赤歯(全 KILLED 実測): `metallib-absent`(rc=24・**出力 file を作らぬ** = scalar 結果を返さぬ)·
`metal-reps-0`(rc=25)· `metal-offset-3`(4B 非倍数 offset → scalar と乖離)·
`metal-fail-silent`(-1 検査削除、metallib 欠落下 rc=24)· `metal-init-silent`(init 検査削除、rc=25)·
`metal-both-silent`(両検査削除 → rc=0 だが **出力≠scalar** ∴ 黙って CPU 相当値を返す経路は無い)·
`metal-flro-magic-be`(出力 endianness 反転)· `metal-no-wait`(bridge の `waitUntilCompleted`
選択子を潰す → rc=134 unrecognized selector)。
緑歯(不変量として採点): `default-is-scalar`(旗無し = Metal 未接触)· `metal-state-reuse`
(同一 process 内 二重 init でも byte 一致)· `metal-offset-inv`(offset=4/28 は scalar と byte 一致)。

**死枝(因つき)**:
- `metal-reps-2`: reps>1 = 同一入力の再走 ∴ out 不変、sat も dispatch 毎に零化される
  (`metal_bridge.s:333`)∴ **FLRO から観測不能** → 偽の歯。観測可能な改変 = `reps=0` に置換。
- `metal-offset-4`: byte_offset は bridge の **staging buffer 内部の置き場**で cur/prev/out 全てに
  同一適用 ∴ 4B 倍数では**不変量**(`../q30_wave_metal/gate.sh` が 0/4/28 を同一結果として採点済)。
  歯として立たぬ故 緑の不変量検査へ降格し、4B 非倍数(3)を歯に据えた。
- 場の大型化: 前任 Vayu の教訓(小場は本体経路を踏まぬ)を継承し 40x40x4tick を追加。
  8x8/9x9 の SHA は A5 と同値(場生成を変えぬ事で回帰性を保った)。

既存門の緑維持(本 lane 実測): `fieldrun/gate.sh`(A1-A5 全歯 + A6)rc=0 ·
`../gate.sh`(fieldlang-asm)rc=0 · `../q30_wave/gate.sh` rc=0 · `../q30_wave_metal/gate.sh` rc=0。
`fieldrun_gate.sh`/`neon_gate.sh` の変異体 link 行に metal 連結を足した(**採点律は無変更**)。

UNVERIFIED: A7(実 journal ≥200 step の三経路 SHA 一致)· A8 · A9 は **未着手**。
`--metal` は arm64 macOS 実機 GPU のみ実測。big-endian host は未検。

## 14. A7 実装記(Prithvi)

`gen_fld_a7.sh`(**.fld 源**生成器・shell のみ)+ `a7_gate.sh`(門・shell のみ)。新規 asm 零
(A6 の実行体を無改造で使用)。`gate.sh` 末尾に `./a7_gate.sh` を追加。

**独立性**: `gen_fld_a7.sh` は `fieldrun`/`fldj_parse.s`/`fieldc` の実装を読まぬ。参照は
`../CONTRACT.md` の文法(`界`/`寫`/`歩`)のみ、出力は人可読 `.fld` テキスト。
門が非註釈行を走査し実装記号(`fieldrun|fldj_parse|fieldc|q20_conv|coef|wave_|.s|.fldj`)零を強制。
初期場は f32 の指数/仮数欄を**整数算で組む**(host 浮動小数 零)。e ∈ [118,137] ∴ |v| ∈ [2^-9,2^10] < 2048
(§2a の rc=21 圏外)· 胞毎に指数・仮数・符号を散らす = **非一様**。

**実走実測(生)**:
```
green   generator-independence     gen_fld_a7.sh の非註釈行に実装参照 零
green   fieldc-compile             fld=11306B fldj=4158B
raw     sha(a7.fld)                b83c386ac4d1bc5a7a0be1dc65954754ab0d152f04d5098c2bc7e1e13a8a51b1
raw     sha(a7.fldj)               11e973d2722dfeb8a38b1774266380f2339f9be7a6a932797113d4daadc86925
raw     fldj-header(48B)            46 4c 44 4a 01 00 00 00 20 00 00 00 20 00 00 00 00 00 80 3f cd cc cc 3d 77 be 7f 3f 00 00 80 3f 00 00 00 00 00 00 00 00 00 00 80 3f 02 00 00 00
green   real-journal-32x32-200step sha=63f868bc673bbd4f7a9cf943f2b48f7f46fd416d0e55d145fa9e3cb97c1a24c9 (scalar==neon==metal, byte 一致)
green   gpu-evidence               metal command status: 4
raw     flro-header                46 4c 52 4f 00 00 00 00 20 00 00 00 20 00 00 00 c8 00 00 00 00 00 00 00 af 00 00 00 00 00 00 00
green   steps>=200                 steps=200 sat=175 (FLRO off16/off24)
green   nonzero-evolution          sha0=7ec0c4c809a0d3d7c3335444197f0b00f383fa4351349d8ea5059c1a8aa29d13
green   nonzero-evolution          sha200=63f868bc… diff_bytes=4083/4128
raw     cells0-first8              2048 4294591538 62639496 4294961970 908908 78316456 4294954185 1084027
raw     cells200-first8            175994898 173395924 170886056 168515726 166302587 164415367 162829334 161576353
KILLED  tooth:step-count-199       sha=4f9e6eed… != sha200
KILLED  tooth:init-field-phase1    sha=8d029911… != sha200
KILLED  tooth:journal-1byte        off=100 64->65 sha=446ea00a… != sha200
KILLED  tooth:max-cells-arg        rc=8 (max_cells=512 < 1024 胞)
green   max-cells=1024-exact       sha=63f868bc… (同一)
gate: fieldrun A7 (real fieldc journal, 32x32, 200 step, 三経路) OK
```
`max_cells` は門引数(既定 4096・`MAXCELLS` 可変)∴ 硬碼零。1024 丁度で同一 SHA・512 で rc=8。

**時間発展が真である証**: 0 step 版(同一 `.fld`、`歩 0`)FLRO の cells = 初期場そのもの
(`steps=0 sat=0`)。200 step 版と 4128B 中 **4083B 相違**。初頭 8 胞も全く別値 ∴ 恒等写像でない。
`steps=200`(off16)· `sat=175`(off24)が wire に載る事も raw で確認(§2c 適合)。
`sat>0` = kernel 内部飽和が 200 tick 中 175 胞回起きた実測値(§2a の入力 reject とは別事象、
三経路とも同値ゆえ parity は保たれる)。

**Vishnu 死枝の反証**: round12 序盤 Vishnu は「FLDJ decoder 不在 ∴ 三経路 ≥200step 一致は不成立」と
REJECT した。本 atom で **decoder は実在し(A1..A4)**、実 `fieldc` 出力 journal 一本から
scalar/neon/metal が 200 step 後に **FLRO byte 一致**(sha `63f868bc…`、実機 GPU 実走)。
∴ 当該死枝は **反証済**。ただし反証されたのは「decoder 不在」の前提であり、
Vishnu が同時に指摘した G1(FFT `wave_step` との bit 一致は主張不可)は**依然有効**
— 本一致は `wave_step_reference` 意味論の内部整合であり、上流 product 経路との parity ではない。

UNVERIFIED: A8(§3d 変異一括表)· A9(上流 D1/G2 修正)= 未着手。
FLRO cell endian は host LE ∴ big-endian host 未検。`--metal` は arm64 macOS 実機のみ。

## 15. A8 実装記(Akasha)

`teeth_kill.sh`(shell のみ・新規 asm 零)。用 = `teeth_kill.sh [--new-only]`。
段1 = §3d **残余変異**を `fieldrun.s` へ **各々単独** sed 適用 → 再 as/ld → 同一入力で無変異と比較。
段2 = `gate.sh`(A1..A7)を実走し `KILLED`/`green` 行を採取 → **一枚表**へ集約。
`gate.sh` からは **呼ばぬ**(段2 が `gate.sh` を呼ぶ故、連結すると無限再帰)∴ A8 は独立門。

**採点律**: 無変異 fieldrun の同一入力 rc/FLRO byte を基準とし、変異体が **観測上乖離**(rc 相違
或 出力 byte 相違)すれば KILLED。基準と同一 = SURVIVED = 門赤。

**段1 実測(生)**:
```
raw     baseline rc: truncation=14 trailing=14 bad-tag=12 bad-len=15 wh-32bit=8
KILLED  truncation-check-removed   baseline rc=14 -> mutant rc=12  fieldrun reject code=12
KILLED  trailing-byte-check-removed baseline rc=14 -> mutant rc=12  fieldrun reject code=12
KILLED  bad-tag-check-removed      baseline rc=12 -> mutant rc=0
KILLED  bad-len-check-removed      baseline rc=15 -> mutant rc=12  fieldrun reject code=12
KILLED  wh-32bit-multiply          baseline rc=8 -> mutant rc=15  fieldrun reject code=15
green   backend-forced-failure     rc=24 出力 file 零 (metal NSError: library not found)
KILLED  metal-init-check-removed   baseline rc=24 -> mutant rc=25  metal NSError: library not found
```
段2 合計: **KILLED=77 green=36 SURVIVED=0**、`gate: fieldrun A8 (teeth_kill 一括表) OK`。

**自省(隠さぬ)— 歯の強度に段差がある**:
`truncation` / `trailing-byte` / `bad-len` の三変異は rc=0 へは**落ちぬ**。当該検査を潰しても
**別の検査**(未知 tag rc=12)が捕える = 多重防御。∴ 之等の歯が実証するのは
「当該検査が唯一の防壁」ではなく「当該検査を外すと **観測上の rc が変わる**(誤分類が起きる)」。
不正入力が rc=0 で通る歯は `bad-tag-check-removed`(rc=0)のみ。**弱い主張と強い主張を混ぜぬ**為に
本節に明記する。真に強い版(rc=0 まで抜ける)を得るには複数検査の同時除去が要り、
之は「各々単独適用」の要件と衝突する ∴ **単独適用を優先**した。

**死枝(因つき)**:
- `rc=7`(`w*h` u64 溢れ)の歯 = **到達不能**。wire の w,h は u32 ∴ 積 < 2^64、`umulh` は常に 0。
  §8(Brahma)の判定を踏襲し、`wh-32bit-multiply`(64bit→32bit 弱化 + 65536^2 wrap)を代替の歯に据えた。
- `mul` 除去のみの歯(`cbnz x9, Lrej_mul` を nop 化)= 上と同理由で観測不能 ∴ 立てず。
- FLRO cell の big-endian 歯 = host が LE のみ ∴ **実測不能**(A4 からの継続 UNVERIFIED)。

既存門の緑維持(本 lane 実測 rc=0): `fieldrun/gate.sh` · `../gate.sh` · `../q30_wave/gate.sh` ·
`../q30_wave_metal/gate.sh`。

UNVERIFIED: A9(上流 D1 `寫 n==w*h` 生産側検査 + G2 十進注釈訂正)= **未着手**。

## 16. A8b 実装記(Akasha・Chandi 審 blocker の閉塞)

**摘出(Chandi, `CHANDI_REVIEW.md`, verdict=赤)**: §2e/§11 は「max_cells = 既定 + 引数(硬碼禁)」と
謳うが `fieldrun.s` に **第二の固定上限 16384**(静的 `.space` arena の実寸)が在り、
`129x128 = 16512` 胞・引数 `16512` が **rc=8** で落ちた。∴ 契約と実装の齟齬。

**どちらが誤りか**: **実装が誤**。契約(硬碼禁)が法であり、静的 arena は之に違反していた。
§11 の「arena 実寸 16384 胞」という記述も、固定上限を正当化する形で書かれていた点で **誤**
(本節で訂正済)。§2e の「`w*h <= 実装 arena 上限`(定数、硬碼禁=既定値+引数)」は
「上限 = **引数由来の max_cells のみ**」と読むのが正。

**修正**: 静的 `_bufA/_bufB/_bufC`(各 64KiB)を撤去し、`n` 確定後に
`mmap(NULL, n*4, PROT_READ|WRITE, MAP_ANON|MAP_PRIVATE, -1, 0)` を **三本**取る
(`MAP_ANON` ∴ 全域 0 = §2d の初期条件を保つ)。判定は `n > max_cells` の **一箇所のみ**。
mmap 失敗 = **rc=19 loud**(新 rc)。

**実測(生)**:
```
green   arena-arg-16512        rc=0 size=66080 (max_cells=16512、旧固定 16384 超)
KILLED  arena-arg-under        rc=8 fieldrun reject code=8            (引数 16511 < 16512 胞)
KILLED  arena-cap-hardcoded    rc=8 output differs from reference     (`ldr x9,[sp,#48]`→`mov x9,#16384` = 硬碼再導入)
```
回帰零(SHA 不変、A8b 時点の実測): `4x4-3tick a85a4cc0ee31…` · `real-journal-32x32-200step ead5a8fff10e…`。
【A9-4 以降の骸】上記二値 = **誤定数時代の値**。真係数への訂正で意図的に壊し、
新値 `33074917e723…` / `63f868bc673b…` へ再凍結済(G12 参照)。
全門 rc=0: `fieldrun/gate.sh` · `teeth_kill.sh`(KILLED=79 green=37 SURVIVED=0)· `../gate.sh` ·
`../q30_wave/gate.sh` · `../q30_wave_metal/gate.sh`。

**残る固定量(隠さぬ)**: 入力 file 緩衝 `_filebuf = 4 MiB`(静的)∴ 4 MiB 超の `.fldj` は読めぬ。
之は arena とは別の資源であり、**本 atom の範囲外**(未修正・**UNVERIFIED** な上限)。
引数化すべきか否かは A9 以降の判断に委ねる。16512 胞の journal = 66110 B ∴ 現要件には十分。

**G1 の存続(Chandi 再確認)**: product 執行路は `world.rs:93-99` → `plane.rs:79-90` `wave_step`(FFT)
∴ fieldrun の Q30 reference stencil とは **別法**。三経路 byte 一致が示すのは
`wave_step_reference` 意味論**内部**の整合のみであり、**product parity は非証明**。之を契約に残す。

## §15 上流 `wave_step_reference` parity 許容誤差(**測定前に定む**・Ganga)

射程 = `../../field/src/plane.rs:93-113`(5点 stencil, f32)対 fieldrun(Q20 cell / Q30 係数)。
**bit 一致は要求せぬ**(固定小数 対 浮動小数 = 別数系)∴ 誤差上限を先に導出して封ずる。

導出(手算・実測前):
- Q20 量子化 ulp `u = 2^-20 = 9.5367431640625e-7`。
- 1 tick の增幅率 `G = (|c_cur| + 4|c_lap| + |c_prev|)/2^30 = (2040216832 + 4*10737419 + 966475008)/2^30
  = 3049633_2... 正確に = 3049638516/2^30 = 2.84040...` (≤ 2.8405 と丸めて用う)。
- 1 tick で新たに入る誤差 `e1 ≤ u`(Q30 積 3 本の丸め ≤ 3*(u/2) と f32 側丸めを一括して u で覆う)。
- ∴ N tick 後の絶対誤差上限 `E(N) = u * Σ_{i=0}^{N-1} G^i`。
```
E(1) = 9.537e-7
E(2) = 3.663e-6
E(3) = 1.406e-5
E(10)= 3.36e-3    (G^N 爆発 ∴ 長走 tick で parity 主張は無意味 — 之も先に認む)
```
**判定律**:
- PASS 条件 = 全胞で `|q20_value - ref_f32| ≤ E(N)`(N = 実行 tick 数)。
- 飽和(sat≠0)を含む走は **parity 対象外**(Q20 飽和は f32 に無き非線形)。
- 相対誤差は `max|ref|` を分母とする一つの値のみ報告(胞毎相対は 0 割 ∴ 用いぬ)。
- 本節は **測定前** に凍結。以後 **基準の後出し改訂を禁ず**。改訂は骸+因を残して別節に。
- **G1 は本節で閉じぬ**: product 執行路 `plane.rs:79-90` = FFT 分光 ∴ 本節が緑でも product parity は非証明のまま存続。

### §15b parity 実測(Ganga・§15 の基準は測定前に凍結済)

経路: `gen_fldj.sh` → `./fieldrun`(Q20/Q30 手ARM64)対 `cargo run -p field --example replay_fldj`
の **`FLDJ_REF=1` 分岐**(既存 example の最小改・新規 file 零)= `plane::wave_step_reference` で Step 執行 → slot0 f32 を吐く。
比較 = `paste`+`awk`(shell のみ)。q20 は `od -An -v -td4 -j32`(`-v` 必須: `*` 圧縮が偽差を生む=骸)。

```
t1 cells=4    1tick  MAXABS=4.563481e-07 BOUND=9.537000e-07 MAXREF=1.860100e+00 MAXREL=2.453352e-07 PASS
t2 cells=16   3tick  MAXABS=9.769963e-14 BOUND=1.406000e-05 MAXREF=3.439523e+00 MAXREL=2.840499e-14 PASS
t4 cells=64   3tick  MAXABS=1.907348e-06 BOUND=1.406000e-05 MAXREF=6.443011e+00 MAXREL=2.960336e-07 PASS
t5 cells=256  6tick  MAXABS=5.722046e-06 BOUND=2.000000e-04 MAXREF=8.524576e+00 MAXREL=6.712411e-07 PASS
t6 cells=1024 10tick MAXABS=1.049042e-05 BOUND=3.360000e-03 MAXREF=8.672419e+00 MAXREL=1.209630e-06 PASS
```
全走 `sat=0`(§15 の飽和除外に該当せず)。傾向: 誤差は tick 数と共に緩やかに増(1.9e-6→1.05e-5)、
上限 `E(N)` の増加(G^N)より遥かに遅い ∴ 誤差は **系統的増幅ではなく量子化雑音の準ランダム蓄積**。
t2 の 9.8e-14 = 一様場 ∴ lap=0 で誤差経路が退化した特異例(裏付けにはならぬ・因を明記して残す)。

**結論**: 真係数下で fieldrun は `wave_step_reference` の意味論に **§15 の許容内で一致**(bit 一致にあらず)。
**G1 は依然存続**(product = `plane.rs:79-90` FFT 分光)。長走(A7 200tick)の parity は
`E(200)` が無意味に発散する ∴ **主張せぬ・未測**(UNVERIFIED)。

## §16 A9-5 残務閉(Ganga)

- **max_slots 引数化**: `fieldrun.s` の `mov x10,#1024` → `ldr x10,[sp,#160]`。frame 160→176B、
  slot160 = max_slots(既定 1024・**第6引数**で上書き)。`fieldrun <in> <out> [max_cells] [max_bytes] [max_slots]`。
  歯: `slots-arg-4`(引数4 と既定 byte 一致)· `slots-arg-under rc=10` · `slots-arg-2000`(旧硬碼超 rc=0)·
  `slots-default-cap rc=10` · `slots-cap-hardcoded`(硬碼再導入 = KILLED)。
- **`../gate.sh` 偽赤**: `$FIELDC` 不在時に `rc=3`(SKIP-ENV)で落とし、検査失敗 `rc=1` と区別。実測 `missing rc=3` / `normal rc=0`。
- 残る固定量: `.bss` 内部固定は fieldrun.s に理由明記済(範囲外)。

## §17 長走 parity 判定基準(**測定前に凍結**・Tulasi)

**因**: §15 の最悪上界 `E(N)=u·ΣG^i` は `G=2.84>1` ∴ N=200 で無意味に発散し、
長走では **何も判定し得ぬ**(骸: §15 の基準を長走に流用 = 死枝、因 = 上界が真値でなく
「増幅が毎tick最悪方向に揃う」最悪世界を測る為)。∴ 別基準を新たに凍結する。

### 統計量(定義)
`d_i = q20_i - ref_i`(全胞 i=1..M、slot0、200step 後)、`ref` = `wave_step_reference` f32。
- `RMSREL(N) = sqrt(mean(d_i^2)) / sqrt(mean(ref_i^2))` — **主基準**
- `MEDREL(N) = median(|d_i|) / sqrt(mean(ref_i^2))` — 副(外れ値耐性)
- `P95REL(N) = p95(|d_i|) / sqrt(mean(ref_i^2))` — 副(裾)
分母 = 場の RMS 振幅(単一の大域量)∴ 胞毎 0 割を避ける(§15 と同思想)。

### 何を保証し、何を保証せぬか
- 保証する: 場**全体**として Q20 執行が reference 意味論から離れる**典型的**距離の大きさ。
- 保証**せぬ**: (a) 任意個別胞の誤差上限(統計量は最悪値を覆わぬ)、(b) bit 一致、
  (c) product 経路(FFT・G1 存続)、(d) 飽和(sat≠0)を含む走。

### 閾(測定前に定む)
`N ∈ {1,10,50,100,200}` で測る。
- **PASS = `RMSREL(200) ≤ 1e-3`**。
  導出: Q20 ulp `u=2^-20≈9.54e-7`、場 RMS 振幅は O(1..10) ∴ 1 step の相対雑音 ~1e-6 級。
  無相関蓄積なら `~sqrt(200)·1e-6 = 1.4e-5`、完全相関(線形)蓄積でも `~200·1e-6 = 2e-4`。
  ∴ 閾 1e-3 = **線形蓄積すら許す上で 5 倍の余裕**、且つ指数増幅は一切通さぬ
  (G^N なら 200 step で天文数字)。**緩くして通す為の選択にあらず**: 之を超えたら
  「量子化雑音の蓄積では説明できぬ」= 真に赤。
- **分類律**(判定は閾のみに依らず、増加則で行う):
  `r = (RMSREL(200)/RMSREL(10))^(1/190)` = 実効毎tick増幅率。
  - `r ≤ 1.001` かつ `RMSREL(200) ≤ 1e-3` → **緩慢蓄積**(量子化雑音)
  - `RMSREL(200)/RMSREL(10) ≲ sqrt(200/10)=4.47` → **無相関(ランダムウォーク)蓄積**
  - `r ≥ 1.01`(=200step で 7 桁級)→ **指数発散** ∴ 「fieldrun は長走で reference から
    乖離する」と明記し、`RMSREL(N) ≤ 1e-3` を満たす最大 N を実用上限として数値で示す。
- 飽和走(FLRO off24 の sat≠0)は本節の対象外 ∴ sat を毎走 raw で貼る。
- 本節は測定前に凍結。**後出し改訂を禁ず**(改訂は骸+因を残し別節に)。
- 測定法: `./fieldrun` 対 `FLDJ_REF=1 replay_fldj`、q20 読出は `od -An -v -td4 -j32`
  (**`-v` 必須** — `*` 圧縮が偽差を生む = §15b の骸を継承)。

### §17b 長走 parity 実測(Tulasi・§17 の基準は測定前に凍結済 `3687b21`)

器 = `fieldrun/longrun_parity.sh`(shell のみ・新規 Rust/C/Swift/Python 零)。
`gen_fld_a7.sh` に `EBASE/ESPAN`(指数欄基点・幅、既定 118/20 = **出力不変**、sha `b83c386a…` 一致確認)
を引数化(硬碼除去)= 非飽和振幅の走を作る為。

**(1) 既定 A7 場(EBASE=118 ESPAN=20)は 200step で飽和 ∴ §17 の対象外**(生):
```
N      sat    RMSREF        RMSREL        MEDREL        P95REL
1      47     7.499309e+02  2.897026e-01  1.271681e-09  1.208098e-07
10     175    1.135583e+03  8.137315e-01  7.699172e-04  1.823790e+00
50     175    3.767576e+02  6.817813e-01  3.619087e-01  1.481627e+00
100    175    3.229889e+02  5.962991e-01  4.922324e-01  1.046310e+00
200    175    3.160220e+02  5.590654e-01  5.297206e-01  8.034321e-01
```
**悪い事は悪いと書く**: 既定 A7 振幅(|v| 最大 2^10)では Q20 が **1 step 目から飽和**(sat=47→175)
∴ fieldrun 出力は f32 reference と **相対 O(1) で全く別物**。三経路 byte 一致(A7 門)は
之を検知せぬ(三経路とも同じ飽和をする)。**A7 門の緑は f32 意味論との一致を意味せぬ**。

**(2) 非飽和振幅(EBASE=118 ESPAN=6, |v|∈[2^-9,2^-4], sat=0 全走)= §17 の射程**(生):
```
N      sat    RMSREF        RMSREL        MEDREL        P95REL        cells
1      0      8.355100e-02  7.986244e-06  5.707139e-06  1.551628e-05  1024
2      0      1.146814e-01  8.779732e-06  5.863331e-06  1.753924e-05  1024
5      0      1.613039e-01  1.213689e-05  8.221767e-06  2.318723e-05  1024
10     0      1.333072e-01  2.182601e-05  1.475504e-05  4.296566e-05  1024
20     0      1.017481e-01  3.757314e-05  2.504321e-05  7.293287e-05  1024
50     0      6.503359e-02  6.354181e-05  4.433670e-05  1.216681e-04  1024
100    0      6.115782e-02  7.469352e-05  5.080122e-05  1.442414e-04  1024
200    0      6.060039e-02  1.101092e-04  7.020222e-05  2.199506e-04  1024
```
- **PASS**: `RMSREL(200)=1.101e-4 ≤ 1e-3`(§17 の閾)。
- **判定 = 緩慢蓄積(量子化雑音のランダムウォーク)、指数発散に非ず**。
  根拠(独立2路): (a) `RMSREL(200)/RMSREL(10)=5.04 ≲ sqrt(20)=4.47` = 無相関蓄積律。
  (b) 全8点の log-log 最小二乗 `RMSREL = 6.60e-6 · N^0.538`(p=0.538≈1/2)。
  指数律 `G^N`(G=2.84)なら 200step で 10^88 級 ∴ **7桁以上の乖離で棄却**。
- **§17 の分類律の欠陥(自認・骸)**: 毎tick率 `r=5.04^(1/190)=1.0085` は「r≤1.001=緩慢」を
  満たさぬが「r≥1.01=発散」も満たさぬ = **穴**。因 = 冪則 `N^0.5` を毎tick一定率で表そうとした誤り。
  訂正 = 判別は **冪指数 p**(p≈0.5 無相関 / p≈1 完全相関 / 指数則は log-log で直線に乗らぬ)で行う。
  閾 1e-3 と主基準 RMSREL は改訂せず(後出し禁)。
- **実用上限**(冪則外挿・EXTRAPOLATED, 未実測):`RMSREL=1e-3` 到達は `N≈1.1e4 step`。
  **但し真の上限は誤差でなく飽和**: 振幅が Q20 域(|v|<2^11)を超えれば 1 step で破綻(上記(1))。
  ∴ 実用則 = 「**sat=0 を保つ限り** 200step の相対誤差は 1.1e-4 級、~10^4 step まで 1e-3 未満」。
- 保証せぬ物(§17 の通り): 個別胞の最悪誤差 · bit 一致 · product(FFT)経路(**G1 存続**) · 飽和走。

### §17c A7 門の飽和限定(Chandi・門碼 `a7_gate.sh` `69f21f2`)

**因**: §17b(1) の実測で、既定 A7 振幅は **1 step 目で飽和**(sat=47→175)∴ fieldrun 出力は
f32 reference と相対 O(1) の別物。然るに A7 門は三経路 byte 一致のみを見る ∴ 緑のまま。
**「A7 緑」を「f32 意味論一致」と読む誤読の余地が門の出力自体に在った** = 誇大主張の温床。

**選択 = ②(SAT ラベル + 非飽和 vector 追加)。①(既定振幅を下げて golden 再凍結)は死枝**:
- 因1: 飽和挙動それ自体が三経路一致の検査対象として価値有り(飽和は分岐を伴う ∴
  scalar/neon/metal が同一に飽和する事は非自明な性質)。①は之を捨てる。
- 因2: 既存 golden `63f868bc…` の再凍結 = 回帰基準の破壊。②は既定走を不変に保つ ∴ 回帰零。
- 因3: 門の欠陥は「振幅が悪い」ではなく「**主張の範囲が明示されぬ**」事 ∴ 治療は表示であり値の変更に非ず。

**規律(sat = 門の一級市民)**:
- 各 vector は飽和級 `SAT|NONSAT` を**事前宣言**し、FLRO off24 の実測 sat と照合。
  食い違いは**両方向で赤**(NONSAT 宣言が飽和 = 退行検知 / SAT 宣言が sat=0 = 宣言腐敗)。
- `sat>0` の走は門の出力行が `SAT` で始まり、`f32 意味論一致を主張せぬ` と明記する。
  当該走が主張するのは**三経路 byte 一致のみ**。
- 非飽和 vector(`NS_EBASE=118 NS_ESPAN=6`・32x32・200step)を新設。之が
  **f32 意味論 parity(§17b(2): RMSREL(200)=1.101e-4 ≤ 1e-3)を主張し得る唯一の A7 走**。
- sat 読出も `od -An -v`(`-v` 必須 = §15b の骸継承)。

**新 golden(実測凍結・Chandi)**:
```
nonsat-journal-32x32-200step  sha=b44380767643728b27d526c765d62e9e8841baccf9532dbd9c3dd7159394f7c3
  (scalar==neon==metal, byte 一致・実機 GPU `metal command status: 4`・sat=0・steps=200)
sha(a7ns.fld) = c5a76e41842b7acfa8889aeba5636ca01adc1e627cfec2cc1cd82b99ce3afbe0
不変 golden: A7 既定 63f868bc673b… / 4x4 33074917e723…(②選択 ∴ 再凍結せず)
```

**§14 及び round12 総括への限定追記**: 「実 fieldc journal 一本から三経路が 200 step 後に
FLRO byte 一致(`63f868bc…`)」は**真**であるが、当該走は **sat=175 の飽和走** ∴
之を以て「fieldrun が f32 `wave_step_reference` 意味論を 200 step 保つ」とは**言えぬ**。
その主張の根拠は非飽和 vector(`b443807676…`・sat=0)と §17b(2) の統計のみ。
G1(product=FFT との parity)は依然**存続**。

**合流記(Lakshmi・`r12/integrated`)**: 本節の総括向け訂正は旧 `ROUND12_SAT_ADDENDUM.md` に別置していたが、
`ROUND12.md` 本文へ合流させ当該 file を廃した(二重管理終)。訂正の原典は本節 §17c と §17b。

### §15c 基準の算術訂正 + fixture 凍結(Chandi・審 Jyestha二番 RED + Rahu 独立検算に応ず)

**(あ) G の訂正(骸+因)**
```
真: SUM = |c_cur| + 4|c_lap| + |c_prev| = 2040216832 + 42949676 + 966475008 = 3049641516
    G   = SUM/2^30 = 2.840199988335371          ← ./coef 実走出力から導出
骸: §15 は SUM=3049638516 · G=2.84040 と書いた = **誤和**(下3桁 641516→638516 の書き損じ)
```
**「旧誤係数由来か単なる誤記か」の実碼判定 = 誤記**。因: 旧誤係数(`c_cur=2040220160`,
`c_prev=-966478272`, A9-4 で骸化)を入れても `G=2.840206` ∴ **どの係数でも 2.8404 は出ぬ**。
∴ 旧値は係数汚染の残滓に非ず、算術の誤り。

**BOUND の再導出(E(N)=u·Σ_{i<N}G^i, u=2^-20)**:
```
E(1)=9.536743e-07 E(2)=3.662300e-06 E(3)=1.135534e-05 E(6)=2.715191e-04 E(10)=1.770156e-02
```
§15b の BOUND 表は之と不整合(t4 3tick=1.406e-05 · t5 6tick=2.000e-04 · t6 10tick=3.360e-03 は
式から出ぬ数)∴ **§15b の PASS 判定は根拠を失う**。測定値自体は棄てぬが、
**§15b は「根拠不整合 ∴ 緑にあらず」と本節で明示的に格下げする**(閾は緩めぬ — 訂正後の
BOUND は t4 で **より厳しく**(1.406e-05→1.135e-05)、t5/t6 では緩い。後出しの都合合わせに非ず)。

**(い) fixture 凍結** — §15b の t1..t6 は生成律が commit されず **再現不能 ∴ UNVERIFIED**(審の指摘=真)。
新門 `parity_gate.sh`(shell のみ)が fixture を**決定論的に生成**し、各走の `sha(fldj)` を吐く。
BOUND は契約本文でなく **`./coef` 実走出力から毎回再導出**(検が算と失敗様式を共有せぬ独立路)。

**(う) 一様場走 = SPECIAL** — `lap≡0` で誤差経路が退化 ∴ PASS 数に算入せぬ(v0-uniform)。

**実測(生・凍結 fixture・`3e2eea9`)**:
```
raw     coef(実走)           c_cur=2040216832 c_lap=10737419 c_prev=-966475008
raw     amplification          G=2.840199988335
v0-uniform cells=4      1tick MAXABS=2.384185e-07 BOUND=9.536743e-07 MAXREF=1.900100e+00 MAXREL=1.254768e-07 sat=0 SPECIAL
         sha(fldj)=02d1f831948eef1d9326d754597c358aa4b1a43e9981007d09598908ecbb0a2d
v1       cells=16     1tick MAXABS=8.903444e-07 BOUND=9.536743e-07 MAXREF=2.109687e-01 MAXREL=4.220267e-06 sat=0 PASS
         sha(fldj)=dd105c33a58be7f40a83d6049f21e31693c0e753823b2285f89641d26161e66e
v2       cells=16     3tick MAXABS=3.207475e-06 BOUND=1.135534e-05 MAXREF=3.456257e-01 MAXREL=9.280198e-06 sat=0 PASS
         sha(fldj)=c07d543f82ab08e0072733ca5648083688d7f4c7d8b3f10c73c75b546d20b984
v3       cells=64     6tick MAXABS=5.654991e-06 BOUND=2.715191e-04 MAXREF=4.373482e-01 MAXREL=1.293018e-05 sat=0 PASS
         sha(fldj)=e0e02074b6011f19ae84a7e647a42af9b2e8e216f6670448469910157420c8fa
v4       cells=256   10tick MAXABS=8.352101e-06 BOUND=1.770156e-02 MAXREF=3.670087e-01 MAXREL=2.275723e-05 sat=0 PASS
         sha(fldj)=9f9f9fc4d243bf44224858b3b6df0bf67555012dbeb4a8dbb34546dabaa60b68
v5       cells=1024  10tick MAXABS=8.538365e-06 BOUND=1.770156e-02 MAXREF=3.582374e-01 MAXREL=2.383438e-05 sat=0 PASS
         sha(fldj)=3633e5bf6f9564b8408612e13359b0319f8f6346966b779964af40e530131fe0
green   parity-vectors         PASS=5 SPECIAL=1
```
**注意(誇大禁)**: v1 は `MAXABS=8.90e-07` 対 `BOUND=9.54e-07` = **余裕 7% のみ**。
1tick の上限は u そのもの ∴ Q20 量子化の下限に張り付いている(異常でなく設計上の下限)。
之を「余裕大」と読むな。N=1 での PASS は**辛勝**である。

**(え) SKIP-ENV 伝播**: `a7_gate.sh`(fieldc/metallib 不在)· `gate.sh`(A7/A10 の rc=3 伝播)·
`teeth_kill.sh` 段2 を **rc=3** へ統一。実測: `FIELDC=./nonexistent` で a7/gate/teeth 全て `rc=3`、
`REPLAY=./nope` で parity `rc=3`、正常時 `rc=0`。

**なお存続**: G1(product=FFT parity 非証明)· §17c の飽和限定 · N>200 未実測。
