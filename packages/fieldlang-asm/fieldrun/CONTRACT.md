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

**G2 (定数注釈の誤り・実測)** 上流 `../CONTRACT.md:37` は `damping=0x3F7FBE77(0.999)` と書くが、
`0x3F7FBE77` の実値 = `0.99896949529647827148`。f32 最近傍 of 0.999 = `0x3F7FC077`(512 ulp 差、
§5 で独立検算)。**bits が法、十進注釈は誤**。fieldrun は bit で比較する故 動作は不変だが、
係数導出を十進から起こすと **必ず外す**。上流注釈の訂正が要る。
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
canonical header → `c_cur=2040220160` `c_lap=10737419` `c_prev=-966478272`。
非 canonical header(例 dt を `0x3E000000`=0.25 に変えた FLDJ)→ **REJECT**(係数を算出せぬ)。

### 3c. 一 tick の完全手検査(2x2、周期 wrap)
`w=h=2`、`cur = [1048576, 0, 0, 0]`(= 1.0,0,0,0 Q20)、`prev = [0,0,0,0]`。
`w==2` ゆえ x-1 と x+1 は同一胞へ wrap(`../q30_wave/CONTRACT.md` の周期律、`w==1`/`h==1` は自身へ wrap)。
```
i=0: lap = cur[1]+cur[1]+cur[2]+cur[2] - 4*cur[0] = 0+0+0+0 - 4194304 = -4194304
     q(c_cur,cur0)  = (2040220160*1048576 + 2^29) >> 30 = 1992403
     q(c_lap,lap)   = (10737419*(-4194304) + 2^29) >> 30 = -41943   (§5d)
     q(c_prev,0)    = (0 + 2^29) >> 30 = 0
     acc = 1992403 - 41943 + 0 = 1950460
i=1: lap = cur[0]+cur[0]+cur[3]+cur[3] - 4*cur[1] = 2097152
     q(c_lap,lap) = 20972 ; 他 0 ∴ acc = 20972
i=2: 同 i=1 -> 20972
i=3: lap = cur[2]+cur[2]+cur[1]+cur[1] - 0 = 0 -> acc = 0
out = [1950460, 20972, 20972, 0], sat = 0
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

f32 decode: `0x3F7FBE77` → e=0x7E=126, f=0x7FBE77=8371319, sig=2^23+f=16759927, 値=`16759927/2^24`。
`0x3DCCCCCD` → e=0x7B=123, f=0x4CCCCD=5033165, sig=13421773, 値=`13421773/2^27`。
```
$ bc <<< 'scale=20; 16759927/2^24'      -> .99896949529647827148   (damping 実値)
$ bc <<< 'scale=20; 13421773/2^27'      -> .10000000149011611938   (dt 実値)
```
`dd = fl32(damping*dt)`: 厳密積 = `16759927*13421773/2^51`。dd≈0.09990 ∈ [2^-4,2^-3)
∴ f32 は `m/2^27`, m∈[2^23,2^24)。`m = round_even(16759927*13421773 / 2^24)`:
```
$ bc: n=16759927*13421773            -> 224947935690571
      n/2^24 = 13407941  rem 13418315 ;  half = 8388608  ->  rem > half ∴ 切上
      m_dd = 13407942                  (dd = 13407942/2^27)
```
`2-dd ∈ [1,2)` ∴ `m2/2^23`。`(2-dd)*2^23 = 2^24 - m_dd/16`:
```
$ bc: a = 2^28 - m_dd = 255027514 ; a/16 = 15939219 rem 10 ; half=8 -> 切上
      m2 = 15939220 ;  c_cur = m2 * 2^7 = 2040220160        (厳密、丸め無し)
```
`1-dd ∈ [0.5,1)` ∴ `m3/2^24`。`(1-dd)*2^24 = 2^24 - m_dd/8`:
```
$ bc: b = 2^27 - m_dd = 120809786 ; b/8 = 15101223 rem 2 ; half=4 -> 切捨
      m3 = 15101223 ;  c_prev = -(m3 * 2^6) = -966478272    (厳密)
```
`k = fl32(dt^2)`(c=dx=1.0 ゆえ courant=dt 厳密)。`k≈0.01 ∈ [2^-7,2^-6)` ∴ `m4/2^30`:
```
$ bc: s = 13421773^2 = 180143990463529 ; s/2^24 = 10737418 rem 9395241 ; half=8388608 -> 切上
      m4 = 10737419 ;  c_lap = m4 = 10737419                (厳密)
```
**独立検算(十進の別経路、上の整数経路と失敗様式を共有せぬ)**:
`c_cur/2^30 = 1.9001030...` vs `2-dd = 2-0.0998969 = 1.9001031` ✓ ·
`c_lap/2^30 = 0.0100000003` vs `k = 0.01` ✓ · `c_prev/2^30 = -0.9001030` vs `dd-1` ✓。
G2 の 512 ulp: `0.999 - 0.99896949529 = 3.0505e-5`、`ulp(0.999)=2^-24=5.9605e-8`、
比 = `511.8 ≈ 512 = 0x200` = `0x3F7FC077 - 0x3F7FBE77` ✓(別経路で一致)。
§3c の量子化(bc 実走、本 lane 実測):
```
$ bc: cc=2040220160; cl=10737419; v=1048576
      (cc*v+2^29)/2^30                  -> 1992403
      cl*(-4194304)+2^29 = -45035462590464 ; floor(/2^30) -> -41943   (算術shift=floor)
      (cl*2097152+2^29)/2^30            -> 20972
```
**独立検算(除算の別経路)**: `c_cur*2^20/2^30 = c_cur/2^10 = 2040220160/1024 = 1992402.5`
丁度 ∴ 半加算後 floor = `1992403` ✓ · `c_lap*2^22/2^30 = 10737419/256 = 41943.043` →
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
