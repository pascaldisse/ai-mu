# 審記② — mu-world arc2 段3 (frozen-policy修 + 振舞gate + RMS)

審者: @kimi (lane `mu-world-adv`, worktree `~/projects/mc-mu-world-adv`)
對象SHA: `64ef041d` (枝 mu-world; 列 6787d2a4→30401377→4d8781c8→64ef041d の頂)
驗法: 異路復推 — ①純mlx機構試驗 ②lr對照AB+pre-fix對照 ③gate再走 ④committed log自parse+RMS再生
附帶: arc1審記 = `~/projects/magic-crystal-mu-v0-adv/verify/REVIEW.md` (commit bdb9d0e7)
G1 (perf): **未驗** — 閑機要 (主令)。本審では觸れず。

## 可否 總判: **成立 (PASS、條付強化)** — 修は正しく必要、gate設計は正當、行動學習は審者の全指定configで獨立再生 (彼より強い)。
條付: ⓵ committed log は committed code 既定値で **bit不再現** (生成系統差、後述) → 彼の value 收斂數値は**未驗**に格下げ。
⓶ 私の既定値 repro では **critic が 900步で發散** (value損 1.035→17.126) — 行動學習は成立するが、critic 安定化は次 lane の課題。
指摘=NIT群 (下記)。①修の正否・③決定性・行動學習の存否は不動。

## ① mx.compile inputs=state 修 — 正否: **正** (兩異路で確認)
- 機構路 (verify/mech_compile_state.py, env無し純mlx):
  重みを實updateで動かし acting logits を比較。
  - compile w/o inputs=state: update後も logits **bit同一** → arc1 bug = 實在
    (acting netは學習を見ない)。
  - compile with inputs=state: logits 變化 max|Δ|=19.89 → 修 = 有效。
- 實env路 (私設計, mu-serve@64ef041d, 150步 eps=0 batch=1):
  - post-fix: lr=0 vs lr=1e-2 → 行動列 **step 8で分岐, 128/150 相違** → 學習が行動に到達。
  - pre-fix對照 (私がinputs=stateを外して再現): **0/150 相違 = bit同一**、
    同時にlossは 1.59 vs 727.76 に大きく乖離 → 「損失は動くのに行動は凍る」
    彼の証の簽名そのままを獨立再現。對照後ファイル復元済 (git checkout)。
- 附言: 此のbugは arc1 の loss-trend gate では**原理的に檢出不可能**だった
  (損失は訓練netのみ測り、acting netを一度も測らず) — arc1審 V1 と併せて
  loss-trend gate の死を確定する證據。彼の「行動が變わらねばならぬ」新standing
  gate は正しい遺物。

## ② 新gate『振舞推移』(行動histogram) — 妥当性: **正當、採用可**
- 數値再生 (verify/parse_world_log.py, committed online_world_log.txt @64ef041d):
  act2窓share [29,96,104,42,83,111] ✓ · raw r 0.1595→1.0125 (6.3x) ✓ ·
  value 0.680→0.107 ✓ · (未公表) reward損 0.804→0.057 も減少。
- 虛PASS否定: lr=0對照では act2 share = 9/150 ≈ 偶然 (18.75) で**推移を持ち得ない**
  (凍結logits=一定分布、trend gateは構造上通らない)。
- 崩壞否定: final150 histogram [4,2,111,6,5,9,7,6], entropy 1.06 nats (max 2.08)。
  單一行動崩壞 (150/150) ではなく健全な73%集中。
- **loss-trend gate の盲点 論証** (要求項):
  1. 総和は下無界の REINFORCE/entropy 項を含む → 符號漂移が「改善」を裝う
     (arc1審測: 減少の69%はpolicy項の符號漂移; arc2でも entropy bonus で
     loss_total は構造的に負化し得る — 彼のcorpse記載通り)。
  2. 空虚目標: slice損は persistence baseline に 100x 負ける (arc1審 V3)。
  3. 非定常目標: RMS scaling・on-policy報酬shiftで同じ數値の意味が時刻で變わる。
  4. 致命的: **損失は訓練netのみを測る** — acting経路の凍結を2arcに亙り見逃した。
  → 振舞gateは代理(損失)ではなく目的物(報酬の來る行動を取るか)を直接測る。
    raw報酬推移との對設計で entropy崩壞の虛PASSも塞ぐ。正しい置換。
  5. 獨立實演 (私の repro900): value損 1.035→17.126 に發散する run で、行動は
     act2 59→132・報酬2.6x と正常に學習 → **loss-trend gate なら「學習している run」
     を FAIL にした**。逆方向の盲点の實在證明 (arc1の符號漂移=順方向)。
- gate自體の盲点 (指摘として登錄):
  NIT-2a 窓依存: 第4窓 42/150 の振れ — PASS判定は「最終窓」でなく「趨勢+報酬對」で讀む運用を。
  NIT-2b 單seed: 種依存の分散は未測 (彼のUNVERIFIEDにも無し → 追加要)。
  NIT-2c G3 runの設定 (lr/eps/batch_size/episode_len) がBENCHに未記錄 → 再現性は
        既定値假定に依存 (③で檢證、下記)。

## ③ G2決定性 再走 — **PASS** (私の實行 @64ef041d)
```
$ python online.py --check-determinism
determinism (same seed -> same final obs, bit-identical): True
determinism (same seed -> same full 80-step log incl. losses/actions, tick_ms excluded): True
```
- 延長檢證 (私): 既定値で900步×2走 → **兩走 bit同一** (tick_ms除く)。長射程決定性=實在、80步gateを900步に私が延長確認。
- 併せて否定済み: CPU stress×8 下 / online.py×2 GPU並行 下でも 80步は靜時と bit同一 → 負荷依存の非決定性は檢出されず (全軸否定)。

### ③b committed log の不再現 (生成系統差) — 正規發見
- 既定値 repro900×2 は互いに bit同一だが、**committed online_world_log.txt とは step 5 で分岐** (steps 0-4 は bit同一)。
- 0-4同一 → seed/初期重み/lr/batch/世界binaryは同一と確定。設定差 (eps等) では 0-4同一かつ5分岐は構成不可能 (候補全て數値的に死)。warmup差仮説・mlx版仮説も實驗死。
- 故に殘る説明: **logは committed code とは微妙に異なる agent側 code 狀態 (段3開発途中) で生成**された可能性が最有力。これ以上は此處からは斬れない。
- 影響評価: log內の數値は內部一貫し私のparseで全再生 (②)。行動學習の主張自體は私の獨立 run が裏付ける (下記③c)。故に致命ではないが、**証跡logは再生成してcommitし直すべき** (再現性 hygiene)。

### ③c 審者の全指定configでの獨立 G3 (格上げ證據)
```
repro900 (既定値 lr=1e-3 eps=0.2 batch=4, 審者run) act2/150: [59, 113, 119, 122, 125, 132]
raw r: first150=0.3987 last150=1.0468 (2.6x)
value loss: first150=1.035 last150=17.126   <- critic 發散
final150 histogram: [4, 2, 132, 2, 3, 2, 2, 3] (88% 集中、崩壞ならず)
```
- 行動學習=既定値で**彼のlogより強く**再現 (act2 88%、報酬2.6x) → 段3の G3 主張「行動が變わる」は完全指定configで審者が獨立確認。
- 然し **value損は發散** (彼のlogの 0.680→0.107 は既定値で不再現) → 彼の value 收斂行は未驗に格下げ。

## ④ 報酬 running-RMS化 — 方向=正、副作用=4種實測 (全て軽微・未報告)
判定式: r / (sqrt(acc/n)+1e-3), acc+=r², episodeを跨いで不reset。決定性=問題なし
(pure f(seed) trajectory関数)。符號=反転なし (r≥0)。
- S1 初期inflate: 非零の最初期 normalized = 4.144 → 2.74 → 1.91 … (self-normalizeで
  ~sqrt(n)まで膨む)。criticの初期TD目標は最大4x過大。収斂するが**過渡は實在**。
- S2 scale漂移: rms_t = 0.316 (first150) → 1.267 (mid) → 1.146 (last)。同一raw報酬が
  到達時刻で別のscaled値を持つ = 非定常目標。然し value MSE 0.107 vs var(scaled r)
  1.057 → R²≈0.90、criticは實質追從 (被害は吸収されている)。
- S3 buffer舊scale混入: buffer_cap=200 に對し 200-tick rms 相對漂移 max 100% / mean 30%。
  古いsampleは古いscaleで保存 → world-model回歸が混合scale目標で學習。
  修正案: bufferには**rawを保存し sample時に rescale** (staleness消滅、1行改)。
- S4 reset境界dip: RMSはresetを跨ぐが報酬量はresetで落ちる → 境界±1步の normalized
  mean 0.427 vs 中盤 0.867 (約2xの周期artifact)。30步周期の擬似季節性。
- **S5 critic發散 (本丸)**: 既定値 900步 repro で value損 1.035→17.126 と後半發散。
  機構: RMS分母が 4x 漂移 (S2) + target net 無し + dyn經由 bootstrap → 動く目標を
  criticが追い、末期に正feedback。行動は發散の間も改善續けた (advantageの符號が
  殘る限り actorは進む) — 然し criticを信じる價值系としては不健康。
  修正案候補: target network (緩慢追從) / value損に重み衰减 / RMSを episode 每に
  freeze するか EMA化 / advantage 正規化 (batch內 z-score)。次 lane の實驗題。
- 判: S1-S4 は過渡・軽微、**S5 は實害** (但し行動學習を殺すには至らず)。全て BENCH 未報告 → S1-S5 を記錄要。

## 指摘一覧 (NIT — 致命なし)
- NIT-1: G3 run設定 (lr/eps/batch/episode_len) をBENCHに記錄せよ (再現は既定値假定)。
- NIT-2a/b/c: 振舞gateの窓依存・單seed分散・(同上設定記錄)。
- NIT-3: RMS副作用 S1-S5 をBENCHに記錄 + buffer raw保存案の檢討。
- NIT-6: committed online_world_log.txt を committed code で再生成・configコマンド明記
  (③b: 現logは既定値でbit不再現、生成系統差)。
- NIT-4: reward損 0.804→0.057 は好材料 — 未公表なのは勿體ない。成分全行必須化
  (arc1審 V1 修正法と同じ) に從えば自然に出る。
- NIT-5: 決定性gateは80步固定 — 900步の本番長では別途未驗 (③延長で私が一部補完)。

## 未驗 (UNVERIFIED — 本審でも殘る)
- G1 ms/tick (閑機要、主令により今審せず)。
- 實GPU dispatch數 (headless profiler無し、繼越)。
- 振舞の收斂 vs 振動 (42/150窓、彼のmark通り)。
- 彼の value 收斂數値 (0.680→0.107) — 既定値で不再現、私の repro では發散 (S5)。
  要: 再現する config/code の同定、或いは S5 對策後の再測。
- 種依存 (seed≠0/7 での振舞趨勢)。
- RMS S1-S4 の長期影響 + S5 の發火條件 (lr・步數・RMS漂移率の相關)。

## gate貼 (生出力)
### ①機構
```
params moved by update: True
frozen-compile logits identical after update: True (True = arc1 bug real: acting net blind to learning)
inputs=state logits identical after update: False (False = fix works: acting net sees learning)
max|dlogits| live: 19.8883113861084
```
### ①實env AB對照
```
post-fix lr0 vs lr1e-2: first-diverge step=8, differing=128/150
pre-fix lr0 vs lr1e-2: first-diverge step=none, differing=0/150
(post-fix final loss_total=1.58998 vs pre-fix=727.75879 — loss動き行動凍る、を再現)
```
### ③決定性
```
determinism (same seed -> same final obs, bit-identical): True
determinism (same seed -> same full 80-step log incl. losses/actions, tick_ms excluded): True
```
### ②④ parse (拔粹)
```
n=900
act2 share per 150: [29, 96, 104, 42, 83, 111]
raw r: first150=0.1595 last150=1.0125
value loss: first150=0.680 last150=0.107
reward loss: first150=0.8037 last150=0.0570
step0: raw=0.00000 normalized=0.00000
normalized of first 10 nonzero raw: [4.144 2.74  1.907 1.312 0.91  0.666 0.524 2.589 1.668 1.007]
scale drift: rms_t first150=0.3162 mid=1.2671 last=1.1460
200-tick staleness: max rel rms drift=1.000 mean=0.301
normalized r at reset boundary(±1): mean=0.427 vs mid-episode: mean=0.867
value-loss floor check: var(scaled r)=1.057 vs last150 value MSE=0.107
final150 action histogram: [4, 2, 111, 6, 5, 9, 7, 6] entropy 1.060 nats
lr=0 control: act2 share per 150: [9] (chance=18.75)
```

### ③b/③c + ④S5 (拔粹)
```
repro1 vs repro2 (900步, 既定値): identical=True
repro1 vs committed log: identical=False first-diff=[5,6,7] (steps 0-4 bit同一)
stress-80 (CPU×8) vs calm: identical=True
concurrent GPU ×2 80步 vs calm: True / A vs B: True
no-categorical-warmup vs committed: first-diff=[0,1,2] (仮説死)

repro900 act2/150: [59, 113, 119, 122, 125, 132]
raw r: first150=0.3987 last150=1.0468
value loss: first150=1.035 last150=17.126   <- S5 critic 發散
final150 histogram: [4, 2, 132, 2, 3, 2, 2, 3]
```

## 証跡 (本worktree verify/)
mech_compile_state.py · parse_world_log.py · runA_lr0.txt · runB_lr1e2.txt ·
runC_prefix_lr0.txt · runD_prefix_lr1e2.txt
