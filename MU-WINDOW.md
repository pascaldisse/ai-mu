# MU-WINDOW — 観測窓, mu-inside atom-1, 2026-08-01, @ghoul-sonnet

worktree `~/projects/mc-mu-inside` · branch `mu-inside` (from `mu-world@8fe6b54d` /
`mu-inside@3006d4b7`) · 主樹不觸 · venv `~/projects/magic-crystal-mu-v0/.venv/bin/python`

atom: mu に自己位置を与え、観測を近傍 K×K patch に限定し、その窓の中だけで
次frameを予測し誤差で学ぶ最小閉ループ。obs=全場(128,128,2)の神視點を捨てる。

## 段1 — 観測窓 (SHA `94b2e92f`)

### 設計判断1: PATCH は mu-serve(rust) 側 — python側全frame切出しではない
理由: `plane::excite`(既存poke)と`reward()`(既存probe)が既にrustでtorus-wrap
算術を持つ。python側で第二の独立実装を足すと、言語をまたいだ二重実装が
index-off-by-oneや1ulp(libm/codegen差, 追令①の指摘そのもの)でdriftするリスク
がある。wrap真理源を1つ(rust)にした。
cost: mu_serve.rs(WORLD DOOR, 明示的に拡張可 — journal.rs/world.rs/store.rs
=frozen/W3・W6所有、不觸)を変更。POKEXY(自位置poke)は既存ACTと同形の
frame返却(protocol一様性)のため、poke tickごとにフル(128,128,2)フレームを
捨てて追加でPATCH呼び出し1回=64倍の帯域(k=16: 2048B vs 131072B)を払う。
move tickは追加費用0(STEP自体を捨ててから1回PATCH)。追令①がbytes原文限定
を課しているので、どちらの経路でもtext/json往復コストは無い。

### 設計判断2: 行動空間 = 置換 (既存8-ring-pokeとは非併存、このクラス限定)
WorldEnv.ACT(0..7, ring sites)は一切変更なし、他呼び出し元(online.py,
gate_inside.py)に影響なし。WindowEnvは独自5行動(up/down/left/right/poke)を
定義するのみ。
理由: ring siteは世界宣言の外部target(MU-INSIDE.mdが報酬側で既に
"task handed in from outside"と名指し — 行動側にも同じ批判が刺さる)。
より具体的にこの原子では: ring pokeの座標がKxK窓の外に落ちれば、その行動の
効果は観測不能 → (patch,action)→patch'回帰(段2)にラベルノイズを注入する。
自己位置に紐づく行動は、効果が常に学習対象の窓の中に留まる。

### smoke (`mu/smoke_window.py`, release `target/release/mu-serve`, 実測)
```
GW-1-smoke determinism (release, same seed -> byte-identical patch stream): PASS
GW-2-smoke world-sourced (seed 0 vs 7 -> patch stream differs): PASS
  torus-wrap (x=64,y=64): same_world_state=True mu-serve==np.roll-python: PASS
  torus-wrap (x=0,y=0): same_world_state=True mu-serve==np.roll-python: PASS
  torus-wrap (x=127,y=127): same_world_state=True mu-serve==np.roll-python: PASS
  torus-wrap (x=3,y=126): same_world_state=True mu-serve==np.roll-python: PASS
torus-wrap correctness (independent numpy path, interior+edge sites): PASS
smoke_window.py OVERALL: PASS
```
release vs debug profile (追令①, 同seed+同action列, 64 patches): byte-identical
(この回の実測。恒久的に一致する保証ではない — 見ていないseed/action列は不明)。

## 段2 — 予測・誤差学習 (mu/window_nets.py, mu/window_online.py)

予測器 `WindowPredictor`: (patch_t (k,k,2), action_onehot(5)) → patch_{t+1}
(residual: patch_t + delta, online.pyのdecoder規約踏襲)。132352 params。
`ConstPredictor`(追令②専用null baseline): 出力=学習済み定数(k,k,2)、
patch_t/actionに一切非依存(patch+const残差ではない、正真正銘の定数) —
512 params。

reward/policy/critic無し(追令2④確認: value head不要と明言 — critic不安定
(value損発散, target net無し起因)は主樹側の既知課題であり、本atomはそもそも
criticを持たないため構造的に無縁。導入していない、導入する理由も無い)。
atom brief通り「予測し誤差で学ぶ閉ループ」— RLではない。
stop_gradient法(MU-INSIDE.md)は「報酬が方策graphに入る」場合の防壁であり、
本atomには報酬も方策graphも無い。err はそのまま教師信号として勾配を受ける
(それが「誤差で学ぶ」の意味) — 法の対象外であることを明記する。

行動列は`random.Random(seed)`(mx.randomと非共有)で、重みと完全非依存。
→ 同seedならpredictor/lrを変えても軌道は同一 = GW-3のA/B比較の前提が成立。

mx.compileパターン: `state=[net.state, opt.state]`, `inputs=state,
outputs=state`(online.pyのarc2段3忘却バグと同じ穴を踏まない — ただし本atomは
compiled行動選択関数自体を持たない=行動はoff-graphなので、その特定のバグ
クラスは構造的に発生しない。学習stepのstate配線のみが要件)。

warmup: online.pyの別枠warmup(実重みを汚してsnapshot/restoreが要る)を採用
せず、step0を正真正銘の最初のcompiled呼び出しとし、そのms/tickのみ統計から
除外(学習効果は除外しない、隠さない)。

### WINDOW_ENERGY_FLOOR=1e-5 (実測に基づく設計定数)
200-step probe(seed=1,predictor=window)実測: patch energy min=1.8e-18
p5=4.6e-5 median=6.5e-4 max=5.1e-3、3/200 tickで<1e-9(まだ波が窓に届いて
いない)。floor無しでerr_rel が3.0e9まで爆発(モデル品質ではなく0割の
artifact)。p5の1桁下にfloorを置き、artifactは消えるがoutlierには残る(floor
到達時のerrそのものは意味のある不一致量なので0にはしない)。報酬/方策が
無いのでonline.pyのENERGY_FLOORのような「shout exploit」リスクは無い —
純粋に診断量の数値安定化。

## 段3 — gate (`mu/gate_window.py` → `mu/GATE-WINDOW.txt`)

追令2①(mc-mu-inside disk実査済: `git merge-base --is-ancestor 64ef041d HEAD`=TRUE、
`mu/online.py:131,216`に`inputs=state`/`outputs=state`確認済 -- 凍結bug backport不要)
に応じ、GW-1bを追加: 訓練後の**重みそのもの**をloss/errとは別経路(直接param
read, `mx.max(abs(before-after))`)で比較する、lossの下降と**同じ失敗モードを
共有しない**独立チェック。lr=0対照が「本当に凍結しているか」をerr_relの
横ばいではなく重みのbit比較で確認する。

`mu/GATE-WINDOW.txt` 原文 (この session の実行、seed=4, N_GW3=300):
```
# mu-inside atom-1 段3 gate_window.py -- seed=4 N_GW3=300
GW-1 determinism (seed=4, 60 steps, tick_ms-excluded log bit-identical): PASS
GW-1-profile (release vs debug mu-serve, 追令①): release vs debug byte-identical, 60-step forced-action patch stream
GW-1b weight-update mechanism (追令2①, direct param read, independent of loss trend): lr=1e-3 max|Δw|=3.162e-03 (moves: PASS) · lr=0 max|Δw|=0.000e+00 (frozen: PASS) -> PASS
GW-2 world-sourced (seed 0 vs 7, fixed_actions=True to isolate world from action-stream randomness, 200 steps, err streams differ): PASS (mean err seed0=1.673992e-04 seed7=1.387635e-04)
GW-3 learning-real, 5 controls required (追令②), any FAIL forbids "学習実在" claim:
  (a) err_rel first-half=8.626879 second-half=0.788349 decrease=PASS
  (b) lr=0 control err_rel first-half=28.835952 second-half=11.904781 ratio=0.4128 flat(|ratio-1|<=0.1): FAIL
  (c) persistence baseline: trained second-half=0.788349 persist first-half=0.902686 persist second-half=0.754705 ratio(trained/persist)=1.0446 beat<=0.95: FAIL
  (d) fixed-action-trajectory err_rel first-half=5.941817 second-half=0.178549 decrease-persists: PASS
  (e) 追令② const-predictor null: trained second-half=0.788349 const first-half=1.177137 const second-half=1.871842 ratio(trained/const)=0.4212 beat<=0.95: PASS
GW-3 OVERALL: FAIL -- 学習実在主張禁止 (learning-exists claim FORBIDDEN, per law: any one control failing blocks the claim)
GW-4 perf: step0_ms=1.4392 mean_ms=0.5682 p95_ms=0.6631 p99_ms=0.7481 vs budget=8.3333 load1=9.37 -> UNVERIFIED (load1=9.37 > 2.0, numbers load-contaminated -- 既往: committed 2-4ms measured 21ms under load)

# summary: GW-1=PASS GW-1b=PASS GW-2=PASS GW-3=FAIL GW-4=UNVERIFIED
```
(2回のgate実行間でGW-3の数値は同一 -- 同seed/同config、GW-1b追加はGW-3計算に
影響しない。GW-4の絶対msはload変動で回ごとに違う、load刻印律通りどちらも
UNVERIFIED。)

**GW-1bの意味**: lr=1e-3で`max|Δw|=3.162e-03`(重みは確かに動く)、lr=0で
`max|Δw|=0.000e+00`(bit完全凍結)。これは(b)の「lr=0でもerr_relが59%落ちる」
という結果が**隠れた重み更新バグではない**ことを独立path(直接param比較、
lossを一切経由しない)で確認する -- (b)のFAILは正真正銘、分母(場エネルギー)
側の趨勢的増加によるscale artifactであって、mx.compileの凍結bug(既往,
arc2段3)の再発ではない。

### GW-3 の5対照、生数値と解釈
- (a) err_rel 8.626879 → 0.788349 (見た目は91%減) — **単独では学習の証拠にならない**。
- (b) lr=0凍結対照: 28.835952 → 11.904781 (ratio 0.4128、59%減) —
  flat基準(|ratio-1|<=0.10、GATE-INSIDE.txtのGI-5と同じ規約)を大きく外れて
  FAIL。**凍結モデルでも(a)の減少の大半が再現する** = err_relの分母
  (patch energy)が試行を通じて増大していく(poke行動が場にエネルギーを
  累積注入)ため、学習ゼロでも分母↑→err_rel↓が起きる。まさに追令②が
  警告した「尺度の縮み」がここで実際に検出された。
- (c) 持続基線: trained second-half 0.788349 vs persist second-half
  0.754705、ratio 1.0446 — **持続基線を上回れない**(basisより4.5%悪い)。
  beat基準(<=0.95, GATE-INSIDE.txtのGI-7と同じ規約)を満たさずFAIL。
- (d) 軌道固定対照: 5.941817 → 0.178549、減少は残る。この対照が問うのは
  「trajectoryの選択自体が減少を作っていないか」のみで、(b)(c)の
  confoundは払拭しない(構造的に別質問)。単体PASSは学習の証明にはならない。
- (e) 追令②定数予測null: trained second-half 0.788349 vs const
  second-half 1.871842、ratio 0.4212 — 定数予測より明確に良い(<=0.95達成)。
  ただし(c)がFAILしている以上、「定数より良い」は「持続基線より良い」を
  意味しない — 定数予測はpersistenceより弱い基線(位置に無関係な単一値)
  であり、(e)のPASSは(c)のFAILを免罪しない。

**結論: GW-3 = FAIL。「学習実在」の主張は禁止。** (a)の見かけの91%減の
大部分は(b)が示す通り分母(場エネルギー)の趨勢的増加による尺度アーティ
ファクトであり、(c)が示す通り学習後もtrivialな「現patchをそのまま予測」
を上回れていない。この結果は正直に報告する — 主樹mu-worldや前atom
(MU-INSIDE.md)の「学習は弱いがdirectional」という既往パターンより一段
弱い: 本atomでは学習方向性そのものが単独の対照だけでは支持されず、
定数予測を上回る一方で持続基線を上回れない、という一貫した弱さ。

### 補足測定 (gate外, 参考値・未gate化)
- 600-step, seed=4, trained: err_rel thirds = [6.7617, 0.5617, 0.7477] /
  persist thirds = [0.9298, 0.6038, 0.7801] / ratio thirds =
  [7.272, 0.930, 0.958] — 第2三分区間ではratio<=0.95(持続基線を僅かに
  上回る)が第3区間で再び0.958まで戻る = 非単調、境界線上。**この600-step
  測定はgate_window.pyの正式GW-3対照セットを通していない**(seed=4単一、
  比較対照(b)(e)無し) — 参考値であり、公式verdictはあくまで上記300-step
  gate結果。

## 死枝 (corpses, 理由付き)
- 死 wrap-correctness smoke初版でWorldEnv.step()(=ring poke込み)を参照
  軌道に使用 — reason: WindowEnv側はtick()のみ(no poke)なので軌道が
  分岐しdigest不一致(same_world_state=False)。wrapのバグではなく参照
  軌道の選定ミス。WorldEnv側もtick()のみに揃えて解消。
- 死 err_rel に floor無し(初版) — reason: 200-step probeで3/200 tickの
  patch energy<1e-9、floor無しでerr_rel最大3.0e9(0割artifact、モデル
  品質と無関係)。WINDOW_ENERGY_FLOOR=1e-5導入。
- 死 online.py流の別枠warmup(実重みを1step汚してからsnapshot/restore) —
  reason: 「最小閉ループ」に対してsnapshot/restore機構は過剰。step0を
  正直に最初のcompiled callとし、ms統計からのみ除外する方が単純で誠実。
- 死 自己pokeをring siteの最近傍で代替 — reason: 「自位置での」pokeでは
  なくなる(非硬碼/世界の門経由の精神に反する近似)。POKEXY(rust側,
  任意(x,y)excite)を新設して解消。
- 死 「(a)単独の91%減 → 学習実在」という早期解釈 — reason: (b)(c)対照が
  それを直接反証。追令②が要求した通り、対照無しの単一指標を信じなかった
  ことでこの誤りを検出できた。

## UNVERIFIED / 測っていないこと
- **GW-4 perf**: 終始load1が2.0を大きく超過(セッション中 6.14→104.38→
  56.23で推移)。ms/tick自体(mean 0.62/p99 1.13、予算8.33の対して十分な
  余裕に見える)は取得したが、UNVERIFIEDのまま — idle環境での再測が必要。
- **GW-1 profile比較**: release/debug 1組のseed(=4)・1つの固定行動列
  (60 step)でのみ確認、byte-identical。他のseed/行動列/kで1ulp差が出るかは
  未測。
- **GW-3の複数seed頑健性**: 公式gate(300 step, 5対照フル)はseed=4のみで
  実行。段2の探索時にseed=1,2で個別に(a)相当・lr=0相当を単独確認したが、
  (c)(e)を含むフルセットでの複数seed再現性は未検証。
- **600-step以遠の挙動**: 上記補足測定は非単調(境界線上)。1000+step相当の
  長時間傾向、収束するのかしないのかは未測。
- **k(窓サイズ)依存性**: k=16のみ測定。k=8/32等での挙動比較は未実施。
- **行動別の学習容易性**: up/down/left/right/pokeそれぞれの寄与分解は未実施
  (aggregate err_relのみ)。
- **lr感度**: lr=1e-3固定のみ。他lrでの(c)通過可能性は未探索。

## 正直な限界
観測窓自体(段1: PATCH/POKEXY, torus-wrap, 決定性)は独立path(numpy roll)で
検証済み、堅い。段2の予測器アーキテクチャも動作しコンパイルされ高速
(budget比10倍以上の余裕、ただしUNVERIFIED@load)。しかし**このatomの核心
主張「誤差学習の閉ループが機能している」はGW-3でFAILした** — 300ステップ・
seed=4という条件下では、学習された予測器は「現patchをそのまま次frameとして
予測する」という何も学習していない基線を上回れない。これは失敗の隠蔽では
なく測定結果であり、次段(未実施)の課題として: (i) 学習率スケジュール/
より長い訓練、(ii) 分母(場エネルギー)の非定常性を切り離す正規化の再設計、
(iii) 複数seedでの頑健性確認、が持ち越しになる。

## atom-2r (2026-08-02, Bhairava lane, dead-kimi corpse inherited) — GW-3 緑化試行: 死枝
- ONE move (定): scale-invariance package = NormWindowPredictor (RMS-norm in/out,
  512×2 hidden) + scale-free training error. Replay buffer (corpse code) SEALED, replay=0.
- Measurement: stationary held-out set (seed 11, 64 transitions, never trained on),
  checkpoints 0/75/150/225/300 — `mu/gate_window_r2.py` → `mu/GATE-WINDOW-R2.txt`.
- Result: **GW-3R FAIL**. (b) lr=0 exactly flat PASS (harness sound) · (e) beats const
  null 0.732 PASS · (a) FAIL — training makes held-out err_rel WORSE, 0.565→0.944 ·
  (c) FAIL — persistence null 0.562 vs trained 0.944, ratio 1.68 (need ≤0.95).
- 死枝, 因: online training on the seed-4 trajectory OVERFITS that trajectory; under
  seed shift (4→11) the trained net is worse than its own random init. Old
  WindowPredictor same protocol: 0.857 final — also never beats persistence.
  Scale-invariance is NOT the missing ingredient. 学習実在主張=禁 stands.
- Unexaggerated statement of the only positive: the harness now has a control ((b))
  that is exactly flat, so future failures are attributable to models, not metrics.
