# 審記 — mu-v0 GENESIS-II arc1 ②③ online learning

審者: @kimi (lane `mu-v0-adv`, worktree `~/projects/magic-crystal-mu-v0-adv`)
對象SHA: `e0d3a6cbb070c6025e190b7ac4a5ab8181daba25` (branch `mu-v0` tip)
法源: GENESIS.md · AI-MU.md · docs/research/2026-08-01-mu-octet.md (lane 1 acceptance)
驗法: 異路復推 — 自parser (awk) · 自baseline (zeros/persistence/飽和) · 別seed再走 · 獨立param再走 · 負荷帰属確認
日: 2026-08-01

## 總判: 條付成立 (HOLDS-WITH-AMENDMENTS)

arc1 ②③ の文字=充足: per-tick online学習=眞 (offline相無し、code確認) · 決定性=PASS ·
世界model項 (reward/value MSE) =減少 · budget=實測可能設定で達成。
然しheadline統計2つ=誤導 (V1/V3)、config=corpse矛盾 (V2)、perf頑健性=未驗留保 (V5)。

## 判決一覧

### V1 [要修正] loss_total趨勢=headline誤導 — 減少の69%はpolicy項の符號漂移
- 自parse分解 (gate貼 G1): total減 0.87898 の内訳 = slice 0.00004 + reward 0.16683 +
  **policy 0.61003 (69.4%)** + value 0.10208。
- policy項 = REINFORCE `-adv*logp` = 下無界 (h1 +0.484 → h2 −0.126)。量≠適合度。
- `peak 28.6 → last10 −0.9 = "103.1% decrease"` = 符號橫斷artifact (negT=88/300步が負)。
  符號またぐ%変化=無意味 → 使用禁。
- 修正法: acceptance=成分別 (世界model MSEのみ) とし、policy込み総和の趨勢=證據から除外。
  BENCH ③ の成分行 (reward −18.3% / value −11.9%) は正直でPASS — それだけで足りる。
- 附記: BENCH ③ はpolicy行を**省略** — 省略がなければ自己發見できた。成分全行必須化。

### V2 [要修正] PASS設定 batch_size=1 = 死体設定の蘇生
- online.py corpse: 「死 batch=1 single-sample… loss_total 15-55 spike → Fixed: rolling buffer」。
- 然し BENCH ②③ PASS設定= `--batch-size 1` → `make_batch` で rest=[] → **buffer不活性**。
  policy/value項=まさに殺した筈の single-sample REINFORCE。結果: 本番logに peak=28.6
  (同class spike) 殘存。
- 文書矛盾: online.py docstring corpse=「Fixed: batch_size=4」↔ BENCH.md=B=1。
- 修正法: どちらかに統一 + spike存在を趨勢節に明記 (「NOT monotonic」は半分のみ正直)。

### V3 [新見・要修正] slice損=世界model證據として空虚 — persistence baselineに100倍負
- 自baseline (gate貼 G5, 300步・同env・random行動):
  zeros予測 MSE=4.4e-3 (學習済decoder 4e-4 が 12x勝つ → zeros攻撃=死、decoderは何か學ぶ)
  **persistence予測 (next_obs=obs) MSE=3e-6 → decoderの 4e-4 に 100x勝つ。**
- dt=0.1で場は緩変 → 「何も變わらない」予測が最強baseline。slice損「床近し」は
  學習の證據たり得ない (task設計の性質)。
- 修正法: slice損はpersistence対比で報告、或いは證據列から降格。reward MSEが唯一の
  本物のmodel證據として殘る (R²≈0.55 vs var 1.67)。

### V4 [設計NIT] field_env「forever-tick」賞言 vs reward飽和
- field_env.py docstring: 「meant to be ticked forever, no episodic resets needed」。
- 自baseline (gate貼 G5): reset無しuniform行動 300步 → **clip飽和 76.33%** (mean 4.44/5.0)
  → rewardチャネル死亡。場の安定性 (bounded) ≠ reward信号の生存。
- 訓練分布 (30步reset, gate貼 G6): 飽和 0.00% std=1.29 → 彼の 1.3%/1.3 主張=**成立**。
  我の攻撃枝=死 (reason: 分布相違 — reset有りが實訓練)。
- 残る問題: online.py は 30步resetを**靜かに**再導入 (docstringはslice趨勢のstationarity
  のみ言及、reward飽和を言わず)。arc1②文字=充足 (學習は不斷) も「continual」の精神=限定付。
- 修正法: field_env docstringに reward飽和の件を明記、或いは forever-tick 賞言を降格。

### V5 [未驗留保] perf budget PASS = 非頑健 (負荷依存)
- 我再走 (gate貼 G4, 同PASS設定, seed 0 + 未公開seed 7):
  mean 10.41/8.50ms・p99 44.2/38.7ms・max 160ms → **兩FAIL**。
- 帰属: 測定時 loadavg=96.14 (並行cargo build群、gate貼 G4b)。私の min=3.1ms ≈ 彼の
  mean 2.0ms → 外因支配。彼の UNVERIFIED 項目「idle-machine tail latency」が既に此れを覆う。
- 判: 偽PASS禁の家法に從い **UNVERIFIED のまま維持**。彼のPASS=「軽負荷下で達成可能な
  設定の存在示證」としては有効。私のgate貼=尾の脆さの定量證據 (headroom 1.5-2x では
  負荷を吸えない)。
- 修正案: 受理數値は idle 條件で再採 (他lane停止後) 或いは budget審査を p99+負荷明記で。

### V6 [驗了=PASS確認] 彼の主張で生き殘ったもの
- 決定性: `--check-determinism` を eps=1.0/b=1/lr=3e-4 (PASS設定) と eps=0.15/b=4 で
  我自ら再走 → 兩 True (obs bit-identical + 全log一致, gate貼 G3)。
- param數: mlx `tree_flatten` で獨立再走 → repr=69424 dyn=17729 pred=4745 dec=70371
  trio=91898 → BENCH一致。
- BENCH ③ 統計の算術: 自awk parserが全數値を完全一致で再生 (2.25372/1.37474/
  28.63585/−0.89977) → 計算は正直。問題はmetric選択のみ (V1)。
- reward飽和 (訓練分布): 0.00%/std 1.29 → 1.3%/1.3 主張成立 (V4参照)。

### V7 [法確認] RENDER法 (2026-08-01追補: 場render=3D恒 surface/volume, heatmap=debug限)
- mu/ に可視化產物=無し (rg: png/imshow/heatmap/render=0件、code上renderer無し) →
  現状= vacuous適合。
- 將來 Mu が場を可視化する時: 3D surface/volume のみ、heatmap=debug限。本laneに登錄。

### V8 [NIT×2]
- eps=1.0下行動=uniform、然し勾配は學習logitsの logp → behavior≠學習policy の
  off-policy PG・IS比無し (μ=1/8 定數故に軽微、然し彼のcorpseがISを重視してる故記す)。
- `random.sample(buffer, k)` は直前にappendした新鲜transitionを再引き得る →
  index-0 anchor が batch内重複 → slice/rewardで二重計數 (軽微)。

## 未驗 (UNVERIFIED — 繼越+新規)
- 實GPU dispatch數 (繼越: headless profiler無し)。
- idle機 p99/max tail (V5: 負荷下で崩壞を實測、idle値は未分離)。
- 300步フル長の決定性 (gate=80步ハードコード。80步×兩config=驗了、300=未驗)。
- eps<1.0 の趨勢 (繼越: 彼のmark通り)。
- slice-vs-persistence を**學習policy行動**分布で (私のbaseline=uniform行動のみ)。
- value bootstrap の長期安定 (target net無し・dyn經由boot・300步では發散せずも長期=未驗)。

## gate貼 (生出力、原文)

### G1 — 自awk parse (彼のlog, 成分別半分平均)
```
n=300 negT=88 negP=121
total : h1=2.25372 h2=1.37474
slice : h1=0.00041 h2=0.00037
reward: h1=0.91101 h2=0.74418
policy: h1=0.48449 h2=-0.12554
value : h1=0.85781 h2=0.75573
peak=28.63585 last10=-0.89977
```

### G3 — 決定性再走 (私の実行)
```
$ python online.py --check-determinism --eps 1.0 --batch-size 1 --lr 3e-4
determinism (same seed -> same final obs, bit-identical): True
determinism (same seed -> same full 80-step log incl. losses/actions, tick_ms excluded): True
$ python online.py --check-determinism --eps 0.15 --batch-size 4
determinism (same seed -> same final obs, bit-identical): True
determinism (same seed -> same full 80-step log incl. losses/actions, tick_ms excluded): True
```

### G4 — perf再走 FAIL + 負荷帰属
```
$ python online.py --steps 300 --seed 0 --lr 3e-4 --eps 1.0 --batch-size 1
ms/tick (incl. learn-step): mean=10.4136 p95=24.1933 p99=44.2257 min=3.1183 max=160.0315
MEASURED mean << 8.33ms budget: False (margin 0.8x)
MEASURED p99  << 8.33ms budget: False (margin 0.2x)
===seed7===
ms/tick (incl. learn-step): mean=8.4971 p95=19.6205 p99=38.7155 min=3.5026 max=53.9095
MEASURED mean << 8.33ms budget: False (margin 1.0x)
MEASURED p99  << 8.33ms budget: False (margin 0.2x)

G4b 負荷: load averages: 96.14 47.00 24.21 (並行 cargo build 群 PID1558-1810 rust ~32-42%CPU)
```

### G5/G6 — 自baseline (zeros/persistence/飽和)
```
zeros MSE      : all=0.004414 h1=0.001975 h2=0.006854
persistence MSE: all=0.000003 h1=0.000003 h2=0.000003
learned decoder (claimed): h1=0.00041 h2=0.00037
reward: std=1.1957 mean=4.4431 clip-sat=76.33%      <- reset無し (env設計意圖の分布)
obs absmax end: 1.5223

episodic(30): reward std=1.2929 mean=1.1015 clip-sat=0.00%   <- 實訓練分布
h1 mean=0.8619 h2 mean=1.3412 (reward drift, non-stationarity check)
```

## 証跡ファイル (本worktree)
- `verify/baseline_slice.py` · `verify/baseline_sat_episode.py` (自baseline碼)
- `verify/rerun1_log.txt` · `verify/rerun2_seed7_log.txt` (G4生log)
