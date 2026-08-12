# AUDIT-INSIDE-R1 — 孫A審 round-1, mu-inside atom-1, 2026-08-01, @kimi
対象: branch `mu-inside` @ 42a6efb8+5e402e12 · gate `mu/GATE-INSIDE.txt` · 報告 `MU-INSIDE.md`
手当遵守: worktree読取·主樹不觸 · 審独立再計算=全て自前走行 (/tmp scripts, --out /tmp → repo原本不觸)
verdict総表: ①HOLDS ②HOLDS ③HOLDS(記述訂正付) ④UNVERIFIED持越(再測不能) ⑤HOLDS(実在·但し弱)

## 審の独立再現 (全gate再走, load1=45.59)
自前走行 =  committed値と **全3モード bit一致** (hist/H/err/err_rel 全桁同一 · GI-1 PASS再現 · GI-2 PASS再現)。
⇒ 報告の決定論的部分は真。ms値のみ負荷比例で変動 (9.46@load9.6 → 14.73@load45.6)。

## ① GI-3判定量 途中変更 (絶対MSE→scale-free) — verdict: HOLDS (gate-shoppingに非ず·但し弱証拠)
時序 (git証拠): 段1 42a6efb8 の gate_inside.py GI-3 = **絶対err_falls** 判定 → 段2 5e402e12 で err_rel へ変更。
変更は結果後 = shoppingの外形は成立。だが物理的裏付けを独立計測:
- 絶対err: 全3モード上昇を独立再現 (+22.0% / +10.0% / +8.0%) → 旧判定なら確かにFAILだった
- 分母非定常 = 真: model無し・固定action cycleの生field energy +6.5%/200tick; 学習方策下で implied energy (err/err_rel) +14〜33%
- 凍結対照 (lr=0): err上昇+9.5% ≈ energy上昇+9.6% = 1:1完全連動, err_rel 1.0087→1.0079 平坦
  ⇒ 絶対MSEは「fieldが大声になる」と学習無しでも上がる。分母混在は実在、scale-free化は正当
- 両量はGATE-INSIDE.txtに残留·MU-INSIDE.mdが変更を自白 = 隠蔽無し
結: 変更は事後だが物理的正当·開示済。但し後段③の通り err_rel 自体もmu操作可能 ⇒ GI-3 PASSの証拠力は弱い。

## ② surprise報酬 stop_gradient 実在 — verdict: HOLDS
code査 (online.py:147): `reward_used = mx.stop_gradient(err) / scale` — 実在。
消費側も全てsg防壁: advantage=sg(td_target−value) · loss_value vs sg(td_target) · loss_reward vs sg(reward_used)。
scale は python float → mx.array (入力·微経路無し)。報酬→重みへの勾配経路は loss_slice 経由のみ
(= 世界模型が自分の将来surpriseを下げる intended learning であり shortcut 非ず)。
GI-1 bit一致再現 = in-graph報酬が決定性を壊していない実証。

## ③ energy正規化報酬却下の論 (分母操作) — verdict: HOLDS (却下は正当·記述の向きは誤り)
論「分母はmuがshoutで膨らませられる」= 計測で真: 分母 +14〜33%/200tick (学習方策下)。
progress/品質信号を正規化すれば偽改善を払う = 却下正当。但し:
- 記述の向き誤り: surprise(maximize err)に対しては正規化は shoutを**罰する**側に働く (分母↑で報酬↓)。
  「inflate by shouting」は surprise では逆向き。progress方向には正しい
- 鏡像exploit未記載: 絶対err報酬は逆方向に操作可能 (shout→絶対err↑→surprise報酬↑ = 騒音TVを自ら作る)。
  実測 err は全モード上昇中 = このexploitは現に作動中
- 設計不整合: 分母操作を理由に却下した正規化量を、GI-3の判定量(err_rel)として採用している。
  今回は凍結対照で分母単独産物でないと切り分け済(⑤)だが、論法としては自分のgateに自分の却下理由が刺さる

## ④ GI-4 FAIL@load1=9.61 idle再測 — verdict: UNVERIFIED持越 (再測不能)
load1 観測推移: 9.61 (建方時) → 11.93 → 45.59 (審再現時) → 15.54 (終盤)。**全session load>2 = idle条件一度も成立せず**。
再現走行: mean 9.46ms@load9.6 → 14.73ms@load45.6 (ms比1.56 ≪ load比4.74 = 負荷比例混入と整合、code退行の証拠無し)。
決定性部分はbit一致で健全。⇒ GI-4は UNVERIFIED のまま、FAILともPASSとも言えず。idle再測は次round宿題。

## ⑤ err_rel .83-.93 学習実在か雑音か — verdict: HOLDS (学習は実在·但し弱い·報告の自己申告は正確)
独立検算3本:
1. **lr=0凍結対照** (200step·同seed·同eps): err_rel 1.0087→1.0079 = 完全平坦 (全3モード同一値 = 方策不変の傍証)。
   学習ありは −3.6〜−11.4%。⇒ err_relの低下は **勾配更新を要求** = 環境単独では落ちない
2. **固定強制軌道·学習OFF·未見seed123** (交絡「学習方策が易状態を選ぶ」を除去): 同一200強制actionを両netに採点
   → trained 0.9175 vs frozen 1.0018 = **init比+8.4%の実予測**。重みが世界知識を保持
3. **生energy probe** (model無し): +6.5%/200tick。分母drift実在するも小、低下の主因ではない
留保: 説明率 ~8-16% のみ (模型弱は真)。600step horizon では trained err_rel thirds 0.876|0.821|**0.840** =
非単調·第3区間で悪化 = 改善は停滞·200step半分比のPASSは浅い証拠。報告の "directional, not good" 表現は正確で誇大無し。

## 附: 審が追加する死枝ではなく宿題
- GI-4 idle再測 (load<2で再実行 → 8.33ms budget判定)
- GI-3強化案: err_rel_falls に lr=0対照のflat線を併記するか、固定held-out軌道でのerr_relをgateに (分母操作を構造的に排除)
- surprise の鏡像exploit (絶対err報酬がshoutを誘発) を MU-INSIDE.md risk欄に追記すべき

証跡: 審scripts=/tmp/audit_r1_controls.py, /tmp/audit_r1_fixedtraj.py · gate再現=/tmp/gate-r1.txt
(repo外·worktree原本一切不觸·commitは本mdのみ)
