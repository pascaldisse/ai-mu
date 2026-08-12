# 審計画 — 言lane (kimi建 F1/F3/F4/F5) · @opus5 審者 · 2026-08-01

輪番: kimi=建 · opus5=審。審は**別経路再導出のみ**を証拠と認む。
建の門を再走するだけ=echo、証拠に非ず(3者1盲点則)。

## 審則(全F共通)
1. **異路則** — 検査は計算と失敗様式を共有せざる事。建の固定値(pin)は建自身が産んだ数 → 審は spec から独立に再導出する。
2. **識別力則** — 門が「壊れた版で落ちる」事を実測せよ。落ちぬ門は空虚(=旧F1門が緑だった理由)。
3. **非空虚則** — 全零/全死/未描画で通る閾値は門に非ず。必ず「何かが起きた」下限を同一門に持たせよ。
4. **貼付則** — 数は端末原文貼り。言い換え・丸め・再計算禁。
5. **死枝則** — 殺した枝は理由付きで残す。

---

## F1 — fnv1a prime parity (`3d6b9342`, branch `speakback-s0`) — **審済 PASS**
検証器: `tools/audit_lang_f1.py`(本審が新規、建の碼と一行も共有せず)
- 素数は**定義から**再導出(2^40+2^8+0xb3)、literal複写せず。
- 両側の定数を**source解析**で抽出(import せず)。
- speak-in導出鎖(fnv1a→word_seed→mix64→word_channels→seed_unit→channel_amp→channel_center)を Python で**仕様から**再実装、field crate の生probe(固定fixture不使用、実行毎に再生成)と bit 比較。
- **識別力**: 旧typo素数で13/13語のchannelが動く事を実測 → pinは空虚に非ず。

結果(原文):
```
spec prime (2^40+2^8+0xb3) = 1099511628211 = 0x100000001b3
  PASS  packages/field/src/lib.rs: fnv1a multiplier(s) ['0x100000001b3']
  PASS  packages/speakback-s0/src/cell_decode.rs: fnv1a multiplier(s) ['0x100000001b3']
  PASS  'fire': seed 0xaa77f578efdfc4b9 ch [26, 9, 20] amp [1056481003, 1064926718, 1055650483]
  PASS  '愛': seed 0x33a7291b74bac894 ch [45, 34, 12] amp [1054038824, 1062560382, 1064670074]
  PASS  discrimination: old prime 0x1000001b3 moves channels for 13/13 words
  PASS  speakback-s0 own suite
AUDIT F1: PASS (13 words, bit-exact across python-spec / field-crate / mirror-constant)
```

### 審指摘 N1(**要修正・記述のみ、碼は正**)
commit題・碼註・docs が field crate の素数を **"non-standard"** と呼ぶ。逆。
`0x100000001b3` = 1099511628211 = **FNV-1a 64 の標準素数**(2^40+2^8+0xb3)。
旧mirrorの `0x1000001b3` = 4294967731 は素数ですらない桁落ちtypo。
→ 「fieldが非標準」と読むと、後続が field 側を"直し"に来る。文言訂正を要求。

### 審指摘 N2(軽・提案)
pin の真値は建が捕った run 由来。本script(`audit_lang_f1.py`)を CI 門に入れ、
pin が漂流した時に**外部真値**で落ちるようにせよ。fixture 保存は不可(期限切れ事実)。

### 未検(F1)
- 実 field へ speak→伝播→s0復号 の**閉ループ**精度は未測(定数一致のみ)。F3審で測る。

---

## F3 — adapter(mirror廃止 → field::speak 直用)審点
- A1 **写し身ゼロ**: `cell_decode.rs` に fnv1a/mix64/seed_unit/GRID/AMP_MIN の**再実装が残っていない**事を grep で示す。残るなら adapter は名ばかり。
- A2 **依存方向**: speakback→field の単方向。field が speakback を知らぬ事(`cargo tree`原文貼)。
- A3 **閉ループ**: 実 world に speak → tick → 觀 → 復号 の語別正解率、語彙≥6、`--nocapture`原文。旧6/6は合成データ由来 → **合成禁止、実場のみ**。
- A4 **識別力**: 語を1つ差し替えると復号が外れる事(混同行列 or 最近傍差)を数で。
- A5 **決定性**: 同seed2回で復号結果 bit 同一。
- A6 **予算**: 觀+復号の ms/utterance、120fps床(8.33ms)に対する比。p99も。

## F4 — 3D化(RENDER法)審点
- B1 **平面回帰禁**: 提示経路に `debug_flat`/heatmap が到達不能である事(碼経路 + `--debug-flat`は明示旗のみ)。
- B2 **高さ=振幅**の証: 振幅を上げた場が**画面上で上がる**(天際線行番号の差)。色ランプでは通らぬ門である事を確認。
- B3 **遮蔽/遠近**: 近景が遠景を隠す・同物が遠いと小さい、を画素数で。
- B4 **純関数性**: render = f(state, view) のみ。clock/rng/step非依存 → 同状態同視点で fb digest 一致、かつ LIVE==REPLAY。
- B5 **p99床**: 平均でなく **p99** を熱い世界(起伏の育った後)で測る。`960073e2`の指摘(実行中に2.9→8.8ms劣化)を必ず再現条件に含めよ。冷起動門は無効。
- B6 **外枠ゼロ**: `cargo tree -p field` に描画crate無し(核純度)。

## F5 — 上限(capacity/K≤d/22 系)審点
- C1 **導出**: 上限式が測定でなく**導出**である事、導出文書の各段が独立に検算可能か。
- C2 **飽和実測**: 上限直下/直上で挙動が**実際に割れる**(語混同率の跳ね)を実測。跳ばぬなら上限は装飾。
- C3 **次元依存**: d, K, ch を振った表は**同一run同一集合**(stitch禁)。
- C4 **崩れ方**: 上限超過時の劣化が緩やかか断崖か。断崖なら運用余裕を規定せよ。

---

## 走らせ方
```
python3 tools/audit_lang_f1.py --truth-repo <field crate を持つ worktree>
```
`--repo` = mirror(speakback-s0)側、`--truth-repo` = field側。
mirror枝には field crate が無い → **自分自身を真値に出来ぬ**構造をscriptが強制する。
