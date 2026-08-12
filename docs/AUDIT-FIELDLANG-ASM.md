# 審 — fieldlang-asm 設計審 checklist + 既存資産表 (審方B-2 atom-1 · @kimi · 2026-08-01)

対: fieldlang asm建方 (枝 fieldlang-asm, worktree mc-fieldlang-asm)。
基盤継承: audit-lang 審計画 F1-F5 (docs/AUDIT-GAIAGO-LANG.md) — 審則はそのまま適用。
法源: GENESIS.md (120fps床·godseed決定性·OSの扉) · ENTROPY.md (state=f(seed,journal)) ·
docs/design/2026-08-01-field-contract.md (ultradeterminism条) · CONTRACT v1 (packages/fieldlang-asm/CONTRACT.md, frozen)。

## 審則 (audit-langより不変で継承)
1. 異路則 — 検査は計算と失敗様式を共有せざる事。定数はCONTRACT/journal.rs原文から独立再導出。
2. 識別力則 — 門は壊れた版で落ちる事を実測。落ちぬ門は空虚。
3. 非空虚則 — 空program/全零で通る門は門に非ず。digestはprogram毎に動く事。
4. 貼付則 — 数は端末原文貼り。丸め禁。
5. 死枝則 — 殺した枝は理由付きで残す。

---

## Checklist A — 決定性 (godseed)
- A1 **compiler = 純関数**: 同一.fld → byte同一.fldj ×3 run (shasum原文)。出力にpath/時刻/address痕なし。
  実測済(2026-08-01, /tmp/flaudit): ex1 ×2 run sha256 `d89a5df9…b30b52` 一致 ✓ — ただし副作用mode非決定 → F2。
- A2 **replay bit-exact**: World::replay(fieldc出力).digest == flgate reference digest == live apply digest。
  accept.sh digest×2 門を LIVE==REPLAY 3者に拡張せよ (現行はreplay×2のみ)。
- A3 **f32=bit列のみ**: .s に FP演算命令ゼロ (fadd/fmul/fcvt grep)。decimal parseは整数mul/add + overflow検査 (lexer code 2)。
  撃 amp・header params は decimal bit pattern の受渡しのみ。
- A4 **LE固定**: header/op全てLE明示。arm64 native store = LE に依存している事を文書化 (V1 port時の地雷)。
- A5 **defaults照合**: driver ._fl_defaults (40B) を CONTRACT §Defaults から審が独立再導出しbit照合
  (w=64 h=64 c=0x3F800000 dt=0x3DCCCCCD damp=0x3F7FBE77 dx=0x3F800000 seed=42 range=0x3F800000 n_slots=16)。
- A6 **error決定性**: 同一bad入力 → 同一error code/line (lex: -(line<<8|code), emit: -(code))。
  fuzz: 乱bytes·切断UTF-8·u64越えint·束 n=0/過大·寫 len過大·界 が2番目以降·觀 単独 → 全て deterministic reject、
  crash/hang/黙ってtrunc なし。界 非先頭 = emit syntax err (code 1) 実測。
- A7 **識別力**: seed差・op差で digest が動く事 (determinism.rs gate 4 に倣う)。壊れた版 (1byte反転 .s) で accept.sh が落ちる事。

## Checklist B — 120fps床 (8.33ms)
- B1 **compile計測**: fieldc を 3 examples + stress program (op数 10⁴級, src 1MiB級) で ms 実測・原文貼。
  V0はoffline tool宣言 → 宣言の証明 = render/frame loop から fieldc への到達不能 (call graph/依存方向)。
- B2 **線形性保証**: lexer/emit は O(tokens)。固定buf (16MiB×3 .bss) = allocator無し → hot path realloc不可。good。
  cap超過は buffull err で deterministic に落ちる事 (A6と連動)。
- B3 **frame侵食ゼロ**: 生成op列は journal append のみ → append()=RAM buffer, flush=off-cadence (journal.rs法)。
  Step{count} の cost は wave tick 側 (probe 0.47ms/frame 実測@field-unified) — compiler門に非ず、だが
  Step{10⁶} の様な悪性programを runtime が naive に1 frameで回さぬ事 = scheduler側門、asm側は文書化要求。
- B4 **p99主義**: 平均でなくp99 (F4-B5継承)。熱い世界で。冷起動門は無効。
- B5 **V1 live-compile の扉**: speak→compile→excite を将来 live に載せるなら、その時点で frame budget 配分表を要求。
  V0で予約のみ。

## Checklist C — OS可搬性 (GOAL∞ OSの扉)
- C1 **syscall面 inventory**: extern symbols 完全列挙 (nm 原文貼)。期待 = _open/_read/_write/_close/_exit の5 + _fl_lex/_fl_emit。
  lexer.s/emit.s は syscallゼロ (純compute) → grep で証明。
- C2 **flag値の所有権**: O_RDONLY=0x0000, O_WRTRC=0x0601 は macOS/BSD encoding (Linuxでは0x241相当)。
  .equ block に隔離済 → 審は「全platform定数が.equにのみ居る」事を grep で確認。core logicにmagic number残留なし。
- C3 **artifact可搬**: .fldj = LE byte stream = arch不変。第2 host (Linux/arm64 or 自前runtime) 出現時に byte同一実測。
  現状 host=macOS/arm64 のみ → **UNVERIFIED** 明記 (1 host での byte決定性は cross-host の証明に非ず)。
- C4 **ring-zeroの扉**: malloc無し・固定.bss・libc5callのみ → freestanding 移行面は driver.s のみ。
  門: 新規 extern 追加は CONTRACT 改訂なしに禁。
- C5 **archの扉**: arm64専 = V0合法 (self-hosting dict 有: commit 7c6bbd04)。backend swap点 = CONTRACT §V1 layer B (journal v2) と同一seam。
- C6 **variadic ABI (F2由来)**: Darwin arm64 では variadic libc call の可変引数は stack 渡し。
  open() は唯一の variadic → F2修正後、printf系・fcntl系など variadic を新規に呼ぶ事は審対象 (原則禁止、raw svc へ)。

## Checklist D — mu-v0/tracr bridge整合
- D1 **op alphabet閉包**: CONTRACT tags 1-6 == journal.rs enum Op (SeedAtom/Excite/Step/Bind/Bundle/WriteRaw) 1:1。
  新op = journal.rs + CONTRACT 同時改訂のみ。片側先行 = FAIL。compiled ⊂ learned をopレベルで保つ鍵。
- D2 **bind意味論**: 縛 = field::ops::bind = FFT-domain multiply (tracr PROOF A: 3実装一致 max err 8.882e-15)。
  審は独立経路: field crate bind 結果 vs bridge/circulant_bridge.py の fft_bind を同一入力で比較 (異路則)。
  compiled binding が learned weight space の到達点である根拠。
- D3 **mu tick budget不侵**: mu-v0 実測 mean 0.91–4.1ms p99<8.33 (mu/BENCH.md)。compiled program の ops は
  同一 two-cadence scheduler を通る → asm導入後も BENCH.md の ms/tick が劣化せぬ事を統合runで実測。
- D4 **觀 = probe整合**: V1 reservation 觀=channel collapse→light。tracrの readout = probe = correlation = matched filter。
  建方案の 觀 意味論が matched-filter readout と矛盾せぬ事 (V1審の核心)。
- D5 **seed唯一性**: 種 slot seed・界 seed 以外の entropy源を compiler が導入せぬ事 (godseed法: 全て=f(seed))。
  asmに乱数・時刻・uuid無し = grep + A1と連動。

---

## Findings register

### F2 (新・実測 2026-08-01 · 重大度: 中-高) — driver.s open() mode = stack garbage
Darwin arm64 ABI: variadic関数の可変引数は register でなく **stack** 渡し。
driver.s は mode を `mov w2, #0644` で渡すが `_open` は w2 を読まず [sp] を読む → mode = 非決定 garbage。
証拠 (原文):
```
$ otool -tv fieldc | grep -B6 "bl.*_open"
0000000100000578	mov	w1, #0x601
000000010000057c	mov	w2, #0x1a4        ← mode 0644 は正しくw2に居るが読まれない
$ ls -l out1.fldj      (umask 0022, fresh create)
-rwxr-----  ... 0740  ← 0644に非ず
$ (umask 0; ./fieldc …) && ls -l
-----w----  ... 0020  ← run条件で変わる = stack garbage確定
```
影響: ①副作用の非決定性 (godseedの外道) ②garbageが r--/--- 系なら後続tool読めず (旧finding
`--wx------` @accept.sh workaround の正体 = 同一bug) ③最悪ケース overwrite時 EACCES → 間欠fail。
要求: mode を stack 経由で渡す修正 (`str w2,[sp]` 系) または raw svc 化。修正後 accept.sh の
workaround (rm -f + chmod u+r) を撤去せよ — workaround残置 = 未修正の証。

### N1 (継承・未クローズ) — "non-standard prime" 文言逆転
audit-lang F1審の指摘。commit 3d6b9342 題が未だ "(0x0000_0100_0000_01b3, non-standard)" — 逆。
0x100000001b3 = FNV-1a 64 **標準**素数。後続が field 側を誤って"直し"に来る地雷。文言訂正を要求し続ける。

### UNVERIFIED 明示
- cross-host byte同一性 (C3): host 1個のみ。
- 觀 reject 実測 (A6): CONTRACT曰く emit MUST reject — lexer kind10 化は済 (049a7df7)、emit側 reject path の gate 実測は審時に行う。
- accept.sh の ASM byte-exact: commit 6f03c91a で PASS 記録。審時に再走 (echo禁 = 異路則により審自身のrunで)。

---

## 既存資産表

| # | 資産 | 所在 (枝:path) | 種別 | 状態 |
|---|------|----------------|------|------|
| 1 | CONTRACT v1 + V1 reservation | field/lang-asm:packages/fieldlang-asm/CONTRACT.md | spec(frozen) | token ABI·glyph grammar·FLDJ target·觀=kind10 reserved |
| 2 | 言語設計書 V0 | field/lang:docs/design/2026-08-01-field-lang.md | spec FINAL 179行 | corpses: Rust compiler·floats·varargs·:new-batch |
| 3 | field contract (W1-W8·laws·gates) | field/lang-asm:docs/design/2026-08-01-field-contract.md | spec | ultradeterminism条·signatures frozen |
| 4 | driver.s 162行 | field/lang-asm (源自 field/lang-driver cb2b0a9e) | 建碼 | libSystem5call·.bss 16MiB×3·defaults struct · **F2 bug保有** |
| 5 | lexer.s 148行 | field/lang-asm (源自 field/lang-lexer a737c484) | 建碼 | kind 1-7,9,10 · overflow/badchar/buffull err · gate 8/8 green |
| 6 | emit.s 187行 | field/lang-asm (源自 field/lang-emit 50db54be) | 建碼 | 界 override·觀 reject·arg-range err · 11 checks green |
| 7 | stub_lexer.s / stub_emit.s | field/lang-asm | gate-only stub | 統合前の driver 単体検証用 |
| 8 | test_lexer.c / test_emit.c | field/lang-asm | gate tool (C, test-only) | compiler に非ず |
| 9 | build.sh | field/lang-asm | build | as+ld -lSystem -syslibroot |
| 10 | examples ex1-3.fld | field/lang-asm:packages/fieldlang-asm/examples/ | fixture | 種+歩 / 界+撃+歩 / 種×2+縛+束+歩 |
| 11 | fieldlang-gate crate (flgate, 218行 lib) | field/lang:packages/fieldlang-gate/ | gate tool (Rust) | reference .fldj generator via frozen journal writer · 10 unit tests |
| 12 | accept.sh 4段門 | field/lang:packages/fieldlang-gate/accept.sh | gate | ref→golden cmp→asm cmp→digest×2 · F2 workaround込 |
| 13 | golden 01-03 .fldj | field/lang:packages/fieldlang-gate/golden/ | snapshot | byte regression tripwire |
| 14 | ACCEPT PASS 記録 | field/lang commit 6f03c91a | 証跡 | asm fieldc byte-exact ×3 · digest deterministic ×2 |
| 15 | field crate (journal.rs FROZEN FLDJ v1) | field/*:packages/field/src/ | core | Op enum 6 tags · World::replay+digest · tests/determinism.rs(W5)·capacity.rs(W8) |
| 16 | 審計画 F1-F5 + 審則 | audit-lang:docs/AUDIT-GAIAGO-LANG.md | 審資産 | 本書が継承 |
| 17 | tools/audit_lang_f1.py | audit-lang:tools/ | 審 tool | spec再導出型 F1 verifier · PASS 13語 bit-exact · N2: CI門入れ提案 |
| 18 | tracr bridge proof | tracr-bridge:bridge/{circulant_bridge.py,PROOF.md} | proof | A: bind=conv=circulant=FFT (8.9e-15) · B: exact recall 8/8 zero-training · C: relax+learn |
| 19 | mu-v0 suite | mu-v0:mu/{field_env,nets,mcts,online,tick_bench}.py + BENCH.md | learned側 | mean 0.91–4.1ms p99<8.33 · determinism bit-identical · UNVERIFIED marks付 |
| 20 | adversary binding verdict | field/lang-asm:docs/research/2026-08-01-field-adversary.md | spec根拠 | §1 one-space-one-opset · §2 compiled⊂learned |
| 21 | arm64 instruction dict (V1 self-hosting) | field/lang-asm commit 7c6bbd04 | 辞典 | as+objdump検証済 |
| 22 | replay_fldj gate example | field/lang-asm commit c714aec8 | gate | read_all→World::replay→digest |
| 23 | speakback-s0 + audit資産(言lane別系) | audit-lang:packages/speakback-s0/, tools/ | 隣接lane | F1審済·F3-F5審点 = 本書A/D軸と相互参照 |

死枝 (審側で確認済・再挑戦禁):
- Rust製compiler (purity decree: layer = gaialang源·手asm·arm64出力、compiler内にRust/C無し) — 理由: 自前主義·V1 self-hostの前提。fieldlang-gate は gate tool と明記され合法。
- floats可変長args言語仕様 (:new-batch等) — 設計書§5 non-goals。
- 審が建のgateを再走してPASSと言う事 — 異路則違反 (echo)。審は自前run+独立再導出のみ。

---

## 建方案到着後の審手順 (atom-2 予告)
1. 方案を Checklist A-D に写像 → 各項 PASS/FAIL/UNVERIFIED + 証拠原文。
2. F2 修正の有無 → 無ければ V0統合前ブロック推奨 (副作用非決定は godseed法に触れる)。
3. V1 (4D W×H×D×C64·界 extended arity·觀) が来た場合: D4 が核心門。grammar knows/compiler ignores の
   分離が保たれているか = V0 byte互換を壊さぬ事 (golden ×3 で実測)。
4. 報告は貼付則に従い端末原文のみ。echoは認めぬ。
