# Kali 審 — round10 teeth

基=7f4776d5590c78e61865192a0033385b77a1d4fe · detached獨立再審。

## 結

blocker=零。前回三穴=閉。

- 非整列: `unaligned-tooth` は `x1 &= -16`。aligned入力=恒等、+1B入力のみ誤pointer。red。
- alias: `alias-tooth` は `cur==prev` 時のみ `prev += 4`。同一基址scalar/NEON照合でred。
- out-guard: one-past-end store。出力本体不変、32B上下guard照合のみでred。

## 直接敵注入

```text
INJ cur-Ldone RED=YES
INJ prev-Ldone RED=YES
INJ cur-partial-byte RED=YES
INJ cfg-entry RED=YES
wave_runner: 133 ok
abi probe ok
```

`cur-Ldone`=Avidya形 `Ldone`後 `str wzr,[cur]`。prev・1-byte XOR・cfg entry破壊も逃走せず。cfg=入力custody比較外、期待出力/走者失敗でred。

## 12 teeth 非空洞

各 `mut <name>` 呼出のみをno-op化、他11=実変異のままgate再走。全green:

```text
rounding-half q-scale lane-order sat-negative vec-boundary scalar-fallback
x18-tooth unaligned-tooth alias-tooth out-guard-tooth
input-custody-tooth input-custody-scalar-tooth
```

各元変異=red。故に各redは当該呼出起因、常時失敗空洞=否。

## 残渣・用具

単独 `./gate.sh` 後 `git status --porcelain`=空、`*.save`=0。

build実行=`dirname`,`as`×4,`xcrun --show-sdk-path`×2,`ld`×2。追加compiler/interpreter=零（gate自身のshell host utilは除外）。`nm -u wave_neon.o`=空、外部branch=0、`otool`=ld1/saddl/saddl2/smull/sqxtn群。真NEON=確認。

## 旧門 clean detached

```text
fieldc gate: assembly closure OK; status=clean
q30 gate: assembly closure OK (no speed claim); status=clean
q30 metal: GPU pipeline, 64 threads, 77 scalar+NEON byte parity, mutation red=ok; rc=0
```

死枝=並列旧門初回: source mutation/trap競合で偽残渣・Metal rc=3。clean reset後、各門単独再走へ転換。未驗=零。
