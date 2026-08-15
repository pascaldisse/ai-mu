# Apasmara atom9 — memory adversary

§base=`2b0cf304f82ebca1531924c41ac5ebb7f0ba7ad8`。

実門:

```text
packages/fieldlang-asm/gate.sh                         → OK
packages/fieldlang-asm/fieldrun/apasmara_atom9.sh       → OK
packages/fieldlang-asm/q30_wave_metal/gate.sh           → OK
packages/fieldlang-asm/fieldrun/teeth_kill.sh           → rc=3 SKIP-ENV
```

raw=`proof/apasmara-atom9-raw.log`。

- raw FLDJ literal header/ops、fieldc/golden/gen_fldj 非依存。
- slot0/slot1: `a=c25f8578f3e…` ≠ `b=a1b81f05d0b0…`。
- unwritten=`z=198b0bf449c9…` = repeated WriteRaw後zero=`repeat=198b0bf449c9…`。
- post-step write: `rotate=ef774c200b4e…` ≠ `resetprev=e6ede0a1c9eb…`。
- slot1→cur mutation: `b`→`c25f8578f3e…`、KILLED。
- 三pointer rotation mutation: `ef774c200b4e…`→`e6ede0a1c9eb…`、KILLED。
- mmap call mutation: rc=19・FLRO absent、KILLED。

旧門:

- `proof/apasmara-atom9-fieldc-gate.log`: golden ex1/ex2/ex3、shape、16MiB exact/over、全OK。
- `proof/apasmara-atom9-metal.log`: frozen138 scalar=NEON=Metal、shader teeth13 + host teeth11 + guard solo、全red=ok。
- `proof/apasmara-atom9-teeth.log`: A8 stage1 KILLED=6 + backend green; stage2=A10 `replay_fldj` absent → rc=3 SKIP-ENV。

partial-cleanup:

- `fieldrun.s:453-461`: `Larena` mmap失敗→`Lreject`→`_exit`。
- `fieldrun.s:151-152,233-241`: file/cur/prev/scratch は順次 mmap、失敗時 munmap unwind 無。
- `apasmara_atom9.sh` の `nm -u fieldrun`: `_munmap` absent。
- 判定=process-exit cleanup のみ。partial allocation中の継続/回復契約は無く、UNVERIFIED。

死枝:

- full teeth=A1..A7:因=`../../../target/debug/examples/replay_fldj` absent、明示 rc=3、偽緑化せず。
- mmap partial unwind 完全実測:因=失敗路は即 `_exit`、生存過程を観測不能。
