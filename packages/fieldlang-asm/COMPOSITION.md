# atom3 合成記録 — vishnu/atom3-trusted-closure

base = 556d007 場語：fieldc実門を固定

## 取込
- 451c56b 場語：実門拒絶群を閉ず (556d007 の子、直に枝先とす)
- 7ef3fb7 組立: driver.s の _open 可変長 ABI 修正 (別系統 fa07d09 基、cherry-pick -x → 8619002)

## 衝突と裁定 (driver.s 出力 open)
両系統が同一欠陥を別形で修す。衝突箇所を手裁定:
- 451c56b 形 `mov w2,#0x1A4 / str w2,[sp]` → 値は正、格納は 4 バイトのみ(上位 4 バイト不定)。
- 7ef3fb7 形 `mov x8,#0644 / str x8,[sp]` → 格納は正、値が誤。アセンブラは `0644` を
  八進でなく**十進 644** と読む(= 0o1204)。
- 採用 = 両者の正しい半分: `mov x8, #0x1A4` + `str x8,[sp]`。
  実測 perm = -rw-r--r-- (gate boundary 行で常時検証)。

## 追加 (71365aa)
driver.s: 読込バッファ満杯時の黙殺を絶つ。満杯 → 1 バイト探査読み。
  EOF(0) → 丁度 BUFSZ の入力として受理 · >0 → `Lerr_toolong` rc=1 明示拒絶 · <0 → `Lerr_read`。
既存の部分読みループは不変。∴ 全入力は受理か拒否のいずれかに落ちる。

## 門 (gate.sh)
既存: 例 3 件 parity+digest · 拒絶 5 件。
追加 `check_size_boundary`:
- 丁度 BUFSZ(注釈で詰めた有効プログラム) → rc0 · perm 0644 ·
  digest が無詰め同プログラムと一致(独立経路の意味検証、金値の焼付でない)
- BUFSZ+1 → rc≠0 · 出力ファイル非生成 · stderr に明示語
- 17MiB → 同上
- 拒絶時に既存出力ファイルを切詰めぬこと

## 変異検査 (真)
`cbz x2, Lread_full` → `Lread_done` に戻すと門は `gate: oversize over accepted` で落ちる。
∴ 門は黙殺退行を実際に捕える。

## 未検証 (UNVERIFIED)
- BUFSZ 超の入力で lexer/emit のバッファ上限 (TOKCAP/出力 16MiB) を越える経路は本 atom の対象外。
- Rust 経路無し = build.sh の grep 検査のみ(既存 451c56b の仕掛け)。
