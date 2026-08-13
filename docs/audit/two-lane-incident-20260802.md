# AUDIT — one worktree / one lane violation indicators, worktree mc-mu-inside
2026-08-02, recorder: Bhairava lane (narigo-msbbtukd8cbqdb). FACTS ONLY.

Prior law: HANDOFF-MU-INSIDE.md "Law for the next lane: one worktree, one lane"
(written after the 2026-08-01 mid-run binary-swap incident).

Observed facts, this session:
1. Session start (~2026-08-02 06:56 local): `git status` showed uncommitted
   changes not made by this lane: ` M mu/window_nets.py`, `?? mu/window_eval.py`.
2. A `read` of `mu/window_online.py` early in the session showed
   `PREDICTORS = ("window", "const")` (no "norm").
3. A `grep` of the same file minutes later, in the same session, showed
   `PREDICTORS = ("window", "const", "norm")` and `git status` then also showed
   ` M mu/window_online.py` — i.e. the file changed on disk between the two
   reads. This lane had made no successful write to it (its one edit attempt
   returned match-failure, which writes nothing).
4. mtimes at detection (epoch, via `stat -f "%m"` at epoch 1785646650):
   `mu/window_nets.py` 1785646601 · `mu/window_online.py` 1785646616 ·
   `mu/window_eval.py` (see commit below).
5. Process check at detection: `ps aux` and `lsof +D mu/` showed no live
   python/mu process holding or writing these files.
6. Defense action: all three files committed immediately as-found:
   commit `78992d8a` ("atom-2r 骸救出"). Subsequent work built on committed
   state; no further external modification was detected through this lane's
   later commits `80b0888b`, `a4d8ea20`.

No attribution is made. Items 1–6 are the complete factual record.
