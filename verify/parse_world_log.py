# ②④ parse of the committed 段3 log (online_world_log.txt @64ef041d) + lr0 control.
# Independent recompute: action-share windows, raw-r trend, value/reward loss
# windows, RMS-replay side effects (first-step inflation, reset boundaries,
# scale drift, 200-tick staleness), final-window histogram (collapse check).
import re, sys
import numpy as np

def load(path):
    rows = []
    for line in open(path):
        d = dict(re.findall(r"(\w+)=([-0-9.e]+)", line))
        rows.append({k: float(v) if k != "action" else int(float(v)) for k, v in d.items()})
    return rows

rows = load("/Users/pascaldisse/projects/mc-mu-world-adv/mu/online_world_log.txt")
n = len(rows)
act = np.array([r["action"] for r in rows])
rraw = np.array([r["r"] for r in rows])
vloss = np.array([r["value"] for r in rows])
rloss = np.array([r["reward"] for r in rows])
print(f"n={n}")

W = 150
shares = [int((act[i:i+W] == 2).sum()) for i in range(0, n - n % W, W)]
print("act2 share per 150:", shares, "(claimed 29->96->104->42->83->111)")
print(f"raw r: first150={rraw[:150].mean():.4f} last150={rraw[-150:].mean():.4f} (claimed 0.1595->1.0125)")
print(f"value loss: first150={vloss[:150].mean():.3f} last150={vloss[-150:].mean():.3f} (claimed 0.680->0.107)")
print(f"reward loss: first150={rloss[:150].mean():.4f} last150={rloss[-150:].mean():.4f}")

# RMS replay (their formula: r / (sqrt(acc/n)+1e-3), acc += r^2, never reset)
acc, cnt, rn = 0.0, 0, []
for r in rraw:
    acc += r * r; cnt += 1
    rn.append(r / ((acc / cnt) ** 0.5 + 1e-3))
rn = np.array(rn)
rms = np.sqrt(np.array([np.cumsum(rraw**2)][0]) / np.arange(1, n + 1))
print("\n-- RMS side-effect hunt --")
print(f"step0: raw={rraw[0]:.5f} normalized={rn[0]:.5f} (self-normalizing: any nonzero r0 -> ~1)")
nz = rraw > 0
print(f"normalized of first 10 nonzero raw: {rn[np.where(nz)[0][:10]].round(3)}")
print(f"raw/std {rraw[nz].std():.3f} -> scaled/std {rn[nz].std():.3f} (variance shape change)")
print(f"scale drift: rms_t first150={rms[149]:.4f} mid={rms[n//2]:.4f} last={rms[-1]:.4f}")
lag = 200
if n > lag:
    drift = np.abs(rms[lag:] - rms[:-lag]) / rms[lag:]
    print(f"200-tick staleness (buffer_cap): max rel rms drift={drift.max():.3f} mean={drift.mean():.3f}")
# episode boundaries (episode_len=30): boundary = steps where step%30==0 (post-reset region)
steps = np.arange(n)
bnd = (steps % 30) <= 1
mid = (steps % 30).between if hasattr(steps % 30, "between") else ((steps % 30) >= 10) & ((steps % 30) <= 20)
print(f"normalized r at reset boundary(±1): mean={rn[bnd].mean():.3f} vs mid-episode: mean={rn[mid].mean():.3f}")
print(f"value-loss floor check: var(scaled r)={rn.var():.3f} vs last150 value MSE={vloss[-150:].mean():.3f}")

# collapse check: final-150 full histogram
hist = np.bincount(act[-150:], minlength=8)
print(f"\nfinal150 action histogram: {hist.tolist()} (top share {hist.max()}/150; collapse=150)")
ent = -np.sum((hist/150) * np.log(np.maximum(hist/150, 1e-9)))
print(f"final150 entropy: {ent:.3f} nats (max {np.log(8):.3f})")

# lr0 control (runA): act2 share must sit at chance -> gate does not fake-pass
ra = load("/Users/pascaldisse/projects/mc-mu-world-adv/verify/runA_lr0.txt")
actA = np.array([r["action"] for r in ra])
print(f"\nlr=0 control: act2 share per 150: {[int((actA[i:i+150]==2).sum()) for i in range(0, len(actA)-len(actA)%150, 150)]} (chance=18.75)")
