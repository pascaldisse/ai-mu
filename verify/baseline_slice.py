# adversary baseline: is slice-MSE ~4e-4 evidence of learning, or vacuous?
# zero-predictor + persistence-predictor MSE vs next_obs, same env, 300 steps.
import sys, numpy as np
sys.path.insert(0, "/Users/pascaldisse/projects/magic-crystal-mu-v0/mu")
from field_env import FieldEnv, NUM_ACTIONS

env = FieldEnv(seed=0)
obs = env.reset(seed=0)
rng = np.random.default_rng(123)
mz, mp, rs, sat = [], [], [], 0
for i in range(300):
    a = int(rng.integers(NUM_ACTIONS))
    nxt, r = env.step(a)
    mz.append(float(np.mean(nxt ** 2)))
    mp.append(float(np.mean((nxt - obs) ** 2)))
    rs.append(r); sat += (r >= env.reward_clip)
    obs = nxt
mz, mp, rs = np.array(mz), np.array(mp), np.array(rs)
h = 150
print(f"zeros MSE      : all={mz.mean():.6f} h1={mz[:h].mean():.6f} h2={mz[h:].mean():.6f}")
print(f"persistence MSE: all={mp.mean():.6f} h1={mp[:h].mean():.6f} h2={mp[h:].mean():.6f}")
print(f"learned decoder (claimed): h1=0.00041 h2=0.00037")
print(f"reward: std={rs.std():.4f} mean={rs.mean():.4f} clip-sat={100*sat/300:.2f}%")
print(f"obs absmax end: {np.abs(obs).max():.4f}")
