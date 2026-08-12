# same attack but on the REAL training distribution: resets every 30 steps,
# seed+episode_idx, uniform random actions (eps=1.0), 300 steps.
import sys, numpy as np
sys.path.insert(0, "/Users/pascaldisse/projects/magic-crystal-mu-v0/mu")
from field_env import FieldEnv, NUM_ACTIONS

env = FieldEnv(seed=0)
obs = env.reset(seed=0)
rng = np.random.default_rng(123)
rs, sat, ep = [], 0, 0
for i in range(300):
    if i > 0 and i % 30 == 0:
        ep += 1; obs = env.reset(seed=0 + ep)
    a = int(rng.integers(NUM_ACTIONS))
    obs, r = env.step(a)
    rs.append(r); sat += (r >= env.reward_clip)
rs = np.array(rs)
print(f"episodic(30): reward std={rs.std():.4f} mean={rs.mean():.4f} clip-sat={100*sat/300:.2f}%")
print(f"h1 mean={rs[:150].mean():.4f} h2 mean={rs[150:].mean():.4f} (reward drift, non-stationarity check)")
