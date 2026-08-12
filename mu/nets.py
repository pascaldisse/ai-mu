"""mu/nets.py — MuZero-style tiny nets over a flattened field slice.

State space: 128x128 = 16384 field slice (adversary amendment §1, docs/research/
2026-08-01-field-adversary.md Q1/Q4: one d=W*H space, no separate renderer module).
Layout: NHWC (mlx.nn.Conv2d convention). CHANNELS = 2 (GENESIS-II arc2 段1:
[cur, prev] = the COMPLETE state of the real field plane, straight out of
mu-serve / mu/world_env.py). Arc1's 3 channels (heat/flow/pressure) belonged
to mu/field_env.py, a python PDE toy — retired as Mu's world: Mu learns from
the engine's own field or it is not learning the world. Channel count is a
Conv2d weight-shape param, not an extra op: dispatch count unchanged.

Three nets, MuZero roles:
  ReprNet  h : obs (B,128,128,1)              -> latent s      (B, LATENT_DIM)
  DynNet   g : (s, a-onehot)                  -> (s', r_pred)  (B,LATENT_DIM),(B,1)
  PredNet  f : s                               -> (policy_logits, value)
  Decoder  d : s -> field slice (B,128,128,1)  [predict-next-field-slice target;
               mirrors ReprNet exactly (verified shapes), used only in train.py]

All nets are tiny conv/MLP, param budget 50k-1M total (see param_count() below).
DynNet and PredNet each use ONE combined output head (split after) instead of
two separate Linear heads, so each is 2 matmul ops (trunk+head), not 3. This
keeps the fused interactive tick (repr 3 + dyn 2 + pred 2 = 7 matmul/conv ops)
inside the 4-8 dispatch hard constraint (see tick_bench.py docstring for the
measurement method/caveats).
"""
import mlx.core as mx
import mlx.nn as nn

GRID = 128           # field slice side; GRID*GRID = 16384 = state space (adversary §1)
LATENT_DIM = 64
NUM_ACTIONS = 8       # discrete "poke" locations/impulses (see field_env.py)
HIDDEN = 128
CHANNELS = 2          # cur, prev — real field plane state (arc2 段1; arc1 toy was 3)


class ReprNet(nn.Module):
    """h(obs) -> latent. obs: (B,128,128,CHANNELS) -> (B,LATENT_DIM). 3 ops (conv,conv,linear)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(CHANNELS, 8, kernel_size=5, stride=4, padding=2)
        self.c2 = nn.Conv2d(8, 16, kernel_size=5, stride=4, padding=2)
        self.fc = nn.Linear(16 * 8 * 8, LATENT_DIM)

    def __call__(self, obs):
        x = nn.relu(self.c1(obs))
        x = nn.relu(self.c2(x))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class DecoderNet(nn.Module):
    """s -> predicted field slice (B,128,128,CHANNELS). Mirrors ReprNet, exact
    inverse shapes (verified: 64 -> 1024 -> 8x8x16 -> 32x32x8 -> 128x128xCHANNELS).
    Used only in train.py/online.py (slice-prediction loss), not in the
    interactive tick."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 16 * 8 * 8)
        self.ct1 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=4, padding=2, output_padding=3)
        self.ct2 = nn.ConvTranspose2d(8, CHANNELS, kernel_size=5, stride=4, padding=2, output_padding=3)

    def __call__(self, s):
        x = self.fc(s)
        x = x.reshape(x.shape[0], 8, 8, 16)
        x = nn.relu(self.ct1(x))
        return self.ct2(x)


class DynNet(nn.Module):
    """(s, a_onehot) -> (s', r). s: (B,LATENT_DIM), a: (B,NUM_ACTIONS).
    2 ops (trunk, combined head) -- head outputs LATENT_DIM+1 columns, split
    into next_state / reward (no separate reward Linear)."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(LATENT_DIM + NUM_ACTIONS, HIDDEN)
        self.head = nn.Linear(HIDDEN, LATENT_DIM + 1)

    def __call__(self, s, a_onehot):
        x = nn.relu(self.trunk(mx.concatenate([s, a_onehot], axis=-1)))
        out = self.head(x)
        return out[:, :LATENT_DIM], out[:, LATENT_DIM:LATENT_DIM + 1]


class PredNet(nn.Module):
    """s -> (policy_logits, value). s: (B,LATENT_DIM).
    2 ops (trunk, combined head) -- head outputs NUM_ACTIONS+1 columns, split
    into policy_logits / value (no separate value Linear)."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.head = nn.Linear(LATENT_DIM, NUM_ACTIONS + 1)

    def __call__(self, s):
        x = nn.relu(self.trunk(s))
        out = self.head(x)
        return out[:, :NUM_ACTIONS], out[:, NUM_ACTIONS:NUM_ACTIONS + 1]


class MuNet(nn.Module):
    """Container for the whole MuZero-style net trio + decoder, so params()
    reports one combined tree (used for optimizer state and param_count)."""

    def __init__(self):
        super().__init__()
        self.repr = ReprNet()
        self.dyn = DynNet()
        self.pred = PredNet()
        self.decoder = DecoderNet()


def action_onehot(a):
    """a: (B,) int array -> (B, NUM_ACTIONS) float one-hot."""
    return (mx.arange(NUM_ACTIONS)[None, :] == a[:, None]).astype(mx.float32)


def param_count(module: nn.Module) -> int:
    return tree_flatten_sizes(module.parameters())


def tree_flatten_sizes(tree) -> int:
    total = 0
    if isinstance(tree, dict):
        for v in tree.values():
            total += tree_flatten_sizes(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            total += tree_flatten_sizes(v)
    else:
        total += tree.size
    return total


if __name__ == "__main__":
    net = MuNet()
    n_repr = param_count(net.repr)
    n_dyn = param_count(net.dyn)
    n_pred = param_count(net.pred)
    n_dec = param_count(net.decoder)
    print(f"repr={n_repr} dyn={n_dyn} pred={n_pred} decoder={n_dec}")
    print(f"total(repr+dyn+pred)={n_repr + n_dyn + n_pred}")
    print(f"total(all incl. decoder)={n_repr + n_dyn + n_pred + n_dec}")

    obs = mx.zeros((2, GRID, GRID, CHANNELS))
    s = net.repr(obs)
    a = mx.array([0, 3])
    s2, r = net.dyn(s, action_onehot(a))
    p, v = net.pred(s)
    dec = net.decoder(s2)
    print("s", s.shape, "s2", s2.shape, "r", r.shape, "p", p.shape, "v", v.shape, "dec", dec.shape)
