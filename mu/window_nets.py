"""mu/window_nets.py -- mu-inside atom-1 段2: tiny (patch,action)->patch'
predictor + a null constant-output baseline (追令② GW-3e). Kept in a
SEPARATE file from mu/nets.py -- this worktree has other atom lanes actively
touching nets.py/GATE-INSIDE.txt concurrently (observed: uncommitted
GATE-INSIDE.txt diff mid-session that this atom did not make); isolating new
code in new files avoids stepping on that work, same rationale as
WindowEnv living in world_env.py as an additive subclass, not an edit.
"""
import mlx.core as mx
import mlx.nn as nn

K_DEFAULT = 16
NUM_WINDOW_ACTIONS = 5
CHANNELS = 2
HIDDEN = 128


class WindowPredictor(nn.Module):
    """(patch_t (B,K,K,2), action_onehot (B,5)) -> patch_{t+1} (B,K,K,2).
    Residual (patch_t + delta) -- same convention as mu/online.py's decoder
    path: a K*K*2-d output head has to carry the whole next patch; residual
    leaves it only the DYNAMICS instead of re-drawing the input."""

    def __init__(self, k=K_DEFAULT, num_actions=NUM_WINDOW_ACTIONS):
        super().__init__()
        self.k = k
        in_dim = k * k * CHANNELS + num_actions
        out_dim = k * k * CHANNELS
        self.fc1 = nn.Linear(in_dim, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, out_dim)

    def __call__(self, patch, a_onehot):
        b = patch.shape[0]
        x = patch.reshape(b, -1)
        x = mx.concatenate([x, a_onehot], axis=-1)
        h = nn.relu(self.fc1(x))
        delta = self.fc2(h).reshape(b, self.k, self.k, CHANNELS)
        return patch + delta


class NormWindowPredictor(nn.Module):
    """atom-2 段1: scale-normalised residual predictor.

    corpse it replaces: WindowPredictor trained on raw MSE. Patch energies
    span decades within one run (1e-18 empty .. 5e-3 wavefront, 段2 measured),
    so a raw-MSE online stream is dominated by whichever ticks happen to be
    loud, and the quiet ticks -- where the shift/dynamics structure is just as
    real -- contribute ~nothing. Here each sample is divided by its own RMS
    before the net sees it and multiplied back after, so the net solves ONE
    scale-free problem: given a normalised patch and an action, what is the
    normalised next patch.

    Structure to be learned is non-trivial (the reason persistence is beatable
    here at all): 4 of the 5 actions TRANSLATE the observation window by
    move_step cells, so patch_{t+1} is (mostly) a SHIFT of patch_t -- known
    exactly for the k-move_step overlapping columns/rows, unknowable for the
    newly-entered band. Persistence predicts no shift at all.
    """

    def __init__(self, k=K_DEFAULT, num_actions=NUM_WINDOW_ACTIONS, hidden=512):
        super().__init__()
        self.k = k
        in_dim = k * k * CHANNELS + num_actions
        out_dim = k * k * CHANNELS
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def __call__(self, patch, a_onehot):
        b = patch.shape[0]
        x = patch.reshape(b, -1)
        s = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + 1e-12)
        xn = x / s
        h = nn.gelu(self.fc1(mx.concatenate([xn, a_onehot], axis=-1)))
        h = nn.gelu(self.fc2(h))
        delta = self.fc3(h)
        return ((xn + delta) * s).reshape(b, self.k, self.k, CHANNELS)


class ConstPredictor(nn.Module):
    """追令② GW-3e null baseline: prediction = ONE learned constant patch,
    trained by the identical online MSE loss/optimizer/trajectory as
    WindowPredictor, but architecturally BLIND to patch_t and action -- no
    residual, no input dependence at all (not `patch + const`, literally
    `const`), so it cannot piggy-back on the persistence signal (GW-3c) or on
    any real dynamics. If this baseline's err_rel falls as much as
    WindowPredictor's, the fall is a scale/normalisation artifact, not
    learned world dynamics -- gate_window.py's GW-3e compares the two."""

    def __init__(self, k=K_DEFAULT):
        super().__init__()
        self.k = k
        self.const = mx.zeros((k, k, CHANNELS))

    def __call__(self, patch, a_onehot):
        b = patch.shape[0]
        return mx.broadcast_to(self.const[None], (b, self.k, self.k, CHANNELS))


def action_onehot(a, num_actions=NUM_WINDOW_ACTIONS):
    return (mx.arange(num_actions)[None, :] == a[:, None]).astype(mx.float32)


def param_count(module: nn.Module) -> int:
    total = 0

    def rec(tree):
        nonlocal total
        if isinstance(tree, dict):
            for v in tree.values():
                rec(v)
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                rec(v)
        else:
            total += tree.size

    rec(module.parameters())
    return total


if __name__ == "__main__":
    net = WindowPredictor()
    cnet = ConstPredictor()
    print("WindowPredictor params:", param_count(net))
    print("ConstPredictor params:", param_count(cnet))
    patch = mx.zeros((3, K_DEFAULT, K_DEFAULT, CHANNELS))
    a = mx.array([0, 1, 4])
    out = net(patch, action_onehot(a))
    cout = cnet(patch, action_onehot(a))
    print("out", out.shape, "cout", cout.shape)
