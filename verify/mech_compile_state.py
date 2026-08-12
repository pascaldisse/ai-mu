# ① mechanism test (reviewer's own path, no env): does mx.compile capture
# net.state BY VALUE (frozen acting policy) and does inputs=state fix it?
# update applied via the proven compiled pattern (as online.py does).
import sys
sys.path.insert(0, "/Users/pascaldisse/projects/mc-mu-world-adv/mu")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from nets import MuNet, GRID, CHANNELS

mx.random.seed(0)
net = MuNet()
opt = optim.Adam(learning_rate=1e-2)
state = [net.state, opt.state]

def _select(obs):
    s = net.repr(obs)
    logits, value = net.pred(s)
    return logits, value

sel_frozen = mx.compile(_select)                      # arc1 form (the bug)
sel_live   = mx.compile(_select, inputs=state)        # arc2 段3 fix

def loss_fn(net, obs):
    s = net.repr(obs)
    logits, value = net.pred(s)
    return (logits ** 2).sum() + (value ** 2).sum()

def _step(obs):
    _, g = nn.value_and_grad(net, loss_fn)(net, obs)
    opt.update(net, g)
    return None

step_fn = mx.compile(_step, inputs=state, outputs=state)

obs = mx.random.normal((1, GRID, GRID, CHANNELS))

def pdigest():
    return hash(tuple(v.sum().item() for _, v in tree_flatten(net.parameters())))

lf1, _ = sel_frozen(obs); ll1, _ = sel_live(obs)
d1 = pdigest()
mx.eval(lf1, ll1, state)

step_fn(obs)          # one real gradient step inside compiled fn (proven pattern)
mx.eval(state)        # state now holds updated params
d2 = pdigest()

lf2, _ = sel_frozen(obs); ll2, _ = sel_live(obs)
mx.eval(lf2, ll2)

print("params moved by update:", d1 != d2)
print("frozen-compile logits identical after update:", bool(mx.array_equal(lf1, lf2)),
      "(True = arc1 bug real: acting net blind to learning)")
print("inputs=state logits identical after update:", bool(mx.array_equal(ll1, ll2)),
      "(False = fix works: acting net sees learning)")
print("max|dlogits| live:", mx.abs(ll2 - ll1).max().item())
