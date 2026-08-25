"""Round 5b: difference-map gate initialization.

At growth time, compare the NEW stream against the OLD data footprint in
feature space. Pre-train the gated-residual's GATE on that difference
(labels: 1 = resembles new data, 0 = old data), so the correction starts
life already open where the world changed. Value head remains zero-init ->
function preserved exactly at init. Gate pre-training uses ONLY stream
membership labels (no targets) - it is pure data-geometry information.

Claim to test: map-initialized gate => faster error descent than random
gate init, same or better retention, equal final floor.
"""
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

SEEDS = [0, 1, 2, 3, 4]
STEPS = 900
LR = 1e-3
BATCH = 128
DIM = 4
CENTER = np.array([0.85, 0.85, 0.15, 0.15])


def f_base(x):
    return (0.5 * np.sin(2 * math.pi * x[:, 0]) + 0.3 * x[:, 1] ** 2
            + 0.2 * x[:, 2] + 0.1 * x[:, 3] * x[:, 0])


def bump(x):
    d2 = np.sum((x - CENTER) ** 2, axis=1)
    return 0.6 * np.exp(-d2 / (2 * 0.20 ** 2))


def sample(n, rng, with_bump):
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    y = f_base(x) + (bump(x) if with_bump else 0.0)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


def sample_shifted(n, rng):
    x = np.clip(CENTER + rng.normal(0, 0.25, size=(n, DIM)), 0.0, 1.0)
    y = f_base(x) + bump(x)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


class Gated(nn.Module):
    def __init__(self, base, units=64):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()
        h = base.net[0].out_features
        self.value = nn.Sequential(nn.Linear(h, units), nn.GELU(),
                                   nn.Linear(units, 1))
        self.gate = nn.Sequential(nn.Linear(h, units // 2), nn.GELU(),
                                  nn.Linear(units // 2, 1))
        with torch.no_grad():
            self.value[-1].weight.zero_()
            self.value[-1].bias.zero_()

    def forward(self, x):
        with torch.no_grad():
            b = self.base(x)
        hh = self.base.net(x)
        corr = self.value(hh).squeeze(-1) * torch.sigmoid(
            self.gate(hh)).squeeze(-1)
        return b + corr


def pretrain_gate_on_difference(model, x_old, x_new, epochs=300):
    """BCE-trains the gate head: 1 for new-stream features, 0 for old."""
    feats_old = model.base.net(torch.tensor(x_old)).detach()
    feats_new = model.base.net(torch.tensor(x_new)).detach()
    X = torch.cat([feats_old, feats_new])
    Y = torch.cat([torch.zeros(len(feats_old)),
                   torch.ones(len(feats_new))])
    opt = torch.optim.Adam(model.gate.parameters(), lr=1e-3)
    for _ in range(epochs):
        idx = torch.randint(0, len(X), (128,))
        logits = model.gate(X[idx]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def train_track(model, x2, y2, seed, steps=STEPS):
    """Returns RMSE curve sampled every 50 steps."""
    torch.manual_seed(seed + 777)
    opt = torch.optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=LR)
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    rng = np.random.default_rng(seed)
    curve = []
    xl, yl = None, None
    for s in range(steps):
        idx = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 50 == 0:
            curve.append(loss.item())
    return curve


def main():
    results = {"random": [], "mapinit": []}
    for seed in SEEDS:
        rng = np.random.default_rng(900 + seed)
        xr, yr = sample(2000, rng, with_bump=False)
        xl, yl = sample(2000, rng, with_bump=True)
        x_new_full, y_new_full = sample_shifted(400, rng)

        torch.manual_seed(seed)
        xb, yb = sample(600, rng, with_bump=False)
        base = TrunkLike()
        fit_base(base, xb, yb, seed)
        pristine = copy.deepcopy(base)

        # hold out validation slice from the stream
        perm = rng.permutation(len(x_new_full))
        vi, ti = perm[:80], perm[80:]
        xv, yv = x_new_full[vi], y_new_full[vi]
        xtr, ytr = x_new_full[ti], y_new_full[ti]

        curves = {}
        for variant in ("random", "mapinit"):
            torch.manual_seed(seed + 31)
            m = Gated(copy.deepcopy(pristine))
            if variant == "mapinit":
                # difference labels come free with the stream:
                # new-stream rows vs a sample of the stage-1 corpus
                pretrain_gate_on_difference(m, xb, xtr, epochs=300)
            c = train_track(m, xtr, ytr, seed)
            curves[variant] = c
            results[variant].append(dict(
                seed=seed, curve=c,
                retain=rmse(m, xr, yr), learn=rmse(m, xl, yl)))

        r_rand = results["random"][-1]
        r_map = results["mapinit"][-1]
        print(f"[seed {seed}] final learn: random={r_rand['learn']:.4f} "
              f"mapinit={r_map['learn']:.4f}   "
              f"retain: {r_rand['retain']:.4f} / {r_map['retain']:.4f}")

    print("\n=== mean over seeds ===")
    for variant in ("random", "mapinit"):
        ls = [r["learn"] for r in results[variant]]
        rs = [r["retain"] for r in results[variant]]
        curves = np.array([r["curve"] for r in results[variant]])
        mc = curves.mean(axis=0)
        # steps-to-half-of-final-descent proxy: first checkpoint below
        # (start+final)/2
        start, final = mc[0], mc[-1]
        half = (start + final) / 2
        k = next((i for i, v in enumerate(mc) if v <= half), len(mc))
        print("{:>8}: learn {:6.4f}+-{:.4f}  retain {:6.4f}+-{:.4f}  "
              "steps-to-half ~{}".format(
                  variant, np.mean(ls), np.std(ls), np.mean(rs), np.std(rs),
                  (k + 1) * 50))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_gate_map_init.json")
    json.dump(results, open(out, "w"), indent=2)
    print("saved:", out)


class TrunkLike(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def fit_base(model, x, y, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt, yt = torch.tensor(x), torch.tensor(y)
    rng = np.random.default_rng(seed + 5)
    for _ in range(STEPS):
        i = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = ((model(xt[i]) - yt[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


if __name__ == "__main__":
    main()
