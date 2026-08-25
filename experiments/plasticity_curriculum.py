"""Round 7: plasticity curriculum - do shift-experienced bases adapt better?

Stage-1 training, two ways (identical total optimizer exposure):
  VANILLA    : straight training to convergence
  CURRICULUM : cycles of [inject small random weight-strength shifts ->
               brief re-convergence], finishing with clean convergence

Then identical gated-residual adaptation (round-2 protocol) on the
shifted stream. Metrics: retain/learn RMSE + convergence speed
(RMSE checkpoint curve) over 5 seeds.

Relation to literature: perturb-and-recover approximates flat-minima
selection (cf. SAM, Foret 2021 - but random directions, not adversarial);
the question is whether flatness transfers to ADAPTATION quality of a
later gated expansion.
"""
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

SEEDS = [0, 1, 2, 3, 4]
STEPS = 900          # stage-2 adaptation budget
LR = 1e-3
BATCH = 128
DIM = 4
CENTER = np.array([0.85, 0.85, 0.15, 0.15])
SIGMA = 0.03         # relative weight-shift strength


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


class Trunk(nn.Module):
    def __init__(self, h=64):
        import torch.nn as nn
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


class Gated(nn.Module):
    def __init__(self, base):
        super().__init__()
        import torch.nn as nn
        self.base = base
        h = base.net[0].out_features
        self.value = nn.Sequential(nn.Linear(h, 64), nn.GELU(),
                                   nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(h, 32), nn.GELU(),
                                  nn.Linear(32, 1))
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


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        return float(np.sqrt(np.mean(
            (model(torch.tensor(x)).numpy() - y) ** 2)))


def train_stage1(seed, curriculum=False):
    """Returns converged base. Curriculum: 5 cycles of
    [shift weights -> 250 recovery steps], final 300 clean steps."""
    import torch.nn as nn
    rng = np.random.default_rng(400 + seed)
    torch.manual_seed(seed)
    model = Trunk(h=64)
    xb, yb = sample(600, rng, with_bump=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def run(steps, perturb=None):
        xt, yt = torch.tensor(xb), torch.tensor(yb)
        r = np.random.default_rng(rng.integers(1 << 30))
        for _ in range(steps):
            i = torch.tensor(r.integers(0, len(xt), BATCH))
            loss = ((model(xt[i]) - yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    if curriculum:
        for cycle in range(5):
            with torch.no_grad():
                for p in model.parameters():
                    scale = SIGMA * (p.detach().abs().mean() + 1e-8)
                    p.add_(torch.randn_like(p) * scale)
            run(250)                      # recover from the shift
        run(300)                          # clean landing
    else:
        run(1550)                         # equal total exposure
    return model


def adapt(base, x2, y2, seed, steps=STEPS):
    import torch.nn as nn
    torch.manual_seed(seed + 31)
    model = Gated(copy.deepcopy(base))
    opt = torch.optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=LR)
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    rng = np.random.default_rng(seed + 77)
    curve = []
    for s in range(steps):
        i = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = ((model(xt[i]) - yt[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 50 == 0:
            curve.append(((model(xt) - yt) ** 2).mean().item() ** 0.5)
    return model, curve


def main():
    rows = []
    curves = {"vanilla": [], "curriculum": []}
    for seed in SEEDS:
        for mode in ("vanilla", "curriculum"):
            base = train_stage1(seed, curriculum=(mode == "curriculum"))
            base_retain = rmse(base, *sample(2000, np.random.default_rng(
                999 + seed), with_bump=False))
            rng = np.random.default_rng(600 + seed)
            xl, yl = sample_shifted(2000, rng)
            xr, yr = sample(2000, rng, with_bump=False)
            m, curve = adapt(base, *sample_shifted(400, rng), seed)
            ret, lrn = rmse(m, xr, yr), rmse(m, xl, yl)
            curves[mode].append(curve)
            rows.append(dict(seed=seed, mode=mode,
                             base_retain=base_retain,
                             retain=ret, learn=lrn))
            print(f"[{mode:>10} s{seed}] base_retain={base_retain:.4f}  "
                  f"after-growth: retain={ret:.4f} learn={lrn:.4f}")

    print("\n=== mean +- std over {} seeds ===".format(len(SEEDS)))
    summary = {}
    for mode in ("vanilla", "curriculum"):
        rs = [r["retain"] for r in rows if r["mode"] == mode]
        ls = [r["learn"] for r in rows if r["mode"] == mode]
        bs = [r["base_retain"] for r in rows if r["mode"] == mode]
        cv = np.array(curves[mode])
        summary[mode] = dict(retain_mean=float(np.mean(rs)),
                             retain_std=float(np.std(rs)),
                             learn_mean=float(np.mean(ls)),
                             learn_std=float(np.std(ls)))
        print("{:>10}:  base_retain {:6.4f}+-{:.4f}   "
              "post-growth retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
              .format(mode, np.mean(bs), np.std(bs),
                      np.mean(rs), np.std(rs), np.mean(ls), np.std(ls)))
        print("            adaptation curve:", " ".join(
            f"{v:.3f}" for v in cv.mean(axis=0)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_plasticity_curriculum.json")
    json.dump(dict(rows=rows, summary=summary,
                   curves={k: np.array(v).tolist()
                           for k, v in curves.items()}),
              open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
