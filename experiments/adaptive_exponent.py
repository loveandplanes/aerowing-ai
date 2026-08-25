"""Round 6: shift-coupled loss exponent (user proposal).

Idea: L = mean(|e|^p) with p adapted to HOW DIFFERENT the new stream is
from the stage-1 corpus:
    big distribution change -> larger p  (aggressive correction)
    similar data            -> smaller p (gentle, noise-robust refinement)

Shift magnitude D = normalized feature-space distance between the new
stream and the stage-1 corpus. p = clip(1 + 3*D, 1, 4).

Protocols chosen so a FIXED p is wrong at one extreme or the other:
  EASY : stream near the old distribution, but 12% of labels carry heavy
         noise spikes -> high p lets spikes dominate (bad), low p robust.
  HARD : clean labels, strongly shifted concentrated stream -> low p
         adapts too slowly (bad), high p corrects aggressively.

Arms: fixed p=1, fixed p=2 (MSE), fixed p=4, ADAPTIVE (ours).
All arms: identical plain fine-tune of the same stage-1 base, same seeds.
"""
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

SEEDS = [0, 1, 2, 3, 4]
STEPS = 700
LR = 1e-3
BATCH = 128
DIM = 4


def f_base(x):
    return (0.5 * np.sin(2 * math.pi * x[:, 0]) + 0.3 * x[:, 1] ** 2
            + 0.2 * x[:, 2] + 0.1 * x[:, 3] * x[:, 0])


def sample_base(n, rng):
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    y = f_base(x)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


def sample_easy(n, rng):
    """Near the old distribution; 12% heavy label-noise spikes."""
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    y = f_base(x) + 0.04 * np.sin(6 * math.pi * x[:, 0])
    y = y + rng.normal(0, 0.01, n)
    spikes = rng.uniform(size=n) < 0.12
    y[spikes] += rng.normal(0, 0.35, int(spikes.sum()))
    return x.astype(np.float32), y.astype(np.float32)


def sample_hard(n, rng):
    """Strongly shifted: concentrated corner stream, clean labels,
    new localized structure."""
    c = np.array([0.85, 0.85, 0.85, 0.15])
    x = np.clip(c + rng.normal(0, 0.18, size=(n, DIM)), 0.0, 1.0)
    d2 = ((x[:, 0] - c[0]) ** 2 + (x[:, 1] - c[1]) ** 2
          + ((x[:, 2] - c[2]) / 0.5) ** 2 + (x[:, 3] - c[3]) ** 2)
    y = f_base(x) + 0.9 * np.exp(-d2 / (2 * 0.15 ** 2)) \
        + 0.25 * x[:, 2]
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


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def shift_distance(x_new, x_old_ref):
    """Normalized feature-space distance between stream and corpus."""
    mn_n, sd_n = x_new.mean(axis=0), x_new.std(axis=0)
    mn_o, sd_o = x_old_ref.mean(axis=0), x_old_ref.std(axis=0)
    d_mean = float(np.mean(np.abs(mn_n - mn_o)))
    d_std = float(np.mean(np.abs(sd_n - sd_o)))
    return float(np.clip((d_mean + d_std) / 2.0, 0.0, 1.0))


def p_loss(pred, target, p):
    e = pred - target
    return (torch.abs(e).clamp_min(1e-8) ** p).mean()


def run_ft(pristine, x2, y2, seed, p_mode, p_fixed, x_ref=None):
    torch.manual_seed(seed)
    model = copy.deepcopy(pristine)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    if p_mode == "adaptive":
        D = shift_distance(x2, x_ref)
        p = float(np.clip(1.0 + 3.0 * D, 1.0, 4.0))
    else:
        p = float(p_fixed)
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        idx = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = p_loss(model(xt[idx]), yt[idx], p)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, p, D if p_mode == "adaptive" else None


def main():
    rows = []
    ARMS = [("p1", "fixed", 1.0),
            ("p2", "fixed", 2.0),
            ("p4", "fixed", 4.0),
            ("ADAPT", "adaptive", None)]

    for proto, sampler in (("EASY", sample_easy), ("HARD", sample_hard)):
        for seed in SEEDS:
            rng = np.random.default_rng(1200 + seed)
            xr, yr = sample_base(2000, rng)
            xl, yl = sampler(2000, rng)
            x2, y2 = sampler(400, rng)

            torch.manual_seed(seed)
            xb, yb = sample_base(600, rng)
            base = Trunk(h=64)
            o = torch.optim.Adam(base.parameters(), lr=LR)
            xb_t, yb_t = torch.tensor(xb), torch.tensor(yb)
            r = np.random.default_rng(seed + 99)
            for _ in range(STEPS * 2):
                i = torch.tensor(r.integers(0, len(xb_t), BATCH))
                loss = ((base(xb_t[i]) - yb_t[i]) ** 2).mean()
                o.zero_grad(); loss.backward(); o.step()
            pristine = copy.deepcopy(base)

            for arm, p_mode, p_fixed in ARMS:
                model, p_used, D = run_ft(pristine, x2, y2, seed,
                                          x_ref=xb, p_mode=p_mode,
                                          p_fixed=p_fixed)
                rows.append(dict(protocol=proto, seed=seed, arm=arm,
                                 p=p_used, D=None if D is None else round(D, 4),
                                 retain=rmse(model, xr, yr),
                                 learn=rmse(model, xl, yl)))
            last = rows[-4:]
            print(f"[{proto} s{seed}] " +
                  "  ".join(f"{a['arm']}(p={a['p']}):"
                            f"{a['retain']:.3f}/{a['learn']:.3f}"
                            for a in last))

    print("\n=== mean +- std over {} seeds ===".format(len(SEEDS)))
    summary = {}
    for proto in ("EASY", "HARD"):
        for arm, _p_mode, _p_fixed in ARMS:
            rs = [r["retain"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            ls = [r["learn"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            key = f"{proto}/{arm}"
            summary[key] = dict(
                retain_mean=float(np.mean(rs)), retain_std=float(np.std(rs)),
                learn_mean=float(np.mean(ls)), learn_std=float(np.std(ls)))
            print("{:>14}:  retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
                  .format(key, np.mean(rs), np.std(rs), np.mean(ls), np.std(ls)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_adaptive_exponent.json")
    json.dump(dict(rows=rows, summary=summary), open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
