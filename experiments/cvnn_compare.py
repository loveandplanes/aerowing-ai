"""Round 7: is complex-native processing worth it for physics surrogates?

Arms (matched training budget, same probes, 5 seeds):
  mlp_base   : real MLP on raw inputs (current production-style architecture)
  cv_modrelu : complex-linear network on Fourier-pair complex features,
               phase-preserving modReLU activation ("complex routing")
  cv_gelu    : same complex layers, componentwise GELU after full complex
               multiplication ("complex product")

Complex features come free: Fourier pairs (cos(w.x), sin(w.x)) are exactly
the real/imaginary parts of e^{i w.x}.
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
N_FREQ = 32
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


class Trunk(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def fourier_pairs(x, n_freq=N_FREQ):
    """Returns real tensor [N, 2*n_freq]: interleaved (cos, sin) pairs."""
    w = torch.linspace(1.0, 8.0, n_freq)[None, :].repeat(x.shape[1], 1)
    proj = (x @ w) * math.pi                 # [N, n_freq]
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)


class CLinear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.wr = nn.Linear(c_in, c_out)
        self.wi = nn.Linear(c_in, c_out)

    def forward(self, zr, zi):
        r = self.wr(zr) - self.wi(zi)
        i = self.wr(zi) + self.wi(zr)
        return r, i


class CVNet(nn.Module):
    def __init__(self, c_units=24, mode="modrelu", n_freq=N_FREQ):
        super().__init__()
        self.mode = mode
        self.n_freq = n_freq
        self.l1 = CLinear(n_freq, c_units)
        self.b1 = nn.Parameter(torch.zeros(c_units))
        self.l2 = CLinear(c_units, c_units)
        self.b2 = nn.Parameter(torch.zeros(c_units))
        self.out = CLinear(c_units, 1)

    def forward(self, x):
        f = fourier_pairs(x, self.n_freq)
        zr, zi = f[:, :self.n_freq], f[:, self.n_freq:]
        zr, zi = self.l1(zr, zi)
        if self.mode == "modrelu":
            mag = torch.sqrt(zr ** 2 + zi ** 2 + 1e-8)
            g = torch.relu(mag + self.b1)
            zr, zi = zr * g / mag, zi * g / mag
        else:  # gelu product: componentwise on both parts
            zr, zi = torch.nn.functional.gelu(zr), \
                torch.nn.functional.gelu(zi)
        zr, zi = self.l2(zr, zi)
        if self.mode == "modrelu":
            mag = torch.sqrt(zr ** 2 + zi ** 2 + 1e-8)
            g = torch.relu(mag + self.b2)
            zr, zi = zr * g / mag, zi * g / mag
        else:
            zr, zi = torch.nn.functional.gelu(zr), \
                torch.nn.functional.gelu(zi)
        rr, ri = self.out(zr, zi)
        return rr.squeeze(-1)


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(2000 + seed)
        xr, yr = sample(2000, rng, with_bump=False)
        xl, yl = sample(2000, rng, with_bump=True)
        x2, y2 = sample_shifted(400, rng)

        models = {
            "mlp_base": Trunk(h=64),
            "cv_modrelu": CVNet(mode="modrelu"),
            "cv_gelu": CVNet(mode="gelu"),
        }
        for arm, model in models.items():
            torch.manual_seed(seed + 13)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            xt, yt = torch.tensor(x2), torch.tensor(y2)
            rr2 = np.random.default_rng(seed + 77)
            for _ in range(STEPS):
                i = torch.tensor(rr2.integers(0, len(xt), BATCH))
                loss = ((model(xt[i]) - yt[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            rows.append(dict(seed=seed, arm=arm,
                             params=n_params(model),
                             retain=rmse(model, xr, yr),
                             learn=rmse(model, xl, yl)))
            print(f"[shifted s{seed}] {arm:>11}: "
                  f"params={n_params(model):5d}  "
                  f"retain={rmse(model, xr, yr):.4f}  "
                  f"learn={rmse(model, xl, yl):.4f}")

    print("\n=== mean +- std over {} seeds ===".format(len(SEEDS)))
    summary = {}
    for arm in ("mlp_base", "cv_modrelu", "cv_gelu"):
        rs = [r["retain"] for r in rows if r["arm"] == arm]
        ls = [r["learn"] for r in rows if r["arm"] == arm]
        ps = [r["params"] for r in rows if r["arm"] == arm][:1]
        summary[arm] = dict(retain_mean=float(np.mean(rs)),
                            retain_std=float(np.std(rs)),
                            learn_mean=float(np.mean(ls)),
                            learn_std=float(np.std(ls)))
        print("{:>11}:  params {:6d}   retain {:6.4f}+-{:.4f}   "
              "learn {:6.4f}+-{:.4f}"
              .format(arm, ps[0], np.mean(rs), np.std(rs),
                      np.mean(ls), np.std(ls)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_cvnn.json")
    json.dump(dict(rows=rows, summary=summary), open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
