"""Round 7: zoned error weighting (user proposal).

Stream concentrated near a new feature over-represents that zone; plain
fine-tuning then drifts away from under-represented regions (measured in
the multistage study). Fix: assign each sample a zone by distance to the
new-feature center, and weight zone-rare samples UP so the effective
gradient balances coverage - no extra stored data required.

Arms: plain_ft (unweighted), zoned_ft (proposal), R2_frozen (reference).
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


def sample_base(n, rng):
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    y = f_base(x)
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


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def zone_weights(x2, cap=6.0):
    d = np.linalg.norm(x2 - CENTER, axis=1)
    near = d < 0.45
    n_near, n_far = int(near.sum()), int((~near).sum())
    w_far = float(np.clip(n_near / max(n_far, 1), 1.0, cap))
    w = np.where(near, 1.0, w_far)
    return w * (len(w) / w.sum())          # mean-preserving normalization


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(1500 + seed)
        xr, yr = sample_base(2000, rng)
        xl, yl = sample_shifted(2000, rng)     # learning probe lives in-zone
        xg, yg = sample_base(2000, rng)        # global probe incl. out-zone
        x2, y2 = sample_shifted(400, rng)

        torch.manual_seed(seed)
        xb, yb = sample_base(600, rng)
        pristine = Trunk(h=64)
        o = torch.optim.Adam(pristine.parameters(), lr=LR)
        xt0, yt0 = torch.tensor(xb), torch.tensor(yb)
        rr = np.random.default_rng(seed + 3)
        for _ in range(STEPS):
            i = torch.tensor(rr.integers(0, len(xt0), BATCH))
            loss = ((pristine(xt0[i]) - yt0[i]) ** 2).mean()
            o.zero_grad(); loss.backward(); o.step()
        pre_ret, pre_lrn = rmse(pristine, xg, yg), rmse(pristine, xl, yl)

        results = {}
        for arm in ("plain_ft", "zoned_ft", "R2_frozen", "zoned_gated"):
            torch.manual_seed(seed + 7)
            if arm == "R2_frozen" or arm == "zoned_gated":
                # frozen-base gated residual, minimal version
                class Gated(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.base = pristine
                        h = pristine.net[0].out_features
                        self.value = nn.Sequential(
                            nn.Linear(h, 64), nn.GELU(), nn.Linear(64, 1))
                        self.gate = nn.Sequential(
                            nn.Linear(h, 32), nn.GELU(), nn.Linear(32, 1))
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
                m = Gated()
                params = [p for p in m.parameters() if p.requires_grad]
            else:
                m = copy.deepcopy(pristine)
                params = m.parameters()

            opt = torch.optim.Adam(params, lr=LR)
            xt, yt = torch.tensor(x2), torch.tensor(y2)
            wz = None
            if arm in ("zoned_ft", "zoned_gated"):
                wz = torch.tensor(zone_weights(x2), dtype=torch.float32)
            rr2 = np.random.default_rng(seed + 50)
            for _ in range(STEPS):
                i = torch.tensor(rr2.integers(0, len(xt), BATCH))
                sq = (m(xt[i]) - yt[i]) ** 2
                loss = (sq * wz[i]).mean() if wz is not None else sq.mean()
                opt.zero_grad(); loss.backward(); opt.step()

            results[arm] = dict(retain=rmse(m, xg, yg),
                                learn=rmse(m, xl, yl))
            rows.append(dict(seed=seed, arm=arm, **results[arm],
                             retain_before=pre_ret, learn_before=pre_lrn))

        print(f"[seed {seed}] " +
              "  ".join(f"{a}: r={results[a]['retain']:.4f} "
                        f"l={results[a]['learn']:.4f}"
                        for a in ("plain_ft", "zoned_ft", "R2_frozen")))

    print("\n=== mean +- std ===")
    for arm in ("plain_ft", "zoned_ft", "R2_frozen", "zoned_gated"):
        rs = [r["retain"] for r in rows if r["arm"] == arm]
        ls = [r["learn"] for r in rows if r["arm"] == arm]
        print("{:>10}:  retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
              .format(arm, np.mean(rs), np.std(rs), np.mean(ls), np.std(ls)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_zoned_loss.json")
    json.dump(rows, open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
