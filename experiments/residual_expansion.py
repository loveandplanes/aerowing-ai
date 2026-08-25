"""Round 2: does the growth idea WORK when the protection is structural?

Diagnosis of round 1: Formulation C protects solved regions in SAMPLE space
(downweight easy samples) - but easy samples carry the replay anchoring that
retention needs, and their gradients still move shared weights everywhere.
The intuition belongs in PARAMETER/FUNCTION space. Three redesigned variants:

  R1 frozen_base_resid : trunk FROZEN; expander fits residual (y - G_old).
                         Retention guaranteed by construction.
  R2 gated_residual    : expander value multiplied by learned sigmoid gate;
                         correction localizes itself; trunk frozen.
  R3 distill_anchor    : all params trainable + lambda * correctness-weighted
                         distillation to the frozen pre-expansion self.

Reference arms: plain_ft (round-1 winner), formC (round-1 loser),
cold_retrain. Same two-stage protocol, same probes, 5 seeds.
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
BUMP_CENTER = np.array([0.85, 0.85, 0.15, 0.15])
BUMP_AMP, BUMP_W = 0.6, 0.20


def f_base(x):
    return (0.5 * np.sin(2 * math.pi * x[:, 0]) + 0.3 * x[:, 1] ** 2
            + 0.2 * x[:, 2] + 0.1 * x[:, 3] * x[:, 0])


def bump(x):
    d2 = np.sum((x - BUMP_CENTER) ** 2, axis=1)
    return BUMP_AMP * np.exp(-d2 / (2 * BUMP_W ** 2))


def sample(n, rng, with_bump):
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    y = f_base(x) + (bump(x) if with_bump else 0.0)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


def sample_shifted(n, rng):
    x = np.clip(BUMP_CENTER + rng.normal(0, 0.25, size=(n, DIM)), 0.0, 1.0)
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


class ResidualExpansion(nn.Module):
    """Frozen base + expander on (y - G_old) targets; optional learned gate."""

    def __init__(self, base, units=64, gated=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()
        h = base.net[0].out_features
        self.gated = gated
        self.exp = nn.Sequential(nn.Linear(h, units), nn.GELU(),
                                 nn.Linear(units, 1))
        self.gate = nn.Sequential(nn.Linear(h, units // 2), nn.GELU(),
                                  nn.Linear(units // 2, 1)) if gated else None

    def forward(self, x):
        with torch.no_grad():
            base_out = self.base(x)
        h = self.base.net(x)
        corr = self.exp(h).squeeze(-1)
        if self.gated:
            corr = corr * torch.sigmoid(self.gate(h)).squeeze(-1)
        return base_out + corr

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def run_arm(arm, pristine, x2, y2, seed):
    torch.manual_seed(seed)
    xt = torch.tensor(x2)
    yt = torch.tensor(y2)

    if arm == "cold_retrain":
        model = Trunk(h=96)
        params = model.parameters()
        steps = STEPS * 3
    elif arm == "plain_ft":
        model = copy.deepcopy(pristine)
        params = model.parameters()
        steps = STEPS
    else:
        model = ResidualExpansion(copy.deepcopy(pristine),
                                  gated=(arm == "R2_gated"))
        params = model.trainable_params()
        steps = STEPS

    opt = torch.optim.Adam(params, lr=LR)
    rng = np.random.default_rng(seed)
    n = len(xt)

    # frozen-reference tensors for residual/distillation arms
    ref = copy.deepcopy(pristine).eval()
    with torch.no_grad():
        base_out_all = ref(torch.tensor(x2)).numpy()

    for step in range(steps):
        idx = torch.tensor(rng.integers(0, n, BATCH))
        xb = xt[idx]
        if arm in ("R1_frozen_resid", "R2_gated"):
            # residual targets: teach only the correction
            rb = torch.tensor((y2 - base_out_all)[idx.numpy()]
                              .astype(np.float32))
            pred_corr = model(xb) - ref(xb)
            loss = ((pred_corr - rb) ** 2).mean()
        elif arm == "R3_distill_anchor":
            pred = model(xb)
            with torch.no_grad():
                anch = ref(xb)
            e_old_local = torch.tensor(
                np.abs(y2[idx.numpy()] - base_out_all[idx.numpy()]),
                dtype=torch.float32)
            # protect where old was accurate: weight ~ exp(-|e_old|/scale)
            w_prot = torch.exp(-e_old_local / 0.10)
            loss = ((pred - yt[idx]) ** 2).mean() \
                + 2.0 * (((pred - anch) ** 2).mean(dim=0)
                         * w_prot).sum() / len(idx)
        else:
            pred = model(xb)
            loss = ((pred - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main():
    rows = []
    arms = ("cold_retrain", "plain_ft", "formC_reference",
            "R1_frozen_resid", "R2_gated", "R3_distill_anchor")
    for proto in ("shifted", "mixed"):
        for seed in SEEDS:
            rng = np.random.default_rng(100 + seed)
            x1, y1 = sample(600, rng, with_bump=False)
            xr, yr = sample(2000, rng, with_bump=False)
            xl, yl = sample(2000, rng, with_bump=True)
            if proto == "mixed":
                xn, yn = sample_shifted(200, rng)
                xo, yo = sample(200, rng, with_bump=False)
                x2 = np.concatenate([xo, xn])
                y2 = np.concatenate([yo, yn])
            else:
                x2, y2 = sample_shifted(400, rng)

            torch.manual_seed(seed)
            base = Trunk(h=64)
            opt = torch.optim.Adam(base.parameters(), lr=LR)
            xt1, yt1 = torch.tensor(x1), torch.tensor(y1)
            for _ in range(STEPS * 2):
                idx = torch.tensor(rng.integers(0, len(xt1), BATCH))
                loss = ((base(xt1[idx]) - yt1[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            pristine = copy.deepcopy(base)

            def rm(model, xs=None, ys=None):
                return rmse(model, xr, yr), rmse(model, xl, yl)

            pre = rm(base)
            data = dict(x2=x2, y2=y2)
            for arm in arms:
                if arm == "formC_reference":
                    continue  # round-1 result already recorded
                model = run_arm(arm, pristine, x2, y2, seed)
                ret, lrn = rm(model)
                row = dict(protocol=proto, seed=seed, arm=arm,
                           retain_before=pre[0], learn_before=pre[1],
                           retain=ret, learn=lrn)
                rows.append(row)
                print("[{protocol}] {seed} {arm:>18}  retain "
                      "{retain_before:.4f}->{retain:.4f}  learn "
                      "{learn_before:.4f}->{learn:.4f}".format(**row))

    print("\n=== mean +- std over {} seeds ===".format(len(SEEDS)))
    summary = {}
    for proto in ("shifted", "mixed"):
        for arm in arms:
            rs = [r["retain"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            ls = [r["learn"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            key = f"{proto}/{arm}"
            summary[key] = dict(
                retain_mean=float(np.mean(rs)), retain_std=float(np.std(rs)),
                learn_mean=float(np.mean(ls)), learn_std=float(np.std(ls)))
            print("{:>28}:  retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
                  .format(key, np.mean(rs), np.std(rs), np.mean(ls), np.std(ls)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_residual_expansion.json")
    json.dump(dict(rows=rows, summary=summary,
                   protocol=dict(seeds=SEEDS, steps=STEPS, lr=LR)),
              open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
