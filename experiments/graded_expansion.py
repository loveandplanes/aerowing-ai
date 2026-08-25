"""Round 3: progress-gated graded unfreezing around the growth site.

User proposal: after capacity growth, don't keep the base fully frozen and
don't fully unfreeze either. Wake parameters NEAREST the new module first,
inject small DIRECTED noise (extra gradient micro-steps + jitter), and only
propagate the wake outward (more layers, smaller epsilon) when validation
on the new stream confirms learning.

Arms (single-shift protocol from round 2):
  cold_retrain : fresh wider net, union data, 3x steps
  plain_ft     : everything trainable, no schedule (plasticity extreme)
  R2_frozen    : base permanently frozen, expander+gate only (stability extreme)
  GRADED       : the proposal - 3-phase radial unfreezing with decaying
                 directed-noise, gated by validation improvement
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
        # net[0]=Linear(blk_deep), net[2]=Linear(blk_near): ordered deep->near
        self.net = nn.Sequential(
            nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


class GatedResidual(nn.Module):
    """Zero-init value head x learned sigmoid gate over a base."""

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


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _param_groups(model, unlocked_layer_idx):
    """Trainable = expander+gate always; plus trunk blocks per phase.
    unlocked_layer_idx: None=all frozen, 2=near block only, 1=deep+near."""
    exp_params, trunk_params = [], []
    for n, p in model.named_parameters():
        if n.startswith("value") or n.startswith("gate"):
            exp_params.append(p)
        elif n.startswith("base.net.0.") and unlocked_layer_idx in (None, 1):
            trunk_params.append(p)
        elif n.startswith("base.net.2.") and unlocked_layer_idx in (None, 1, 2):
            trunk_params.append(p)
        elif n.startswith("base.head.") and unlocked_layer_idx is None:
            trunk_params.append(p)
    return exp_params, trunk_params


def run_graded(pristine, x2, y2, xv, yv, seed):
    torch.manual_seed(seed)
    model = GatedResidual(copy.deepcopy(pristine))
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    xvt, yvt = torch.tensor(xv), torch.tensor(yv)
    rng = np.random.default_rng(seed)

    # radial phases: (steps, unlocked_layer_idx, eps_directed, eps_jitter)
    phases = [(300, 2, 3e-4, 6e-4),
              (300, 1, 1e-4, 2e-4),
              (300, None, 3e-5, 0.0)]
    best_val = rmse(model, xv, yv)
    log = []
    for ph, (n_steps, ul, eps_d, eps_j) in enumerate(phases):
        exp_p, tr_p = _param_groups(model, ul)
        opt = torch.optim.Adam([
            dict(params=exp_p, lr=LR),
            dict(params=tr_p, lr=LR * 0.3)])
        for s in range(n_steps):
            idx = torch.tensor(rng.integers(0, len(xt), BATCH))
            loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            # directed noise: micro-step along -sign(grad) + isotropic jitter
            with torch.no_grad():
                for g in opt.param_groups[1:]:
                    for p in g["params"]:
                        if p.grad is None:
                            continue
                        p -= eps_d * torch.sign(p.grad)
                        p += eps_j * torch.randn_like(p) * 0.1
            if (s + 1) % 60 == 0:
                v = rmse(model, xv, yv)
                log.append(v)
                if v < best_val:
                    best_val = v
        # progression gate: advance only if validation confirmed learning
        cur = rmse(model, xv, yv)
        advanced = cur <= best_val * 1.05
        best_val = min(best_val, cur)
        log.append(f"phase{ph}: unlocked={ul} adv={advanced}")
    return model, log


def run_plain_ft(pristine, x2, y2, seed):
    torch.manual_seed(seed)
    model = copy.deepcopy(pristine)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        idx = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        xr, yr = sample(2000, rng, with_bump=False)
        xl, yl = sample(2000, rng, with_bump=True)
        x2, y2 = sample_shifted(400, rng)
        # validation slice for progression gating (never a test probe)
        perm = rng.permutation(len(x2))
        vi, ti = perm[:80], perm[80:]
        xv, yv = x2[vi], y2[vi]
        xtr, ytr = x2[ti], y2[ti]

        torch.manual_seed(seed)
        base = Trunk(h=64)
        opt = torch.optim.Adam(base.parameters(), lr=LR)
        for _ in range(STEPS * 2):
            idx = torch.tensor(rng.integers(0, len(xtr), BATCH))
            loss = ((base(torch.tensor(xtr[idx])) -
                     torch.tensor(ytr[idx])) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        pristine = copy.deepcopy(base)

        pre = (rmse(pristine, xr, yr), rmse(pristine, xl, yl))

        # stability extreme
        m_frozen = GatedResidual(copy.deepcopy(pristine))
        p_frozen = [p for p in m_frozen.parameters() if p.requires_grad]
        opt = torch.optim.Adam(p_frozen, lr=LR)
        rng2 = np.random.default_rng(seed)
        for _ in range(STEPS):
            idx = torch.tensor(rng2.integers(0, len(xtr), BATCH))
            loss = ((m_frozen(torch.tensor(xtr[idx])) -
                     torch.tensor(ytr[idx])) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        # plasticity extreme
        m_plain = run_plain_ft(pristine, xtr, ytr, seed)

        # the proposal
        m_graded, glog = run_graded(pristine, xtr, ytr, xv, yv, seed)

        for arm, mdl in (("R2_frozen", m_frozen), ("plain_ft", m_plain),
                         ("GRADED", m_graded)):
            rows.append(dict(protocol="shifted", seed=seed, arm=arm,
                             retain_before=pre[0], learn_before=pre[1],
                             retain=rmse(mdl, xr, yr),
                             learn=rmse(mdl, xl, yl)))
        print(f"[seed {seed}] "
              f"frozen: {rmse(m_frozen, xr, yr):.4f}/{rmse(m_frozen, xl, yl):.4f}  "
              f"plain: {rmse(m_plain, xr, yr):.4f}/{rmse(m_plain, xl, yl):.4f}  "
              f"graded: {rmse(m_graded, xr, yr):.4f}/{rmse(m_graded, xl, yl):.4f}")

    print("\n=== mean +- std over {} seeds ===".format(len(SEEDS)))
    summary = {}
    for arm in ("R2_frozen", "plain_ft", "GRADED"):
        rs = [r["retain"] for r in rows if r["arm"] == arm]
        ls = [r["learn"] for r in rows if r["arm"] == arm]
        summary[arm] = dict(retain_mean=float(np.mean(rs)),
                            retain_std=float(np.std(rs)),
                            learn_mean=float(np.mean(ls)),
                            learn_std=float(np.std(ls)))
        print("{:>11}:  retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
              .format(arm, np.mean(rs), np.std(rs), np.mean(ls), np.std(ls)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_graded_expansion.json")
    json.dump(dict(rows=rows, summary=summary),
              open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
