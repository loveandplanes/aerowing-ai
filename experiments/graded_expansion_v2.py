"""Round 4: the graded-unfreezing WIN CONDITION test.

Stage-1 world ignores x3 entirely (f_base has no x3 term), so the frozen
trunk learns ~zero sensitivity along x3. The new requirement is a bump
localized ALONG x3 - expressible only if the x3 pathway grows.

Prediction:
  R2_frozen : plateaus - cannot represent x3-selective correction
  GRADED    : unlocks the deep block -> x3 pathway can form -> wins
  plain_ft  : fits but forgets catastrophically
"""
import copy
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graded_expansion import Trunk, rmse, run_graded, run_plain_ft, \
    GatedResidual, STEPS, LR, BATCH, DIM

CENTER = np.array([0.85, 0.85, 0.90, 0.15])   # note: x3 = 0.90


def f_base_no_x3(x):
    return (0.5 * np.sin(2 * math.pi * x[:, 0]) + 0.3 * x[:, 1] ** 2
            + 0.1 * x[:, 3] * x[:, 0])


def new_bump(x):
    d2 = ((x[:, 0] - CENTER[0]) ** 2 + (x[:, 1] - CENTER[1]) ** 2
          + ((x[:, 2] - CENTER[2]) / 0.35) ** 2
          + (x[:, 3] - CENTER[3]) ** 2)
    return 0.6 * np.exp(-d2 / (2 * 0.20 ** 2))


def sample_base(n, rng):
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    y = f_base_no_x3(x)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


def sample_new(n, rng):
    x = np.clip(CENTER + rng.normal(0, 0.25, size=(n, DIM)), 0.0, 1.0)
    y = f_base_no_x3(x) + new_bump(x)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


SEEDS = [0, 1, 2, 3, 4]


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        xr, yr = sample_base(2000, rng)                 # retention: no-x3 world
        xl, yl = sample_new(2000, rng)                  # learning: x3-selective
        x2, y2 = sample_new(400, rng)

        torch.manual_seed(seed)
        base = Trunk(h=64)
        opt = torch.optim.Adam(base.parameters(), lr=LR)
        xt1, yt1 = torch.tensor(x1 := sample_base(600, rng)[0],
                                dtype=torch.float32), \
            torch.tensor(sample_base(600, rng)[1], dtype=torch.float32)
        # deterministic paired base-set (avoid double sampling drift)
        xb, yb = sample_base(600, rng)
        xt1, yt1 = torch.tensor(xb), torch.tensor(yb)
        for _ in range(STEPS * 2):
            idx = torch.tensor(rng.integers(0, len(xt1), BATCH))
            loss = ((base(xt1[idx]) - yt1[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        # measure how dead the x3 pathway is in the frozen trunk
        with torch.no_grad():
            w3 = float(base.net[0].weight[:, 2].abs().mean())
        pristine = copy.deepcopy(base)

        m_frozen = GatedResidual(copy.deepcopy(pristine))
        p_frozen = [p for p in m_frozen.parameters() if p.requires_grad]
        o1 = torch.optim.Adam(p_frozen, lr=LR)
        xt = torch.tensor(x2)
        yt = torch.tensor(y2)
        r2 = np.random.default_rng(seed)
        for _ in range(STEPS):
            i = torch.tensor(r2.integers(0, len(xt), BATCH))
            loss = ((m_frozen(xt[i]) - yt[i]) ** 2).mean()
            o1.zero_grad(); loss.backward(); o1.step()

        m_plain = run_plain_ft(copy.deepcopy(pristine), x2, y2, seed)

        # graded with longer phases: give the x3 pathway room to form
        m_graded = run_graded_long(copy.deepcopy(pristine), x2, y2,
                                   xv=torch.tensor(x2[:80]),
                                   yv=torch.tensor(y2[:80]), seed=seed)

        res = {
            "R2_frozen": (rmse(m_frozen, xl, yl), rmse(m_frozen, xr, yr)),
            "plain_ft": (rmse(m_plain, xl, yl), rmse(m_plain, xr, yr)),
            "GRADED": (rmse(m_graded, xl, yl), rmse(m_graded, xr, yr)),
        }
        rows.append(dict(seed=seed, w3_abs=w3, **{
            k: dict(learn=v[0], retain=v[1]) for k, v in res.items()}))
        print(f"[seed {seed}] |w_x3|={w3:.4f}  "
              + "  ".join(f"{k}: learn={v[0]:.4f} retain={v[1]:.4f}"
                          for k, v in res.items()))

    print("\n=== mean over {} seeds (learn RMSE / retain RMSE) ===".format(
        len(SEEDS)))
    for arm in ("R2_frozen", "plain_ft", "GRADED"):
        ls = [r[arm]["learn"] for r in rows]
        rs = [r[arm]["retain"] for r in rows]
        print("{:>10}:  learn {:6.4f}+-{:.4f}   retain {:6.4f}+-{:.4f}"
              .format(arm, np.mean(ls), np.std(ls), np.mean(rs), np.std(rs)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_graded_wincondition.json")
    json.dump(rows, open(out, "w"), indent=2, default=float)
    print("saved:", out)


def run_graded_long(pristine, x2, y2, xv, yv, seed):
    """Same radial schedule but 400 steps per phase."""
    import graded_expansion as ge
    torch.manual_seed(seed)
    model = ge.GatedResidual(copy.deepcopy(pristine))
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    xvt, yvt = xv, yv
    rng = np.random.default_rng(seed)
    phases = [(400, 2, 3e-4, 6e-4),
              (400, 1, 1e-4, 2e-4),
              (400, None, 3e-5, 0.0)]
    best_val = rmse(model, xv.numpy(), yv.numpy())
    for n_steps, ul, eps_d, eps_j in phases:
        exp_p, tr_p = _param_groups_local(model, ul)
        opt = torch.optim.Adam([
            dict(params=exp_p, lr=LR),
            dict(params=tr_p, lr=LR * 0.3)])
        for s in range(n_steps):
            idx = torch.tensor(rng.integers(0, len(xt), BATCH))
            loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for g in opt.param_groups[1:]:
                    for p in g["params"]:
                        if p.grad is None:
                            continue
                        p -= eps_d * torch.sign(p.grad)
                        p += eps_j * torch.randn_like(p) * 0.1
            if (s + 1) % 80 == 0:
                v = rmse(model, xvt.numpy(), yvt.numpy())
                best_val = min(best_val, v)
    return model


def _param_groups_local(model, unlocked_layer_idx):
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


if __name__ == "__main__":
    main()
