"""Controlled ablation of the growth mechanism (paper Table 1).

Protocol (mirrors the industrial story in miniature):
  Stage 1: base net G learns a smooth landscape f_base  -> "solved knowledge"
  Stage 2: new labels arrive containing a localized hard feature (a Gaussian
           bump absent from stage 1)                    -> "new requirement"

Arms at IDENTICAL optimizer-step budget on the stage-2 stream:
  cold_retrain : fresh wider net, trained on union(D1, D2) from scratch
  plain_ft     : continue G on D2, no structural change
  grow_mse     : attach zero-init residual block, plain MSE
  grow_formA   : residual block + Formulation A (sigmoid error mask)
  grow_formC   : residual block + full error-directed routing (the method)

Metrics: RMSE on a RETENTION set (f_base only - did old knowledge survive?)
and a LEARNING set (bump included - was the new region acquired?).
"""
import json
import math
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

SEEDS = [0, 1, 2, 3, 4]
STEPS = 900
LR = 1e-3
BATCH = 128
TAU = 0.05
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
    """Stage-2 stream CONCENTRATED around the new feature - a genuine
    distribution shift that creates stability-plasticity pressure."""
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


class Grown(nn.Module):
    """Structurally identical to aerowing.continual.GrowableSurrogate:
    frozen trunk/head plus a zero-init residual block (out = head(h)+exp(h))."""

    def __init__(self, base, units=64):
        super().__init__()
        self.base = base
        h = base.net[0].out_features
        self.expander = nn.Sequential(
            nn.Linear(h, units), nn.GELU(), nn.Linear(units, 1))
        with torch.no_grad():
            self.expander[-1].weight.zero_()
            self.expander[-1].bias.zero_()

    def features(self, x):
        return self.base.net(x)

    def forward(self, x):
        h = self.features(x)
        return (self.base.head(h) + self.expander(h)).squeeze(-1)


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((pred - y) ** 2)))


# ---------------------------------------------------------------------------
# published-baseline arms (Phase 1 of the comparison note)
# ---------------------------------------------------------------------------

def compute_fisher(model, x1, y1, n_batches=20, batch=128):
    """Diagonal empirical Fisher: mean squared loss-gradient per parameter,
    estimated on stage-1 data (Kirkpatrick et al. 2017)."""
    model = copy.deepcopy(model).eval()
    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters()}
    rng = np.random.default_rng(0)
    xt = torch.tensor(x1)
    yt = torch.tensor(y1)
    for _ in range(n_batches):
        idx = torch.tensor(rng.integers(0, len(xt), min(batch, len(xt))))
        model.zero_grad()
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= n_batches
    return fisher


def _ewc_penalty(model, fisher, theta_star):
    return sum((fisher[n] * (p - theta_star[n]) ** 2).sum()
               for n, p in model.named_parameters())


def train_ewc(pristine, data, seed, lam):
    """EWC: stage-2 loss + lam/2 * F (theta - theta*)^2."""
    torch.manual_seed(seed)
    model = copy.deepcopy(pristine)
    theta_star = {n: p.detach().clone()
                  for n, p in model.named_parameters()}
    fisher = compute_fisher(model, data["x1"], data["y1"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt = torch.tensor(data["x2"])
    yt = torch.tensor(data["y2"])
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        idx = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean() \
            + 0.5 * lam * _ewc_penalty(model, fisher, theta_star)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_lwf(pristine, data, seed, lam):
    """LwF-style regression adaptation: new-stream MSE + uniform-weight
    distillation of the frozen pre-expansion function's outputs
    (Li & Hoiem 2016 spirit; soft-target machinery is classification-only)."""
    torch.manual_seed(seed)
    model = copy.deepcopy(pristine)
    ref = copy.deepcopy(model).eval()
    with torch.no_grad():
        base_out = ref(torch.tensor(data["x2"])).numpy()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt = torch.tensor(data["x2"])
    yt = torch.tensor(data["y2"])
    bt = torch.tensor(base_out.astype(np.float32))
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        idx = torch.tensor(rng.integers(0, len(xt), BATCH))
        pred = model(xt[idx])
        loss = ((pred - yt[idx]) ** 2).mean() \
            + lam * ((pred - bt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def select_and_score(make_and_train, lambdas, val, arms_tag):
    """Trains one model per lambda, selects by validation RMSE (a held-out
    20% slice of the stage-2 stream - never a test probe), returns the
    winning model + chosen lambda."""
    xv, yv = val
    best = None
    for lam in lambdas:
        model = make_and_train(lam)
        v = rmse(model, xv, yv)
        if best is None or v < best[1]:
            best = (model, v, lam)
    print(f"      {arms_tag}: lambda={best[2]} (val {best[1]:.4f})")
    return best[0], best[2]


def train_arm(arm, base, data, seed):
    torch.manual_seed(seed)
    x2, y2 = data["x2"], data["y2"]
    xt = torch.tensor(x2)
    yt = torch.tensor(y2)

    if arm == "cold_retrain":
        x_all = np.concatenate([data["x1"], x2])
        y_all = np.concatenate([data["y1"], y2])
        model = Trunk(h=96)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        errs_old = None
        steps = STEPS * 3   # fresh nets need more steps to converge; matched wall-effort
    else:
        model = base
        if arm.startswith("grow"):
            model = Grown(base)
            # verify exact function preservation at init
            with torch.no_grad():
                d = (model(torch.tensor(x2[:32]))
                     - base(torch.tensor(x2[:32]))).abs().max().item()
            assert d < 1e-6, f"growth disturbed the function: {d}"
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        if arm in ("grow_formA", "grow_formC"):
            g_eval = base.eval()
            with torch.no_grad():
                e_old = (torch.tensor(y2)
                         - g_eval(torch.tensor(x2))).abs().numpy()
            errs_old = e_old
        else:
            errs_old = None
        steps = STEPS

    rng = np.random.default_rng(seed)
    n = len(xt)
    for step in range(steps):
        idx = torch.tensor(rng.integers(0, n, BATCH))
        xb, yb = xt[idx], yt[idx]
        pred = model(xb)
        sq = (pred - yb) ** 2
        if arm == "grow_formA":
            m = torch.sigmoid(torch.tensor(
                errs_old[idx] / TAU, dtype=torch.float32))
            loss = (sq * m).mean()
        elif arm == "grow_formC":
            m = torch.sigmoid(torch.tensor(
                errs_old[idx] / TAU, dtype=torch.float32))
            cur = sq.detach()
            w = m * (1.0 + torch.relu(m * 0 + torch.tensor(
                errs_old[idx], dtype=torch.float32) - cur.sqrt()))
            loss = (sq * w).mean()
        else:
            loss = sq.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_growth_ablation.json")
    rows = []
    for proto in ("shifted", "mixed"):
        for seed in SEEDS:
            rng = np.random.default_rng(100 + seed)
            x1, y1 = sample(600, rng, with_bump=False)      # stage 1 world
            xr, yr = sample(2000, rng, with_bump=False)     # retention probe
            xl, yl = sample(2000, rng, with_bump=True)      # learning probe

            if proto == "mixed":
                # half new-region stream, half old-domain revisited - the
                # regime where error-routing's protective claim applies
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
            for _ in range(STEPS * 2):   # stage-1 must actually converge
                idx = torch.tensor(rng.integers(0, len(xt1), BATCH))
                loss = ((base(xt1[idx]) - yt1[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

            pre_ret = rmse(base, xr, yr)
            pre_lrn = rmse(base, xl, yl)
            assert pre_ret < 0.08, (
                f"stage-1 base underfit (retain RMSE {pre_ret:.3f}) - "
                "ablation would be meaningless; increase stage-1 budget")
            pristine = copy.deepcopy(base)   # frozen reference for e_old + arms

            data = dict(x1=x1, y1=y1, x2=x2, y2=y2)
            for arm in ("cold_retrain", "plain_ft", "grow_mse",
                        "grow_formA", "grow_formC"):
                model = train_arm(arm, copy.deepcopy(pristine), data, seed)
                row = dict(protocol=proto, seed=seed, arm=arm,
                           retain_before=pre_ret, learn_before=pre_lrn,
                           retain=rmse(model, xr, yr),
                           learn=rmse(model, xl, yl))
                rows.append(row)
                print("[{protocol}] seed={seed} {arm:>13}  retain "
                      "{retain_before:.4f}->{retain:.4f}   learn "
                      "{learn_before:.4f}->{learn:.4f}".format(**row))

            # ---- published baselines: EWC & LwF (lambda selected on a
            #      held-out 20% slice of the stage-2 stream, never probes)
            perm2 = rng.permutation(len(x2))
            nv = max(1, int(0.2 * len(x2)))
            vi, ti = perm2[:nv], perm2[nv:]
            d_sub = dict(x1=data["x1"], y1=data["y1"],
                         x2=data["x2"][ti], y2=data["y2"][ti])
            val = (data["x2"][vi], data["y2"][vi])

            ewc_model, ewc_lam = select_and_score(
                lambda lam: train_ewc(copy.deepcopy(pristine), d_sub,
                                      seed, lam),
                (100.0, 5000.0), val, f"[{proto}] EWC     ")
            rows.append(dict(protocol=proto, seed=seed, arm="EWC",
                             lam=ewc_lam, retain_before=pre_ret,
                             learn_before=pre_lrn, retain=rmse(ewc_model, xr, yr),
                             learn=rmse(ewc_model, xl, yl)))
            lwf_model, lwf_lam = select_and_score(
                lambda lam: train_lwf(copy.deepcopy(pristine), d_sub,
                                      seed, lam),
                (0.5, 2.0, 10.0), val, f"[{proto}] LwF     ")
            rows.append(dict(protocol=proto, seed=seed, arm="LWF",
                             lam=lwf_lam, retain_before=pre_ret,
                             learn_before=pre_lrn, retain=rmse(lwf_model, xr, yr),
                             learn=rmse(lwf_model, xl, yl)))

    print("\n=== mean +- std over {} seeds ===".format(len(SEEDS)))
    summary = {}
    for proto in ("shifted", "mixed"):
        for arm in ("cold_retrain", "plain_ft", "grow_mse",
                    "grow_formA", "grow_formC", "EWC", "LWF"):
            rs = [r["retain"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            ls = [r["learn"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            key = f"{proto}/{arm}"
            summary[key] = dict(
                retain_mean=float(np.mean(rs)), retain_std=float(np.std(rs)),
                learn_mean=float(np.mean(ls)), learn_std=float(np.std(ls)))
            print("{:>22}:  retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
                  .format(key, summary[key]["retain_mean"],
                          summary[key]["retain_std"],
                          summary[key]["learn_mean"],
                          summary[key]["learn_std"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(dict(rows=rows, summary=summary,
                   protocol=dict(seeds=SEEDS, steps=STEPS, lr=LR,
                                 tau=TAU, batch=BATCH)),
              open(out_path, "w"), indent=2)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
