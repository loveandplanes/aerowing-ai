"""Round 8: the user's distribution-weight idea - Bayesian selective plasticity.

Each weight becomes a distribution w = mu + sigma*eps (reparameterization
trick -> trained entirely through normal backprop). sigma is LEARNED
per-weight: weights critical to stage-1 knowledge can reduce their own
uncertainty (soft self-protection); weights needed for the new region stay
uncertain (free to adapt). KL(anchor to stage-1 posterior) replaces both
hard freezing and sample masking.

Protocol: shifted stream (round-3 forgetting pressure).
Arms: plain_ft | bayes_ft (proposal) | R2_frozen (reference).
"""
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

SEEDS = [0, 1, 2, 3, 4]
STEPS = 900
LR = 1e-3
BATCH = 128
DIM = 4
CENTER = np.array([0.85, 0.85, 0.15, 0.15])
LAMBDA_KL = 1e-2


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


class BayesLinear(nn.Module):
    """Weight as learned distribution: w = mu + softplus(rho)*eps.
    Prior anchors at the stage-1 posterior (mu_p, sigma_p), not zero."""

    def __init__(self, d_in, d_out, prior_mu=None, prior_sigma=0.02):
        super().__init__()
        init_std = math.sqrt(2.0 / d_in)
        if prior_mu is not None:
            self.mu = nn.Parameter(prior_mu.clone())
        else:
            self.mu = nn.Parameter(torch.empty(d_out, d_in).normal_(
                0.0, init_std))
        self.rho = nn.Parameter(torch.full((d_out, d_in), -5.0))
        if prior_mu is not None:
            self.b_mu = nn.Parameter(torch.zeros(d_out))
        else:
            self.b_mu = nn.Parameter(torch.zeros(d_out))
        self.b_rho = nn.Parameter(torch.full((d_out,), -5.0))
        self.register_buffer(
            "prior_mu", prior_mu.clone() if prior_mu is not None
            else torch.zeros(d_out))
        self.register_buffer(
            "prior_mu_b", torch.zeros(d_out))
        self.prior_sigma = prior_sigma

    def forward(self, x):
        s = tnf.softplus(self.rho)
        sb = tnf.softplus(self.b_rho)
        w = self.mu + s * torch.randn_like(s)
        b = self.b_mu + sb * torch.randn_like(sb)
        return torch.nn.functional.linear(x, w, b)

    def kl_to_prior(self):
        """KL(N(mu,sig) || N(mu_p, sig_p)) summed - anchors at the stage-1
        posterior so the network stays near what it already knows."""
        s = tnf.softplus(self.rho)
        var_q = s ** 2
        var_p = self.prior_sigma ** 2
        kld = (0.5 * (math.log(var_p) - torch.log(var_q)
               + (var_q + (self.mu - self.prior_mu) ** 2) / var_p - 1.0)).sum()
        sb = tnf.softplus(self.b_rho)
        var_bq = sb ** 2
        kb = (0.5 * (math.log(var_p) - torch.log(var_bq)
              + (var_bq + (self.b_mu - self.prior_mu_b) ** 2)
              / var_p - 1.0)).sum()
        return kld + kb


class BayesNet(nn.Module):
    def __init__(self, h=64, base=None):
        super().__init__()
        src = dict(base.named_parameters()) if base is not None else None
        pm1 = src["net.0.weight"] if src else None
        self.l1 = BayesLinear(DIM, h,
                              prior_mu=pm1, prior_sigma=0.02)
        if src is not None:
            self.l1.b_mu = nn.Parameter(src["net.0.bias"].clone())
            self.l1.prior_mu_b = src["net.0.bias"].detach().clone()
        self.l2 = BayesLinear(h, h,
                              prior_mu=src["net.2.weight"] if src else None,
                              prior_sigma=0.02)
        if src is not None:
            self.l2.b_mu = nn.Parameter(src["net.2.bias"].clone())
            self.l2.prior_mu_b = src["net.2.bias"].detach().clone()
        self.out = BayesLinear(h, 1,
                               prior_mu=src["head.weight"] if src else None,
                               prior_sigma=0.02)
        if src is not None:
            self.out.b_mu = nn.Parameter(src["head.bias"].clone())
            self.out.b_mu_prior = None
            self.out.register_buffer("prior_mu_b",
                                     src["head.bias"].detach().clone())
        # initialize mus from stage-1 weights (knowledge init)
        if src is not None:
            with torch.no_grad():
                self.l1.mu.copy_(src["net.0.weight"])
                self.l1.b_mu.copy_(src["net.0.bias"])
                self.l2.mu.copy_(src["net.2.weight"])
                self.l2.b_mu.copy_(src["net.2.bias"])
                self.out.mu.copy_(src["head.weight"])
                self.out.b_mu.copy_(src["head.bias"])

    def forward(self, x):
        h = tnf.gelu(self.l1(x))
        h = tnf.gelu(self.l2(h))
        return self.out(h).squeeze(-1)

    def kl(self):
        return self.l1.kl_to_prior() + self.l2.kl_to_prior() \
            + self.out.kl_to_prior()

    def deterministic_forward(self, x):
        saved = [(m.rho.detach().clone()) for m in
                 (self.l1, self.l2, self.out)]
        for m, blk in zip((self.l1, self.l2, self.out),
                          (self.l1, self.l2, self.out)):
            pass
        # evaluate with eps=0: temporarily zero rho effect
        outs = []
        orig = [blk.rho.detach().clone() for blk in
                (self.l1, self.l2, self.out)]
        for blk in (self.l1, self.l2, self.out):
            blk.rho.data.fill_(-30.0)   # sigma ~ 0 -> w = mu
        with torch.no_grad():
            outs = self.forward(x)
        for blk, r in zip((self.l1, self.l2, self.out), orig):
            blk.rho.data.copy_(r)
        return outs


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(2500 + seed)
        xr, yr = sample(2000, rng, with_bump=False)
        xl, yl = sample_shifted(2000, rng)
        x2, y2 = sample_shifted(400, rng)

        torch.manual_seed(seed)
        xb, yb = sample(600, rng, with_bump=False)

        # ---- shared stage-1 base (plain MLP) ----
        base = Trunk(h=64)
        o = torch.optim.Adam(base.parameters(), lr=LR)
        xt0, yt0 = torch.tensor(xb), torch.tensor(yb)
        rr = np.random.default_rng(seed + 9)
        for _ in range(STEPS * 2):
            i = torch.tensor(rr.integers(0, len(xt0), BATCH))
            loss = ((base(xt0[i]) - yt0[i]) ** 2).mean()
            o.zero_grad(); loss.backward(); o.step()
        pristine = copy.deepcopy(base)
        pre_ret, pre_lrn = rmse(pristine, xr, yr), rmse(pristine, xl, yl)

        results = {}
        for arm in ("plain_ft", "bayes_ft", "R2_frozen"):
            if arm == "plain_ft":
                m = copy.deepcopy(pristine)
                fit_plain(m, x2, y2, seed)
            elif arm == "bayes_ft":
                m = BayesNet(h=64, base=pristine)
                fit_bayes(m, x2, y2, seed)
            else:
                import torch.nn as tnn

                class Gated(tnn.Module):
                    def __init__(self):
                        super().__init__()
                        self.base = copy.deepcopy(pristine)
                        for p in self.base.parameters():
                            p.requires_grad = False
                        h = pristine.net[0].out_features
                        self.value = tnn.Sequential(
                            tnn.Linear(h, 64), tnn.GELU(), tnn.Linear(64, 1))
                        self.gate = tnn.Sequential(
                            tnn.Linear(h, 32), tnn.GELU(), tnn.Linear(32, 1))
                        with torch.no_grad():
                            self.value[-1].weight.zero_()
                            self.value[-1].bias.zero_()

                    def forward(self, xx):
                        with torch.no_grad():
                            b = self.base(xx)
                        hh = self.base.net(xx)
                        corr = self.value(hh).squeeze(-1) * torch.sigmoid(
                            self.gate(hh)).squeeze(-1)
                        return b + corr
                m = Gated()
                fit_plain(m, x2, y2, seed,
                          params=[p for p in m.parameters() if p.requires_grad])
            results[arm] = dict(retain=rmse(m, xr, yr),
                                learn=rmse(m, xl, yl))
            rows.append(dict(seed=seed, arm=arm, **results[arm],
                             retain_before=pre_ret, learn_before=pre_lrn))

        print(f"[seed {seed}] " +
              "  ".join(f"{a}: r={results[a]['retain']:.4f} "
                        f"l={results[a]['learn']:.4f}"
                        for a in ("plain_ft", "bayes_ft", "R2_frozen")))

    print("\n=== mean +- std ===")
    for arm in ("plain_ft", "bayes_ft", "R2_frozen"):
        rs = [r["retain"] for r in rows if r["arm"] == arm]
        ls = [r["learn"] for r in rows if r["arm"] == arm]
        print("{:>10}:  retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
              .format(arm, np.mean(rs), np.std(rs), np.mean(ls), np.std(ls)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_bayes_plasticity.json")
    json.dump(rows, open(out, "w"), indent=2)
    print("saved:", out)


def fit_plain(model, x2, y2, seed, params=None):
    torch.manual_seed(seed + 21)
    opt = torch.optim.Adam(params or model.parameters(), lr=LR)
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    rng = np.random.default_rng(seed + 41)
    for _ in range(STEPS):
        i = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = ((model(xt[i]) - yt[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def fit_bayes(model, x2, y2, seed):
    """User spec: sigma/deviation decreases as it learns; once small,
    the network runs on the mean values alone."""
    torch.manual_seed(seed + 22)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    rng = np.random.default_rng(seed + 42)
    for step in range(STEPS):
        frac = step / STEPS
        # full anchor during first 60% -> protection;
        # anneal to ~5% by the end -> plasticity for the new region,
        # then the network effectively runs on mu (point values)
        lam_t = LAMBDA_KL * (1.0 if frac < 0.6
                             else max(0.05, 1.0 - (frac - 0.6) / 0.4))
        i = torch.tensor(rng.integers(0, len(xt), BATCH))
        pred = model(xt[i])
        loss = ((pred - yt[i]) ** 2).mean() + lam_t * model.kl()
        opt.zero_grad(); loss.backward(); opt.step()


if __name__ == "__main__":
    main()
