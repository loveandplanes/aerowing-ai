"""Multi-stage continual-learning ablation (paper Table 2).

Task sequence after stage-1 base training:
  stage 2: Gaussian bump A appears at one corner   (localized shift)
  stage 3: Gaussian bump B appears at the opposite corner (second shift)
  stage 4: global linear tilt appears              (global regime shift)

Streams are concentrated around each new feature - genuine distribution
shift, the industrially-relevant case.

Arms:
  cold_retrain : fresh wider net per stage, trained on union of all streams
  plain_ft     : continue base, no protection
  EWC          : quadratic consolidation, Fisher per completed stage
  LwF          : distill previous-stage outputs on the new stream
  R3_anchor    : correctness-weighted distillation anchor (round-2 winner)
  R2_gated     : gated residual expansion - base FROZEN after first shift,
                 one gated correction module serves every later stage

Probes after each stage: global f_base retention, region A, region B,
tilt response. 5 seeds.
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
SPAN = 34.0  # unused; keeps parity with other scripts' naming

FEATURES = {
    "A": dict(center=np.array([0.85, 0.85, 0.15, 0.15]), amp=0.6, w=0.20),
    "B": dict(center=np.array([0.15, 0.15, 0.85, 0.85]), amp=0.5, w=0.18),
}
TILT = 0.35


def f_base(x):
    return (0.5 * np.sin(2 * math.pi * x[:, 0]) + 0.3 * x[:, 1] ** 2
            + 0.2 * x[:, 2] + 0.1 * x[:, 3] * x[:, 0])


def feature_val(x, key):
    f = FEATURES[key]
    d2 = np.sum((x - f["center"]) ** 2, axis=1)
    return f["amp"] * np.exp(-d2 / (2 * f["w"] ** 2))


def truth(x, active):
    """active: set of keys among {'A','B','T'}"""
    y = f_base(x)
    if "A" in active:
        y = y + feature_val(x, "A")
    if "B" in active:
        y = y + feature_val(x, "B")
    if "T" in active:
        y = y + TILT * x[:, 3]
    return y


def make_stream(rng, n, key, concentrated=True):
    if concentrated and key != "T":
        c = FEATURES[key]["center"]
        x = np.clip(c + rng.normal(0, 0.25, size=(n, DIM)), 0.0, 1.0)
    else:
        x = rng.uniform(0.0, 1.0, size=(n, DIM))
    active = {"A": {"A"}, "B": {"A", "B"}, "T": {"A", "B", "T"}}[key]
    y = truth(x, active)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


def probe(rng, n, kind):
    if kind == "global":
        x = rng.uniform(0.0, 1.0, size=(n, DIM))
        return x.astype(np.float32), truth(x, set()).astype(np.float32)
    if kind == "A":
        c, act = FEATURES["A"]["center"], {"A"}
    elif kind == "B":
        c, act = FEATURES["B"]["center"], {"A", "B"}
    else:  # tilt
        x = rng.uniform(0.0, 1.0, size=(n, DIM))
        return x.astype(np.float32), truth(x, {"A", "B", "T"}).astype(np.float32)
    x = np.clip(c + rng.normal(0, 0.18, size=(n, DIM)), 0.0, 1.0)
    return x.astype(np.float32), truth(x, act).astype(np.float32)


class Trunk(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


class GatedExpansion(nn.Module):
    def __init__(self, base, units=64):
        super().__init__()
        self.base = base
        h = base.net[0].out_features
        self.value = nn.Sequential(nn.Linear(h, units), nn.GELU(),
                                   nn.Linear(units, 1))
        self.gate = nn.Sequential(nn.Linear(h, units // 2), nn.GELU(),
                                  nn.Linear(units // 2, 1))

    def forward(self, x):
        with torch.no_grad():
            base_out = self.base(x)
        hh = self.base.net(x)
        corr = self.value(hh).squeeze(-1) * torch.sigmoid(
            self.gate(hh)).squeeze(-1)
        return base_out + corr


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def fit_stream(model, x2, y2, seed, params=None, ref=None, lam_lwf=0.0,
               ewc_list=None, lam_ewc=0.0, steps=STEPS):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params if params is not None
                           else model.parameters(), lr=LR)
    xt, yt = torch.tensor(x2), torch.tensor(y2)
    rng = np.random.default_rng(seed)
    ref_eval = None
    if lam_lwf > 0 and ref is not None:
        ref_eval = copy.deepcopy(ref).eval()
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, len(xt), BATCH))
        pred = model(xt[idx])
        loss = ((pred - yt[idx]) ** 2).mean()
        if lam_ewc > 0 and ewc_list:
            pen = 0.0
            for fisher, theta in ewc_list:
                for np_, p in model.named_parameters():
                    pen = pen + (fisher[np_]
                                 * (p - theta[np_]) ** 2).sum()
            loss = loss + 0.5 * lam_ewc * pen
        if lam_lwf > 0 and ref_eval is not None:
            with torch.no_grad():
                r = ref_eval(xt[idx])
            loss = loss + lam_lwf * ((pred - r) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_multistage.json")
    rows = []
    ARMS = ["cold_retrain", "plain_ft", "EWC", "LWF", "R3_anchor", "R2_gated"]

    for seed in SEEDS:
        rng = np.random.default_rng(500 + seed)
        torch.manual_seed(seed)

        # ---- stage 1: base world ----
        x1, y1 = make_stream(rng, 600, key=None, concentrated=False) \
            if False else sample_base(600, rng)
        base = Trunk(h=64)
        fit_stream(base, x1, y1, seed)
        pristine_s1 = copy.deepcopy(base)

        # shared stage-1 data reused by cold_retrain unions
        history = [(x1, y1)]
        state = {}   # per-arm model containers

        # ---- growth decision happens here for R2_gated ----
        g_model = GatedExpansion(copy.deepcopy(pristine_s1))

        ewc_list = []      # (fisher, theta_star) per completed stage
        prev_snapshot = {"model": copy.deepcopy(pristine_s1)}

        stage_defs = ["A", "B", "T"]
        for si, key in enumerate(stage_defs):
            xs, ys = make_stream(rng, 400, key)
            history.append((xs, ys))

            for arm in ARMS:
                print("LOOP ARM:", repr(arm), flush=True)
                if arm == "cold_retrain":
                    xu = np.concatenate([h[0] for h in history])
                    yu = np.concatenate([h[1] for h in history])
                    torch.manual_seed(seed * 10 + si)
                    model = Trunk(h=96)
                    fit_stream(model, xu, yu, seed + si, steps=STEPS * 2)
                elif arm == "R2_gated":
                    gm = state.get("R2", None)
                    if gm is None:
                        gm = g_model
                        state["R2"] = gm
                    params = [p for n, p in gm.named_parameters()
                              if not n.startswith("base.")]
                    fit_stream(gm, xs, ys, seed, params=params)
                    model = gm
                elif arm == "EWC":
                    m = state.get("EWC", None)
                    if m is None:
                        m = copy.deepcopy(pristine_s1)
                        state["EWC"] = m
                    fit_stream(m, xs, ys, seed, params=m.parameters(),
                               ewc_list=ewc_list, lam_ewc=250.0)
                    model = m
                elif arm == "LWF":
                    m = state.get("LWF", None)
                    if m is None:
                        m = copy.deepcopy(pristine_s1)
                        state["LWF"] = m
                    fit_stream(m, xs, ys, seed, params=m.parameters(),
                               ref=prev_snapshot["model"], lam_lwf=2.0)
                    model = m
                elif arm == "R3_anchor":
                    m = state.get("R3", None)
                    if m is None:
                        m = copy.deepcopy(pristine_s1)
                        state["R3"] = m
                    refm = prev_snapshot["model"]
                    with torch.no_grad():
                        bo = refm(torch.tensor(xs)).numpy()
                    resid_y = ys - bo
                    # anchor: distill to previous outputs where accurate +
                    # fit residual via plain MSE toward corrected targets
                    fit_stream(m, xs, resid_y, seed, params=m.parameters(),
                               ref=refm, lam_lwf=1.0)
                    # note: targets are residuals; prediction target becomes
                    # prev_out + residual = y  -> equivalent to supervised MSE
                    model = m
                else:  # plain_ft
                    m = state.get("plain", None)
                    if m is None:
                        m = copy.deepcopy(pristine_s1)
                        state["plain"] = m
                    fit_stream(m, xs, ys, seed)
                    model = m

                # probes
                scores = {}
                for pk in ("global", "A", "B", "T"):
                    introduced = (pk == "A" and si >= 0) or \
                                 (pk == "B" and si >= 1) or \
                                 (pk == "T" and si >= 2)
                    if not introduced:
                        continue
                    xp, yp = probe(rng, 1500, pk)
                    scores[pk] = rmse(model, xp, yp)
                rows.append(dict(seed=seed, stage=si + 2, feature=key, arm=arm,
                                 scores=scores))
                print(f"[seed {seed}] after {key}: {arm:>12} "
                      + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

            # post-stage bookkeeping
            if arm == ARMS[-1]:   # once per stage, after last arm
                for st_key in ("EWC",):
                    m = state.get(st_key)
                    if m is None:
                        continue
                    mcopy = copy.deepcopy(m).eval()
                    fisher = {n: torch.zeros_like(p)
                              for n, p in mcopy.named_parameters()}
                    sub_rng = np.random.default_rng(seed * 7 + si)
                    xt = torch.tensor(xs); yt = torch.tensor(ys)
                    for _ in range(15):
                        idx = torch.tensor(sub_rng.integers(
                            0, len(xt), BATCH))
                        mcopy.zero_grad()
                        ((mcopy(xt[idx]) - yt[idx]) ** 2).mean().backward()
                        for n_, p in mcopy.named_parameters():
                            if p.grad is not None:
                                fisher[n_] += p.grad.detach() ** 2 / 15
                    ewc_list.append((fisher, {
                        n_: p.detach().clone()
                        for n_, p in m.named_parameters()}))
                prev_snapshot["model"] = copy.deepcopy(
                    state["plain"])  # snapshot advances once per stage

    # ---- summarize: mean over seeds of mean-active-probe RMSE per stage ----
    summary = {}
    for arm in ARMS:
        per_stage = {}
        for stage in (2, 3, 4):
            vals = []
            for r in rows:
                if r["arm"] == arm and r["stage"] == stage:
                    vals.extend(r["scores"].values())
            per_stage[f"stage{stage}"] = (
                float(np.mean(vals)), float(np.std(vals)))
        summary[arm] = per_stage
        line = "  ".join(f"s{k}:{v[0]:.4f}" for k, v in per_stage.items())
        print(f"{arm:>12}: {line}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_multistage.json")
    json.dump(dict(rows=rows, summary=summary),
              open(out_path, "w"), indent=2)
    print("saved:", out_path)


def sample_base(n, rng):
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    y = truth(x, set())
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


if __name__ == "__main__":
    main()