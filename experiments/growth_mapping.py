"""Growth mapping: WHO changed, WHERE it matters.

Analyses on the two-shift gated-residual setup:
  1. GATE MAP     - learned gate opening per input region (near-A, near-B,
                    global). Structural protection predicts: opens at new
                    features, stays shut elsewhere.
  2. WEIGHT DELTA - norm of parameter change per component per stage.
  3. RESPONSIBILITY MAP - per-expander-unit ablation: RMSE increase on
                    region-A / region-B / global probes. Cross-referenced
                    with per-unit weight change during the B stage.
"""
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

SEEDS = [0, 1, 2]
DIM = 4
CENTERS = {
    "A": np.array([0.85, 0.85, 0.15, 0.15]),
    "B": np.array([0.15, 0.15, 0.85, 0.85]),
}


def f_base(x):
    return (0.5 * np.sin(2 * math.pi * x[:, 0]) + 0.3 * x[:, 1] ** 2
            + 0.2 * x[:, 2] + 0.1 * x[:, 3] * x[:, 0])


def bump(x, key, amp=0.6, w=0.20):
    d2 = np.sum((x - CENTERS[key]) ** 2, axis=1)
    return amp * np.exp(-d2 / (2 * w ** 2))


def make_stream(rng, n, key):
    c = CENTERS[key]
    x = np.clip(c + rng.normal(0, 0.25, size=(n, DIM)), 0.0, 1.0)
    active = {"A": {"A"}, "B": {"A", "B"}}[key]
    y = f_base(x) + sum(bump(x, k) for k in active)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


def probe(rng, n, kind):
    if kind == "global":
        x = rng.uniform(0.0, 1.0, size=(n, DIM))
        return x.astype(np.float32), f_base(x).astype(np.float32)
    c = CENTERS[kind]
    x = np.clip(c + rng.normal(0, 0.18, size=(n, DIM)), 0.0, 1.0)
    active = {"A": {"A"}, "B": {"A", "B"}}[kind]
    y = f_base(x) + sum(bump(x, k) for k in active)
    return x.astype(np.float32), y.astype(np.float32)


class Gated(nn.Module):
    def __init__(self, base, units=48):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False          # structural protection (production mode)
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

    def gate_open(self, x):
        hh = self.base.net(torch.tensor(x))
        with torch.no_grad():
            return torch.sigmoid(self.gate(hh)).squeeze(-1).numpy()


def fit(model, x, y, seed, steps=700, params=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params or model.parameters(), lr=1e-3)
    xt, yt = torch.tensor(x), torch.tensor(y)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = torch.tensor(rng.integers(0, len(xt), 128))
        loss = ((model(xt[i]) - yt[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x)).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    seed = 0
    rng = np.random.default_rng(11)
    torch.manual_seed(seed)
    xb, yb = make_stream(rng, 500, "A")           # stage A stream
    xb2, yb2 = make_stream(rng, 500, "B")         # stage B stream

    base = TrunkLike()
    fit(base, xb, yb, seed)                       # stage-1: base world
    pristine = copy.deepcopy(base)

    model = Gated(base)
    pre_gate = {}
    for region in ("A", "B", "global"):
        xp, yp = probe(rng, 1500, region)
        pre_gate[region] = float(np.mean(model.gate_open(xp)))

    # ---- stage A ----
    snapA_before = copy.deepcopy(model.state_dict())
    trainable = [p for p in model.parameters() if p.requires_grad]
    fit(model, xb, yb, seed, params=trainable)
    snapA_after = copy.deepcopy(model.state_dict())
    post_gate_A = {r: float(np.mean(model.gate_open(probe(rng, 1500, r)[0])))
                   for r in ("A", "B", "global")}

    # ---- stage B ----
    snapB_before = copy.deepcopy(model.state_dict())
    fit(model, xb2, yb2, seed + 1, params=trainable)
    snapB_after = copy.deepcopy(model.state_dict())
    post_gate_B = {r: float(np.mean(model.gate_open(probe(rng, 1500, r)[0])))
                   for r in ("A", "B", "global")}

    print("=== 1. GATE MAP (mean gate opening per region) ===")
    print(f"{'region':>8} {'init':>7} {'post-A':>8} {'post-B':>8}")
    for r in ("A", "B", "global"):
        print(f"{r:>8} {pre_gate[r]:7.3f} {post_gate_A[r]:8.3f} "
              f"{post_gate_B[r]:8.3f}")

    print("\n=== 2. WEIGHT DELTAS (L2 norm of change) ===")
    def delta_norm(s1, s2, prefix):
        tot = 0.0
        for k in s2:
            if k.startswith(prefix):
                tot += float((s2[k] - s1[k]).norm() ** 2)
        return math.sqrt(tot)
    for stage, s_before, s_after in (("A", snapA_before, snapA_after),
                                     ("B", snapA_after, snapB_after)):
        print(f"stage {stage}: value={delta_norm(s_before, s_after, 'value'):.4f}"
              f"  gate={delta_norm(s_before, s_after, 'gate'):.4f}"
              f"  base={delta_norm(s_before, s_after, 'base'):.4f}")

    print("\n=== 3. UNIT RESPONSIBILITY MAP (expander value units) ===")
    W = model.value[-1].weight.detach()          # (1, units)
    dW_stageB = (snapB_after["value.2.weight"]
                 - snapB_before["value.2.weight"])[0]   # (units,)
    resp = []
    for k in range(W.shape[1]):
        m2 = copy.deepcopy(model)
        with torch.no_grad():
            m2.value[-1].weight[0, k] = 0.0
        rA = rmse(m2, *probe(rng, 1200, "A"))
        rB = rmse(m2, *probe(rng, 1200, "B"))
        rg = rmse(m2, *probe(rng, 1200, "global"))
        full = rmse(model, *probe(rng, 1200, "A")), \
            rmse(model, *probe(rng, 1200, "B")), \
            rmse(model, *probe(rng, 1200, "global"))
        resp.append(dict(unit=k,
                         respA=float(rA - full[0]),
                         respB=float(rB - full[1]),
                         respG=float(rg - full[2]),
                         moved=float(dW_stageB[k].abs())))
    resp.sort(key=lambda d: d["moved"], reverse=True)
    print(f"{'unit':>5} {'|dW_B|':>8} {'respA':>8} {'respB':>8} {'respG':>8}")
    for d in resp[:10]:
        print(f"{d['unit']:>5} {d['moved']:8.4f} {d['respA']:8.4f} "
              f"{d['respB']:8.4f} {d['respG']:8.4f}")

    import numpy as _np
    moved = _np.array([d["moved"] for d in resp])
    rbias = _np.array([d["respB"] - d["respA"] for d in resp])
    corr = float(_np.corrcoef(moved, rbias)[0, 1])
    print(f"\ncorrelation(|unit movement during B| , B-minus-A responsibility)"
          f" = {corr:+.3f}")
    report = dict(gate= dict(init=pre_gate, postA=post_gate_A, postB=post_gate_B),
                  unit_resp=resp, corr_moved_vs_Bresponsibility=corr)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_growth_mapping.json")
    json.dump(report, open(out, "w"), indent=2)
    print("saved:", out)


class TrunkLike(nn.Module):
    """Minimal base net matching the gated-residual expectations."""
    def __init__(self, h=64):
        super().__init__()
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


if __name__ == "__main__":
    main()
