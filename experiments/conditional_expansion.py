"""Round 5: conditional escalation - plasticity must be EARNED.

Policy (user proposal): the base stays frozen while the gated correction
trains. Escalation to modifying the frozen base happens only when the
validation trajectory shows (1) real learning happened, (2) progress has
plateaued, and (3) error is still material. Directed-noise magnitude then
scales with the measured improvement.

Protocols:
  EASY : ordinary bump (frozen features sufficient -> should NOT escalate)
  HARD : x3-starved world, x3-localized bump (escalation required)

Reference arms: R2_frozen (stability extreme), GRADED_unconditional
(round-4 winner), plain_ft (plasticity extreme).
"""
import copy
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graded_expansion import (GatedResidual, Trunk, rmse, run_plain_ft,
                             f_base, bump, sample, sample_shifted,
                             STEPS, LR, BATCH, DIM)
from graded_expansion_v2 import f_base_no_x3, new_bump, sample_new


def sample_easy(n, rng):
    return sample_shifted(n, rng)


def sample_hard_base(n, rng):
    # x3 pinned constant: its input weights receive exactly zero signal
    x = rng.uniform(0.0, 1.0, size=(n, DIM))
    x[:, 2] = 0.5
    y = f_base_no_x3(x)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


def sample_hard_new(n, rng):
    x = np.clip(CENTER_H + rng.normal(0, 0.25, size=(n, DIM)), 0.0, 1.0)
    y = f_base_no_x3(x) + hard_bump(x)
    return x.astype(np.float32), (y + rng.normal(0, 0.01, n)).astype(np.float32)


CENTER_H = np.array([0.85, 0.85, 0.90, 0.15])
SEEDS = [0, 1, 2, 3, 4]


def hard_bump(x):
    d2 = ((x[:, 0] - CENTER_H[0]) ** 2 + (x[:, 1] - CENTER_H[1]) ** 2
          + ((x[:, 2] - CENTER_H[2]) / 0.30) ** 2
          + (x[:, 3] - CENTER_H[3]) ** 2)
    return 0.6 * np.exp(-d2 / (2 * 0.18 ** 2))


class _Gated(nn.Module):
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


class EscalationPolicy:
    """Fires when: learned (total improvement > 0), plateaued recently,
    and error still material. noise_scale grows with improvement."""

    def __init__(self, window=15, plateau_frac=0.02, min_improve=0.01):
        self.window, self.plateau_frac, self.min_improve = \
            window, plateau_frac, min_improve
        self.hist = []

    def observe(self, val_rmse):
        self.hist.append(float(val_rmse))

    def ready_to_escalate(self):
        h = self.hist
        if len(h) < 2 * self.window:
            return False, 0.0
        total = h[0] - h[-1]
        recent = h[-2 * self.window] - h[-self.window]
        recent2 = h[-self.window] - h[-1]
        gain_recent = (recent + recent2) / max(abs(recent) + 1e-9, 1e-9)
        plateaued = abs(recent2) / max(h[-self.window], 1e-9) \
            < self.plateau_frac
        learned = total / max(h[0], 1e-9) > self.min_improve
        return bool(learned and plateaued), max(total, 0.0) / max(h[0], 1e-9)


PHASES = [(300, 2), (300, 1), (300, None)]   # deep -> near -> head
EPS_BASE = 3e-4


def run_conditional(pristine, xt, yt, xv, yv, seed, max_level=3):
    torch.manual_seed(seed)
    xt = torch.as_tensor(xt, dtype=torch.float32)
    yt = torch.as_tensor(yt, dtype=torch.float32)
    xv = torch.as_tensor(np.asarray(xv), dtype=torch.float32)
    yv = torch.as_tensor(np.asarray(yv), dtype=torch.float32)
    model = _Gated(copy.deepcopy(pristine))
    policy = EscalationPolicy()
    rng = np.random.default_rng(seed)
    level = 0
    escalated_at = None
    log = []

    def param_sets(lvl):
        exp_p, tr_p = [], []
        for n, p in model.named_parameters():
            if n.startswith("value") or n.startswith("gate"):
                exp_p.append(p)
            elif lvl >= 2 or (lvl == 1 and n.startswith("base.net.")) \
                    or (lvl >= 1 and n.startswith("base.net.2.")
                        and not n.startswith("base.head")):
                tr_p.append(p)
        return exp_p, tr_p

    exp_p, _ = param_sets(0)
    opt = torch.optim.Adam(exp_p, lr=LR)
    step = 0
    while step < STEPS:
        idx = torch.tensor(rng.integers(0, len(xt), BATCH))
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 15 == 0:
            policy.observe(rmse(model, xv.numpy(), yv.numpy()))
        fire, strength = policy.ready_to_escalate()
        if fire and level < max_level:
            level += 1
            exp_p, tr_p = param_sets(level)
            opt = torch.optim.Adam([
                dict(params=exp_p, lr=LR),
                dict(params=tr_p, lr=LR * 0.3)])
            log.append((step, level, round(strength, 3)))
            policy.__init__()   # reset observation window after escalation
        elif fire and level >= 1:
            # regulate directed noise by measured improvement strength
            with torch.no_grad():
                for g in opt.param_groups[1:]:
                    for p in g["params"]:
                        if p.grad is None:
                            continue
                        p -= EPS_BASE * strength * torch.sign(p.grad)
    return model, log


def run_graded_uncond(pristine, xt, yt, seed):
    """Unconditional graded schedule (round-4 winner), same radial phases."""
    torch.manual_seed(seed)
    model = _Gated(copy.deepcopy(pristine))
    rng = np.random.default_rng(seed)
    scopes = [("near", 300), ("deep+near", 400), ("all", 400)]

    def params_for(scope):
        out = []
        for n, p in model.named_parameters():
            if n.startswith("value") or n.startswith("gate"):
                out.append(p)
            elif scope == "near" and n.startswith("base.net.2."):
                out.append(p)
            elif scope == "deep+near" and n.startswith("base.net."):
                out.append(p)
            elif scope == "all":
                out.append(p)
        return [p for p in out] or list(model.parameters())

    xt_t, yt_t = torch.tensor(xt), torch.tensor(yt)
    for scope, n_steps in scopes:
        opt = torch.optim.Adam(params_for(scope), lr=LR)
        for _ in range(n_steps):
            idx = torch.tensor(rng.integers(0, len(xt_t), BATCH))
            loss = ((model(xt_t[idx]) - yt_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    rows = []
    protocols = {
        "EASY": lambda rng: sample_easy(400, rng),
        "HARD": lambda rng: sample_hard_new(400, rng),
    }
    bases = {
        "EASY": lambda rng: sample(600, rng, with_bump=False),
        "HARD": lambda rng: sample_hard_base(600, rng),
    }
    probes = {
        "EASY": dict(retain=lambda rng: sample(2000, rng, with_bump=False),
                     learn=lambda rng: sample_shifted(2000, rng)),
        "HARD": dict(retain=lambda rng: sample_hard_base(2000, rng),
                     learn=lambda rng: sample_hard_new(2000, rng)),
    }

    for proto in ("EASY", "HARD"):
        for seed in SEEDS:
            rng = np.random.default_rng(700 + seed)
            xr, yr = probes[proto]["retain"](rng)
            xl, yl = probes[proto]["learn"](rng)
            x2, y2 = protocols[proto](rng)

            torch.manual_seed(seed)
            bdata = bases[proto](rng)
            base = Trunk(h=64)
            o = torch.optim.Adam(base.parameters(), lr=LR)
            xb, yb = torch.tensor(bdata[0]), torch.tensor(bdata[1])
            for _ in range(STEPS * 2):
                i = torch.tensor(rng.integers(0, len(xb), BATCH))
                loss = ((base(xb[i]) - yb[i]) ** 2).mean()
                o.zero_grad(); loss.backward(); o.step()
            pristine = copy.deepcopy(base)
            pre = (rmse(pristine, xr, yr), rmse(pristine, xl, yl))

            m_cond, esc_log = run_conditional(copy.deepcopy(pristine),
                                              x2, y2,
                                              torch.tensor(x2[:80]),
                                              torch.tensor(y2[:80]), seed)
            m_r2 = _Gated(copy.deepcopy(pristine))
            p2 = [p for p in m_r2.parameters() if p.requires_grad]
            o2 = torch.optim.Adam(p2, lr=LR)
            r2r = np.random.default_rng(seed)
            for _ in range(STEPS):
                i = torch.tensor(r2r.integers(0, len(x2), BATCH))
                l2 = ((m_r2(torch.tensor(x2[i])) -
                       torch.tensor(y2[i])) ** 2).mean()
                o2.zero_grad(); l2.backward(); o2.step()

            m_gr = run_graded_uncond(pristine, x2, y2, seed)
            m_pl = run_plain_ft(pristine, x2, y2, seed)

            for arm, mdl, extra in (
                    ("plain_ft", m_pl, {}),
                    ("R2_frozen", m_r2, {}),
                    ("GRADED_uncond", m_gr, {}),
                    ("CONDITIONAL", m_cond,
                     {"escalated_at": esc_log[0][0] if esc_log else None,
                      "levels": len(esc_log)})):
                rows.append(dict(protocol=proto, seed=seed, arm=arm,
                                 retain_before=pre[0], learn_before=pre[1],
                                 retain=rmse(mdl, xr, yr),
                                 learn=rmse(mdl, xl, yl), **extra))
            r = rows[-4:]
            print(f"[{proto} seed {seed}] " +
                  "  ".join(f"{x['arm']}:{x['retain']:.3f}/{x['learn']:.3f}"
                            for x in r) +
                  (f"  esc@{r[-1]['escalated_at']},lv{r[-1]['levels']}"
                   if r[-1].get("escalated_at") else ""))

    print("\n=== mean +- std ===")
    summary = {}
    for proto in ("EASY", "HARD"):
        for arm in ("plain_ft", "R2_frozen", "GRADED_uncond", "CONDITIONAL"):
            rs = [r["retain"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            ls = [r["learn"] for r in rows
                  if r["arm"] == arm and r["protocol"] == proto]
            key = f"{proto}/{arm}"
            summary[key] = dict(
                retain_mean=float(np.mean(rs)), retain_std=float(np.std(rs)),
                learn_mean=float(np.mean(ls)), learn_std=float(np.std(ls)))
            print("{:>26}:  retain {:6.4f}+-{:.4f}   learn {:6.4f}+-{:.4f}"
                  .format(key, np.mean(rs), np.std(rs), np.mean(ls), np.std(ls)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_conditional.json")
    json.dump(dict(rows=rows, summary=summary), open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
