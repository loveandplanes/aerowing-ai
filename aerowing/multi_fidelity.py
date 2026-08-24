"""
Multi-Fidelity Study Harness — "the money slide".

Quantifies the variable-fidelity value proposition of the continuous-learning
loop the way a preliminary-design team would have to: how prediction error
against a high-fidelity reference falls as expensive labels accrue, versus
the flat error of the cheap low-fidelity model alone.

Fidelity pairing (deterministic, seeded):
    low  = VLM-only inviscid evaluation (VLMSolver3D): CL, CDi, CM, span
           efficiency, fuel volume. This is what a company gets at ~zero cost.
    high = full AeroEngine3D stack (VLM + compressible boundary layer +
           Korn-Mason wave drag). Its relation to the low model
           (truth = low + profile drag + wave drag + L/D shift) is the same
           bias structure a converged CFD campaign produces, and it is
           generated with exactly the code the ship runs.

What the sweep shows (four trajectories):
    VLM-only  : RMSE vs truth, constant — spending more on expensive labels
                changes nothing unless the labels feed the loop.
    flywheel  : the surrogate, warm-started on VLM labels and refined with
                quality-gated high-fidelity labels through the production
                path (AeroDataLake + ContinualTrainer + holdout promotion
                gate) — the error descends toward the truth stand-in.
    cold      : a surrogate trained ONLY on the high-fidelity labels (raw
                CFD-style training, no cheap model). The textbook
                variable-fidelity result: at any small/medium label budget
                the warm model beats the cold one — the cheap model
                multiplies the information in expensive labels, and the two
                converge as label coverage saturates the domain.
    (implicit) full-CFD-everything: zero error at (almost) every label's cost.

All flywheel/cold trajectories share the same loss objective
(inverse-variance weighted MSE + identical physics terms), the same label
budget, and the same seed family — the only difference is the VLM warm-start.

Everything is seeded and deterministic: run it twice, get the same curve.

Study domain: the design space keeps the full planform/CST diversity of the
production dataset, but the FLIGHT ENVELOPE is narrowed to a cruise band
(alpha ~1-4.5 deg, M ~0.76-0.85, Re ~1.5e7-3e7) — the regime preliminary
design actually sweeps. This is both the industry pattern and an honest
statement about data economics: a sparse 40-D sample cannot resolve a full
envelope, so a study across alpha -1..8/M 0.3..0.85 at these volumes would
show a stagnant curve (that stall is itself the finding the study is built
to expose).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .geometry.wing_3d import Wing3D
from .solvers.aero_engine import AeroEngine3D
from .solvers.vlm_3d import VLMSolver3D
from .continual import (
    AeroDataLake,
    ContinualTrainer,
    _Y_DIM,
    _physics_terms,
    design_to_input,
    random_exploration_design,
)

Y_NAMES = [
    "cl", "cd", "cd_induced", "cd_profile", "cd_wave",
    "cm", "l_over_d", "span_efficiency", "fuel_volume_m3",
]

# Cruise flight band: (alpha_deg range, mach range, log10(Re) range).
# See the module docstring for the domain-rationale discussion.
CRUISE_BAND = [
    (1.0, 4.5),
    (0.76, 0.85),
    (7.2, 7.5),
]


def evaluate_low_fi(wing: Wing3D, alpha_deg: float, mach: float,
                    reynolds: float) -> np.ndarray:
    """9-D label from the low-fidelity model: VLM inviscid physics only.
    Profile/wave drag are absent by construction — that is exactly the gap
    the expensive model must fill (and the surrogate must learn)."""
    vlm = VLMSolver3D(wing, num_chordwise=8, num_spanwise=14)
    r = vlm.solve(alpha_deg=alpha_deg, mach=mach)
    cl = float(r["cl"])
    cdi = float(r["cd_induced"])
    cm = float(r["cm"])
    e = float(r["span_efficiency"])
    l_over_d = cl / max(cdi, 1e-5)
    fuel = float(wing.compute_internal_fuel_volume())
    return np.array([cl, cdi, cdi, 0.0, 0.0, cm, l_over_d, e, fuel])


def evaluate_high_fi(wing: Wing3D, alpha_deg: float, mach: float,
                     reynolds: float) -> np.ndarray:
    """9-D label from the high-fidelity stand-in: full AeroEngine3D stack
    (VLM + compressible boundary layer + Korn-Mason wave drag)."""
    res = AeroEngine3D(wing, num_chordwise=8, num_spanwise=14).evaluate(
        alpha_deg=alpha_deg, mach=mach, reynolds=reynolds)
    return np.array([
        res.cl, res.cd, res.cd_induced, res.cd_profile, res.cd_wave,
        res.cm, res.l_over_d, res.span_efficiency, res.fuel_volume,
    ])


def sample_case(rng: np.random.Generator,
                flight_band: Sequence[Tuple[float, float]] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One (x_40, y_low, y_high) triple. Design diversity matches the
    production dataset generator; flight conditions are drawn from a cruise
    band (see module docstring for why)."""
    if flight_band is None:
        flight_band = CRUISE_BAND
    alpha_lo, alpha_hi = flight_band[0]
    mach_lo, mach_hi = flight_band[1]
    logre_lo, logre_hi = flight_band[2]
    design_37 = random_exploration_design(rng)
    alpha_deg = float(rng.uniform(alpha_lo, alpha_hi))
    mach = float(rng.uniform(mach_lo, mach_hi))
    reynolds = float(10 ** rng.uniform(logre_lo, logre_hi))
    wing = Wing3D.from_parameter_vector(design_37, name="Study_Wing")
    x = design_to_input(design_37, alpha_deg, mach, reynolds)
    return x, evaluate_low_fi(wing, alpha_deg, mach, reynolds), \
        evaluate_high_fi(wing, alpha_deg, mach, reynolds)


def _column_rmse(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    err = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    return {name: float(err[i]) for i, name in enumerate(Y_NAMES)}


def _column_mean_abs_gap(low: np.ndarray, high: np.ndarray) -> Dict[str, float]:
    gap = np.mean(np.abs(high - low), axis=0)
    return {name: float(gap[i]) for i, name in enumerate(Y_NAMES)}


@dataclass
class StudyResult:
    """The money-slide dataset: one error-vs-CFD-budget curve plus baselines."""

    seed: int
    n_designs: int
    n_holdout: int
    budgets: List[int]
    cfd_units: List[int]
    vlm_baseline: Dict[str, float]
    truth_gap: Dict[str, float]
    points: List[Dict[str, Any]] = field(default_factory=list)
    truth: str = "engine"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "n_designs": self.n_designs,
            "n_holdout": self.n_holdout,
            "budgets": self.budgets,
            "cfd_units": self.cfd_units,
            "y_names": Y_NAMES,
            "vlm_baseline": self.vlm_baseline,
            "truth_gap": self.truth_gap,
            "points": self.points,
            "truth": self.truth,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudyResult":
        return cls(
            seed=d["seed"], n_designs=d["n_designs"], n_holdout=d["n_holdout"],
            budgets=d["budgets"], cfd_units=d["cfd_units"],
            vlm_baseline=d["vlm_baseline"], truth_gap=d["truth_gap"],
            points=d["points"], truth=d.get("truth", "engine"))

    def table(self, key: str = "cd") -> str:
        """Human-readable error-vs-budget table for one target quantity."""
        lines = []
        mode = ("real CFD lake rows" if self.truth == "lake"
                else "AeroEngine3D stand-in (synthetic truth)")
        lines.append(f"\nMulti-fidelity study (seed {self.seed}, truth: {mode}) "
                     f"— RMSE of {key} vs high-fidelity truth "
                     f"(holdout n={self.n_holdout})")
        lines.append("-" * 78)
        vlm_b = self.vlm_baseline.get(key)
        gap = self.truth_gap.get(key)
        lines.append(f"  VLM-only (any budget):        "
                     f"{'n/a' if vlm_b is None else f'{vlm_b:.6f}'}")
        lines.append(f"  Truth vs VLM mean gap:        "
                     f"{'n/a' if gap is None else f'{gap:.6f}'}")
        lines.append("-" * 78)
        lines.append("  CFD units | promoted | flywheel |   cold   | vs VLM-only")
        lines.append("-" * 78)
        for p in self.points:
            rmse_v = p["rmse"].get(key)
            delta_s = "     n/a"
            rmse_s = "   n/a "
            if rmse_v is not None and vlm_b is not None:
                rmse_s = f"{rmse_v:7.5f}"
                delta = rmse_v / max(vlm_b, 1e-12) - 1.0
                delta_s = f"{delta * 100:7.1f}%"
            cold = p.get("rmse_cold", {}).get(key)
            cold_str = "   n/a " if cold is None else f"{cold:7.5f}"
            lines.append(f"  {p['cfd_units']:>9d} | "
                         f"{'yes' if p['promoted'] else 'no ':>8s} | "
                         f"{rmse_s} | "
                         f"{cold_str} | "
                         f"{delta_s}")
        lines.append("-" * 78)
        return "\n".join(lines)

    def ascii_curve(self, key: str = "cd", width: int = 60) -> str:
        """Small terminal chart: flywheel (F) vs cold (C) vs VLM-only (|)."""
        vals = [p["rmse"][key] for p in self.points if p["rmse"].get(key) is not None]
        vals += [p.get("rmse_cold", {}).get(key)
                 for p in self.points if p.get("rmse_cold", {}).get(key)]
        vlm_v = self.vlm_baseline.get(key)
        if vlm_v is None and not vals:
            return f"\n{key}: no delivered values under the truth mask."
        vmax = max(max(vals), 1e-12)
        lines = [f"\n{key} RMSE vs high-fi truth — flywheel (F), cold (C), "
                 f"VLM-only (|) [scale 0..{vmax:.5f}]"]
        if vlm_v is not None:
            vlm_bar = int(round(vlm_v / vmax * width))
            lines.append("  VLM-only " + "|" * max(vlm_bar, 1) + " (flat)")
        for p in self.points:
            rv = p["rmse"].get(key)
            if rv is None:
                lines.append(f"  {p['cfd_units']:>6d} CFD   (no delivered values)")
                continue
            bar = int(round(rv / vmax * width))
            cold = p.get("rmse_cold", {}).get(key)
            cold_str = ""
            if cold is not None:
                c_bar = int(round(cold / vmax * width))
                cold_str = f"   C: {'C' * max(c_bar, 1)}"
            lines.append(f"  {p['cfd_units']:>6d} CFD  "
                         + "F" * max(bar, 1) + cold_str)
        return "\n".join(lines)


def _delivered_columns(mask: np.ndarray) -> np.ndarray:
    """Columns delivered in EVERY row (conservative AND over row masks)."""
    if mask is None or len(mask) == 0:
        return np.ones(len(Y_NAMES))
    return np.asarray(mask, dtype=float).min(axis=0)


def _mask_metrics(metrics: Dict[str, Optional[float]],
                  col_ok: np.ndarray) -> Dict[str, Optional[float]]:
    """Undelivered truth columns report as None instead of fake numbers."""
    return {name: (float(v) if (ok and v is not None) else None)
            for (name, v), ok in zip(metrics.items(), col_ok)}


def _load_cfd_truth(lake_path: str):
    """Accepted real-CFD rows from a lake, as (x[N,40], y_hi[N,9], mask[N,9]).

    Uses AeroDataLake.cfd_rows so the engine stand-in ('*_study' sources)
    can never masquerade as ground truth."""
    from .continual import AeroDataLake
    lake = AeroDataLake(lake_path)
    try:
        x, y_hi, masks, _ids = lake.cfd_rows()
    finally:
        lake.close()
    return x, y_hi, masks


class MultiFidelityStudy:
    """Runs the deterministic error-vs-CFD-budget sweep end to end.

    For every budget the flywheel leg is rebuilt from scratch: candidate
    designs enter the lake as VLM labels (source 'vlm'), the surrogate is
    warm-started on them, a fixed unseen holdout receives only high-fidelity
    labels (never trained on), and then the first `budget` candidates are
    (re-)labeled at high fidelity and the model is refined with the real
    ContinualTrainer — promotion gate included.

    Truth modes (`truth=`):
      * "engine" — AeroEngine3D stand-in labels. Fully synthetic: useful to
        exercise the mechanism and compare learner recipes, but it is NOT a
        claim about real-CFD accuracy (the stand-in shares its physics with
        the cheap prior).
      * "lake"   — accepted rows whose source marks real CFD (via
        AeroDataLake.cfd_rows) act as truth; partial-delivery masks are
        honored end to end and undelivered metrics report as None.
    """

    # Training is expressed as fixed optimizer-step budgets so the sweep is
    # comparable across pool sizes (raw epochs are meaningless when every
    # epoch is a single batch). The CLI epochs args are *floors*.
    WARM_STEP_TARGET = 600
    FINE_STEP_TARGET = 200
    COLD_STEP_TARGET = 300
    WARM_BATCH = 32

    def __init__(self, seed: int = 1337):
        self.seed = int(seed)

    def _seed_all(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def run(self, n_designs: int = 160, n_holdout: int = 48,
            budgets: Sequence[int] = (0, 16, 32, 64, 128),
            warm_epochs: int = 18, warm_lr: float = 1e-3,
            finetune_epochs: int = 6, finetune_lr: float = 1e-4,
            flight_band: Optional[Sequence[Tuple[float, float]]] = None,
            workdir: Optional[str] = None, verbose: bool = True,
            truth: str = "engine",
            lake_path: Optional[str] = None) -> StudyResult:
        if truth not in ("engine", "lake"):
            raise ValueError("truth must be 'engine' or 'lake'")
        self._seed_all()
        budgets = [int(b) for b in budgets]
        rng = np.random.default_rng(self.seed)
        n_cand = max(int(n_designs), len(budgets))
        n_hold = max(int(n_holdout), 4)
        base_dir = workdir or tempfile.mkdtemp(prefix="mf_study_")
        os.makedirs(base_dir, exist_ok=True)

        if truth == "lake":
            x_all, y_hi_all, m_all = _load_cfd_truth(lake_path)
            need = n_cand + n_hold
            if len(x_all) < need:
                raise ValueError(
                    f"truth='lake' needs {need} accepted CFD rows but the "
                    f"lake at {lake_path} exposes {len(x_all)}")
            if verbose:
                print(f"[mf-study] using {need} of {len(x_all)} real CFD rows "
                      f"from {lake_path} (seed {self.seed}) ...")
            perm = rng.permutation(len(x_all))[:need]
            cases, mask_rows = [], []
            for idx in perm:
                xv = x_all[idx]
                wing = Wing3D.from_parameter_vector(np.asarray(xv[:37]),
                                                    name="study_cfd")
                y_lo = evaluate_low_fi(wing, float(xv[37]), float(xv[38]),
                                       10.0 ** float(xv[39]))
                cases.append((xv, y_lo, np.asarray(y_hi_all[idx], dtype=float)))
                mask_rows.append(m_all[idx])
            mask_cases = np.asarray(mask_rows, dtype=float)
        else:
            if verbose:
                print(f"[mf-study] sampling {n_cand} candidates + {n_hold} "
                      f"unseen holdout designs (seed {self.seed}) ...")
            cases = [sample_case(rng, flight_band=flight_band)
                     for _ in range(n_cand + n_hold)]
            mask_cases = np.ones((n_cand + n_hold, len(Y_NAMES)))

        cand = cases[:n_cand]
        hold = cases[n_cand:]
        m_cand_hi = mask_cases[:n_cand]
        m_hold = mask_cases[n_cand:]
        x_hold = np.array([h[0] for h in hold], dtype=float)
        y_hold_hi = np.array([h[2] for h in hold], dtype=float)
        y_hold_lo = np.array([h[1] for h in hold], dtype=float)
        x_cand = np.array([c[0] for c in cand], dtype=float)
        y_cand_lo = np.array([c[1] for c in cand], dtype=float)
        y_cand_hi = np.array([c[2] for c in cand], dtype=float)

        vlm_baseline = _mask_metrics(
            _column_rmse(y_hold_lo, y_hold_hi), _delivered_columns(m_hold))
        truth_gap = _mask_metrics(
            _column_mean_abs_gap(y_cand_lo, y_cand_hi),
            _delivered_columns(m_cand_hi))

        points: List[Dict[str, Any]] = []
        for budget in budgets:
            if verbose:
                print(f"[mf-study] budget {budget} CFD units ...")
            pt = self._run_budget(
                budget, x_cand, y_cand_lo, y_cand_hi, x_hold, y_hold_hi,
                y_hold_lo,
                warm_epochs=warm_epochs, warm_lr=warm_lr,
                finetune_epochs=finetune_epochs, finetune_lr=finetune_lr,
                base_dir=base_dir, verbose=verbose,
                m_cand_hi=m_cand_hi, m_hold=m_hold)
            points.append(pt)

        if workdir is None:
            shutil.rmtree(base_dir, ignore_errors=True)

        return StudyResult(
            seed=self.seed, n_designs=n_cand, n_holdout=n_hold,
            budgets=budgets, cfd_units=list(budgets),
            vlm_baseline=vlm_baseline, truth_gap=truth_gap, points=points,
            truth=truth)

    # ------------------------------------------------------------------
    def _run_budget(self, budget: int, x_cand: np.ndarray,
                    y_cand_lo: np.ndarray, y_cand_hi: np.ndarray,
                    x_hold: np.ndarray, y_hold_hi: np.ndarray,
                    y_hold_lo: np.ndarray,
                    warm_epochs: int, warm_lr: float,
                    finetune_epochs: int, finetune_lr: float,
                    base_dir: str, verbose: bool,
                    m_cand_hi: Optional[np.ndarray] = None,
                    m_hold: Optional[np.ndarray] = None) -> Dict[str, Any]:
        run_dir = os.path.join(base_dir, f"budget_{budget}")
        os.makedirs(run_dir, exist_ok=True)
        lake_path = os.path.join(run_dir, "lake.sqlite")
        ckpt_path = os.path.join(run_dir, "model.pt")

        lake = AeroDataLake(lake_path)
        try:
            # 1. cheap labels first: the whole candidate pool at VLM fidelity
            for x, y in zip(x_cand, y_cand_lo):
                lake.append(x, y, source="vlm", y_vlm=y, accepted=True)
            # 2. unseen holdout: high-fidelity labels only, never trained on
            #    (paired cheap label stored so the flywheel-vs-cheap closure
            #    KPI has an honest baseline on the same rows)
            for i, (x, y, ylo) in enumerate(zip(x_hold, y_hold_hi, y_hold_lo)):
                mask = ([int(v) for v in m_hold[i]]
                        if m_hold is not None else None)
                lake.append(x, y, source="truth_holdout", accepted=True,
                            y_vlm=ylo, y_cfd=y, in_holdout=True, mask=mask)

            # 3. warm-start the surrogate on the cheap labels with the SAME loss the
            #    ContinualTrainer fine-tune will use: inverse-variance
            #    weighted MSE (scale-free across the 9 targets, so drag gets
            #    as much attention as fuel volume) plus the identical
            #    physics terms. A fixed optimizer-step budget keeps small
            #    and large pools comparable.
            weights = self._label_weights(y_cand_lo)
            self._warm_start(x_cand, y_cand_lo, weights, ckpt_path,
                             warm_steps=self.WARM_STEP_TARGET, lr=warm_lr)

            # 4. expensive labels for the first `budget` candidates only
            lake.set_meta("last_processed_id", lake.last_id())
            for i in range(min(budget, len(x_cand))):
                mask = ([int(v) for v in m_cand_hi[i]]
                        if m_cand_hi is not None else [1] * len(Y_NAMES))
                lake.append(x_cand[i], y_cand_hi[i], source="cfd_study",
                            mask=mask,
                            accepted=True, y_vlm=y_cand_lo[i],
                            y_cfd=y_cand_hi[i])

            # 5. the real production loop: continual fine-tune + promotion gate.
            #    Again epochs are a floor; the step target keeps the
            #    refinement comparable whether few or many labels arrived.
            replay_rows = min(512, len(x_cand) - min(budget, len(x_cand)))
            fine_steps_per_epoch = max(
                int(np.ceil((min(budget, len(x_cand)) + replay_rows)
                            / self.WARM_BATCH)), 1)
            fine_epochs_eff = max(int(finetune_epochs), int(np.ceil(
                self.FINE_STEP_TARGET / fine_steps_per_epoch)))
            fin = ContinualTrainer(lake, checkpoint_path=ckpt_path,
                                   device="cpu", seed=self.seed,
                                   label_weights=weights)
            summary = {"updated": False}
            if budget > 0:
                summary = fin.update(epochs=fine_epochs_eff,
                                     lr=finetune_lr, min_new_samples=budget,
                                     verbose=False)

            masks = (np.asarray(m_hold, dtype=float)
                     if m_hold is not None else np.ones_like(y_hold_hi))
            hold_mse = fin._eval_mse(x_hold, y_hold_hi, masks)
            pred = fin.model(torch.tensor(x_hold, dtype=torch.float32))
            rmse = _column_rmse(pred.detach().numpy(), y_hold_hi)

            # 6. the cold trajectory: same label budget, same loss objective,
            #    but NO cheap-model warm-start. Only meaningful for budget > 0
            #    (at budget 0 it would train on nothing).
            rmse_cold = {name: None for name in Y_NAMES}
            hold_mse_cold = None
            if budget > 0:
                # cold step budget scales with budget so every hi label gets
                # ~80 epochs of exposure (the warm model got ~120 epochs over
                # its cheap pool); floor keeps tiny budgets from under-fitting.
                labels_in = min(budget, len(x_cand))
                steps_per_epoch_cold = max(
                    int(np.ceil(labels_in / self.WARM_BATCH)), 1)
                cold_steps = min(4000, max(
                    self.COLD_STEP_TARGET,
                    int(80 * steps_per_epoch_cold)))
                weights_hi = self._label_weights(y_cand_hi)
                cold_ckpt = os.path.join(run_dir, "cold.pt")
                self._warm_start(x_cand[:labels_in], y_cand_hi[:labels_in],
                                 weights_hi, cold_ckpt,
                                 warm_steps=cold_steps,
                                 lr=warm_lr)
                cold = ContinualTrainer(lake,
                                        checkpoint_path=cold_ckpt,
                                        device="cpu", seed=self.seed,
                                        label_weights=weights_hi)
                cold_hold_mse = cold._eval_mse(x_hold, y_hold_hi, masks)
                with torch.no_grad():
                    cold_pred = cold.model(
                        torch.tensor(x_hold, dtype=torch.float32))
                rmse_cold = _column_rmse(cold_pred.numpy(), y_hold_hi)
                hold_mse_cold = float(cold_hold_mse)
                cold.lake = None  # shared with the outer lake; no double close

            # undelivered truth columns must not report fabricated errors
            col_ok = _delivered_columns(m_hold)
            rmse = _mask_metrics(rmse, col_ok)
            rmse_cold = _mask_metrics(rmse_cold, col_ok)
        finally:
            lake.close()

        pt = {
            "cfd_units": budget,
            "updated": bool(summary.get("updated", False)),
            "promoted": bool(summary.get("promoted", False)),
            "holdout_mse": float(hold_mse),
            "holdout_mse_cold": hold_mse_cold,
            "parameters": int(fin.model.n_params()),
            "rmse": rmse,
            "rmse_cold": rmse_cold,
        }
        return pt

    # ------------------------------------------------------------------
    def _label_weights(self, y: np.ndarray) -> np.ndarray:
        """Inverse-variance per-column loss weights, normalized to O(1).

        Columns that are (effectively) constant in the label set carry no
        information and are excluded from the loss (weight 0) — otherwise a
        zero-variance column (e.g. profile drag in the low-fidelity labels,
        which is 0 by construction) would get an astronomically large
        inverse-variance weight and destroy training. The surviving weights
        are normalized so every active column contributes comparably."""
        var = np.var(y, axis=0)
        floor = 1e-12 * max(float(np.max(var)), 1e-12)
        w = np.where(var > floor, 1.0 / np.maximum(var, floor), 0.0)
        nnz = int(np.count_nonzero(w))
        if nnz:
            w = w / (np.sum(w) / nnz)
        else:
            w = np.ones_like(w)
        return w

    def _warm_start(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                    ckpt_path: str, warm_steps: int, lr: float):
        """Warm-start AeroSurrogate3D on the cheap labels.

        The input and target normalizers are fitted on this corpus and saved
        with the checkpoint (input z-score before the Fourier embedding;
        targets trained in standardized space, so every labeled column
        contributes comparably — this fixes the CL-variance collapse caused
        by raw-scale losses dominated by the fuel-volume column). The loss
        mirrors the ContinualTrainer update objective: standardized weighted
        MSE + the same trefftz/drag-sum physics terms at 0.25, evaluated on
        the denormalized predictions.
        """
        from .models.surrogate_3d import AeroSurrogate3D
        model = AeroSurrogate3D()
        xm = x.mean(axis=0); xs = x.std(axis=0)
        xs[xs < 1e-9 * max(float(xs.max()), 1e-9)] = 1.0
        ym = y.mean(axis=0); ys = y.std(axis=0)
        ys[ys < 1e-9 * max(float(ys.max()), 1e-9)] = 1.0
        model.x_stdz.set(xm, xs)
        model.y_stdz.set(ym, ys)
        zy = (y - ym) / ys
        w = np.var(zy, axis=0)
        w = np.where(w > 1e-12, 1.0 / np.maximum(w, 1e-12), 0.0)
        w = w / (w.sum() / max(int(np.count_nonzero(w)), 1))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        w_t = torch.tensor(w, dtype=torch.float32)
        stats = (ym, ys)
        xt = torch.tensor(x, dtype=torch.float32)
        zt = torch.tensor(zy, dtype=torch.float32)
        n = x.shape[0]
        b = min(self.WARM_BATCH, n)
        model.train()
        perm = torch.randperm(n)
        ptr = 0
        for _ in range(max(int(warm_steps), 1)):
            if ptr + b > n:
                perm = torch.randperm(n)
                ptr = 0
            idx = perm[ptr:ptr + b]
            ptr += b
            opt.zero_grad()
            pred = model.forward_raw(xt[idx])
            mse = ((pred - zt[idx]) ** 2 * w_t[None, :]).mean()
            loss_trefftz, loss_drag_sum = _physics_terms(pred, xt[idx], stats)
            loss = mse + 0.25 * loss_trefftz + 0.25 * loss_drag_sum
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        torch.save({"surrogate_state": model.state_dict()}, ckpt_path)