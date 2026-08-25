"""
Continual / Lifelong Learning Engine for AeroWing AI Pro.

Builds the closed-loop "the tool keeps improving" concept on a persistent
foundation. Inspired by airfoil_ai's novel approach (Error-Directed Gradient
Routing + function-preserving capacity expansion + closed-loop refinement),
which existed only as an in-memory batch demo; this module makes the same
ideas durable: every CFD run a team performs anyway becomes a training label.

Pipeline:
    CFD run -> quality gate -> data lake (append-only, SQLite)
            -> diversity sampler (exploration tax)
            -> incremental fine-tune from last checkpoint (replay buffer)
            -> holdout validation gate (no self-confirming bias)
            -> promote checkpoint only if no regression
            -> automatic capacity growth when validation plateaus

The four feedback controllers:
  1. Label quality gate  - only converged, physically-plausible runs enter.
  2. Exploration vs exploitation - a randomized design is injected every
     `interval` proposals so the dataset keeps covering the design space.
  3. Self-confirming bias guard - a fixed holdout set (never trained on)
     decides promotion; a checkpoint that regresses on it is refused.
  4. Incremental training - fine-tune from the last checkpoint with a small
     learning rate plus a replay buffer of older samples (no catastrophic
     forgetting, no retrain-from-scratch).

Growth (airfoil_ai connection):
  `GrowableSurrogate.add_capacity()` widens the network with a
  zero-initialized residual expansion block so the old function is preserved
  exactly (G'(x) == G(x) at init). After growth, fine-tuning applies
  error-directed gradient weights (Formulation C of airfoil_ai/error_routing.py):

      w_i = sigmoid(|e_old|_i / tau) * (1 + relu(|e_old|_i - |e_new|_i))

  so already-solved regions receive ~no gradient, regions where the new model
  improved get reinforced, and degraded regions are stabilized (no bonus).
  Checkpoints stay compatible with the web server (only the base surrogate
  state is saved under "surrogate_state").
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .geometry.wing_3d import Wing3D
from .geometry.cst_3d import CSTAirfoil3D
from .models.surrogate_3d import AeroSurrogate3D

# ---------------------------------------------------------------------------
# label schema
# ---------------------------------------------------------------------------

_X_DIM = 40   # 37 design params + [alpha_deg, mach, log10(Re)]
_Y_DIM = 9    # [CL, CD, CDi, CDp, CDw, CM, L/D, e, fuel_volume]
_Y_NAMES = [
    "cl", "cd", "cd_induced", "cd_profile", "cd_wave",
    "cm", "l_over_d", "span_efficiency", "fuel_volume_m3",
]
_Y_ALL_MASK = [1] * _Y_DIM

# plausible physical envelopes for coefficients (quality gate)
_CL_MIN, _CL_MAX = -0.6, 2.5
_CD_MAX = 0.35
_CM_ABS_MAX = 2.5


# ---------------------------------------------------------------------------
# random exploration design (same distribution as dataset_3d)
# ---------------------------------------------------------------------------

def random_exploration_design(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draws a random 37-D design vector (planform + root/tip CST) so that
    exploration covers the same space the original dataset was sampled from."""
    rng = rng if rng is not None else np.random.default_rng()
    span = float(rng.uniform(15.0, 55.0))
    ar = float(rng.uniform(6.0, 12.0))
    taper = float(rng.uniform(0.22, 0.75))
    sweep = float(rng.uniform(5.0, 35.0))
    dihedral = float(rng.uniform(1.0, 5.0))
    twist_root = float(rng.uniform(0.0, 3.0))
    twist_tip = float(rng.uniform(-4.0, 0.0))

    naca_root = f"{rng.integers(0, 4)}{rng.integers(2, 5)}{rng.integers(12, 16):02d}"
    naca_tip = f"{rng.integers(0, 3)}{rng.integers(2, 5)}{rng.integers(8, 12):02d}"
    af_root = CSTAirfoil3D.from_naca4(naca_root, order=6)
    af_tip = CSTAirfoil3D.from_naca4(naca_tip, order=6)
    af_root.weights_upper += rng.normal(0.0, 0.01, size=af_root.weights_upper.shape)
    af_root.weights_lower += rng.normal(0.0, 0.01, size=af_root.weights_lower.shape)
    af_tip.weights_upper += rng.normal(0.0, 0.01, size=af_tip.weights_upper.shape)
    af_tip.weights_lower += rng.normal(0.0, 0.01, size=af_tip.weights_lower.shape)

    wing = Wing3D(
        span=span, aspect_ratio=ar, taper_ratio=taper, sweep_le_deg=sweep,
        dihedral_deg=dihedral, twist_root_deg=twist_root, twist_tip_deg=twist_tip,
        root_airfoil=af_root, tip_airfoil=af_tip,
        name="Exploration_Design",
    )
    return wing.to_parameter_vector()


def design_to_input(design_37: np.ndarray, alpha_deg: float, mach: float,
                    reynolds: float) -> np.ndarray:
    """Appends the flight condition to a 37-D design vector -> 40-D lake row."""
    flight = np.array([alpha_deg, mach, np.log10(max(reynolds, 1e4))], dtype=float)
    return np.concatenate([np.asarray(design_37, dtype=float), flight])


# ---------------------------------------------------------------------------
# SU2 artifact parsing (tolerant; anything exotic is left to the gate)
# ---------------------------------------------------------------------------

def parse_su2_forces(text: str) -> Dict[str, float]:
    """Extracts TOTAL CL / CD / CS / CMx / CMy / CMz (or Total Lift / Total Drag)
    from an SU2 forces breakdown file or stdout log."""
    out: Dict[str, float] = {}
    for key in ("cl", "cd", "cs", "cmx", "cmy", "cmz"):
        m = re.search(
            rf"Total\s+\b{key}\b\s*[:=]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            text, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))
    if "cl" not in out:
        m = re.search(
            r"Total\s+Lift\s*[:=]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text, re.I)
        if m:
            out["cl"] = float(m.group(1))
    if "cd" not in out:
        m = re.search(
            r"Total\s+Drag\s*[:=]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text, re.I)
        if m:
            out["cd"] = float(m.group(1))
    return out


def cm_guard(fly_y: np.ndarray, eng_y: np.ndarray,
             thr: float = 0.07) -> np.ndarray:
    """cm-side production fallback (documented guard, 25-run experiment).

    The flywheel adds zero on strong-pitching (deep-negative-CMz) cases —
    the calibrated engine's cm tracks SU2 CMz there and the learner gains
    nothing beyond it. Whenever either source predicts |cm| > thr, the
    guarded output keeps the calibrated engine's cm just as the trusted
    prior; the CL/CD columns pass through unchanged.
    """
    out = np.array(fly_y, dtype=float, copy=True)
    if out.ndim != 2 or out.shape[1] < 6 or eng_y.shape != out.shape:
        return out
    eng_cm = np.asarray(eng_y, dtype=float)[:, 5]
    fly_cm = out[:, 5]
    strong = (np.abs(eng_cm) > thr) | (np.abs(fly_cm) > thr)
    out[strong, 5] = eng_cm[strong]
    return out


def parse_su2_residuals(text: str) -> List[float]:
    """Extracts the first RMS_* residual column of an SU2 convergence table.

    SU2 prints tables like:  |  Time_Iter | Outer_Iter | Inner_Iter | RMS_PRESSURE ...
    We locate the header line containing 'RMS_', take the first RMS column,
    then read its values from the following numeric rows."""
    lines = text.splitlines()
    rms_cols: List[str] = []
    header_idx = -1
    for i, ln in enumerate(lines):
        cols = re.findall(r"RMS_[A-Za-z0-9_]+", ln)
        if cols:
            rms_cols = cols
            header_idx = i
            break
    if not rms_cols or header_idx < 0:
        return []
    head_parts = [p.strip() for p in lines[header_idx].split("|")]
    target = head_parts.index(rms_cols[0]) if rms_cols[0] in head_parts else 1
    vals: List[float] = []
    for ln in lines[header_idx + 1:]:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) <= target:
            continue
        val = parts[target].replace(" ", "").replace(",", "")
        if re.fullmatch(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", val):
            vals.append(float(val))
    return vals


# ---------------------------------------------------------------------------
# label quality gate (controller #1)
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    accepted: bool
    reasons: List[str] = field(default_factory=list)


class CfdQualityGate:
    """Decides whether a CFD run is trustworthy enough to become a label.

    Three families of checks:
      * convergence  - residual history present, decreasing, final below tol
      * plausibility - coefficients within physical envelopes (finite, sane)
      * effort       - the run actually iterated enough to be meaningful

    Anything failing a hard check is rejected with its reasons recorded, so
    the data lake never silently trains on garbage (the #1 killer of real-world
    feedback loops).
    """

    def __init__(self, residual_tol: float = 1e-5, min_iterations: int = 100,
                 min_residual_points: int = 20):
        self.residual_tol = float(residual_tol)
        self.min_iterations = int(min_iterations)
        self.min_residual_points = int(min_residual_points)

    def _check_forces(self, forces: Dict[str, float]) -> List[str]:
        reasons = []
        cl = forces.get("cl")
        cd = forces.get("cd")
        cmz = forces.get("cmz")
        if cl is None or cd is None:
            reasons.append("missing CL or CD in forces")
            return reasons
        if not (math.isfinite(cl) and math.isfinite(cd)):
            reasons.append("non-finite CL/CD")
            return reasons
        if not (_CL_MIN <= cl <= _CL_MAX):
            reasons.append(f"CL={cl:.4f} outside [{_CL_MIN}, {_CL_MAX}]")
        if not (0.0 < cd <= _CD_MAX):
            reasons.append(f"CD={cd:.4f} outside (0, {_CD_MAX}]")
        if cmz is not None and math.isfinite(cmz) and abs(cmz) > _CM_ABS_MAX:
            reasons.append(f"|CMz|={abs(cmz):.4f} > {_CM_ABS_MAX}")
        return reasons

    def _check_convergence(self, residuals: Sequence[float]) -> List[str]:
        reasons = []
        if len(residuals) < self.min_residual_points:
            reasons.append(f"only {len(residuals)} residual points (< {self.min_residual_points})")
            return reasons
        if not (math.isfinite(residuals[0]) and math.isfinite(residuals[-1])):
            reasons.append("non-finite residuals")
            return reasons
        if residuals[-1] >= residuals[0]:
            reasons.append(f"residuals not decreasing ({residuals[-1]:.3e} >= {residuals[0]:.3e})")
        if residuals[-1] >= self.residual_tol:
            reasons.append(f"final residual {residuals[-1]:.3e} >= tol {self.residual_tol:.1e}")
        return reasons

    def gate(self, forces: Dict[str, float],
             residuals: Optional[Sequence[float]] = None,
             iterations: Optional[int] = None) -> GateResult:
        reasons = self._check_forces(forces)
        if residuals is not None:
            reasons += self._check_convergence(list(residuals))
        if iterations is not None and iterations < self.min_iterations:
            reasons.append(f"only {iterations} iterations (< {self.min_iterations})")
        if reasons:
            return GateResult(accepted=False, reasons=reasons)
        return GateResult(accepted=True)

    def gate_su2_text(self, text: str) -> GateResult:
        residuals = parse_su2_residuals(text)
        return self.gate(parse_su2_forces(text), residuals=residuals if residuals else None)


# ---------------------------------------------------------------------------
# persistence: append-only data lake (SQLite, stdlib only)
# ---------------------------------------------------------------------------

class AeroDataLake:
    """Append-only store of (input, label, provenance, quality) records.

    Schema
    ------
      samples(id, ts, x JSON[40], y JSON[9], mask JSON[9], y_vlm JSON|NULL,
              y_cfd JSON|NULL, source, accepted, gate_reason,
              is_exploration, in_holdout)
      meta(key, value)                 -- counters / last-processed id
      history(id, ts, kind, payload)   -- audit trail of every action

    Holdout rows are chosen once and never trained on; promotion decisions are
    made exclusively against them (controller #3).
    """

    _CONN: sqlite3.Connection

    def __init__(self, path: str):
        self.path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS samples (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts TEXT NOT NULL,
                 x TEXT NOT NULL,
                 y TEXT NOT NULL,
                 mask TEXT NOT NULL,
                 y_vlm TEXT,
                 y_cfd TEXT,
                 source TEXT NOT NULL,
                 accepted INTEGER NOT NULL DEFAULT 0,
                 gate_reason TEXT DEFAULT '',
                 is_exploration INTEGER NOT NULL DEFAULT 0,
                 in_holdout INTEGER NOT NULL DEFAULT 0
               )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS meta (
                 key TEXT PRIMARY KEY, value TEXT NOT NULL
               )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS history (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT
               )"""
        )
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore

    # -- meta --------------------------------------------------------------
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def set_meta(self, key: str, value: Any):
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        self._conn.commit()

    def log(self, kind: str, payload: Any = None):
        self._conn.execute(
            "INSERT INTO history(ts, kind, payload) VALUES(?, ?, ?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), kind,
             json.dumps(payload) if payload is not None else None))
        self._conn.commit()

    def history_vals(self, kind: str, key: str) -> List[float]:
        rows = self._conn.execute(
            "SELECT payload FROM history WHERE kind=? ORDER BY id", (kind,)).fetchall()
        out = []
        for r in rows:
            if not r[0]:
                continue
            try:
                p = json.loads(r[0])
            except json.JSONDecodeError:
                continue
            v = p.get(key)
            if v is not None:
                out.append(float(v))
        return out

    # -- append ------------------------------------------------------------
    def append(self, x: Sequence[float], y: Sequence[float], source: str,
               mask: Optional[Sequence[int]] = None, accepted: bool = True,
               gate_reason: str = "", y_vlm: Optional[Sequence[float]] = None,
               y_cfd: Optional[Sequence[float]] = None,
               is_exploration: bool = False, in_holdout: bool = False) -> int:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size != _X_DIM or y.size != _Y_DIM:
            raise ValueError(f"row must be {_X_DIM}D input / {_Y_DIM}D label, "
                             f"got {x.size} / {y.size}")
        if mask is not None:
            mask = list(mask)
            if len(mask) != _Y_DIM:
                raise ValueError(f"mask must be {_Y_DIM}-D, got {len(mask)}")
        elif "cfd" in source.lower():
            raise ValueError(
                "CFD-source rows must declare an explicit 9-D mask naming the "
                "outputs the solver actually delivered; defaulting to all-ones "
                "would train undelivered columns toward their zero placeholders")
        else:
            mask = list(_Y_ALL_MASK)
        cur = self._conn.execute(
            "INSERT INTO samples(ts, x, y, mask, y_vlm, y_cfd, source, accepted, "
            "gate_reason, is_exploration, in_holdout) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(x.tolist()),
             json.dumps(y.tolist()), json.dumps(mask),
             json.dumps(list(y_vlm)) if y_vlm is not None else None,
             json.dumps(list(y_cfd)) if y_cfd is not None else None,
             source, 1 if accepted else 0, gate_reason,
             1 if is_exploration else 0, 1 if in_holdout else 0))
        self._conn.commit()
        return int(cur.lastrowid)

    # -- reads -------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(accepted), SUM(is_exploration), SUM(in_holdout) "
            "FROM samples").fetchone()
        return {"samples": int(row[0] or 0), "accepted": int(row[1] or 0),
                "exploration": int(row[2] or 0), "holdout": int(row[3] or 0)}

    def _fetch(self, where: str, params: Tuple = ()) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        rows = self._conn.execute(
            f"SELECT id, x, y, mask FROM samples WHERE {where} ORDER BY id",
            params).fetchall()
        ids = [int(r[0]) for r in rows]
        if not rows:
            return (np.empty((0, _X_DIM)), np.empty((0, _Y_DIM)),
                    np.empty((0, _Y_DIM)), ids)
        x = np.array([json.loads(r[1]) for r in rows], dtype=float)
        y = np.array([json.loads(r[2]) for r in rows], dtype=float)
        mask = np.array([json.loads(r[3]) for r in rows], dtype=float)
        return x, y, mask, ids

    def train_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        return self._fetch("accepted=1 AND in_holdout=0")

    def holdout_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        return self._fetch("accepted=1 AND in_holdout=1")

    def holdout_cheap_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray, List[int]]:
        """Holdout rows that also carry a paired cheap label (y_vlm) - the
        subset on which the flywheel-vs-cheap closure KPI is computed."""
        rows = self._conn.execute(
            "SELECT id, x, y, mask, y_vlm FROM samples "
            "WHERE accepted=1 AND in_holdout=1 AND y_vlm IS NOT NULL ORDER BY id"
        ).fetchall()
        ids = [int(r[0]) for r in rows]
        if not rows:
            return (np.empty((0, _X_DIM)), np.empty((0, _Y_DIM)),
                    np.empty((0, _Y_DIM)), np.empty((0, _Y_DIM)), ids)
        x = np.array([json.loads(r[1]) for r in rows], dtype=float)
        y = np.array([json.loads(r[2]) for r in rows], dtype=float)
        mask = np.array([json.loads(r[3]) for r in rows], dtype=float)
        yv = np.array([json.loads(r[4]) for r in rows], dtype=float)
        return x, y, mask, yv, ids

    def newer_than(self, last_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        return self._fetch("accepted=1 AND in_holdout=0 AND id > ?", (int(last_id),))

    def cfd_rows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Accepted rows whose source marks real CFD truth ('cfd' in source).

        These are the only rows allowed to act as ground-truth labels in
        validation/study contexts: the engine stand-in must never masquerade
        as truth. Masks ride along so partial deliveries (e.g. SU2 forces:
        CL/CD/CMz only) stay honest in every consumer."""
        return self._fetch("accepted=1 AND LOWER(source) LIKE '%cfd%' "
                           "AND LOWER(source) NOT LIKE '%study%'")

    def ids_by_source(self, source: str) -> List[int]:
        rows = self._conn.execute(
            "SELECT id FROM samples WHERE source LIKE ? ORDER BY id",
            (f"%{source}%",)).fetchall()
        return [int(r[0]) for r in rows]

    def rows_by_ids(self, ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_list, y_list, m_list = [], [], []
        for i in ids:
            r = self._conn.execute(
                "SELECT x, y, mask FROM samples WHERE id=?", (int(i),)).fetchone()
            if r:
                x_list.append(json.loads(r[0]))
                y_list.append(json.loads(r[1]))
                m_list.append(json.loads(r[2]))
        x = np.array(x_list, dtype=float) if x_list else np.zeros((0, _X_DIM))
        y = np.array(y_list, dtype=float) if y_list else np.zeros((0, _Y_DIM))
        mask = np.array(m_list, dtype=float) if m_list else np.zeros((0, _Y_DIM))
        return x, y, mask

    def ensure_holdout(self, n: int = 15, seed: int = 42) -> int:
        """Flags up to `n` accepted non-holdout rows as holdout. Never unflags."""
        rows = self._conn.execute(
            "SELECT id FROM samples WHERE accepted=1 AND in_holdout=0 ORDER BY id",
        ).fetchall()
        if len(rows) <= n:
            return self.stats()["holdout"]
        rng = np.random.default_rng(seed)
        picked = rng.choice([r[0] for r in rows], size=n, replace=False)
        for sid in picked:
            self._conn.execute("UPDATE samples SET in_holdout=1 WHERE id=?", (int(sid),))
        self._conn.commit()
        return self.stats()["holdout"]

    def last_id(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM samples").fetchone()
        return int(row[0])


# ---------------------------------------------------------------------------
# controller #2: exploration vs exploitation
# ---------------------------------------------------------------------------

class DiversitySampler:
    """Injects a randomized design every `interval` proposals so the dataset
    keeps covering the design space instead of collapsing onto the region the
    optimizer favours (exploration tax). The counter persists in the lake so
    the cadence survives restarts."""

    def __init__(self, lake: AeroDataLake, interval: int = 10,
                 rng: Optional[np.random.Generator] = None):
        self.lake = lake
        self.interval = max(int(interval), 1)
        self.rng = rng if rng is not None else np.random.default_rng()

    def proposal(self, requested: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Returns (design_37, is_exploration). Every `interval`-th call the
        requested design is replaced by a random one, flagged for the lake."""
        counter = int(self.lake.get_meta("exploration_counter", 0))
        self.lake.set_meta("exploration_counter", counter + 1)
        if counter % self.interval == 0:
            return random_exploration_design(self.rng), True
        return np.asarray(requested, dtype=float), False


# ---------------------------------------------------------------------------
# airfoil_ai connection: Error-Directed Gradient Routing (Formulation C)
# ---------------------------------------------------------------------------

def error_directed_weights(err_old: np.ndarray, err_new: np.ndarray,
                           tau: float = 1e-3) -> np.ndarray:
    """Per-sample gradient weights for capacity-expansion fine-tuning.

    Formulation C of airfoil_ai/error_routing.py:
        w_i = sigmoid(|e_old|_i / tau) * (1 + relu(|e_old|_i - |e_new|_i))

    * solved regions (|e_old| ~ 0)        -> w ~ 0.5 (minimal attention)
    * improved regions (|e_new| < |e_old|) -> w > 1  (reinforced learning)
    * degraded regions (|e_new| > |e_old|) -> w ~ sigmoid(|e_old|/tau) (no bonus)
    """
    e_old = np.abs(np.asarray(err_old, dtype=float))
    e_new = np.abs(np.asarray(err_new, dtype=float))
    tau = max(float(tau), 1e-9)
    return (1.0 / (1.0 + np.exp(-e_old / tau))) * (1.0 + np.maximum(0.0, e_old - e_new))


# ---------------------------------------------------------------------------
# capacity growth (function-preserving, exact)
# ---------------------------------------------------------------------------

class GrowableSurrogate(nn.Module):
    """AeroSurrogate3D wrapper whose capacity can grow without disturbing the
    learned function. `add_capacity` attaches a zero-initialized residual
    expansion block on top of the base encoder, so at init it outputs exactly
    the base prediction (G'(x) == G(x)), matching the spirit of airfoil_ai's
    function-preserving widening while being exactly safe for any
    normalization used inside the base network."""

    def __init__(self, base: Optional[AeroSurrogate3D] = None):
        super().__init__()
        self.base = base if base is not None else AeroSurrogate3D()
        self.expander: Optional[nn.Module] = None

    def add_capacity(self, units: int = 128) -> int:
        """Appends Linear(hidden->units) + GELU + Linear(units->out) with the
        final layer zero-initialized -> exact function preservation (returns 1
        if capacity was added, 0 if already grown)."""
        if self.expander is not None:
            return 0
        h = self.base.hidden_dim
        o = self.base.head[2].out_features
        block = nn.Sequential(
            nn.Linear(h, units),
            nn.GELU(),
            nn.Linear(units, o),
        )
        with torch.no_grad():
            block[-1].weight.zero_()
            block[-1].bias.zero_()
        self.expander = block
        return 1

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base.fourier(self.base.x_stdz(x))
        h = self.base.input_proj(feat)
        for block in self.base.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        out = self.base.head(h)
        if self.expander is not None:
            out = out + self.expander(h)
        return self.base.y_stdz.inverse(out)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in standardized target space (for balanced losses)."""
        h = self.hidden(x)
        out = self.base.head(h)
        if self.expander is not None:
            out = out + self.expander(h)
        return out

    def normalize(self, y: np.ndarray) -> np.ndarray:
        """z-scores physical targets with the output normalizer stats."""
        s = self.base.y_stdz
        if not s.is_fitted():
            return np.asarray(y, dtype=float)
        return (np.asarray(y, dtype=float) - s.mean.numpy()) / (s.std.numpy() + 1e-9)

    # checkpoint compatibility: the web server loads "surrogate_state" into a
    # bare AeroSurrogate3D; growth weights ride along under their own key.
    def base_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.base.state_dict()

    def growth_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.expander is None:
            return None
        prefix = "expander."
        return {k[len(prefix):]: v for k, v in self.state_dict().items()
                if k.startswith(prefix)}

    def load_growth_state(self, state: Dict[str, torch.Tensor]):
        if state and self.expander is not None:
            self.expander.load_state_dict(state)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# controller #3 + #4: incremental fine-tune, replay, holdout promotion
# ---------------------------------------------------------------------------

def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor,
                weights: Optional[np.ndarray] = None) -> torch.Tensor:
    """Per-sample MSE restricted to labeled dimensions (B,).

    `weights` (optional, len == y-dim) rescales each target column before
    squaring; inverse-variance weights (1/var) are the standard way to make
    a multi-output loss scale-free when columns live on very different
    physical magnitudes (fuel volume vs drag coefficients)."""
    sq = (pred - target) ** 2
    sq = sq * mask
    if weights is not None:
        w = torch.as_tensor(np.asarray(weights, dtype=float),
                            dtype=sq.dtype, device=sq.device)
        sq = sq * w[None, :]
    return sq.sum(dim=1)


def _physics_terms(pred: torch.Tensor, xt: torch.Tensor,
                   stats: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trefftz + drag-decomposition consistency terms (mirrors the physics
    part of AeroSurrogate3D.compute_physics_loss).

    `pred` may be in standardized target space when `stats` = (y_mean,
    y_std) is passed; the terms are always evaluated in physical units."""
    if stats is not None:
        ym = torch.as_tensor(np.asarray(stats[0], dtype=float),
                             dtype=pred.dtype, device=pred.device)
        ys = torch.as_tensor(np.asarray(stats[1], dtype=float),
                             dtype=pred.dtype, device=pred.device)
        pred = pred * ys + ym
    cl = pred[:, 0]
    cd = pred[:, 1]
    cdi, cdp, cdw = pred[:, 2], pred[:, 3], pred[:, 4]
    e = torch.clamp(pred[:, 7], min=0.3, max=1.1)
    ar = torch.clamp(xt[:, 1], min=2.0, max=20.0)
    loss_trefftz = nn.functional.mse_loss(cdi, (cl ** 2) / (np.pi * ar * e + 1e-6))
    loss_drag_sum = nn.functional.mse_loss(cd, cdi + cdp + cdw)
    return loss_trefftz, loss_drag_sum


class ContinualTrainer:
    """Incremental learner with promotion gating.

    update(): pulls newly accepted samples from the lake, mixes them with a
    replay buffer of older rows, fine-tunes the surrogate from the last
    checkpoint (small LR), then promotes the new state only if the fixed
    holdout set did not regress. Nothing is ever trained on the holdout.

    With auto_grow, once the accepted CFD corpus has grown sufficiently while
    holdout improvement has plateaued, capacity is added; for one update the
    error-directed gradient weights (airfoi Formulation C) steer training so
    already-solved regions stay untouched.
    """

    def __init__(self, lake: AeroDataLake,
                 checkpoint_path: str = "checkpoints/aerowing_models.pt",
                 device: str = "cpu", seed: Optional[int] = None,
                 label_weights: Optional[Sequence[float]] = None):
        self.lake = lake
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._rng = np.random.default_rng(seed)
        self.label_weights = (None if label_weights is None
                              else np.asarray(label_weights, dtype=float))
        self.model = GrowableSurrogate().to(device)
        self._load_checkpoint()
        self.last_processed = int(self.lake.get_meta("last_processed_id", 0))

    # -- checkpoint machinery ------------------------------------------------
    def _load_checkpoint(self) -> bool:
        if not os.path.exists(self.checkpoint_path):
            return False
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.base.load_state_dict(ckpt["surrogate_state"], strict=False)
        growth = ckpt.get("surrogate_growth_state")
        if growth:
            units = int(next(iter(growth.values())).shape[0])
            self.model.add_capacity(units=units)
            self.model.load_growth_state(growth)
        return True

    def _save_checkpoint(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)) or ".",
                    exist_ok=True)
        torch.save({
            "surrogate_state": self.model.base_state_dict(),
            "surrogate_growth_state": self.model.growth_state_dict(),
            "generator_state": self._carry_generator_state(),
        }, self.checkpoint_path)

    def _carry_generator_state(self) -> Optional[Dict[str, torch.Tensor]]:
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            return torch.load(self.checkpoint_path, map_location=self.device).get("generator_state")
        except Exception:
            return None

    def _mse_weights(self) -> Optional[np.ndarray]:
        """Column loss weights.

        With an output normalizer fitted (standardized targets) the MSE is
        already balanced across columns, and the raw-scale inverse-variance
        weights (which were calibrated for physical units) would badly
        misweight the standardized loss — so they are dropped. Without a
        normalizer (legacy checkpoints) the caller-supplied weights apply.
        """
        if self.model.base.y_stdz.is_fitted():
            return None
        return self.label_weights

    def _eval_mse(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        if x.shape[0] == 0:
            return float("nan")
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            yt = torch.tensor(self.model.normalize(y), dtype=torch.float32,
                              device=self.device)
            mt = torch.tensor(mask, dtype=torch.float32, device=self.device)
            pred = self.model.forward_raw(xt)
            return float(_masked_mse(pred, yt, mt, self._mse_weights())
                         .mean().item())

    def _stats(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        s = self.model.base.y_stdz
        if not s.is_fitted():
            return None
        return s.mean.numpy(), s.std.numpy()

    def _ensure_stats(self, x: np.ndarray, y: np.ndarray):
        """Falls back to computing normalizer stats from the update batch when
        the checkpoint carries none (legacy checkpoints)."""
        if not self.model.base.x_stdz.is_fitted() and x.shape[0] >= 32:
            xs = x.std(axis=0); xs[xs < 1e-9 * max(float(xs.max()), 1e-9)] = 1.0
            self.model.base.x_stdz.set(x.mean(axis=0), xs)
        if not self.model.base.y_stdz.is_fitted() and y.shape[0] >= 32:
            ys = y.std(axis=0); ys[ys < 1e-9 * max(float(ys.max()), 1e-9)] = 1.0
            self.model.base.y_stdz.set(y.mean(axis=0), ys)

    # -- replay buffer (controller #4 piece) ---------------------------------
    def _replay_ids(self, exclude: set, size: int) -> List[int]:
        """Older accepted ids, biased toward recent rows so the model neither
        forgets the space nor the newest physics.

        As the lake fills, the prior FADES: (a) the recency split decays from
        50/50 toward 20/80 (the biased cheap pool must yield to accumulated
        truth), and (b) vlm-source rows in the backfill are sampled with a
        probability that decays as CFD-quality rows accumulate, so the
        flywheel is anchored by real labels once enough exist."""
        _, _, _, ids = self.lake.train_batch()
        ids = [i for i in ids if i not in exclude]
        if not ids:
            return []
        ids = np.asarray(ids)
        rng = self._rng
        n_total = int(len(ids) + len(exclude))
        fade = 0.5 if n_total < 300 else max(0.2, 0.5 - 0.3 * (n_total - 300) / 1700.0)
        n_cfd_total = len(self.lake.ids_by_source("cfd"))
        keep_vlm = max(0.25, 1.0 - n_cfd_total / 40.0)
        cfd_ids = set(self.lake.ids_by_source("cfd"))
        if len(ids) <= size:
            # Lake fits inside the replay budget: still down-sample VLM rows
            # (the prior must fade as truth accumulates) while keeping every
            # CFD-quality row; else early-returning the pool wholesale would
            # anchor the update to the biased cheap labels (money-slide
            # finding: flywheel stuck at pool RMSE). The pool is also capped
            # at 2x the number of incoming new samples so the fine-tune is
            # actually steered by fresh truth (cm, the engine's weakest
            # output, is otherwise pinned to the engine prior).
            rest_cfd = [int(i) for i in ids if i in cfd_ids]
            pool_max = max(2 * len(exclude), 8)
            try:
                pool_max = max(int(os.environ.get("AERO_POOL_CAP", pool_max)), 0)
            except ValueError:
                pass
            rest_vlm = [int(i) for i in ids if i not in cfd_ids]
            if len(rest_vlm) > pool_max:
                keep = rng.choice(len(rest_vlm), size=pool_max, replace=False)
                rest_vlm = [rest_vlm[k] for k in sorted(keep)]
            rest_vlm = [int(i) for i in rest_vlm if rng.random() < keep_vlm]
            return (rest_cfd + rest_vlm)[:size]
        recent = int(fade * size)
        tail = ids[-recent:].tolist()
        rest = [int(i) for i in ids[:-recent].tolist()]
        n_rest = size - len(tail)
        rest_cfd = [i for i in rest if i in cfd_ids]
        rest_vlm = [i for i in rest if i not in cfd_ids]
        chosen = list(tail)
        if rest_cfd:
            n_cfd = min(len(rest_cfd), int(0.7 * n_rest))
            chosen += [int(i) for i in rng.choice(rest_cfd, size=n_cfd,
                                                  replace=False)]
        room = n_rest - (len(chosen) - len(tail))
        if rest_vlm and room > 0:
            pool_vlm = [i for i in rest_vlm if rng.random() < keep_vlm]
            n_vlm = min(room, len(pool_vlm))
            chosen += [int(i) for i in rng.choice(pool_vlm, size=n_vlm,
                                                  replace=False)]
        if len(chosen) < size:
            extra = [i for i in rest_cfd + rest_vlm if i not in chosen]
            chosen += extra[:size - len(chosen)]
        return chosen[:size]

    # -- the update loop ------------------------------------------------------
    def update(self, epochs: int = 6, lr: float = 1e-4, batch_size: int = 32,
               replay_size: int = 512, min_new_samples: int = 8,
               promote_tol: float = 0.02, auto_grow: bool = False,
               growth_min_new: int = 800, growth_plateau_updates: int = 4,
               growth_min_rel_gain: float = 5e-3, tau: float = 1e-3,
               new_sample_weight: float = 1.0, train_noise: float = 0.0,
               error_routing: bool = False,
               verbose: bool = True) -> Dict[str, Any]:
        x_new, y_new, m_new, ids_new = self.lake.newer_than(self.last_processed)
        if x_new.shape[0] < min_new_samples:
            return {"updated": False,
                    "reason": f"only {x_new.shape[0]} new accepted samples "
                              f"(need {min_new_samples})"}

        x_hold, y_hold, m_hold, _ = self.lake.holdout_batch()
        if x_hold.shape[0] < 3:
            self.lake.ensure_holdout()
            x_hold, y_hold, m_hold, _ = self.lake.holdout_batch()

        # -- optional capacity growth (plateau trigger) ----------------------
        grew = False
        if auto_grow and self.model.expander is None:
            n_cfd = len(self.lake.ids_by_source("cfd"))
            hist = self.lake.history_vals("promote", "holdout_mse")
            if n_cfd >= growth_min_new and len(hist) >= growth_plateau_updates:
                recent = hist[-growth_plateau_updates:]
                # improvement rate across the window edge: flat OR regressing
                # histories stay below the threshold; a still-descending
                # series has large positive gain and must NOT grow (the
                # previous min()-based form read zero gain off any monotone
                # decline and triggered mid-improvement).
                gain = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-12)
                if gain < growth_min_rel_gain:
                    self.model.add_capacity()
                    grew = True
                    hist_this = {"n_cfd": n_cfd, "holdout_tail": recent}
                    self.lake.log("grow", hist_this)
                    if verbose:
                        print(f"[continual] capacity grown: {self.model.n_params()} params")

        # -- replay + new data ------------------------------------------------
        replay = self._replay_ids(set(ids_new), replay_size)
        if replay:
            x_r, y_r, m_r = self.lake.rows_by_ids(replay)
            x = np.concatenate([x_new, x_r])
            y = np.concatenate([y_new, y_r])
            msk = np.concatenate([m_new, m_r])
        else:
            x, y, msk = x_new, y_new, m_new

        if grew and error_routing:
            # Error-directed routing (Formulation C): OFF by default.
            # Controlled ablation (experiments/growth_ablation.py) found it
            # underperforms plain fine-tuning under both shifted and mixed
            # exposure - suppressing gradients on already-solved samples
            # removes the replay anchoring that retention depends on. Kept
            # as a research path; re-enable only with new evidence.
            err_old = self._per_sample_errs(x, y, msk)

        # -- incremental fine-tune from last checkpoint (controller #4) --------
        self._ensure_stats(x, y)
        stats = self._stats()
        y_all = self.model.normalize(y)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        n = x.shape[0]
        perm = self._rng.permutation(n)
        train_loss = 0.0
        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xt = torch.tensor(x[idx], dtype=torch.float32, device=self.device)
                yt = torch.tensor(y_all[idx], dtype=torch.float32, device=self.device)
                mt = torch.tensor(msk[idx], dtype=torch.float32, device=self.device)
                if train_noise > 0.0:
                    with torch.no_grad():
                        xt = xt + torch.randn_like(xt) * train_noise
                optimizer.zero_grad()
                pred = self.model.forward_raw(xt)
                # physics regularization (mirrors AeroSurrogate3D.compute_physics_loss)
                loss_trefftz, loss_drag_sum = _physics_terms(pred, xt, stats)
                mse_part = _masked_mse(pred, yt, mt, self._mse_weights())
                if new_sample_weight != 1.0:
                    w_row = torch.ones(len(idx), dtype=torch.float32,
                                       device=self.device)
                    n_new = x_new.shape[0]
                    for r, i in enumerate(idx):
                        if i < n_new:
                            w_row[r] = float(new_sample_weight)
                    mse_part = mse_part * w_row
                if grew and error_routing:
                    # Formulation C: error-directed per-sample weighting
                    # (disabled by default - see update() docstring note)
                    err_new = mse_part.detach().cpu().numpy()
                    w = error_directed_weights(err_old[idx], err_new, tau=tau)
                    w_t = torch.tensor(w, dtype=torch.float32, device=self.device)
                    mse_part = mse_part * w_t
                loss = mse_part.mean() + 0.25 * loss_trefftz + 0.25 * loss_drag_sum
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            train_loss += epoch_loss / n
        train_loss = train_loss / int(epochs)

        # -- holdout promotion gate (controller #3) ----------------------------
        self.model.eval()
        new_mse = self._eval_mse(x_hold, y_hold, m_hold)
        baseline = float(self.lake.get_meta("holdout_mse_baseline", float("nan")))
        if math.isnan(baseline) or baseline <= 0:
            promoted = True
        else:
            promoted = (new_mse - baseline) / baseline <= promote_tol

        if promoted:
            self._save_checkpoint()
            self.lake.set_meta("holdout_mse_baseline", new_mse)
            self.lake.log("promote", {"holdout_mse": new_mse, "params": self.model.n_params()})
        else:
            self.lake.log("promotion_refused",
                          {"new_holdout_mse": new_mse, "baseline": baseline,
                           "regress": (new_mse - baseline) / baseline})

        self.last_processed = int(self.lake.last_id())
        self.lake.set_meta("last_processed_id", self.last_processed)

        # -- cheap-model closure KPI: flywheel vs cheap on the same holdout rows
        #    gap closure % = 100 * (1 - fly_rmse / cheap_rmse). The KPI the
        #    loop exists for: how much closer to the expensive truth the
        #    flywheel is than the cheap model alone (>= 30% = doing its job,
        #    <= 0 = fall back to cheap). Reporting only - never blocks update.
        fly_hold_rmse, cheap_hold_rmse, closure = None, None, None
        try:
            xc, yc, _, yv, _ = self.lake.holdout_cheap_batch()
            if xc.shape[0] >= 3:
                self.model.eval()
                with torch.no_grad():
                    pf = self.model(torch.tensor(xc, dtype=torch.float32,
                                                 device=self.device)).cpu().numpy()
                fly_hold_rmse, cheap_hold_rmse, closure = {}, {}, {}
                for ci, name in ((0, "cl"), (1, "cd"), (5, "cm")):
                    fr = float(np.sqrt(np.mean((pf[:, ci] - yc[:, ci]) ** 2)))
                    cr = float(np.sqrt(np.mean((yv[:, ci] - yc[:, ci]) ** 2)))
                    fly_hold_rmse[name] = fr
                    cheap_hold_rmse[name] = cr
                    closure[name] = (round(100.0 * (1.0 - fr / cr), 1)
                                     if cr > 1e-9 else None)
        except Exception as exc:  # reporting metric; never break the loop
            closure = {"error": str(exc)}

        summary = {"updated": True, "new_samples": len(ids_new),
                   "replay_samples": len(replay), "train_loss": float(train_loss),
                   "holdout_mse": new_mse, "holdout_baseline": baseline,
                   "fly_holdout_rmse": fly_hold_rmse,
                   "cheap_holdout_rmse": cheap_hold_rmse,
                   "closure_pct": closure,
                   "promoted": promoted, "grew": grew, "epochs": int(epochs),
                   "parameters": self.model.n_params()}
        if verbose:
            print("[continual] " + json.dumps(summary, indent=2))
        return summary

    def _per_sample_errs(self, x: np.ndarray, y: np.ndarray, msk: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            yt = torch.tensor(self.model.normalize(y), dtype=torch.float32,
                              device=self.device)
            mt = torch.tensor(msk, dtype=torch.float32, device=self.device)
            return _masked_mse(self.model.forward_raw(xt), yt, mt,
                               self._mse_weights()).cpu().numpy()