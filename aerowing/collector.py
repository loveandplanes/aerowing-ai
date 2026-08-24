"""
Batch data acquisition for the continual learning flywheel.

SU2BatchCollector sweeps a directory tree of CFD run outputs (one run per
subdirectory, following the convention emitted by `aerowing export su2` +
`aerowing continual ingest`), quality-gates every run, and appends each
accepted result to the AeroDataLake — zero manual steps.

Per-run convention (all optional except the forces file):
    forces file       forces_breakdown.dat | *forces*.dat|txt | *.log | *.out
                      (anything containing "Total CL" / "Total Lift")
    design spec       design.json | case.json | wing.json  (schema of
                      `aerowing continual ingest --design-json`), or a global
                      --design-json default for runs without their own
    convergence       history*.txt | *convergence* | residuals*  (RMS table;
                      otherwise parsed from the forces file when present)

Idempotency: every ingested forces file path is recorded in the lake meta
(`collected_files`); a re-run of the sweep only processes new files, so the
collector can be scheduled (cron / CI) safely.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry.wing_3d import Wing3D
from .continual import (
    AeroDataLake,
    CfdQualityGate,
    design_to_input,
    parse_su2_forces,
    parse_su2_residuals,
)

DESIGN_PATTERNS = ("design.json", "case.json", "wing.json")
FORCES_PATTERNS = (
    "forces_breakdown.dat", "forces*.dat", "forces*.txt", "*.log", "*.out",
    "stdout*.txt", "history*.txt", "*convergence*", "residuals*.txt",
)
CONV_PATTERNS = ("history*.txt", "*convergence*", "residuals*.txt")


# ---------------------------------------------------------------------------
# shared spec helpers (also used by the CLI ingest command)
# ---------------------------------------------------------------------------

def design_spec_to_input(spec: Dict[str, Any]) -> np.ndarray:
    """Turns a design spec dict into the 40-D lake row input.

    Accepted forms:
      {"x": [40 floats]}                          full input row
      {"design": [37 floats], "flight": {...}}    37-D vector + flight condition
      {"wing": {planform...}, "flight": {...}}     planform dict + flight condition
    """
    if "x" in spec:
        row = np.asarray(spec["x"], dtype=float)
        if row.size != 40:
            raise ValueError(f"'x' must have 40 elements, got {row.size}")
        return row
    flight = spec.get("flight", {})
    design = np.asarray(spec.get("design", []), dtype=float)
    if design.size == 0 and "wing" in spec:
        w = spec["wing"]
        wing = Wing3D(
            span=w.get("span", 30.0), aspect_ratio=w.get("aspect_ratio", 9.5),
            taper_ratio=w.get("taper_ratio", 0.28),
            sweep_le_deg=w.get("sweep_le_deg", 27.5),
            dihedral_deg=w.get("dihedral_deg", 3.5),
            twist_root_deg=w.get("twist_root_deg", 2.0),
            twist_tip_deg=w.get("twist_tip_deg", -2.5),
            name="Ingest_Wing")
        design = wing.to_parameter_vector()
    if design.size != 37:
        raise ValueError(f"design vector must have 37 elements, got {design.size}")
    return design_to_input(
        design,
        alpha_deg=flight.get("alpha_deg", 2.5),
        mach=flight.get("mach", 0.8),
        reynolds=flight.get("reynolds", 2.5e7))


def label_from_forces(forces: Dict[str, float]) -> Tuple[np.ndarray, List[int]]:
    """Maps parsed SU2 forces to the 9-D surrogate label + mask (only the
    dimensions actually measured are labeled)."""
    y = np.zeros(9)
    mask = [0] * 9
    for idx, key in [(0, "cl"), (1, "cd"), (5, "cmz")]:
        if key in forces and forces[key] is not None:
            y[idx] = forces[key]
            mask[idx] = 1
    return y, mask


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

@dataclass
class CollectedRun:
    """One CFD run found under the sweep root."""
    directory: str
    forces_path: str = ""
    design: Optional[Dict[str, Any]] = None
    forces_text: str = ""
    convergence_text: str = ""


def _find_first(directory: str, patterns: Sequence[str]) -> str:
    for pat in patterns:
        hits = glob.glob(os.path.join(directory, pat))
        if hits:
            return hits[0]
    return ""


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


class SU2BatchCollector:
    """Sweeps CFD output directories and ingests quality-gated runs.

    Idempotent: already-ingested forces files are tracked in the lake meta and
    skipped on subsequent sweeps (safe to run on a schedule).
    """

    def __init__(self, lake: AeroDataLake, gate: Optional[CfdQualityGate] = None,
                 source: str = "su2", default_design: Optional[Dict[str, Any]] = None):
        self.lake = lake
        self.gate = gate if gate is not None else CfdQualityGate()
        self.source = source
        self.default_design = default_design

    # -- discovery -----------------------------------------------------------
    def discover(self, root: str, recurse: bool = True) -> List[CollectedRun]:
        """Finds run directories: any directory containing a forces-capable
        file (with a design spec next to it, or covered by the default)."""
        runs: List[CollectedRun] = []
        if recurse:
            it = (os.path.join(dirpath, dirname) for dirpath, dirnames, _ in
                  os.walk(root) for dirname in dirnames)
            dirs = [root] + list(it)
        else:
            dirs = [root]
        for d in sorted(set(dirs)):
            forces_path = _find_first(d, FORCES_PATTERNS)
            if not forces_path:
                continue
            text = _read_text(forces_path)
            if not re.search(r"Total\s+(?:CL|Lift)\b", text, re.IGNORECASE):
                continue
            design = self._find_design(d)
            conv_path = _find_first(d, CONV_PATTERNS)
            conv = _read_text(conv_path) if conv_path else ""
            runs.append(CollectedRun(
                directory=d, forces_path=forces_path, design=design,
                forces_text=text, convergence_text=conv))
        return runs

    def _find_design(self, directory: str) -> Optional[Dict[str, Any]]:
        path = _find_first(directory, DESIGN_PATTERNS)
        if path:
            try:
                return json.loads(_read_text(path))
            except json.JSONDecodeError:
                return None
        return self.default_design

    # -- ingestion -------------------------------------------------------------
    def _processed_files(self) -> set:
        return set(self.lake.get_meta("collected_files", []))

    def _mark_processed(self, path: str):
        files = self._processed_files()
        files.add(os.path.abspath(path))
        self.lake.set_meta("collected_files", sorted(files))

    def collect(self, root: str, recurse: bool = True, dry_run: bool = False,
                auto_update: bool = False, update_min_new: int = 8,
                verbose: bool = True) -> Dict[str, Any]:
        """Sweeps `root` and ingests every new quality-gated run.

        Returns a summary dict; with `auto_update` the ContinualTrainer is
        invoked afterwards (skipped for dry runs)."""
        runs = self.discover(root, recurse=recurse)
        processed = self._processed_files()
        summary = {"runs_found": len(runs), "new": 0, "accepted": 0,
                   "rejected": 0, "skipped_duplicates": 0, "rejected_reasons": {}}
        for run in runs:
            if os.path.abspath(run.forces_path) in processed:
                summary["skipped_duplicates"] += 1
                continue
            if dry_run:
                summary["new"] += 1
                continue
            summary["new"] += 1
            self._ingest_run(run, summary)
            self._mark_processed(run.forces_path)

        if auto_update and not dry_run and summary["accepted"] > 0:
            from .continual import ContinualTrainer
            trainer = ContinualTrainer(self.lake)
            upd = trainer.update(min_new_samples=update_min_new, verbose=False)
            summary["update"] = upd

        if verbose:
            print(f"[collect] {summary}")
        return summary

    def _ingest_run(self, run: CollectedRun, summary: Dict[str, Any]):
        combined = run.forces_text + "\n" + run.convergence_text
        residuals = parse_su2_residuals(combined)
        forces = parse_su2_forces(combined)
        reasons = list(self.gate.gate(
            forces, residuals=residuals if residuals else None).reasons)

        x_row: Optional[np.ndarray] = None
        if run.design is None:
            reasons.append("no design spec (design.json / case.json / wing.json "
                           "or --design-json default)")
        else:
            try:
                x_row = design_spec_to_input(run.design)
            except ValueError as exc:
                reasons.append(f"bad design spec: {exc}")

        if not reasons and x_row is not None:
            y_row, mask = label_from_forces(forces)
            sid = self.lake.append(
                x_row, y_row, source=f"cfd:{self.source}", mask=mask,
                accepted=True, y_vlm=run.design.get("y_vlm") if run.design else None,
                gate_reason="")
            summary["accepted"] += 1
            self.lake.log("collect", {"run": run.directory, "sample": sid})
        else:
            summary["rejected"] += 1
            key = "; ".join(reasons) or "no forces parsed"
            summary["rejected_reasons"][key] = summary["rejected_reasons"].get(key, 0) + 1
            self.lake.log("collect_rejected",
                          {"run": run.directory, "reason": "; ".join(reasons)})