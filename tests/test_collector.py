"""
Tests for the SU2 batch collector (aerowing.collector): directory sweep,
per-run quality gating, idempotent re-runs, dry runs, and shared
design/label helpers.
"""

import json
import os

import numpy as np

from aerowing.collector import (
    SU2BatchCollector,
    design_spec_to_input,
    label_from_forces,
)
from aerowing.continual import AeroDataLake, parse_su2_forces


def _residual_table(residuals):
    lines = ["|  Time_Iter  |  Outer_Iter  |  Inner_Iter  |      RMS_PRESSURE     |"]
    for i, r in enumerate(residuals):
        lines.append(f"|  {i * 5:<11d} |  0           |  {i * 5:<11d} |  {r:.6e}         |")
    return "\n".join(lines)


def _decay(n=120, start=1e-1, end=1e-6):
    return [start * (end / start) ** (i / (n - 1)) for i in range(n)]


def _divergent(n=120):
    return _decay(n, start=1e-5, end=7e-4)


def _forces(cl, cd, cmz=0.0):
    return ("|--- Forces breakdown (zone 0) ---\n"
            f"Total CL: {cl:.4f}\nTotal CD: {cd:.4f}\n"
            "Total CMx: 0.0000\nTotal CMy: 0.0000\n"
            f"Total CMz: {cmz:.4f}\n")


WING_SPEC = {
    "wing": {"span": 32.0, "aspect_ratio": 9.2, "taper_ratio": 0.30,
             "sweep_le_deg": 28.0, "dihedral_deg": 3.5,
             "twist_root_deg": 2.0, "twist_tip_deg": -2.5},
    "flight": {"alpha_deg": 2.5, "mach": 0.80, "reynolds": 2.5e7},
}


def _make_run_tree(tmp_path):
    root = tmp_path / "runs"
    good_hist = _residual_table(_decay())

    run_a = root / "run_a"
    run_a.mkdir(parents=True)
    (run_a / "design.json").write_text(json.dumps(WING_SPEC), encoding="utf-8")
    (run_a / "forces_breakdown.dat").write_text(_forces(0.52, 0.021), encoding="utf-8")
    (run_a / "history.txt").write_text(good_hist, encoding="utf-8")

    run_b = root / "run_b"
    run_b.mkdir()
    (run_b / "design.json").write_text(json.dumps(WING_SPEC), encoding="utf-8")
    (run_b / "run.log").write_text(
        _residual_table(_divergent()) + _forces(8.5, -0.1), encoding="utf-8")

    run_c = root / "run_c"
    run_c.mkdir()
    (run_c / "forces_breakdown.dat").write_text(_forces(0.48, 0.019), encoding="utf-8")

    run_d = root / "run_d"
    run_d.mkdir()
    x40 = list(np.concatenate([WING_SPEC["wing"]["span"] * np.ones(37),
                               [2.5, 0.8, np.log10(2.5e7)]]))
    (run_d / "case.json").write_text(json.dumps({"x": x40}), encoding="utf-8")
    (run_d / "forces_breakdown.dat").write_text(_forces(0.61, 0.024, -0.10), encoding="utf-8")
    (run_d / "history.txt").write_text(good_hist, encoding="utf-8")

    run_e = root / "run_e"
    run_e.mkdir()
    (run_e / "design.json").write_text(json.dumps(WING_SPEC), encoding="utf-8")
    # no forces file at all -> not a run

    (root / "junk.txt").write_text("not a run\n", encoding="utf-8")
    return root


def test_discover_finds_runs_with_forces(tmp_path):
    root = _make_run_tree(tmp_path)
    lake = AeroDataLake(str(tmp_path / "lake.sqlite"))
    collector = SU2BatchCollector(lake)
    runs = collector.discover(str(root))
    names = sorted(os.path.basename(r.directory) for r in runs)
    assert names == ["run_a", "run_b", "run_c", "run_d"]  # run_e excluded
    lake.close()


def test_collect_gates_and_is_idempotent(tmp_path):
    root = _make_run_tree(tmp_path)
    lake = AeroDataLake(str(tmp_path / "lake.sqlite"))
    collector = SU2BatchCollector(lake)
    summary = collector.collect(str(root), verbose=False)
    assert summary["new"] == 4
    assert summary["accepted"] == 2      # run_a, run_d
    assert summary["rejected"] == 2      # run_b (gate), run_c (no design)
    assert lake.stats()["samples"] == 2
    assert lake.stats()["accepted"] == 2
    # rejected run reasons are recorded
    reasons = " | ".join(summary["rejected_reasons"].keys())
    assert "not decreasing" in reasons
    assert "no design spec" in reasons
    # idempotent second sweep: nothing new
    summary2 = collector.collect(str(root), verbose=False)
    assert summary2["new"] == 0
    assert summary2["skipped_duplicates"] == 4
    assert lake.stats()["samples"] == 2
    lake.close()


def test_collect_dry_run_writes_nothing(tmp_path):
    root = _make_run_tree(tmp_path)
    lake = AeroDataLake(str(tmp_path / "lake.sqlite"))
    collector = SU2BatchCollector(lake)
    summary = collector.collect(str(root), dry_run=True, verbose=False)
    assert summary["new"] == 4
    assert lake.stats()["samples"] == 0
    # processed-file tracking is untouched by dry runs
    assert collector._processed_files() == set()
    lake.close()


def test_collect_default_design_covers_missing_spec(tmp_path):
    root = _make_run_tree(tmp_path)
    lake = AeroDataLake(str(tmp_path / "lake.sqlite"))
    collector = SU2BatchCollector(lake, default_design=WING_SPEC)
    summary = collector.collect(str(root), verbose=False)
    assert summary["accepted"] == 3      # run_c now has a design
    assert summary["rejected"] == 1
    lake.close()


def test_helpers_design_spec_and_label():
    row = design_spec_to_input(WING_SPEC)
    assert row.shape == (40,)
    assert abs(row[-1] - np.log10(2.5e7)) < 1e-9
    full = design_spec_to_input({"x": list(np.arange(40, dtype=float))})
    assert np.allclose(full, np.arange(40))
    try:
        design_spec_to_input({"design": [1.0, 2.0]})
        assert False, "should raise for wrong design length"
    except ValueError:
        pass

    forces = parse_su2_forces(_forces(0.5, 0.02, -0.1))
    y, mask = label_from_forces(forces)
    assert y[0] == 0.5 and y[1] == 0.02 and y[5] == -0.1
    assert mask == [1, 1, 0, 0, 0, 1, 0, 0, 0]
    assert sum(mask) == 3