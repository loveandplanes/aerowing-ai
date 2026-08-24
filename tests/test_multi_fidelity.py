"""
Tests for the multi-fidelity study harness (the "money slide"): the
deterministic error-vs-CFD-budget curve showing the continuous-learning loop
descending toward high-fidelity truth while the VLM-only baseline stays flat.
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from aerowing.multi_fidelity import (
    MultiFidelityStudy,
    StudyResult,
    Y_NAMES,
    _column_mean_abs_gap,
    _column_rmse,
    evaluate_high_fi,
    evaluate_low_fi,
    sample_case,
)
from aerowing.continual import AeroDataLake, design_to_input, \
    random_exploration_design


def _small_run(seed=7, budgets=(0, 8, 16), designs=36, holdout=12,
               warm_epochs=4, finetune_epochs=2):
    study = MultiFidelityStudy(seed=seed)
    return study.run(n_designs=designs, n_holdout=holdout, budgets=budgets,
                     warm_epochs=warm_epochs, finetune_epochs=finetune_epochs,
                     verbose=False)


def test_study_is_deterministic():
    r1 = _small_run(seed=11).to_dict()
    r2 = _small_run(seed=11).to_dict()
    assert r1 == r2


def test_low_fi_high_fi_gap_is_the_learnable_bias():
    rng = np.random.default_rng(3)
    x, y_lo, y_hi = sample_case(rng)
    assert x.shape == (40,)
    assert y_lo.shape == (9,)
    assert y_hi.shape == (9,)
    # The expensive model adds profile (+ possibly wave) drag over VLM-only;
    # its induced drag is the transonically-CL-scaled VLM value (CDi ~ CL^2 at
    # fixed span efficiency, so CDi follows (cl_hi/cl_lo)^2 when the calibration
    # corrects lift away from the raw Prandtl-Glauert value).
    assert y_hi[1] > y_lo[1]
    assert y_hi[3] > 0.0  # profile drag present in high-fi, absent in low-fi
    assert np.isclose(y_hi[1], y_hi[2] + y_hi[3] + y_hi[4], atol=1e-12)
    assert np.isclose(y_hi[2], y_lo[2] * (y_hi[0] / y_lo[0]) ** 2, atol=1e-12)


def test_loop_improves_on_its_own_warm_baseline():
    """The robust, small-pool-safe claims of the sweep:
    (a) any label spending leaves the flywheel below its own VLM-warm
    baseline (budget 0) on cd RMSE against truth;
    (b) the production promotion-gate metric (holdout MSE, same weighting
    across budgets) descends from budget 0 to the largest budget.

    The warm-beats-cold claim is asserted at design-suite scale in the CLI
    demo; at tiny pools (n<64) both trajectories are too noisy for a strict
    unit assertion (that noise is itself an honest finding: label budgets
    below ~n/4 barely register)."""
    res = _small_run(seed=5, budgets=(0, 8, 16, 32), designs=48, holdout=16,
                     warm_epochs=6, finetune_epochs=4)
    p0 = next(p for p in res.points if p["cfd_units"] == 0)
    spenders = [p for p in res.points if p["cfd_units"] > 0]
    assert len(spenders) == 3
    for p in spenders:
        assert p["rmse"]["cd"] < p0["rmse"]["cd"]
        assert 0.0 < p["rmse"]["cd"]
        assert math.isfinite(p["holdout_mse"])
    assert spenders[-1]["holdout_mse"] < p0["holdout_mse"]
    # the cold trajectory exists and is evaluable at every spender
    for p in spenders:
        assert p["rmse_cold"]["cd"] is not None and p["rmse_cold"]["cd"] > 0.0
        assert p["holdout_mse_cold"] is not None


def test_study_structure_and_holdout_accounting():
    res = _small_run(seed=9)
    d = res.to_dict()
    assert d["seed"] == 9
    assert d["cfd_units"] == [0, 8, 16]
    assert d["budgets"] == [0, 8, 16]
    assert len(d["points"]) == 3
    for pt in d["points"]:
        assert set(pt["rmse"]) == set(d["y_names"])
        assert pt["cfd_units"] in (0, 8, 16)
        assert pt["holdout_mse"] >= 0.0
        assert pt["parameters"] > 0
        if pt["cfd_units"] > 0:
            assert set(pt["rmse_cold"]) == set(d["y_names"])
            assert pt["holdout_mse_cold"] >= 0.0
            assert all(v is not None for v in pt["rmse_cold"].values())
        else:
            assert pt["rmse_cold"]["cd"] is None  # no cold trajectory at 0
    json.dumps(d)  # fully JSON-serializable


def test_holdout_rows_flagged_and_never_trainable():
    path = os.path.join(tempfile.mkdtemp(), "lake.sqlite")
    lake = AeroDataLake(path)
    try:
        x = np.zeros(40)
        y = np.zeros(9)
        rid = lake.append(x, y, source="vlm", accepted=True)
        hid = lake.append(x, y, source="truth_holdout", accepted=True,
                          in_holdout=True)
        # the flagged row is invisible to training queries
        _, _, _, train_ids = lake.train_batch()
        assert rid in train_ids and hid not in train_ids
        _, _, _, new_ids = lake.newer_than(0)
        assert hid not in new_ids
        # ...but visible to the holdout set
        _, _, _, hold_ids = lake.holdout_batch()
        assert hid in hold_ids and rid not in hold_ids
    finally:
        lake.close()


def test_rmse_and_gap_helpers():
    pred = np.array([[0.5, 0.01, 0.0, 0.0, 0.0, -0.05, 12.0, 0.9, 30.0],
                     [0.6, 0.02, 0.0, 0.0, 0.0, -0.06, 13.0, 0.9, 31.0],
                     [0.55, 0.015, 0.0, 0.0, 0.0, -0.055, 12.5, 0.9, 30.5]])
    truth = np.array([[0.5, 0.015, 0.0, 0.0, 0.0, -0.05, 12.0, 0.9, 30.0],
                      [0.6, 0.025, 0.0, 0.0, 0.0, -0.06, 13.0, 0.9, 31.0],
                      [0.55, 0.020, 0.0, 0.0, 0.0, -0.055, 12.5, 0.9, 30.5]])
    rmse = _column_rmse(pred, truth)
    gap = _column_mean_abs_gap(pred, truth)
    assert set(rmse) == set(Y_NAMES)
    assert abs(rmse["cl"] - 0.0) < 1e-12
    assert abs(rmse["cd"] - 0.005) < 1e-12
    assert abs(gap["cd"] - 0.005) < 1e-12


def _fake_cfd_lake(path: str, n: int = 9):
    """A temp lake with n accepted real-CFD-style rows (partial SU2 delivery)."""
    lake = AeroDataLake(path)
    rng = np.random.default_rng(11)
    for _ in range(n):
        design = random_exploration_design(rng)
        alpha = float(rng.uniform(1.5, 3.5))
        mach = float(rng.uniform(0.76, 0.84))
        re = float(10 ** rng.uniform(7.1, 7.3))
        x = design_to_input(design, alpha, mach, re)
        y = np.zeros(len(Y_NAMES))
        y[0] = float(rng.uniform(0.15, 0.45))   # CL
        y[1] = float(rng.uniform(0.012, 0.05))  # CD
        y[5] = float(rng.uniform(-0.10, -0.01)) # CMz
        lake.append(x, y, source="cfd:su2",
                    mask=[1, 1, 0, 0, 0, 1, 0, 0, 0], accepted=True)
    lake.close()


def test_lake_truth_mode_runs_and_masks_undelivered(tmp_path):
    """truth='lake': real CFD rows are the ground truth; partial-delivery
    masks flow through training/eval and undelivered metrics report None."""
    lake_file = str(tmp_path / "truth.sqlite")
    _fake_cfd_lake(lake_file, n=12)
    study = MultiFidelityStudy(seed=5)
    # tiny step budgets keep this test fast; mechanics are what is asserted
    study.WARM_STEP_TARGET = 60
    study.FINE_STEP_TARGET = 20
    study.COLD_STEP_TARGET = 30
    res = study.run(n_designs=6, n_holdout=3, budgets=(0, 3),
                    warm_epochs=1, finetune_epochs=1,
                    workdir=str(tmp_path / "work"), verbose=False,
                    truth="lake", lake_path=lake_file)
    assert res.truth == "lake"
    assert len(res.points) == 2
    assert res.vlm_baseline["cd"] is not None
    assert res.vlm_baseline["fuel_volume_m3"] is None
    for p in res.points:
        assert p["rmse"]["cd"] is not None
        assert p["rmse"]["cl"] is not None
        assert p["rmse"]["fuel_volume_m3"] is None
        assert p["rmse_cold"]["cd"] is not None or p["cfd_units"] == 0
        assert p["holdout_mse"] >= 0.0
    d = res.to_dict()
    json.dumps(d)
    assert StudyResult.from_dict(d).truth == "lake"
    assert "real CFD lake rows" in res.table(key="cd")


def test_lake_truth_mode_requires_enough_rows(tmp_path):
    lake_file = str(tmp_path / "small.sqlite")
    _fake_cfd_lake(lake_file, n=4)
    study = MultiFidelityStudy(seed=5)
    try:
        study.run(n_designs=6, n_holdout=3, budgets=(0,),
                  workdir=str(tmp_path / "w"), verbose=False,
                  truth="lake", lake_path=lake_file)
        assert False, "should refuse when CFD rows are insufficient"
    except ValueError:
        pass
