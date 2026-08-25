"""Policy tests for the novel capacity-growth trigger.

The growth mechanism (zero-init residual expansion + error-directed
fine-tuning) only ever fires through `ContinualTrainer.update(auto_grow=True)`
when three gates hold simultaneously:

  1. enough real CFD mass has accrued        (growth_min_new)
  2. enough promotion history exists          (growth_plateau_updates)
  3. holdout improvement across the window is below the relative threshold
     (plateau) — and a STILL-IMPROVING history must NOT grow

These pin that decision policy; the mechanics of the grown block are covered
in test_continual.py (function preservation, state-dict compatibility).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from aerowing.continual import AeroDataLake, ContinualTrainer
from aerowing.models.dataset_3d import generate_synthetic_wing_dataset


def _seed_lake(path, n_vlm=10, n_cfd=6):
    lake = AeroDataLake(path)
    x_data, y_data = generate_synthetic_wing_dataset(
        num_samples=n_vlm + n_cfd, seed=3)
    k = 0
    for _ in range(n_vlm):
        lake.append(x_data[k], y_data[k], source="vlm")
        k += 1
    for _ in range(n_cfd):
        lake.append(x_data[k], y_data[k], source="cfd:su2",
                    mask=[1] * len(y_data[k]))
        k += 1
    lake.ensure_holdout(n=4)
    return lake


def _log_history(lake, values):
    for v in values:
        lake.log("promote", {"holdout_mse": v})


PLATEAU = [0.50, 0.4985, 0.498, 0.4979]      # gain ~ 0.4% < 0.5% threshold
IMPROVING = [0.60, 0.52, 0.45, 0.38]         # gain = 36.7% — must NOT grow


def test_auto_grow_triggers_on_plateau(tmp_path):
    lake = _seed_lake(str(tmp_path / "lake.sqlite"))
    try:
        _log_history(lake, PLATEAU)
        trainer = ContinualTrainer(lake, checkpoint_path=str(tmp_path / "m.pt"))
        base_params = trainer.model.n_params()
        assert trainer.model.expander is None
        summary = trainer.update(epochs=1, lr=1e-3, min_new_samples=0,
                                 auto_grow=True, growth_min_new=5,
                                 verbose=False)
        assert summary["updated"] is True
        assert trainer.model.expander is not None, "plateau must grow capacity"
        assert trainer.model.n_params() > base_params
        # the decision is audited in the lake history
        assert len(lake.history_vals("grow", "n_cfd")) == 1
    finally:
        lake.close()


def test_auto_grow_holds_off_while_improving(tmp_path):
    """Regression guard: the original min()-based trigger read zero gain off
    ANY monotone decline and grew mid-improvement."""
    lake = _seed_lake(str(tmp_path / "lake.sqlite"))
    try:
        _log_history(lake, IMPROVING)
        trainer = ContinualTrainer(lake, checkpoint_path=str(tmp_path / "m.pt"))
        summary = trainer.update(epochs=1, lr=1e-3, min_new_samples=0,
                                 auto_grow=True, growth_min_new=5,
                                 verbose=False)
        assert summary["updated"] is True
        assert trainer.model.expander is None, \
            "a still-improving model must not grow"
        assert len(lake.history_vals("grow", "n_cfd")) == 0
    finally:
        lake.close()


def test_auto_grow_requires_cfd_mass(tmp_path):
    lake = _seed_lake(str(tmp_path / "lake.sqlite"), n_cfd=2)
    try:
        _log_history(lake, PLATEAU)
        trainer = ContinualTrainer(lake, checkpoint_path=str(tmp_path / "m.pt"))
        trainer.update(epochs=1, lr=1e-3, min_new_samples=0,
                       auto_grow=True, growth_min_new=5, verbose=False)
        assert trainer.model.expander is None, \
            "growth needs real CFD mass first"
    finally:
        lake.close()


def test_gated_capacity_preserves_function_and_freezes_base(tmp_path):
    """Production growth mode (validated in experiments/residual_expansion.py):
    gated correction preserves the function exactly at init (zero-init value
    head), and the post-growth fine-tune freezes the base network so solved
    knowledge is protected structurally."""
    import torch
    from aerowing.continual import GrowableSurrogate

    g = GrowableSurrogate()
    g.eval()
    x = torch.randn(8, 40)
    before = g(x).clone()
    assert g.add_capacity(units=32, gated=True) == 1
    assert g.add_capacity(units=32, gated=True) == 0      # idempotent
    with torch.no_grad():
        after = g(x)
    assert torch.allclose(before, after, atol=1e-6), \
        "gated capacity must preserve the function exactly at init"

    lake = _seed_lake(str(tmp_path / "lake2.sqlite"))
    try:
        _log_history(lake, PLATEAU)
        trainer = ContinualTrainer(lake, checkpoint_path=str(tmp_path / "m.pt"))
        summary = trainer.update(epochs=1, lr=1e-3, min_new_samples=0,
                                 auto_grow=True, growth_min_new=5,
                                 verbose=False)
        assert summary["updated"] is True
        assert trainer.model.expander is not None
        # structural protection: base frozen, gated correction trainable
        assert all(p.requires_grad is False
                   for p in trainer.model.base.parameters())
        assert any(p.requires_grad for p in trainer.model.expander.parameters())
        # the decision is audited with its mode (n_cfd at trigger time = 6)
        hist = json.loads(json.dumps(
            lake.history_vals("grow", "n_cfd")))
        assert hist == [6]
    finally:
        lake.close()
