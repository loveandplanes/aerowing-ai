"""
Tests for the continual / lifelong learning engine (aerowing.continual):
quality gate, SU2 parsing, data lake, exploration cadence, error-directed
gradient weighting, function-preserving capacity growth, and the
fine-tune + holdout promotion loop.
"""

import json
import os
import tempfile

import numpy as np
import torch

from aerowing.continual import (
    AeroDataLake,
    ContinualTrainer,
    CfdQualityGate,
    DiversitySampler,
    GrowableSurrogate,
    error_directed_weights,
    parse_su2_forces,
    parse_su2_residuals,
    random_exploration_design,
    design_to_input,
    _X_DIM,
    _Y_DIM,
)
from aerowing.models.dataset_3d import generate_synthetic_wing_dataset

def _su2_log(residuals, cl, cd, cmz):
    lines = ["|  Time_Iter  |  Outer_Iter  |  Inner_Iter  |      RMS_PRESSURE     |"]
    for i, r in enumerate(residuals):
        lines.append(f"|  {i * 5:<11d} |  0           |  {i * 5:<11d} |  {r:.6e}         |")
    lines += [
        "|--- Forces breakdown (zone 0) ---",
        f"Total CL: {cl:.4f}",
        f"Total CD: {cd:.4f}",
        "Total CMx: 0.0000",
        "Total CMy: 0.0000",
        f"Total CMz: {cmz:.4f}",
    ]
    return "\n".join(lines)


def _decaying_residuals(n=120, start=1e-1, end=1e-6):
    return [start * (end / start) ** (i / (n - 1)) for i in range(n)]


def _diverging_residuals(n=120, start=1e-5, end=7e-4):
    return [start * (end / start) ** (i / (n - 1)) for i in range(n)]


GOOD_SU2 = _su2_log(_decaying_residuals(), 0.55, 0.0234, -0.1123)
BAD_SU2 = _su2_log(_diverging_residuals(), 8.5, -0.1, 0.0)


# ---------------------------------------------------------------------------
# parsing + gate
# ---------------------------------------------------------------------------

def test_parse_su2_forces():
    forces = parse_su2_forces(GOOD_SU2)
    assert forces["cl"] == 0.55
    assert abs(forces["cd"] - 0.0234) < 1e-9
    assert abs(forces["cmz"] + 0.1123) < 1e-9


def test_parse_su2_residuals():
    residuals = parse_su2_residuals(GOOD_SU2)
    assert len(residuals) >= 100
    assert abs(residuals[0] - 1e-1) < 1e-9
    assert abs(residuals[-1] - 1e-6) < 1e-9


def test_gate_accepts_converged_su2():
    result = CfdQualityGate().gate_su2_text(GOOD_SU2)
    assert result.accepted, result.reasons


def test_gate_rejects_divergent_and_implausible():
    result = CfdQualityGate().gate_su2_text(BAD_SU2)
    assert not result.accepted
    assert any("not decreasing" in r for r in result.reasons)
    assert any("CL" in r for r in result.reasons)


def test_gate_rejects_missing_forces():
    result = CfdQualityGate().gate({"cd": 0.02})
    assert not result.accepted
    assert any("missing CL or CD" in r for r in result.reasons)


def test_gate_rejects_few_iterations():
    good = {"cl": 0.5, "cd": 0.02}
    assert CfdQualityGate(min_iterations=100).gate(good, iterations=10).accepted is False


# ---------------------------------------------------------------------------
# data lake
# ---------------------------------------------------------------------------

def _sample_row():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, _X_DIM)
    y = rng.uniform(0, 1, _Y_DIM)
    return x, y


_FULL_MASK = [1] * _Y_DIM
_SU2_PARTIAL_MASK = [1, 1, 0, 0, 0, 1, 0, 0, 0]  # CL/CD/CMz only


def test_lake_append_stats_and_holdout(tmp_path):
    lake = AeroDataLake(str(tmp_path / "lake.sqlite"))
    x, y = _sample_row()
    lake.append(x, y, source="cfd:su2", mask=_FULL_MASK, accepted=True)
    lake.append(x + 0.1, y, source="vlm", accepted=False, gate_reason="bad")
    stats = lake.stats()
    assert stats["samples"] == 2
    assert stats["accepted"] == 1

    lake.ensure_holdout(n=15)
    assert lake.stats()["holdout"] == 0  # only 1 accepted row -> nothing flagged
    lake.append(x + 0.2, y, source="vlm")
    for i in range(30):
        lake.append(x + 0.01 * i, y + 0.01 * i, source="cfd:su2",
                    mask=_FULL_MASK)
    lake.ensure_holdout(n=5)
    assert lake.stats()["holdout"] == 5
    assert lake.ids_by_source("cfd")  # source pattern match

    x_tr, y_tr, m_tr, ids = lake.train_batch()
    assert ids and x_tr.ndim == 2 and x_tr.shape[1] == _X_DIM
    lake.close()


def test_lake_append_validation_error(tmp_path):
    lake = AeroDataLake(str(tmp_path / "lake2.sqlite"))
    try:
        lake.append([1.0, 2.0], [0.0] * _Y_DIM, source="vlm")
        assert False, "should raise"
    except ValueError:
        pass
    lake.close()


def test_lake_newer_than_and_rejected_never_train(tmp_path):
    lake = AeroDataLake(str(tmp_path / "lake3.sqlite"))
    for i in range(12):
        x, y = _sample_row()
        lake.append(x, y, source="cfd:su2" if i % 2 else "vlm",
                    mask=_FULL_MASK if i % 2 else None,
                    accepted=bool(i % 2), gate_reason="" if i % 2 else "rejected")
    x_tr, _, _, ids = lake.train_batch()
    assert len(ids) == 6
    _, _, _, new_ids = lake.newer_than(3)
    assert new_ids == sorted(new_ids)
    lake.close()


def test_lake_cfd_rows_require_explicit_mask(tmp_path):
    """A CFD row without a declared mask would default to all-ones and train
    undelivered columns toward their zero placeholders — refused at the seam."""
    lake = AeroDataLake(str(tmp_path / "lake4.sqlite"))
    try:
        x, y = _sample_row()
        lake.append(x, y, source="cfd:su2")
        assert False, "cfd append without mask should raise"
    except ValueError:
        pass
    # partial mask survives the round-trip; undelivered columns stay masked out
    lake.append(x, y, source="cfd:su2", mask=_SU2_PARTIAL_MASK)
    cfd_id = lake.ids_by_source("cfd")[0]
    # wrong-length mask refused, non-CFD sources keep the all-ones default
    try:
        lake.append(x + 1.0, y, source="cfd:su2", mask=[1, 1])
        assert False, "wrong-length mask should raise"
    except ValueError:
        pass
    lake.append(x + 2.0, y, source="vlm")
    _, _, m_tr, ids = lake.train_batch()
    assert list(m_tr[ids.index(cfd_id)]) == [1.0, 1.0, 0.0, 0.0, 0.0,
                                             1.0, 0.0, 0.0, 0.0]
    lake.close()


# ---------------------------------------------------------------------------
# exploration cadence (controller #2)
# ---------------------------------------------------------------------------

def test_diversity_sampler_cadence(tmp_path):
    lake = AeroDataLake(str(tmp_path / "lake4.sqlite"))
    sampler = DiversitySampler(lake, interval=3, rng=np.random.default_rng(7))
    requested = np.concatenate([np.full(7, 30.0), np.zeros(30)])
    flags = []
    for _ in range(9):
        _, is_expl = sampler.proposal(requested)
        flags.append(is_expl)
    assert np.sum(flags) == 3  # calls 0, 3, 6
    assert lake.get_meta("exploration_counter") == 9  # persisted cadence
    lake.close()


def test_random_exploration_design_is_valid_design():
    design = random_exploration_design(np.random.default_rng(1))
    assert design.shape == (37,)
    assert design[0] >= 15.0 and design[0] <= 55.0  # span bounds


# ---------------------------------------------------------------------------
# error-directed gradient routing (airfoi Formulation C)
# ---------------------------------------------------------------------------

def test_error_directed_weights_properties():
    # tiny old error -> half-weight (sigmoid(0) = 0.5, no improvement bonus)
    w_solved = error_directed_weights(np.array([1e-9]), np.array([1e-9]))
    assert abs(w_solved[0] - 0.5) < 0.2
    # improved region (error dropped) -> reinforced weight > 1
    w_improved = error_directed_weights(np.array([1.0]), np.array([0.2]))
    assert w_improved[0] > 1.0
    # degraded region (error grew) -> no bonus, stays at sigmoid level
    w_degraded = error_directed_weights(np.array([1.0]), np.array([2.0]))
    assert abs(w_degraded[0] - 1.0) < 0.2
    # monotonic: more improvement -> higher weight
    w1 = error_directed_weights(np.array([1.0]), np.array([0.9]))
    w2 = error_directed_weights(np.array([1.0]), np.array([0.1]))
    assert w2[0] > w1[0]
    # all weights finite and non-negative
    w = error_directed_weights(np.array([0.5, 2.0]), np.array([1.0, 0.5]))
    assert np.all(np.isfinite(w)) and np.all(w >= 0.0)


# ---------------------------------------------------------------------------
# capacity growth preserves the learned function exactly
# ---------------------------------------------------------------------------

def test_growable_surrogate_preserves_function():
    from aerowing.models.surrogate_3d import AeroSurrogate3D
    base = AeroSurrogate3D()
    base.eval()
    growable = GrowableSurrogate(base=base)
    growable.add_capacity(units=64)
    x = torch.randn(4, _X_DIM)
    with torch.no_grad():
        out_old = base(x)
        out_new = growable(x)
    diff = (out_new - out_old).abs().max().item()
    assert diff < 1e-6, f"capacity growth disturbed the function: {diff}"
    base_params = sum(p.numel() for p in base.parameters())
    assert growable.n_params() > base_params  # capacity really grew


def test_growable_surrogate_state_dict_compat():
    growable = GrowableSurrogate()
    growable.add_capacity(units=32)
    state = growable.base_state_dict()
    assert "input_proj.0.weight" in state  # base-only, web-server compatible
    growth = growable.growth_state_dict()
    assert growth is not None and "2.weight" in growth


# ---------------------------------------------------------------------------
# continual trainer: fine-tune + promotion gate (controllers #3+#4)
# ---------------------------------------------------------------------------

def _make_lake_and_trainer(tmp_path, samples: int = 12):
    x_data, y_data = generate_synthetic_wing_dataset(num_samples=samples, seed=42)
    lake = AeroDataLake(str(tmp_path / "lake.sqlite"))
    for i in range(samples):
        lake.append(x_data[i], y_data[i], source="vlm")
    lake.ensure_holdout(n=min(4, samples))
    ckpt = str(tmp_path / "checkpoints" / "models.pt")
    trainer = ContinualTrainer(lake, checkpoint_path=ckpt)
    return lake, trainer


def test_trainer_first_update_promotes_and_saves(tmp_path):
    lake, trainer = _make_lake_and_trainer(tmp_path)
    summary = trainer.update(epochs=2, lr=1e-3, min_new_samples=8, verbose=False)
    assert summary["updated"] is True
    assert summary["promoted"] is True  # no baseline yet -> first promotion
    assert os.path.exists(trainer.checkpoint_path)
    assert trainer.last_processed == lake.last_id()
    # checkpoint compatible with the bare web-server loader
    ckpt = torch.load(trainer.checkpoint_path, map_location="cpu")
    assert "surrogate_state" in ckpt
    lake.close()


def test_trainer_refuses_regression_on_holdout(tmp_path):
    lake, trainer = _make_lake_and_trainer(tmp_path)
    trainer.update(epochs=2, lr=1e-3, min_new_samples=8, verbose=False)
    before = torch.load(trainer.checkpoint_path, map_location="cpu")
    # append a fresh batch of samples so the second update has work to do
    x_data, y_data = generate_synthetic_wing_dataset(num_samples=12, seed=99)
    for i in range(12):
        lake.append(x_data[i], y_data[i], source="vlm")
    # force an impossible baseline -> any update regresses -> refusal
    lake.set_meta("holdout_mse_baseline", 1e-9)
    summary = trainer.update(epochs=2, lr=1e-3, min_new_samples=8, verbose=False)
    assert summary["updated"] is True
    assert summary["promoted"] is False, "must refuse regression on holdout"
    after = torch.load(trainer.checkpoint_path, map_location="cpu")
    w_before = before["surrogate_state"]["input_proj.0.weight"].clone()
    w_after = after["surrogate_state"]["input_proj.0.weight"].clone()
    assert torch.equal(w_before, w_after), "refused update must not touch checkpoint"
    lake.close()


def test_trainer_skips_without_min_new_samples(tmp_path):
    lake, trainer = _make_lake_and_trainer(tmp_path)
    summary = trainer.update(epochs=1, min_new_samples=50, verbose=False)
    assert summary["updated"] is False
    lake.close()


def test_trainer_partial_labels_and_growth_path(tmp_path):
    """CFD rows carrying only CL/CD/CM (masked) train without breaking."""
    lake = AeroDataLake(str(tmp_path / "lake.sqlite"))
    x_data, y_data = generate_synthetic_wing_dataset(num_samples=10, seed=7)
    for i in range(10):
        y = np.zeros(_Y_DIM)
        y[0], y[1], y[5] = y_data[i, 0], y_data[i, 1], y_data[i, 5]
        mask = [1, 1, 0, 0, 0, 1, 0, 0, 0]
        lake.append(x_data[i], y, source="cfd:custom", mask=mask)
    lake.ensure_holdout(n=2)
    trainer = ContinualTrainer(lake, checkpoint_path=str(tmp_path / "m.pt"))
    summary = trainer.update(epochs=1, lr=1e-3, min_new_samples=4, verbose=False)
    assert summary["updated"] is True
    assert summary["promoted"] is True
    # growth path: add capacity, then one fine-tune epoch with error-directed weights
    trainer.model.add_capacity(units=32)
    summary2 = trainer.update(epochs=1, lr=1e-3, min_new_samples=0, verbose=False)
    assert summary2["updated"] is True
    lake.close()


def test_design_to_input():
    design = np.full(37, 0.5)
    row = design_to_input(design, alpha_deg=3.0, mach=0.8, reynolds=2.5e7)
    assert row.shape == (_X_DIM,)
    assert row[-3] == 3.0 and row[-2] == 0.8
    assert abs(row[-1] - np.log10(2.5e7)) < 1e-9