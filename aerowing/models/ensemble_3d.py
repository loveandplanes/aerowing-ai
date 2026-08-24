"""
Deep-Ensemble Uncertainty Quantification (UQ) for the aerodynamic surrogate.

Trains K physics-regularized AeroSurrogate3D members with distinct random
seeds and treats the across-member spread as the per-prediction uncertainty
band. Rationale:
  - cheap and robust (no extra dependencies, no Bayesian machinery),
  - calibrated-ish by construction for in-distribution inputs, and
  - informative for the continuous-learning flywheel: the band tells where
    the model is unsure, i.e. where an expensive CFD label is worth most.

The bands are meant to SHRINK as the learning loop accrues labels - the
concrete "the AI gets more confident over time" claim of the project.

API:
    ens = train_ensemble_surrogate(dataset, n_members=4, epochs=30, seeds=(...))
    mean, std = ens.predict_batch(x)                # x: [B, 40] -> [B, 9] each
    out = ens.predict_wing(wing_param_vector, ...)  # dict with *_uncertainty
    ens.save("checkpoints/aerowing_ensemble.pt")
    ens = EnsembleSurrogate3D.load("checkpoints/aerowing_ensemble.pt")
"""

import numpy as np
import torch

from .surrogate_3d import AeroSurrogate3D
from .trainer_3d import AeroTrainer3D
from .dataset_3d import WingDataset3D

OUTPUT_NAMES = [
    "cl",
    "cd",
    "cd_induced",
    "cd_profile",
    "cd_wave",
    "cm",
    "l_over_d",
    "span_efficiency",
    "fuel_volume_m3",
]

DEFAULT_MEMBER_SEEDS = (1001, 1007, 1013, 1019, 1031)


class EnsembleSurrogate3D:
    """Mean prediction + per-datum uncertainty from a deep ensemble."""

    def __init__(self, members, seeds=None):
        assert len(members) >= 2, "an ensemble needs at least 2 members"
        self.members = list(members)
        self.seeds = (
            list(seeds) if seeds is not None
            else [None] * len(members)
        )

    @property
    def n_members(self) -> int:
        return len(self.members)

    def predict_batch(self, x: np.ndarray) -> tuple:
        """x: [B, 40] inputs -> (mean, std) arrays, each [B, 9]."""
        x_t = torch.tensor(np.asarray(x, dtype=np.float32))
        outs = []
        for member in self.members:
            member.eval()
            with torch.no_grad():
                outs.append(member(x_t).numpy())
        stack = np.stack(outs, axis=0)          # [K, B, 9]
        return stack.mean(axis=0), stack.std(axis=0)

    def predict_wing(self, wing_param_vector: np.ndarray,
                     alpha_deg: float = 2.5, mach: float = 0.82,
                     reynolds: float = 2.5e7) -> dict:
        """High-level inference returning value + `_uncertainty` for every output.

        `wing_param_vector` is the 37-D wing parameter vector; the flight
        condition is appended internally (same convention as AeroSurrogate3D).
        """
        flight_cond = np.array([alpha_deg, mach, np.log10(max(reynolds, 1e4))])
        x = np.concatenate([np.asarray(wing_param_vector, dtype=np.float64),
                            flight_cond]).reshape(1, -1)
        mean, std = self.predict_batch(x)
        m, s = mean[0], std[0]
        out = {}
        for i, name in enumerate(OUTPUT_NAMES):
            out[name] = float(m[i])
            out[name + "_uncertainty"] = float(s[i])
        return out

    def save(self, path: str):
        """Checkpoint the whole ensemble for later loading."""
        torch.save({
            "format": "aerowing_ensemble_v1",
            "seeds": self.seeds,
            "member_hidden_dim": self.members[0].hidden_dim,
            "members": [m.state_dict() for m in self.members],
        }, path)

    @staticmethod
    def load(path: str) -> "EnsembleSurrogate3D":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint.get("format") == "aerowing_ensemble_v1", \
            f"unrecognized ensemble checkpoint format: {path}"
        hidden = int(checkpoint.get("member_hidden_dim", 256))
        members = [AeroSurrogate3D(hidden_dim=hidden)
                   for _ in checkpoint["members"]]
        for m, state in zip(members, checkpoint["members"]):
            m.load_state_dict(state)
        m.eval()
        return EnsembleSurrogate3D(
            members, seeds=checkpoint.get("seeds"))


def train_ensemble_surrogate(
    dataset: WingDataset3D,
    n_members: int = 5,
    epochs: int = 30,
    seeds=None,
    batch_size: int = 32,
    lr: float = 1e-3,
    physics_weight: float = 0.25,
    hidden_dim: int = 128,
    verbose: bool = True,
) -> EnsembleSurrogate3D:
    """Train K physics-regularized surrogate members with distinct seeds.

    Each member is initialized and trained under its own torch seed, so the
    Fourier embedding, the random weight init and the train/val split all
    differ across members - exactly what makes the spread a real (if
    heuristic) uncertainty estimate rather than a rounding artifact.

    `hidden_dim` defaults to 128 (vs the flagship surrogate's 256): leaner
    members overfit their individual train/val splits less, so the spread
    carries more signal and less optimization chaos (measured: pooled
    error-rank correlation ~0.52 at 640 samples / 40 epochs / 5 members,
    stable across member-seed sets).
    """
    if seeds is None:
        seeds = list(DEFAULT_MEMBER_SEEDS[:n_members])
        if len(seeds) < n_members:
            seeds = [1000 + 6 * i for i in range(n_members)]
    assert len(seeds) == n_members, "seeds must match the member count"

    members = []
    for i, seed in enumerate(seeds, 1):
        torch.manual_seed(seed)
        np.random.seed(seed)
        member = AeroSurrogate3D(hidden_dim=hidden_dim)
        trainer = AeroTrainer3D(surrogate=member)
        trainer.train_surrogate(
            dataset, epochs=epochs, batch_size=batch_size, lr=lr,
            physics_weight=physics_weight, verbose=False)
        members.append(member)
        if verbose:
            print(f"[ensemble] member {i}/{n_members} trained (seed {seed})")

    return EnsembleSurrogate3D(members, seeds=seeds)


def uncertainty_label(mean: np.ndarray, std: np.ndarray,
                      name: str, width: float = 2.0) -> str:
    """Human-readable `CL 0.6123 +/- 0.0123` (band = width * std)."""
    idx = OUTPUT_NAMES.index(name)
    return (f"{name} {mean[idx]:.4f} +/- {width * std[idx]:.4f} "
            f"({width:.0f}-sigma)")