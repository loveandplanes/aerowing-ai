"""
3D Wing Dataset Generator and PyTorch Dataset Pipeline.
Generates aerospace-grade multi-fidelity aerodynamic datasets across flight envelopes.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from ..geometry.wing_3d import Wing3D
from ..geometry.cst_3d import CSTAirfoil3D
from ..solvers.aero_engine import AeroEngine3D


class WingDataset3D(Dataset):
    """PyTorch Dataset container for 3D Wing aerodynamic data."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


def generate_synthetic_wing_dataset(
    num_samples: int = 120,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a physically consistent 3D wing dataset using Latin Hypercube design sampling
    evaluated with the high-fidelity 3D AeroEngine.
    """
    np.random.seed(seed)
    x_list = []
    y_list = []

    for i in range(num_samples):
        # 1. Sample Planform Parameters
        span = float(np.random.uniform(15.0, 55.0))
        ar = float(np.random.uniform(6.0, 12.0))
        taper = float(np.random.uniform(0.22, 0.75))
        sweep = float(np.random.uniform(5.0, 35.0))
        dihedral = float(np.random.uniform(1.0, 5.0))
        twist_root = float(np.random.uniform(0.0, 3.0))
        twist_tip = float(np.random.uniform(-4.0, 0.0))

        # 2. Sample Airfoils (CST parameterization)
        # Choose a base NACA 4-digit code and perturb weights
        naca_root = f"{np.random.randint(0, 4)}{np.random.randint(2, 5)}{np.random.randint(12, 16):02d}"
        naca_tip = f"{np.random.randint(0, 3)}{np.random.randint(2, 5)}{np.random.randint(8, 12):02d}"
        
        af_root = CSTAirfoil3D.from_naca4(naca_root, order=6)
        af_tip = CSTAirfoil3D.from_naca4(naca_tip, order=6)

        # Small random perturbation to CST weights
        af_root.weights_upper += np.random.normal(0.0, 0.01, size=af_root.weights_upper.shape)
        af_root.weights_lower += np.random.normal(0.0, 0.01, size=af_root.weights_lower.shape)
        af_tip.weights_upper += np.random.normal(0.0, 0.01, size=af_tip.weights_upper.shape)
        af_tip.weights_lower += np.random.normal(0.0, 0.01, size=af_tip.weights_lower.shape)

        wing = Wing3D(
            span=span,
            aspect_ratio=ar,
            taper_ratio=taper,
            sweep_le_deg=sweep,
            dihedral_deg=dihedral,
            twist_root_deg=twist_root,
            twist_tip_deg=twist_tip,
            root_airfoil=af_root,
            tip_airfoil=af_tip,
            name=f"SampleWing_{i:04d}",
        )

        # 3. Sample Flight Condition
        alpha_deg = float(np.random.uniform(-1.0, 8.0))
        mach = float(np.random.uniform(0.3, 0.85))
        reynolds = float(10 ** np.random.uniform(6.0, 7.6))

        # 4. Evaluate with 3D AeroEngine
        engine = AeroEngine3D(wing, num_chordwise=8, num_spanwise=14)
        res = engine.evaluate(alpha_deg=alpha_deg, mach=mach, reynolds=reynolds)

        # Input vector: [wing_params (37), alpha, mach, log10_re] -> 40 dims
        flight_cond = np.array([alpha_deg, mach, np.log10(reynolds)])
        x_vec = np.concatenate([wing.to_parameter_vector(), flight_cond])

        # Target vector: [CL, CD, CDi, CDp, CDw, CM, L/D, e, fuel_vol] -> 9 dims
        y_vec = np.array([
            res.cl,
            res.cd,
            res.cd_induced,
            res.cd_profile,
            res.cd_wave,
            res.cm,
            res.l_over_d,
            res.span_efficiency,
            res.fuel_volume,
        ])

        x_list.append(x_vec)
        y_list.append(y_vec)

        if verbose and (i + 1) % 25 == 0:
            print(f"Generated {i + 1}/{num_samples} 3D wing aero samples...")

    return np.array(x_list), np.array(y_list)
