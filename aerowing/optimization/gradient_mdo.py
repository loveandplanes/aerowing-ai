"""
Gradient-Based 3D Wing Shape Optimization using Neural Autograd.
Performs SQP shape refinement on the AI aerodynamic surrogate.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from scipy.optimize import minimize
from ..geometry.wing_3d import Wing3D
from ..models.surrogate_3d import AeroSurrogate3D


class GradientOptimizer3D:
    """
    Fast Gradient-Based SQP Shape Optimizer for 3D Wing Design.
    """

    def __init__(
        self,
        surrogate: AeroSurrogate3D,
        target_cl: float = 0.55,
        mach: float = 0.82,
        reynolds: float = 2.5e7,
    ):
        self.surrogate = surrogate
        self.target_cl = target_cl
        self.mach = mach
        self.reynolds = reynolds

    def optimize(
        self,
        initial_wing: Wing3D,
        max_iter: int = 40,
    ) -> Dict[str, Any]:
        """
        Runs SQP optimization to maximize L/D subject to CL >= target_cl.
        The objective gradient is computed analytically through the neural
        surrogate with PyTorch autograd (no numerical finite differences).
        """
        x0 = initial_wing.to_parameter_vector()
        planform_dim = 7

        bounds = []
        # Bounds for planform [span, AR, taper, sweep, dihedral, twist_r, twist_t]
        bounds.extend([
            (20.0, 45.0),
            (6.0, 14.0),
            (0.18, 0.60),
            (5.0, 38.0),
            (1.0, 6.0),
            (0.0, 4.0),
            (-5.0, 0.0),
        ])
        # CST parameter bounds
        for i in range(planform_dim, len(x0)):
            val = x0[i]
            bounds.append((val - 0.15, val + 0.15))

        def _flight_cond() -> np.ndarray:
            return np.array([2.5, self.mach, np.log10(max(self.reynolds, 1e4))])

        def objective(x: np.ndarray) -> float:
            res = self.surrogate.predict_wing(x, alpha_deg=2.5, mach=self.mach, reynolds=self.reynolds)
            # Penalty for missing target CL
            cl_penalty = 100.0 * max(0.0, self.target_cl - res["cl"]) ** 2
            # Minimize -L/D + penalty
            return -res["l_over_d"] + cl_penalty

        def objective_jac(x: np.ndarray) -> np.ndarray:
            """Analytic gradient via PyTorch autograd through the surrogate."""
            self.surrogate.eval()
            x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            flight_cond = torch.tensor(_flight_cond(), dtype=torch.float32)
            x_in = torch.cat([x_t, flight_cond]).unsqueeze(0)
            with torch.enable_grad():
                pred = self.surrogate(x_in)[0]  # [CL, CD, CDi, CDp, CDw, CM, L/D, e, fuel]
                cl_pred = pred[0]
                ld_pred = pred[6]
                cl_penalty = 100.0 * torch.clamp(self.target_cl - cl_pred, min=0.0) ** 2
                obj = -ld_pred + cl_penalty
                obj.backward()
            return x_t.grad.detach().cpu().numpy().astype(np.float64)

        res_opt = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )

        optimized_params = res_opt.x
        optimized_wing = Wing3D.from_parameter_vector(optimized_params, name="Opt_Wing_SQP")
        telemetry = self.surrogate.predict_wing(optimized_params, mach=self.mach, reynolds=self.reynolds)

        return {
            "optimized_wing": optimized_wing,
            "optimized_params": optimized_params,
            "telemetry": telemetry,
            "success": res_opt.success,
            "message": res_opt.message,
        }
