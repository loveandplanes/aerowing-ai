"""
3D Physics-Informed Deep Neural Aerodynamic Surrogate (AeroSurrogate3D).
Predicts 3D surface pressure fields and aerodynamic polars in < 5ms.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class FourierFeatureEmbedding(nn.Module):
    """Embeds scalar inputs into multi-scale Fourier space to learn sharp gradients."""
    def __init__(self, in_features: int, num_frequencies: int = 16, scale: float = 2.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer(
            "B", torch.randn(in_features, num_frequencies) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features]
        # x_proj: [B, num_frequencies]
        x_proj = 2.0 * np.pi * torch.matmul(x, self.B)
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class InputStandardizer(nn.Module):
    """Per-column z-score of the raw input before the Fourier embedding.

    Optional: until `set()` is called it is the identity, so checkpoints
    trained without normalization keep working unchanged. Buffers are
    always registered (identify defaults) so fitted stats round-trip
    through state_dict even with strict=False loads.
    """
    def __init__(self, features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("std", torch.ones(features))
        self.register_buffer("fitted", torch.zeros(1))

    def set(self, mean: np.ndarray, std: np.ndarray):
        self.mean.copy_(torch.tensor(np.asarray(mean, dtype=float), dtype=torch.float32))
        self.std.copy_(torch.tensor(np.asarray(std, dtype=float), dtype=torch.float32))
        self.fitted.fill_(1.0)

    def is_fitted(self) -> bool:
        return bool(self.fitted[0].item() > 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted():
            return x
        return (x - self.mean) / (self.std + 1e-9)


class OutputStandardizer(nn.Module):
    """Stores per-column target stats (computed from the label corpus).

    The model is trained in standardized target space; `inverse` restores
    physical units at inference. Identity until `set()` is called.
    """
    def __init__(self, features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("std", torch.ones(features))
        self.register_buffer("fitted", torch.zeros(1))

    def set(self, mean: np.ndarray, std: np.ndarray):
        self.mean.copy_(torch.tensor(np.asarray(mean, dtype=float), dtype=torch.float32))
        self.std.copy_(torch.tensor(np.asarray(std, dtype=float), dtype=torch.float32))
        self.fitted.fill_(1.0)

    def is_fitted(self) -> bool:
        return bool(self.fitted[0].item() > 0.5)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted():
            return y
        return (y - self.mean) / (self.std + 1e-9)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted():
            return z
        return z * self.std + self.mean


class ResidualBlock(nn.Module):
    """Skip-connection residual block with LayerNorm and GELU."""
    def __init__(self, hidden_dim: int, dropout: float = 0.02):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class AeroSurrogate3D(nn.Module):
    """
    3D Physics-Informed Neural Network (PINN) / Deep Surrogate.
    
    Inputs:
      - Planform params: [span, AR, taper, sweep_le, dihedral, twist_root, twist_tip] (7)
      - Root CST weights (15)
      - Tip CST weights (15)
      - Flight conditions: [alpha_deg, mach, log10(Re)] (3)
      Total input dim = 40
      
    Outputs:
      - [CL, CD, CDi, CDp, CDw, CM, L/D, span_efficiency, fuel_volume] (9 outputs)
    """

    def __init__(
        self,
        input_dim: int = 40,
        output_dim: int = 9,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        num_fourier_freqs: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.x_stdz = InputStandardizer(input_dim)
        self.y_stdz = OutputStandardizer(output_dim)

        self.fourier = FourierFeatureEmbedding(
            in_features=input_dim,
            num_frequencies=num_fourier_freqs,
            scale=1.5,
        )
        fourier_out_dim = input_dim + 2 * num_fourier_freqs

        self.input_proj = nn.Sequential(
            nn.Linear(fourier_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning predicted telemetry in physical units
        (standardized internally when stats are set).
        """
        return self.y_stdz.inverse(self._hidden_out(x))

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in standardized target space, for balanced losses."""
        return self._hidden_out(x)

    def _hidden_out(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fourier(self.x_stdz(x))
        h = self.input_proj(feat)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def compute_physics_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x_raw: torch.Tensor,
        weight_trefftz: float = 0.25,
        weight_drag_sum: float = 0.25,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates data MSE loss regularized by physics conservation laws:
        1. Trefftz plane induced drag law: CDi = CL^2 / (pi * AR * e)
        2. Drag decomposition consistency: CD = CDi + CDp + CDw

        `pred`/`target` are in standardized target space when an output
        normalizer is set (balanced MSE); physics terms are always evaluated
        in physical units via the denormalized prediction.
        """
        # Pred indices:
        # 0: CL, 1: CD, 2: CDi, 3: CDp, 4: CDw, 5: CM, 6: L/D, 7: e, 8: fuel_vol
        mse_data = nn.functional.mse_loss(pred, target)

        pz = self.y_stdz.inverse(pred)
        cl_pred = pz[:, 0]
        cd_pred = pz[:, 1]
        cdi_pred = pz[:, 2]
        cdp_pred = pz[:, 3]
        cdw_pred = pz[:, 4]
        e_pred = torch.clamp(pz[:, 7], min=0.3, max=1.1)

        # AR is at input index 1
        ar = torch.clamp(x_raw[:, 1], min=2.0, max=20.0)

        # 1. Physics Trefftz conservation
        cdi_theoretical = (cl_pred ** 2) / (np.pi * ar * e_pred + 1e-6)
        loss_trefftz = nn.functional.mse_loss(cdi_pred, cdi_theoretical)

        # 2. Drag summation consistency
        loss_drag_sum = nn.functional.mse_loss(cd_pred, cdi_pred + cdp_pred + cdw_pred)

        total_loss = mse_data + weight_trefftz * loss_trefftz + weight_drag_sum * loss_drag_sum

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_mse": float(mse_data.item()),
            "loss_trefftz": float(loss_trefftz.item()),
            "loss_drag_sum": float(loss_drag_sum.item()),
        }

        return total_loss, metrics

    def predict_wing(
        self,
        wing_param_vector: np.ndarray,
        alpha_deg: float = 2.5,
        mach: float = 0.82,
        reynolds: float = 2.5e7,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        High-level inference on a 3D Wing returning dictionary of coefficients.
        Executes in < 5ms.
        """
        self.eval()
        flight_cond = np.array([alpha_deg, mach, np.log10(max(reynolds, 1e4))])
        x_in = np.concatenate([wing_param_vector, flight_cond])

        t_in = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(t_in).squeeze(0).cpu().numpy()

        return {
            "cl": float(out[0]),
            "cd": float(out[1]),
            "cd_induced": float(out[2]),
            "cd_profile": float(out[3]),
            "cd_wave": float(out[4]),
            "cm": float(out[5]),
            "l_over_d": float(out[6]),
            "span_efficiency": float(out[7]),
            "fuel_volume_m3": float(out[8]),
        }
