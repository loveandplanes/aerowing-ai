"""
3D Generative Wing Conditional VAE (GenerativeWingVAE3D).
Synthesizes novel 3D wing geometries conditioned on aerodynamic flight mission targets.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class GenerativeWingVAE3D(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for 3D Wing Inverse Design.
    
    Condition Vector c (5 dims):
      [target_cl, target_mach, target_ar, target_l_over_d, target_log10_re]
      
    Geometry Vector x (37 dims):
      [span, AR, taper, sweep_le, dihedral, twist_root, twist_tip, root_cst (15), tip_cst (15)]
    """

    def __init__(
        self,
        geom_dim: int = 37,
        cond_dim: int = 5,
        latent_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (192, 192, 128),
    ):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        # Encoder: (geom + cond) -> [mu, log_var]
        enc_in = geom_dim + cond_dim
        enc_layers = []
        in_d = enc_in
        for h_d in hidden_dims:
            enc_layers.append(nn.Linear(in_d, h_d))
            enc_layers.append(nn.LayerNorm(h_d))
            enc_layers.append(nn.GELU())
            in_d = h_d
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder: (latent + cond) -> reconstructed geom
        dec_in = latent_dim + cond_dim
        dec_layers = []
        in_d = dec_in
        for h_d in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_d, h_d))
            dec_layers.append(nn.LayerNorm(h_d))
            dec_layers.append(nn.GELU())
            in_d = h_d
        dec_layers.append(nn.Linear(hidden_dims[0], geom_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xc = torch.cat([x, c], dim=-1)
        h = self.encoder_body(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zc = torch.cat([z, c], dim=-1)
        return self.decoder(zc)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

    def compute_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_kld: float = 0.005,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes Reconstruction MSE + KL-Divergence loss."""
        recon_loss = nn.functional.mse_loss(recon_x, x)
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        total_loss = recon_loss + beta_kld * kld_loss

        return total_loss, {
            "loss_vae": float(total_loss.item()),
            "loss_recon": float(recon_loss.item()),
            "loss_kld": float(kld_loss.item()),
        }

    def generate(
        self,
        target_cl: float = 0.55,
        target_mach: float = 0.82,
        target_ar: float = 9.5,
        target_l_over_d: float = 19.0,
        target_reynolds: float = 2.5e7,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Synthesizes a new 3D wing parameter vector from aerodynamic targets.
        """
        self.eval()
        cond = np.array([
            target_cl,
            target_mach,
            target_ar,
            target_l_over_d,
            np.log10(max(target_reynolds, 1e4)),
        ], dtype=np.float32)

        t_cond = torch.tensor(cond, device=device).unsqueeze(0)
        z = torch.randn(1, self.latent_dim, device=device)

        with torch.no_grad():
            synth_x = self.decode(z, t_cond).squeeze(0).cpu().numpy()

        # Enforce physical positivity on geometry
        synth_x[0] = max(synth_x[0], 2.0)    # span >= 2m
        synth_x[1] = max(synth_x[1], 3.0)    # AR >= 3
        synth_x[2] = np.clip(synth_x[2], 0.15, 1.0) # taper in [0.15, 1.0]
        synth_x[3] = np.clip(synth_x[3], 0.0, 45.0) # sweep in [0, 45 deg]

        return synth_x
