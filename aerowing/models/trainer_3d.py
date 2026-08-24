"""
Neural Model Training Pipeline for AeroWing AI Pro.
Trains AeroSurrogate3D with physics constraints and GenerativeWingVAE3D with KL annealing.
"""

from typing import Dict, Any, Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .surrogate_3d import AeroSurrogate3D
from .generator_3d import GenerativeWingVAE3D
from .dataset_3d import WingDataset3D, generate_synthetic_wing_dataset


class AeroTrainer3D:
    """
    Unified Training Orchestrator for 3D Neural Surrogate and 3D Generative VAE.
    """

    def __init__(
        self,
        surrogate: Optional[AeroSurrogate3D] = None,
        generator: Optional[GenerativeWingVAE3D] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.surrogate = surrogate.to(self.device) if surrogate else AeroSurrogate3D().to(self.device)
        self.generator = generator.to(self.device) if generator else GenerativeWingVAE3D().to(self.device)

    def train_surrogate(
        self,
        dataset: WingDataset3D,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        physics_weight: float = 0.25,
        val_split: float = 0.15,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Trains the physics-informed 3D aerodynamic surrogate.

        Input and target normalizers are fitted on the dataset (input z-score
        before the Fourier embedding; standardized-target MSE so every column
        contributes comparably instead of being dominated by fuel volume).
        Physics terms are always evaluated on denormalized predictions.
        """
        if not isinstance(dataset, WingDataset3D):
            dataset = WingDataset3D(
                np.asarray(dataset[0], dtype=float),
                np.asarray(dataset[1], dtype=float),
            )
        x_arr = np.asarray(dataset.x_data.numpy(), dtype=float)
        y_arr = np.asarray(dataset.y_data.numpy(), dtype=float)
        xm = x_arr.mean(axis=0); xs = x_arr.std(axis=0)
        xs[xs < 1e-9 * max(float(xs.max()), 1e-9)] = 1.0
        ym = y_arr.mean(axis=0); ys = y_arr.std(axis=0)
        ys[ys < 1e-9 * max(float(ys.max()), 1e-9)] = 1.0
        self.surrogate.x_stdz.set(xm, xs)
        self.surrogate.y_stdz.set(ym, ys)

        val_size = max(int(len(dataset) * val_split), 2)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.surrogate.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "val_loss": [], "loss_trefftz": []}

        self.surrogate.train()
        for epoch in range(epochs):
            total_train_loss = 0.0
            total_trefftz_loss = 0.0

            for x_b, y_b in train_loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                pred = self.surrogate.forward_raw(x_b)
                loss, metrics = self.surrogate.compute_physics_loss(
                    pred, y_b, x_b, weight_trefftz=physics_weight, weight_drag_sum=physics_weight
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.surrogate.parameters(), max_norm=2.0)
                optimizer.step()

                total_train_loss += metrics["loss_total"] * len(x_b)
                total_trefftz_loss += metrics["loss_trefftz"] * len(x_b)

            scheduler.step()
            train_loss = total_train_loss / len(train_ds)
            trefftz_loss = total_trefftz_loss / len(train_ds)

            # Validation
            self.surrogate.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    p_val = self.surrogate.forward_raw(x_val)
                    v_loss, _ = self.surrogate.compute_physics_loss(p_val, y_val, x_val)
                    total_val_loss += v_loss.item() * len(x_val)

            val_loss = total_val_loss / len(val_ds)
            self.surrogate.train()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["loss_trefftz"].append(trefftz_loss)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == epochs - 1):
                print(f"[Surrogate Epoch {epoch+1:03d}/{epochs}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Trefftz Loss: {trefftz_loss:.5f}")

        return history

    def train_generator(
        self,
        dataset: WingDataset3D,
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 8e-4,
        beta_kld: float = 0.005,
        kl_anneal: bool = True,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Trains the 3D Conditional VAE inverse wing generator.

        With kl_anneal=True the KL weight is cyclically annealed: beta ramps
        from 0 up to beta_kld across each annealing cycle, letting the
        generator first focus on reconstruction before progressively
        regularizing the latent space.
        """
        if not isinstance(dataset, WingDataset3D):
            dataset = WingDataset3D(
                np.asarray(dataset[0], dtype=float),
                np.asarray(dataset[1], dtype=float),
            )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.generator.parameters(), lr=lr)

        cycle_length = max(epochs // 4, 5)
        history = {"loss_vae": [], "loss_recon": [], "loss_kld": [], "beta_kld": []}
        self.generator.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_kld = 0.0

            # Cyclic KL annealing: ramp beta from 0 to beta_kld each cycle
            if kl_anneal:
                cycle_pos = (epoch % cycle_length) / cycle_length
                beta_epoch = beta_kld * min(1.0, cycle_pos * 2.0)
            else:
                beta_epoch = beta_kld

            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)

                # Geometry vector (first 37 elements of x_b)
                x_geom = x_b[:, :37]
                # Condition vector: [CL, Mach, AR, L/D, log10_Re]
                # y_b[0] = CL, x_b[38] = Mach, x_b[1] = AR, y_b[6] = L/D, x_b[39] = log10_Re
                cond = torch.stack([
                    y_b[:, 0],
                    x_b[:, 38],
                    x_b[:, 1],
                    y_b[:, 6],
                    x_b[:, 39],
                ], dim=-1)

                optimizer.zero_grad()
                recon_x, mu, logvar = self.generator(x_geom, cond)
                loss, metrics = self.generator.compute_loss(recon_x, x_geom, mu, logvar, beta_kld=beta_epoch)
                loss.backward()
                optimizer.step()

                total_loss += metrics["loss_vae"] * len(x_b)
                total_recon += metrics["loss_recon"] * len(x_b)
                total_kld += metrics["loss_kld"] * len(x_b)

            history["loss_vae"].append(total_loss / len(dataset))
            history["loss_recon"].append(total_recon / len(dataset))
            history["loss_kld"].append(total_kld / len(dataset))
            history["beta_kld"].append(float(beta_epoch))

            if verbose and ((epoch + 1) % 10 == 0 or epoch == epochs - 1):
                print(f"[VAE Epoch {epoch+1:03d}/{epochs}] Loss: {total_loss/len(dataset):.5f} | Recon: {total_recon/len(dataset):.5f} | KLD: {total_kld/len(dataset):.5f} | beta: {beta_epoch:.5f}")

        return history

    def save_checkpoint(self, path: str):
        """Saves model weights to checkpoint directory."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "surrogate_state": self.surrogate.state_dict(),
            "generator_state": self.generator.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Loads model weights from checkpoint directory."""
        checkpoint = torch.load(path, map_location=self.device)
        self.surrogate.load_state_dict(checkpoint["surrogate_state"])
        self.generator.load_state_dict(checkpoint["generator_state"])
