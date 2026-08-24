"""
Transonic and Supersonic Wave Drag Engine.
Calculates shock wave drag rise using the Korn-Mason formulation and supersonic Ackeret theory.
"""

from typing import Dict, Any
import numpy as np
from ..geometry.wing_3d import Wing3D


class WaveDragEngine3D:
    """
    Transonic Shock Drag and Supersonic Wave Drag Estimator.
    """

    def __init__(self, wing: Wing3D, korn_tech_factor: float = 0.95):
        self.wing = wing
        self.kappa_a = float(korn_tech_factor)  # 0.95 for supercritical, 0.87 for classical NACA

    def compute_wave_drag(
        self,
        cl: float,
        mach: float,
    ) -> Dict[str, Any]:
        """
        Computes the transonic/supersonic wave drag coefficient CDw.
        """
        m = float(mach)
        cl_val = max(float(cl), 0.0)

        # Average effective thickness-to-chord ratio
        root_tc = self.wing.root_airfoil.get_max_thickness()
        tip_tc = self.wing.tip_airfoil.get_max_thickness()
        avg_tc = 0.5 * (root_tc + tip_tc)

        cos_sweep = np.cos(np.radians(self.wing.sweep_le_deg))
        cos_sweep = max(cos_sweep, 0.2)

        # Korn-Mason Drag Divergence Mach Number M_dd:
        # M_dd = kappa_A / cos(sweep) - (t/c)_eff / cos^2(sweep) - 0.1 * CL / cos^3(sweep)
        # Standard 3D Korn formula:
        m_dd = (self.kappa_a / cos_sweep) - (avg_tc / (cos_sweep ** 2)) - (0.10 * cl_val / (cos_sweep ** 3))
        m_dd = float(np.clip(m_dd, 0.65, 1.25))

        # Critical Mach Number (onset of drag rise): M_crit ~ M_dd - 0.08
        m_crit = m_dd - 0.08

        if m <= m_crit:
            cd_wave = 0.0
        elif m < 1.0:
            # Transonic drag rise (Lock's 4th-power formulation)
            # Delta CDw = 20 * (M - M_crit)^4
            cd_wave = 20.0 * ((m - m_crit) ** 4)
        else:
            # Supersonic wave drag (Ackeret / Linearized supersonic airfoil theory)
            # CDw = (4 * alpha^2 + 4 * (t/c)^2) / sqrt(M^2 - 1)
            beta_super = np.sqrt(m ** 2 - 1.0 + 1e-4)
            cd_wave_supersonic = (4.0 * (avg_tc ** 2) + 0.05 * (cl_val ** 2)) / (beta_super * cos_sweep)
            cd_wave = float(cd_wave_supersonic)

        cd_wave = float(np.clip(cd_wave, 0.0, 0.5))

        return {
            "cd_wave": float(cd_wave),
            "m_crit": float(m_crit),
            "m_dd": float(m_dd),
        }
