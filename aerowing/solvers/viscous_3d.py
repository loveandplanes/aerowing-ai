"""
3D Viscous Boundary-Layer and Skin Friction Drag Engine.
Calculates Reynolds-dependent, compressible skin friction and Hoerner form-factor pressure drag.
"""

from typing import Dict, Any
import numpy as np
from ..geometry.wing_3d import Wing3D


class ViscousEngine3D:
    """
    3D Compressible Viscous Boundary-Layer and Form-Factor Drag Engine.
    """

    def __init__(self, wing: Wing3D):
        self.wing = wing

    def compute_skin_friction_coefficient(
        self,
        reynolds: float,
        mach: float = 0.0,
        turbulent_fraction: float = 0.95,
    ) -> float:
        """
        Computes flat-plate equivalent skin friction coefficient Cf
        with van Driest II compressibility correction.
        """
        re = max(float(reynolds), 1e4)
        m = max(float(mach), 0.0)

        # Incompressible turbulent Cf (Schoenherr / Prandtl-Schlichting formula)
        cf_turb_incomp = 0.455 / (np.log10(re) ** 2.58)

        # Laminar Cf (Blasius)
        cf_lam_incomp = 1.328 / np.sqrt(re)

        # Combined incompressible Cf based on transition location
        cf_incomp = (1.0 - turbulent_fraction) * cf_lam_incomp + turbulent_fraction * cf_turb_incomp

        # Compressibility correction (van Driest II / Sommer & Short T-prime factor)
        gamma = 1.4
        t_ratio = 1.0 + 0.178 * (m ** 2)  # Recovery temperature ratio
        cf_comp = cf_incomp / (t_ratio ** 0.65)

        return float(cf_comp)

    def compute_profile_drag(
        self,
        reynolds_mac: float = 2.5e7,
        mach: float = 0.0,
        num_stations: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculates spanwise profile drag integral including DATCOM 3D form factor:
        CDp = (2 / S_ref) * int_0^{b/2} Cf(y) * (1 + k_sweep(y)) * c(y) dy
        """
        etas = np.linspace(0.0, 1.0, num_stations)
        y_coords = etas * self.wing.semi_span
        chords = []
        cf_values = []
        cdp_sectional = []

        cos_sweep = np.cos(np.radians(self.wing.sweep_le_deg))

        for eta in etas:
            sec = self.wing.get_interpolated_section(eta)
            tc = sec.airfoil.get_max_thickness()
            chord = sec.chord

            # Sectional Reynolds number
            re_sec = reynolds_mac * (chord / self.wing.mac)
            cf = self.compute_skin_friction_coefficient(re_sec, mach=mach)

            # DATCOM / Hoerner 3D Form factor k
            # k = 2*(t/c) + 60*(t/c)^4
            form_factor_2d = 1.0 + 2.0 * tc + 60.0 * (tc ** 4)
            # 3D sweep reduction factor
            form_factor_3d = 1.0 + (form_factor_2d - 1.0) * (cos_sweep ** 2)

            # Sectional profile drag: c_dp = 2 * Cf * (1 + k)
            cdp_sec = 2.0 * cf * form_factor_3d

            chords.append(chord)
            cf_values.append(cf)
            cdp_sectional.append(cdp_sec)

        chords = np.array(chords)
        cdp_sectional = np.array(cdp_sectional)

        # Integrate along span: Total CDp = (2 / S_ref) * int cdp(y) * c(y) dy
        integrand = cdp_sectional * chords
        semi_drag_area = np.trapezoid(integrand, y_coords)
        cd_profile_total = (2.0 / self.wing.s_ref) * semi_drag_area

        return {
            "cd_profile": float(cd_profile_total),
            "etas": etas,
            "cf_mean": float(np.mean(cf_values)),
            "cdp_sectional": cdp_sectional,
        }
