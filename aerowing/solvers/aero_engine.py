"""
Unified Multi-Fidelity 3D Aerodynamic Evaluation Engine.
Coordinates 3D Vortex Lattice Method, Viscous Boundary-Layer, and Transonic Wave Drag solvers.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..geometry.wing_3d import Wing3D
from .vlm_3d import VLMSolver3D
from .viscous_3d import ViscousEngine3D
from .wave_drag import WaveDragEngine3D


@dataclass
class AeroResult3D:
    """Complete 3D aerodynamic polar and structural telemetry."""
    cl: float
    cd: float
    cd_induced: float
    cd_profile: float
    cd_wave: float
    cm: float
    l_over_d: float
    span_efficiency: float
    fuel_volume: float
    wetted_area: float
    etas: np.ndarray
    cl_spanwise: np.ndarray
    delta_cp_matrix: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cl": round(float(self.cl), 5),
            "cd": round(float(self.cd), 5),
            "cd_induced": round(float(self.cd_induced), 5),
            "cd_profile": round(float(self.cd_profile), 5),
            "cd_wave": round(float(self.cd_wave), 5),
            "cm": round(float(self.cm), 5),
            "l_over_d": round(float(self.l_over_d), 2),
            "span_efficiency": round(float(self.span_efficiency), 4),
            "fuel_volume_m3": round(float(self.fuel_volume), 2),
            "wetted_area_m2": round(float(self.wetted_area), 2),
            "etas": self.etas.tolist(),
            "cl_spanwise": self.cl_spanwise.tolist(),
            "delta_cp_matrix": self.delta_cp_matrix.tolist(),
        }


class AeroEngine3D:
    """
    Unified Multi-Fidelity Aerodynamic Evaluation Pipeline.
    """

    # Calibrated transonic lift correction (see _transonic_lift): fitted on
    # SU2 RANS training cases only; holdout CL RMSE 0.018 vs 0.189 uncorrected.
    TRANS_LIFT_A0 = 0.7999
    TRANS_LIFT_A1 = 0.3783

    # Calibrated transonic drag augmentation (see _transonic_drag): adds the
    # wave-drag and profile-drag deficit missing from the inviscid + boundary
    # layer model at M 0.76-0.85. Wave drag scales with CL^2 by physics, so
    # there is no constant term: c0=0 (the previous 0.01406 flat offset was
    # an artifact of fitting SU2 RANS cases with a skin-friction deficit and
    # double-counted the profile drag already supplied by the viscous model).
    # c1=0.15071 kept on holdout-RANS fit; NOT refit against the M6 polar in
    # isolation because the engine's transonic lift correction still predicts
    # CL ~0.16 vs the experimental 0.26 at M6 (see benchmarks docstring), so
    # a drag-only refit would merely compensate for that lift shortfall.
    TRANS_DRAG_C0 = 0.0
    TRANS_DRAG_C1 = 0.15071

    def __init__(
        self,
        wing: Wing3D,
        num_chordwise: int = 12,
        num_spanwise: int = 24,
        transonic_lift_correction: bool = True,
        transonic_drag_correction: bool = True,
    ):
        self.wing = wing
        self.transonic_lift_correction = bool(transonic_lift_correction)
        self.transonic_drag_correction = bool(transonic_drag_correction)
        self.vlm = VLMSolver3D(wing, num_chordwise=num_chordwise, num_spanwise=num_spanwise)
        self.viscous = ViscousEngine3D(wing)
        self.wave = WaveDragEngine3D(wing)

    def _pg_beta(self, mach: float) -> float:
        """Effective Prandtl-Glauert beta exactly as VLMSolver3D.solve uses it."""
        if mach > 0.0:
            beta_lim = np.sqrt(1.0 - self.vlm.pg_limit_mach ** 2)
            return max(np.sqrt(1.0 - mach ** 2), beta_lim)
        return 1.0

    @staticmethod
    def _transonic_lift(cl_incomp: float, a0: float, a1: float) -> float:
        """Maps incompressible VLM lift to RANS-level transonic lift.

        CL_rans = a0 * CL_incomp / (1 - a1 * CL_incomp), i.e. a lift loss
        factor h = a0 / (1 - a1 * CL_incomp) that starts ~0.80 at low lift
        (shock losses at M 0.78-0.85) and recovers toward 1 as lift builds,
        as observed across the SU2 RANS corpus. The rational form stays
        well-behaved for CL_incomp < 1.
        """
        den = max(1.0 - a1 * cl_incomp, 0.2)
        return float(a0 * cl_incomp / den)

    @staticmethod
    def _transonic_drag(cl: float, mach: float, c0: float, c1: float) -> float:
        """Transonic drag augmentation: dCD = c0 + c1 * M * CL^2.

        Only the c1 * M * CL^2 term is physical (wave drag rises with
        transonic lift); there is no constant term, so c0 must stay zero.
        Calibrated at M 0.76-0.85; blended out below M = 0.78 so subsonic
        cases are untouched.
        """
        assert c0 >= 0.0 and c1 >= 0.0, "drag augmentation constants must be non-negative"
        return float(c0 + c1 * mach * cl * cl)

    def evaluate(
        self,
        alpha_deg: float = 2.5,
        mach: float = 0.82,
        reynolds: float = 2.5e7,
        beta_deg: float = 0.0,
    ) -> AeroResult3D:
        """
        Executes coupled multi-fidelity analysis:
        1. 3D VLM for lift, induced drag, and circulation
        2. Viscous boundary-layer for Reynolds-dependent skin friction & form drag
        3. Korn-Mason transonic wave drag
        4. Internal fuel volume and wetted area
        """
        # 1. Potential flow solution (VLM)
        vlm_res = self.vlm.solve(alpha_deg=alpha_deg, mach=mach, beta_deg=beta_deg)

        # 2. Viscous drag solution
        visc_res = self.viscous.compute_profile_drag(reynolds_mac=reynolds, mach=mach)

        # Transonic lift correction (calibrated against SU2 RANS). Blended in
        # over M = 0.70-0.78 so subsonic/compressible-subcritical cases keep
        # the raw Prandtl-Glauert lift, and fully applied by M = 0.78.
        cl_pg = vlm_res["cl"]
        if self.transonic_lift_correction:
            cl_incomp = cl_pg * self._pg_beta(mach)
            cl_corr = self._transonic_lift(cl_incomp, self.TRANS_LIFT_A0,
                                           self.TRANS_LIFT_A1)
            w = float(np.clip((mach - 0.70) / (0.78 - 0.70), 0.0, 1.0))
            cl = cl_pg + w * (cl_corr - cl_pg)
            den = cl_pg if abs(cl_pg) > 1e-9 else float(np.copysign(1e-9, cl_pg))
            ratio = cl / den
            cm = vlm_res["cm"] * ratio
        else:
            cl, cm = cl_pg, vlm_res["cm"]

        # 3. Wave drag solution (fed with the corrected lift)
        wave_res = self.wave.compute_wave_drag(cl=cl, mach=mach)

        # Total drag sum: CD = CDi + CDp + CDw (+ calibrated transonic aug).
        # CDi is far-field, proportional to CL^2: when the transonic lift blend
        # changes cl away from the raw PG value, the induced drag must follow
        # (cl/cl_pg)^2 so the reported (cl, cd_induced) pair stays consistent
        # (span efficiency implied by the pair matches the VLM value at all M).
        cd_i = vlm_res["cd_induced"]
        if self.transonic_lift_correction and abs(cl_pg) > 1e-9:
            cd_i = cd_i * (cl / cl_pg) ** 2
        cd_p = visc_res["cd_profile"]
        cd_w = wave_res["cd_wave"]
        if self.transonic_drag_correction:
            w = float(np.clip((mach - 0.70) / (0.78 - 0.70), 0.0, 1.0))
            cd_w = cd_w + w * self._transonic_drag(cl, mach,
                                                   self.TRANS_DRAG_C0,
                                                   self.TRANS_DRAG_C1)
        cd_total = cd_i + cd_p + cd_w

        # Aerodynamic efficiency L/D
        l_over_d = cl / max(cd_total, 1e-5)

        # Geometric metrics
        fuel_vol = self.wing.compute_internal_fuel_volume()
        wet_area = self.wing.compute_wetted_area()

        return AeroResult3D(
            cl=cl,
            cd=cd_total,
            cd_induced=cd_i,
            cd_profile=cd_p,
            cd_wave=cd_w,
            cm=cm,
            l_over_d=l_over_d,
            span_efficiency=vlm_res["span_efficiency"],
            fuel_volume=fuel_vol,
            wetted_area=wet_area,
            etas=vlm_res["etas_spanwise"],
            cl_spanwise=vlm_res["cl_spanwise"],
            delta_cp_matrix=vlm_res["delta_cp_matrix"],
        )
