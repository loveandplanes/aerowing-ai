"""
3D Parametric Wing Geometry Engine.
Supports multi-section swept, tapered, twisted, and non-planar aerospace wings.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from .cst_3d import CSTAirfoil3D


@dataclass
class WingSection:
    """A defining spanwise station along the semi-span."""
    y_fraction: float        # Normalized spanwise station eta = 2*y / b in [0, 1]
    chord: float             # Local chord length (m)
    x_le: float              # Leading edge x coordinate (m)
    z_le: float              # Leading edge z coordinate (m)
    twist_deg: float         # Geometric twist angle (deg)
    airfoil: CSTAirfoil3D    # Cross-sectional aerodynamic profile


class Wing3D:
    """
    Parametric 3D Aerospace Wing representation.
    
    Includes automated planform sizing, multi-station lofting,
    structural fuel volume computation, and 3D surface mesh generation.
    """

    def __init__(
        self,
        span: float = 30.0,
        aspect_ratio: float = 9.5,
        taper_ratio: float = 0.28,
        sweep_le_deg: float = 27.5,
        dihedral_deg: float = 3.5,
        twist_root_deg: float = 2.0,
        twist_tip_deg: float = -2.5,
        root_airfoil: Optional[CSTAirfoil3D] = None,
        tip_airfoil: Optional[CSTAirfoil3D] = None,
        name: str = "AeroWing_3D",
        sections: Optional[List[WingSection]] = None,
    ):
        self.name = name
        self.span = float(span)
        self.aspect_ratio = float(aspect_ratio)
        self.taper_ratio = float(taper_ratio)
        self.sweep_le_deg = float(sweep_le_deg)
        self.dihedral_deg = float(dihedral_deg)
        self.twist_root_deg = float(twist_root_deg)
        self.twist_tip_deg = float(twist_tip_deg)

        # Planform Reference Area: S = b^2 / AR
        self.s_ref = (self.span ** 2) / self.aspect_ratio
        self.semi_span = self.span / 2.0

        # Root chord & Tip chord from trapezoidal relations:
        # S = 2 * (b/2) * (c_root + c_tip) / 2 = (b/2) * c_root * (1 + lambda)
        self.root_chord = (2.0 * self.s_ref) / (self.span * (1.0 + self.taper_ratio))
        self.tip_chord = self.root_chord * self.taper_ratio

        # Mean Aerodynamic Chord (MAC): c_bar = (2/3) * c_root * (1 + lambda + lambda^2) / (1 + lambda)
        lam = self.taper_ratio
        self.mac = (2.0 / 3.0) * self.root_chord * ((1.0 + lam + lam**2) / (1.0 + lam))
        self.y_mac = (self.span / 6.0) * ((1.0 + 2.0 * lam) / (1.0 + lam))

        # Default standard profiles if not provided
        if root_airfoil is None:
            self.root_airfoil = CSTAirfoil3D.from_naca4("0014", order=6)
        else:
            self.root_airfoil = root_airfoil

        if tip_airfoil is None:
            self.tip_airfoil = CSTAirfoil3D.from_naca4("0010", order=6)
        else:
            self.tip_airfoil = tip_airfoil

        # Initialize defining stations
        if sections is not None:
            self.sections = sorted(sections, key=lambda s: s.y_fraction)
        else:
            self._build_default_sections()

    def _build_default_sections(self):
        """Constructs Root and Tip stations with linear planform variation."""
        tan_sweep = np.tan(np.radians(self.sweep_le_deg))
        tan_dihedral = np.tan(np.radians(self.dihedral_deg))

        # Root station (eta = 0.0)
        sec_root = WingSection(
            y_fraction=0.0,
            chord=self.root_chord,
            x_le=0.0,
            z_le=0.0,
            twist_deg=self.twist_root_deg,
            airfoil=self.root_airfoil,
        )

        # Tip station (eta = 1.0)
        sec_tip = WingSection(
            y_fraction=1.0,
            chord=self.tip_chord,
            x_le=self.semi_span * tan_sweep,
            z_le=self.semi_span * tan_dihedral,
            twist_deg=self.twist_tip_deg,
            airfoil=self.tip_airfoil,
        )

        self.sections = [sec_root, sec_tip]

    def get_interpolated_section(self, eta: float) -> WingSection:
        """
        Interpolates chord, leading-edge coordinates, twist, and CST profile
        at any spanwise location eta = 2y / b in [0, 1].
        """
        eta = float(np.clip(eta, 0.0, 1.0))
        y_fracs = [s.y_fraction for s in self.sections]

        # Handle boundary cases
        if eta <= y_fracs[0]:
            return self.sections[0]
        if eta >= y_fracs[-1]:
            return self.sections[-1]

        # Find enclosing stations
        for i in range(len(self.sections) - 1):
            s0 = self.sections[i]
            s1 = self.sections[i + 1]
            if s0.y_fraction <= eta <= s1.y_fraction:
                t = (eta - s0.y_fraction) / (s1.y_fraction - s0.y_fraction)
                chord = (1.0 - t) * s0.chord + t * s1.chord
                x_le = (1.0 - t) * s0.x_le + t * s1.x_le
                z_le = (1.0 - t) * s0.z_le + t * s1.z_le
                twist = (1.0 - t) * s0.twist_deg + t * s1.twist_deg

                # CST weight interpolation
                wu = (1.0 - t) * s0.airfoil.weights_upper + t * s1.airfoil.weights_upper
                wl = (1.0 - t) * s0.airfoil.weights_lower + t * s1.airfoil.weights_lower
                dz_te = (1.0 - t) * s0.airfoil.dz_te + t * s1.airfoil.dz_te
                airfoil = CSTAirfoil3D(weights_upper=wu, weights_lower=wl, dz_te=dz_te)

                return WingSection(
                    y_fraction=eta,
                    chord=chord,
                    x_le=x_le,
                    z_le=z_le,
                    twist_deg=twist,
                    airfoil=airfoil,
                )

        return self.sections[-1]

    def compute_internal_fuel_volume(self, num_stations: int = 50) -> float:
        """
        Calculates the internal wing tank volume (m^3) across both semi-spans
        by integrating cross-sectional area over span.
        Assumes usable tank occupies 85% of total internal volume (rib/spar structure allowance).
        """
        etas = np.linspace(0.0, 0.85, num_stations)  # Tanks typically end at 85% semi-span
        y_coords = etas * self.semi_span
        areas = []

        for eta in etas:
            sec = self.get_interpolated_section(eta)
            # Area in dimensional units: A_section = area_norm * chord^2
            norm_area = sec.airfoil.get_cross_sectional_area()
            areas.append(norm_area * (sec.chord ** 2))

        # Trapezoidal integration along semi-span * 2 (both wings) * 0.85 structural efficiency
        semi_volume = np.trapezoid(areas, y_coords)
        total_fuel_volume = 2.0 * semi_volume * 0.85
        return float(total_fuel_volume)

    def compute_wetted_area(self, num_stations: int = 50) -> float:
        """Computes total 3D wetted area S_wet (m^2) for upper & lower surfaces."""
        etas = np.linspace(0.0, 1.0, num_stations)
        y_coords = etas * self.semi_span
        perimeters = []

        for eta in etas:
            sec = self.get_interpolated_section(eta)
            x, zu, zl = sec.airfoil.evaluate(num_points=100)
            # Arc lengths of upper and lower curves
            dx = np.diff(x) * sec.chord
            dzu = np.diff(zu) * sec.chord
            dzl = np.diff(zl) * sec.chord
            p_upper = np.sum(np.sqrt(dx**2 + dzu**2))
            p_lower = np.sum(np.sqrt(dx**2 + dzl**2))
            perimeters.append(p_upper + p_lower)

        semi_wetted_area = np.trapezoid(perimeters, y_coords)
        return float(2.0 * semi_wetted_area)

    def generate_surface_mesh_3d(
        self,
        num_chordwise: int = 40,
        num_spanwise: int = 40,
        symmetric: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generates structured 3D surface coordinates (X, Y, Z) and normal vectors.
        Returns full lofted geometry for CAD export, 3D WebGL rendering, and CFD.
        """
        etas = np.linspace(0.0, 1.0, num_spanwise)
        
        # Chordwise parameterization with cosine clustering at LE and TE
        beta = np.linspace(0.0, np.pi, num_chordwise)
        xi = 0.5 * (1.0 - np.cos(beta))

        # Arrays for right semi-span (y >= 0)
        X_upper = np.zeros((num_spanwise, num_chordwise))
        Y_upper = np.zeros((num_spanwise, num_chordwise))
        Z_upper = np.zeros((num_spanwise, num_chordwise))

        X_lower = np.zeros((num_spanwise, num_chordwise))
        Y_lower = np.zeros((num_spanwise, num_chordwise))
        Z_lower = np.zeros((num_spanwise, num_chordwise))

        for i, eta in enumerate(etas):
            sec = self.get_interpolated_section(eta)
            y_val = eta * self.semi_span

            # Evaluate normalized airfoil
            _, zu_norm, zl_norm = sec.airfoil.evaluate(num_points=num_chordwise)

            # Dimensional coordinates before twist rotation
            xc = xi * sec.chord
            zu_dim = zu_norm * sec.chord
            zl_dim = zl_norm * sec.chord

            # Quarter-chord rotation for geometric twist
            rad_twist = np.radians(sec.twist_deg)  # Positive twist = nose-up incidence
            cos_t = np.cos(rad_twist)
            sin_t = np.sin(rad_twist)

            # Upper surface rotation about quarter-chord
            x_rel_u = xc - 0.25 * sec.chord
            z_rel_u = zu_dim
            x_rot_u = 0.25 * sec.chord + (x_rel_u * cos_t + z_rel_u * sin_t)
            z_rot_u = -x_rel_u * sin_t + z_rel_u * cos_t

            # Lower surface rotation
            x_rel_l = xc - 0.25 * sec.chord
            z_rel_l = zl_dim
            x_rot_l = 0.25 * sec.chord + (x_rel_l * cos_t + z_rel_l * sin_t)
            z_rot_l = -x_rel_l * sin_t + z_rel_l * cos_t

            # Global coordinate placement
            X_upper[i, :] = sec.x_le + x_rot_u
            Y_upper[i, :] = y_val
            Z_upper[i, :] = sec.z_le + z_rot_u

            X_lower[i, :] = sec.x_le + x_rot_l
            Y_lower[i, :] = y_val
            Z_lower[i, :] = sec.z_le + z_rot_l

        return {
            "X_upper": X_upper,
            "Y_upper": Y_upper,
            "Z_upper": Z_upper,
            "X_lower": X_lower,
            "Y_lower": Y_lower,
            "Z_lower": Z_lower,
            "semi_span": self.semi_span,
            "s_ref": self.s_ref,
            "mac": self.mac,
            "aspect_ratio": self.aspect_ratio,
        }

    def to_parameter_vector(self) -> np.ndarray:
        """
        Flattens the entire 3D wing into a vector representation for AI surrogates
        and optimization algorithms.
        [span, AR, taper, sweep_le, dihedral, twist_root, twist_tip, root_cst..., tip_cst...]
        """
        planform = np.array([
            self.span,
            self.aspect_ratio,
            self.taper_ratio,
            self.sweep_le_deg,
            self.dihedral_deg,
            self.twist_root_deg,
            self.twist_tip_deg,
        ])
        root_vec = self.root_airfoil.to_vector()
        tip_vec = self.tip_airfoil.to_vector()
        return np.concatenate([planform, root_vec, tip_vec])

    @classmethod
    def from_parameter_vector(cls, vec: np.ndarray, cst_order: int = 6, name: str = "Synth_Wing") -> "Wing3D":
        """Reconstructs a Wing3D object from a parameter vector."""
        vec = np.asarray(vec, dtype=float)
        span, ar, taper, sweep, dihedral, twist_r, twist_t = vec[:7]

        cst_len = (cst_order + 1) * 2 + 1
        root_vec = vec[7 : 7 + cst_len]
        tip_vec = vec[7 + cst_len : 7 + 2 * cst_len]

        root_af = CSTAirfoil3D.from_vector(root_vec)
        tip_af = CSTAirfoil3D.from_vector(tip_vec)

        return cls(
            span=span,
            aspect_ratio=ar,
            taper_ratio=taper,
            sweep_le_deg=sweep,
            dihedral_deg=dihedral,
            twist_root_deg=twist_r,
            twist_tip_deg=twist_t,
            root_airfoil=root_af,
            tip_airfoil=tip_af,
            name=name,
        )
