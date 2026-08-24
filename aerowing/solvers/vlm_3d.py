"""
3D Non-Planar Vortex Lattice Method (VLM) Solver.
Calculates spanwise circulation, lift, downwash, induced drag, and surface Cp distributions.

Physics notes
-------------
* Horseshoe vortices are placed on the wing mean surface, with the bound filament at
  the panel quarter-chord and the boundary condition enforced at the three-quarter-chord
  collocation point (classic Campbell-type lattice).
* Symmetry: only the right semi-span is meshed; the image horseshoe system of the left
  semi-span is superposed analytically in the influence matrix.
* Compressibility: the incompressible system is solved first, then the Prandtl-Glauert
  rule for lifting surfaces is applied by scaling the perturbation potential (and hence
  the circulation) by 1/beta with beta = sqrt(1 - M^2). This yields
  C_L, C_m -> C_L/beta and, at fixed geometry/alpha, C_Di -> C_Di/beta^2, which is
  equivalent to C_Di = C_L^2 / (pi * AR * e) with the span efficiency e invariant.
  Valid for subcritical Mach numbers (M < ~0.95); supersonic flow is not a VLM regime.
* Induced drag is integrated in the far-field (Trefftz) plane using the discrete
  trailing-filament system, including the mirrored semi-span contribution.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from ..geometry.wing_3d import Wing3D

_EPS = 1e-9


def _horseshoe_velocity_matrix(
    field_points: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    wake_length: float = 100.0,
) -> np.ndarray:
    """
    Vectorized Biot-Savart evaluation for every (field point, horseshoe) pair.

    Parameters
    ----------
    field_points : (K, 3) collocation points
    p1, p2 : (M, 3) horseshoe bound-filament endpoints

    Returns
    -------
    (K, M, 3) induced velocity at each field point from each unit-strength horseshoe.
    """
    P = field_points[:, None, :]          # (K, 1, 3)
    A1 = p1[None, :, :]                   # (1, M, 3)
    A2 = p2[None, :, :]                   # (1, M, 3)
    wake = np.array([wake_length, 0.0, 0.0])

    def segment(verts_a: np.ndarray, verts_b: np.ndarray) -> np.ndarray:
        """Velocity from segments a->b, broadcast over (K, M)."""
        r1 = P - verts_a                  # (K, M, 3)
        r2 = P - verts_b
        r0 = verts_b - verts_a            # (1, M, 3)
        r1_mag = np.linalg.norm(r1, axis=-1)
        r2_mag = np.linalg.norm(r2, axis=-1)
        cross = np.cross(r1, r2)          # (K, M, 3)
        cross_sq = np.sum(cross ** 2, axis=-1)
        denom = np.maximum(cross_sq, _EPS)
        dot1 = np.sum(r0 * r1, axis=-1) / np.maximum(r1_mag, _EPS)
        dot2 = np.sum(r0 * r2, axis=-1) / np.maximum(r2_mag, _EPS)
        factor = np.where(cross_sq > _EPS, (dot1 - dot2) / (4.0 * np.pi * denom), 0.0)
        return cross * factor[..., None]

    v_bound = segment(A1, A2)
    v_left = segment(A1 + wake, A1)
    v_right = segment(A2, A2 + wake)
    return v_bound + v_left + v_right


class VLMSolver3D:
    """
    High-Performance 3D Vortex Lattice Aerodynamic Physics Solver.

    The geometric influence matrix is built once per instance (vectorized NumPy)
    and reused across operating-point changes (alpha / Mach / sideslip).
    """

    def __init__(
        self,
        wing: Wing3D,
        num_chordwise: int = 12,
        num_spanwise: int = 24,
        wake_length: Optional[float] = None,
        pg_limit_mach: float = 0.85,
    ):
        if num_chordwise < 2 or num_spanwise < 2:
            raise ValueError("num_chordwise and num_spanwise must both be >= 2")
        self.wing = wing
        self.nx = int(num_chordwise)
        self.ny = int(num_spanwise)
        # Scale-invariant far wake by default (~100 mean aerodynamic chords)
        self.wake_length = float(wake_length) if wake_length else 100.0 * wing.mac
        # Linearized compressibility is frozen beyond this Mach: as drag divergence
        # is approached, shock formation progressively offsets the linear
        # Prandtl-Glauert amplification, and extrapolating 1/beta towards M -> 1
        # is unphysical. 0.85 is a standard conceptual-design validity limit
        # for transonic cruise cases (M = 0.82-0.85).
        self.pg_limit_mach = float(pg_limit_mach)
        self.num_panels = self.nx * self.ny
        self._A_matrix: Optional[np.ndarray] = None
        self._build_lattice()

    # ------------------------------------------------------------------ geometry
    def _build_lattice(self):
        """Constructs panel corners, bound vortices, collocation points, and normals."""
        etas = np.linspace(0.0, 1.0, self.ny + 1)
        # Cosine chordwise distribution
        beta = np.linspace(0.0, np.pi, self.nx + 1)
        xi = 0.5 * (1.0 - np.cos(beta))

        grid_x = np.zeros((self.ny + 1, self.nx + 1))
        grid_y = np.zeros((self.ny + 1, self.nx + 1))
        grid_z = np.zeros((self.ny + 1, self.nx + 1))

        for j, eta in enumerate(etas):
            sec = self.wing.get_interpolated_section(eta)
            y_val = eta * self.wing.semi_span
            rad_twist = np.radians(sec.twist_deg)
            cos_t, sin_t = np.cos(rad_twist), np.sin(rad_twist)

            # Mean-surface (camber) ordinate interpolated onto the panel xi grid
            x_af, zu_af, zl_af = sec.airfoil.evaluate(num_points=60)
            zc_af = 0.5 * (zu_af + zl_af)
            zc = np.interp(xi, x_af, zc_af) * sec.chord
            xc = xi * sec.chord

            # Quarter-chord rotation for geometric twist
            x_rel = xc - 0.25 * sec.chord
            x_rot = 0.25 * sec.chord + (x_rel * cos_t + zc * sin_t)
            z_rot = -x_rel * sin_t + zc * cos_t

            grid_x[j, :] = sec.x_le + x_rot
            grid_y[j, :] = y_val
            grid_z[j, :] = sec.z_le + z_rot

        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

        # Panels: (j, i) -> panel index k = j * nx + i
        self.bound_p1 = np.zeros((self.num_panels, 3))
        self.bound_p2 = np.zeros((self.num_panels, 3))
        self.collocation = np.zeros((self.num_panels, 3))
        self.normals = np.zeros((self.num_panels, 3))
        self.panel_area = np.zeros(self.num_panels)
        self.panel_dy = np.zeros(self.num_panels)
        self.panel_dx = np.zeros(self.num_panels)
        self.panel_eta = np.zeros(self.num_panels)

        k = 0
        for j in range(self.ny):
            eta_mid = 0.5 * (etas[j] + etas[j + 1])
            for i in range(self.nx):
                p00 = np.array([grid_x[j, i], grid_y[j, i], grid_z[j, i]])
                p01 = np.array([grid_x[j, i + 1], grid_y[j, i + 1], grid_z[j, i + 1]])
                p10 = np.array([grid_x[j + 1, i], grid_y[j + 1, i], grid_z[j + 1, i]])
                p11 = np.array([grid_x[j + 1, i + 1], grid_y[j + 1, i + 1], grid_z[j + 1, i + 1]])

                # Bound vortex at 1/4 chord line
                bp1 = p00 + 0.25 * (p01 - p00)
                bp2 = p10 + 0.25 * (p11 - p10)

                # Collocation point at 3/4 chord line
                cp1 = p00 + 0.75 * (p01 - p00)
                cp2 = p10 + 0.75 * (p11 - p10)
                cp = 0.5 * (cp1 + cp2)

                # Outward panel normal from diagonals
                diag1 = p11 - p00
                diag2 = p01 - p10
                norm_vec = np.cross(diag1, diag2)
                norm_mag = np.linalg.norm(norm_vec)
                if norm_mag > 1e-12:
                    norm_vec /= norm_mag
                else:
                    norm_vec = np.array([0.0, 0.0, 1.0])

                # Ensure normal points upwards (positive z component)
                if norm_vec[2] < 0:
                    norm_vec = -norm_vec

                self.bound_p1[k] = bp1
                self.bound_p2[k] = bp2
                self.collocation[k] = cp
                self.normals[k] = norm_vec
                self.panel_area[k] = 0.5 * norm_mag
                self.panel_dy[k] = np.abs(grid_y[j + 1, i] - grid_y[j, i])
                self.panel_dx[k] = np.linalg.norm(p01 - p00)
                self.panel_eta[k] = eta_mid
                k += 1

    # ------------------------------------------------------------------ influence
    def _influence_matrix(self) -> np.ndarray:
        """
        Builds the aerodynamic influence coefficient matrix A with A[k, m] equal to the
        wall-normal velocity induced at collocation point k by a unit-strength
        horseshoe m and its mirrored (left semi-span) image. Cached per instance.
        """
        if self._A_matrix is not None:
            return self._A_matrix

        wake = self.wake_length
        # Right semi-span horseshoes
        v_right = _horseshoe_velocity_matrix(
            self.collocation, self.bound_p1, self.bound_p2, wake
        )

        # Mirrored left semi-span (reflect y; reverse bound endpoints to keep
        # the mirrored circuit orientation consistent with symmetric loading)
        p1_sym = self.bound_p2.copy()
        p1_sym[:, 1] *= -1.0
        p2_sym = self.bound_p1.copy()
        p2_sym[:, 1] *= -1.0
        v_left = _horseshoe_velocity_matrix(
            self.collocation, p1_sym, p2_sym, wake
        )

        # (K, M, 3) dot (K, 1, 3) -> (K, M)
        A = np.einsum("kmi,ki->km", v_right + v_left, self.normals)
        self._A_matrix = A
        return A

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        alpha_deg: float = 2.5,
        mach: float = 0.0,
        beta_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Solves the linear aerodynamic system for circulation and force coefficients.

        The incompressible system is solved exactly; the Prandtl-Glauert correction is
        then applied to the circulation (Gamma -> Gamma / beta), which consistently
        scales C_L and C_m by 1/beta and the Trefftz-plane induced drag by 1/beta^2
        at fixed geometry and incidence.

        Note: sideslip (beta_deg != 0) breaks strict left/right loading symmetry; the
        mirrored-image superposition is retained as an approximation for small angles.
        """
        # Compressibility factor beta = sqrt(1 - M^2), frozen at the linear-validity
        # limit; VLM is a subcritical method (see __init__ for the rationale).
        # The factor stays frozen at beta_lim for all M >= pg_limit_mach so the
        # solution is continuous across the Mach sweep (no beta reset at 0.95+).
        if mach > 0.0:
            beta_lim = np.sqrt(1.0 - self.pg_limit_mach ** 2)
            beta_pg = max(np.sqrt(1.0 - mach ** 2), beta_lim)
        else:
            beta_pg = 1.0

        alpha_rad = np.radians(alpha_deg)
        beta_rad = np.radians(beta_deg)

        # Free-stream velocity vector (unit magnitude)
        v_inf = np.array([
            np.cos(alpha_rad) * np.cos(beta_rad),
            -np.sin(beta_rad),
            np.sin(alpha_rad),
        ])

        A_mat = self._influence_matrix()
        rhs = -(self.normals @ v_inf)

        try:
            gamma = np.linalg.solve(A_mat, rhs)
        except np.linalg.LinAlgError:
            gamma, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)

        # Prandtl-Glauert correction on the perturbation potential (see module docstring)
        gamma = gamma / beta_pg

        # ---------------------------------------------------------------- forces
        gamma_matrix = gamma.reshape((self.ny, self.nx))
        panel_dy_matrix = self.panel_dy.reshape((self.ny, self.nx))

        # Total circulation per spanwise strip
        gamma_spanwise = np.sum(gamma_matrix, axis=1)
        dy_spanwise = panel_dy_matrix[:, 0]
        # Strip centroid stations eta_j = (j + 1/2) / ny
        etas_spanwise = (np.arange(self.ny) + 0.5) / self.ny

        # Total lift coefficient (both semi-spans): C_L = 4/S * sum(Gamma * dy)
        cl_total = (4.0 / self.wing.s_ref) * np.sum(gamma_spanwise * dy_spanwise)

        # Sectional lift coefficient c_l(y) = 2 * Gamma(y) / chord(y)
        chords_spanwise = np.array(
            [self.wing.get_interpolated_section(eta).chord for eta in etas_spanwise]
        )
        cl_spanwise = 2.0 * gamma_spanwise / (chords_spanwise + 1e-12)

        # ------------------------------------------------- Trefftz plane induced drag
        # Standard vortex-lattice far-field integration: trailing filaments of strength
        # dGamma_k = Gamma_k - Gamma_{k-1} shed at the strip edges (root and tip edges
        # closed by symmetry / Kutta conditions), downwash evaluated at strip centroids.
        # The mirrored semi-span contributes with reversed circulation sign.
        strip_edges = np.arange(self.ny + 1) / self.ny            # edge stations eta_k
        y_edges = strip_edges * self.wing.semi_span
        gamma_ext = np.concatenate([[gamma_spanwise[0]], gamma_spanwise, [0.0]])
        d_gamma = np.diff(gamma_ext)                              # (ny+1,) shed strengths

        y_c = etas_spanwise * self.wing.semi_span
        w_trefftz = np.zeros(self.ny)
        for j in range(self.ny):
            yj = y_c[j]
            dw = np.sum(d_gamma / (yj - y_edges)) - np.sum(d_gamma / (yj + y_edges))
            w_trefftz[j] = dw / (4.0 * np.pi)

        cd_induced = (4.0 / self.wing.s_ref) * np.sum(gamma_spanwise * w_trefftz * dy_spanwise)
        cd_induced = float(np.clip(cd_induced, 0.0, 2.0))

        # Span efficiency factor: e = C_L^2 / (pi * AR * C_Di)
        if cd_induced > 1e-6 and abs(cl_total) > 1e-4:
            span_efficiency = float((cl_total ** 2) / (np.pi * self.wing.aspect_ratio * cd_induced))
        else:
            span_efficiency = 0.95
        span_efficiency = float(np.clip(span_efficiency, 0.4, 1.05))

        # Pitching moment about the quarter-chord of the MAC. The leading edge
        # is a straight line (x_le linear in y), hence x_ref is the exact
        # quarter-chord station of the MAC: x_c/4(y_mac) = x_LE(y_mac) + mac/4.
        # The panel lift acts through its bound vortex (quarter-chord filament);
        # using the collocation (3/4-chord) position here would inject a spurious
        # nx-dependent nose-down moment of order -0.5*(panel chord)/mac * CL.
        x_ref = self.wing.y_mac * np.tan(np.radians(self.wing.sweep_le_deg)) + 0.25 * self.wing.mac
        x_bv = 0.5 * (self.bound_p1[:, 0] + self.bound_p2[:, 0])
        d_lift = gamma * self.panel_dy
        moment_y = float(np.sum(d_lift * (x_ref - x_bv)))
        cm_total = float((4.0 / (self.wing.s_ref * self.wing.mac)) * moment_y)

        # Surface pressure jump Delta Cp across panels: Delta Cp = 2 * Gamma / dx
        delta_cp = 2.0 * gamma / (self.panel_dx + 1e-8)
        delta_cp_matrix = delta_cp.reshape((self.ny, self.nx))

        return {
            "cl": float(cl_total),
            "cd_induced": cd_induced,
            "cm": cm_total,
            "span_efficiency": span_efficiency,
            "etas_spanwise": etas_spanwise,
            "cl_spanwise": cl_spanwise,
            "gamma_spanwise": gamma_spanwise,
            "w_trefftz": w_trefftz,
            "delta_cp_matrix": delta_cp_matrix,
        }
