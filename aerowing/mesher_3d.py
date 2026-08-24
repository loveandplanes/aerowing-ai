"""
Template Volume Mesher for the Parametric Wing3D Family.

The SU2 surface exporter only emits triangles on the wing skin; a running CFD
case needs a closed volume around it: a far field (~15 MAC away), near-wall
spacing sized for y+ ~ 1, geometric growth into the far field, and a resolved
wake region behind the trailing edge. Generic CAD meshing is an unsolved
general problem, but the parametric Wing3D family (CST airfoil lofted along a
straight-swept, dihedral, twisted planform) admits a single-block structured
hexahedral O-grid:

  * per spanwise station the airfoil section is discretized as a closed loop
    with cosine clustering at the leading edge and at the (slit-regularized)
    trailing edge - the slit is what resolves the wake cut;
  * each loop point is pushed radially outward from the section quarter-chord
    to the far-field ring with a geometric progression starting at the y+
    wall spacing (flat-plate skin-friction estimate);
  * stations run from the inboard plug (constant root section extended to the
    far field), across the wing (cosine-clustered), to the tip chain (chord
    shrinking to a point at the far field);
  * hexahedra are formed between consecutive rings; boundary markers are the
    wall (wing skin k=0 quads) and the far field (outer ring + axial caps +
    plug/tip-cone surfaces).

Every cell is checked for positive Jacobian determinant - inverted cells are
reported, not shipped. This is the honest engineering step between "surface
triangles" and "CFD runs that feed the continuous-learning flywheel".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import os
import numpy as np

from .geometry.wing_3d import Wing3D

# SU2 element type ids
SU2_HEXA8 = 12
SU2_QUAD4 = 9


class _ShiftedSection:
    """Lightweight WingSection stand-in with modified chord/LE position."""

    def __init__(self, sec, chord: Optional[float] = None,
                 dx: float = 0.0, dz: float = 0.0):
        self.airfoil = sec.airfoil
        self.twist_deg = sec.twist_deg
        self.chord = sec.chord if chord is None else float(chord)
        self.x_le = sec.x_le + float(dx)
        self.z_le = sec.z_le + float(dz)


def _cosine_cluster(n: int) -> np.ndarray:
    """n points cosine-clustered on [0, 1] (dense at both ends)."""
    beta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


def _signed_hexa_volume(p: np.ndarray, cell: np.ndarray) -> float:
    """Corner-0 Jacobian determinant of a hexahedron.

    det(n1-n0, n3-n0, n4-n0) with the cell's fixed node convention; a
    positive value means a left-handed-consistent, non-inverted cell.
    """
    c = p[cell]
    a = c[1] - c[0]
    b = c[3] - c[0]
    d = c[4] - c[0]
    return float(np.dot(a, np.cross(b, d)))


@dataclass
class VolumeMesh3D:
    """Structured hexahedral O-grid volume mesh around a Wing3D."""

    wing: Wing3D
    points: np.ndarray                    # (N, 3)
    cells: np.ndarray                     # (M, 8) hexa node indices
    wall_faces: np.ndarray                # (Fw, 4) quads, wing skin
    far_faces: np.ndarray                 # (Ff, 4) quads, far field
    y_all: np.ndarray                     # station y positions
    n_loop: int                           # loop points per ring
    n_layers: int                         # rings per station (k=0..n_layers-1)
    first_cell_height: float              # y1 (m), wall layer height
    far_radius: float                     # r_ff (m)
    meta: Dict[str, float] = field(default_factory=dict)

    # -- index helpers -----------------------------------------------------
    @property
    def n_stations(self) -> int:
        return len(self.y_all)

    # -- quality -----------------------------------------------------------
    def min_jacobian(self) -> float:
        """Minimum Jacobian determinant over all hexahedra (6-tet split)."""
        return min(_signed_hexa_volume(self.points, self.cells[i])
                   for i in range(len(self.cells)))

    def validate(self) -> Dict[str, float]:
        """Returns quality stats; negative jacobians mean inverted cells."""
        jacs = np.array([_signed_hexa_volume(self.points, self.cells[i])
                         for i in range(len(self.cells))])
        return {
            "n_nodes": int(len(self.points)),
            "n_cells": int(len(self.cells)),
            "n_wall_faces": int(len(self.wall_faces)),
            "n_far_faces": int(len(self.far_faces)),
            "min_jacobian": float(jacs.min()),
            "mean_jacobian": float(jacs.mean()),
            "inverted_cells": int((jacs <= 0.0).sum()),
            "first_cell_height": float(self.first_cell_height),
            "far_radius": float(self.far_radius),
            "y_plus_target": float(self.meta.get("y_plus", float("nan"))),
        }

    # -- export ------------------------------------------------------------
    def export_su2(self, filepath: str) -> str:
        """Writes the volume mesh in native SU2 format (.su2)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".",
                    exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write("NDIME= 3\n")
            f.write("NELEM= %d\n" % len(self.cells))
            for i, cell in enumerate(self.cells):
                f.write("%d %d %d %d %d %d %d %d %d %d\n" % (
                    SU2_HEXA8, cell[0], cell[1], cell[2], cell[3],
                    cell[4], cell[5], cell[6], cell[7], i))
            f.write("NPOIN= %d\n" % len(self.points))
            for i, p in enumerate(self.points):
                f.write("%.10e %.10e %.10e %d\n" % (p[0], p[1], p[2], i))
            f.write("NMARK= 2\n")
            f.write("MARKER_TAG= wing\n")
            f.write("MARKER_ELEMS= %d\n" % len(self.wall_faces))
            for face in self.wall_faces:
                f.write("%d %d %d %d %d\n" % (SU2_QUAD4, face[0], face[1],
                                              face[2], face[3]))
            f.write("MARKER_TAG= farfield\n")
            f.write("MARKER_ELEMS= %d\n" % len(self.far_faces))
            for face in self.far_faces:
                f.write("%d %d %d %d %d\n" % (SU2_QUAD4, face[0], face[1],
                                              face[2], face[3]))
        return filepath


class VolumeMesher3D:
    """
    Builds the single-block structured O-grid around a Wing3D.

    Mesh sizing follows the physics: first-cell height from a y+ = 1 wall
    spacing with a flat-plate skin-friction estimate, geometric growth out to
    a far field of ``far_field_mult`` MACs. The default ring/layer/station
    counts are for a production-ish cruise mesh; tests and ``--coarse`` use
    relaxed resolution (the O-grid topology is resolution-independent).
    """

    def __init__(
        self,
        wing: Wing3D,
        *,
        n_loop: int = 96,                # loop points per ring (odd: TE split)
        n_layers: Optional[int] = None,  # rings per station (auto from growth)
        growth: float = 1.2,             # geometric layer growth ratio
        far_field_mult: float = 15.0,    # far-field radius, in MACs
        y_plus: float = 1.0,             # target wall y+
        n_stations: int = 48,            # wing stations (cosine-clustered)
        root_plug: int = 6,              # inboard constant-root extension
        tip_chain: int = 10,             # outboard shrinking tip extension
        te_slit: float = 0.0025,         # TE slit half-width, fraction of chord
        rho: float = 1.225,              # freestream density (kg/m^3)
        mu: float = 1.7894e-5,           # dynamic viscosity (Pa*s)
        u_inf: float = 272.0,            # freestream speed (m/s)
        reynolds: Optional[float] = None,  # chord-based Re (default: from mac)
        mirror_full_span: bool = False,  # model both semi-spans explicitly
    ):
        if n_loop < 9:
            raise ValueError("n_loop must be >= 9 (TE split needs room)")
        if growth <= 1.0:
            raise ValueError("growth must be > 1.0")
        if far_field_mult <= 1.0:
            raise ValueError("far_field_mult must be > 1.0")
        if y_plus <= 0.0:
            raise ValueError("y_plus must be > 0.0")

        self.wing = wing
        # full-span mirroring: the semi-span O-grid with an inboard plug
        # models a constant-chord inboard extension, NOT the wind-tunnel
        # reflection plane - on the ONERA M6 that convention collapsed lift
        # by ~3x vs experiment. Mirroring both semi-spans restores the
        # correct root closure at 2x cell count (no symmetry BC needed).
        self.mirror_full_span = bool(mirror_full_span)
        # ring size is odd: upper surface shares the LE point with the lower
        # surface, while the TE is split into two nodes across the wake slit
        self.ring_size = int(n_loop) if int(n_loop) % 2 == 1 else int(n_loop) - 1
        self.growth = float(growth)
        self.far_field_mult = float(far_field_mult)
        self.y_plus = float(y_plus)
        self.n_stations = int(n_stations)
        self.root_plug = int(root_plug)
        self.tip_chain = int(tip_chain)
        self.te_slit = float(te_slit)
        self.rho = float(rho)
        self.mu = float(mu)
        self.u_inf = float(u_inf)
        if reynolds is not None:
            self.reynolds = float(reynolds)
        else:
            self.reynolds = self.rho * self.u_inf * wing.mac / self.mu
        self.far_radius = self.far_field_mult * wing.mac
        self.first_cell_height = self._first_cell_height()
        if n_layers is None:
            # geometric layers such that the outermost offset reaches the
            # far field: g^(R-1) - 1 == (r_ff/y1)(g-1), rounded up so the
            # far field is at least the requested radius (overshoot bounded
            # by one growth factor; first-cell height lands <= y1)
            ratio = (self.far_radius / self.first_cell_height) * (growth - 1.0)
            self.n_layers = int(math.ceil(math.log1p(ratio) / math.log(growth))) + 1
            self.n_layers = max(self.n_layers, 4)
        else:
            self.n_layers = int(n_layers)

    # -- sizing ------------------------------------------------------------
    def _first_cell_height(self) -> float:
        """y1 from y+ target: flat-plate skin friction (1/7-power law)."""
        cf = 0.026 * (self.reynolds ** (-1.0 / 7.0))
        u_tau = self.u_inf * math.sqrt(cf / 2.0)
        return self.y_plus * self.mu / (self.rho * u_tau)

    # -- section geometry --------------------------------------------------
    def _section_loop(self, airfoil, chord: float, x_le: float, z_le: float,
                      twist_deg: float) -> np.ndarray:
        """Closed loop of ring_size points around one section (global coords)."""
        n_side = (self.ring_size + 1) // 2
        x_af, zu, zl = airfoil.evaluate(num_points=n_side)

        # upper: TE -> LE (reversed x), lower: LE -> TE (forward x); TE split
        pts = []
        half_slit = self.te_slit * chord
        for i in range(n_side - 1, -1, -1):
            z = zu[i] if i > 0 else 0.5 * (zu[0] + zl[0])
            if i == n_side - 1:
                z += half_slit            # upper TE above the slit
            pts.append([x_af[i], z])
        for i in range(1, n_side):
            z = zl[i]
            if i == n_side - 1:
                z -= half_slit            # lower TE below the slit
            pts.append([x_af[i], z])
        pts = np.asarray(pts, dtype=float)          # (n_loop, 2), unit chord

        # blunt-TE clamp: the radial rays from the quarter-chord are nearly
        # horizontal at the TE, so the wedge cells flanking the slit fold if
        # the surface rises into the TE. Cap each TE point at its neighbor's
        # height (the mesh airfoil gets a vertical flat cap <= the local
        # surface rise - the standard O-grid "TE cut" treatment).
        pts[0, 1] = min(pts[0, 1], pts[1, 1])
        pts[-1, 1] = max(pts[-1, 1], pts[-2, 1])

        # twist about quarter chord (same convention as wing_3d)
        rad = math.radians(twist_deg)
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        x_rel = pts[:, 0] - 0.25
        z_rel = pts[:, 1]
        x_rot = 0.25 + (x_rel * cos_t + z_rel * sin_t)
        z_rot = -x_rel * sin_t + z_rel * cos_t

        # re-apply the clamp after twist: rotation can re-order heights when
        # the station has negative (washout) twist
        z_rot[0] = min(z_rot[0], z_rot[1])
        z_rot[-1] = max(z_rot[-1], z_rot[-2])

        loop = np.empty((len(pts), 3), dtype=float)
        loop[:, 0] = x_le + x_rot * chord
        loop[:, 1] = 0.0                                # y set by caller
        loop[:, 2] = z_le + z_rot * chord
        return loop

    def _quarter_chord(self, airfoil, chord: float, x_le: float,
                       z_le: float, twist_deg: float) -> np.ndarray:
        """Global quarter-chord point of a section (the O-grid center).

        Twist rotates about the quarter chord, so this point is fixed
        regardless of twist angle.
        """
        return np.array([x_le + 0.25 * chord, 0.0, z_le])

    def _stations_full_span(self) -> Dict[str, List]:
        """Symmetric two-sided station schedule (ascending y).

        Left tip chain | left wing (eta 1 -> 0+) | root (eta 0) |
        right wing (eta 0+ -> 1) | right tip chain.

        Mirroring rules: chord, x_le and twist are unchanged across the
        y=0 plane (sweep/twist are symmetric); z_le keeps its sign so
        dihedral continues upward on both sides.
        """
        wing = self.wing
        semi = wing.semi_span
        out: Dict[str, List] = {
            "y": [], "chord": [], "airfoil": [], "x_le": [], "z_le": [],
            "twist": [],
        }

        def add(y, sec):
            out["y"].append(float(y))
            out["chord"].append(sec.chord)
            out["airfoil"].append(sec.airfoil)
            out["x_le"].append(sec.x_le)
            out["z_le"].append(sec.z_le)
            out["twist"].append(sec.twist_deg)

        etas = _cosine_cluster(self.n_stations)
        tip_sec = wing.get_interpolated_section(1.0)

        # left tip chain: from -far_radius up to just inside the left tip;
        # chord shrinks to a point going outboard (mirror of the right chain)
        y_end = self.far_radius
        left_chain = (-semi * np.geomspace(
            y_end / semi, 1.001, self.tip_chain))          # ascending y
        for n, y in enumerate(left_chain):
            t = 1.0 - n / max(len(left_chain) - 1, 1)      # 0 at far end...
            chord = tip_sec.chord * max(0.02, (y_end - abs(y))
                                        / (y_end - semi))
            add(y, _ShiftedSection(tip_sec, chord))

        # left wing: eta 1 -> 0+ (exclusive), mirrored
        for eta in etas[::-1]:
            if eta <= 1e-12:
                continue
            add(-semi * eta, wing.get_interpolated_section(eta))

        # root on the symmetry plane exactly once
        add(0.0, wing.get_interpolated_section(0.0))

        # right wing: eta 0+ -> 1
        for eta in etas:
            if eta <= 1e-12:
                continue
            add(semi * eta, wing.get_interpolated_section(eta))

        # right tip chain (same construction as the half-span mesher)
        dx_le_dy, dz_le_dy = self._tip_slopes()
        chain_ys = semi * np.geomspace(1.001, y_end / semi, self.tip_chain)
        for y in chain_ys:
            t = (y - semi) / (y_end - semi)
            chord = tip_sec.chord * max(0.02, (1.0 - t))
            add(y, _ShiftedSection(tip_sec, chord,
                                   dx=dx_le_dy * (y - semi),
                                   dz=dz_le_dy * (y - semi)))

        out["n_plug"] = 0
        out["wing_first"] = len(left_chain)
        out["wing_last"] = len(out["y"]) - len(chain_ys)
        return out

    def _tip_slopes(self):
        """LE slope extrapolation used by the outboard chains."""
        wing = self.wing
        etas = _cosine_cluster(self.n_stations)
        prev_x_le = prev_z_le = None
        last_x_le = last_z_le = None
        for eta in etas:
            sec = wing.get_interpolated_section(eta)
            if eta >= 1.0 - 1e-12:
                prev_x_le, prev_z_le = sec.x_le, sec.z_le
            last_x_le, last_z_le = sec.x_le, sec.z_le
        if prev_x_le is None:
            prev_x_le, prev_z_le = last_x_le, last_z_le
        d = 2.0 * (self.n_stations - 2)
        return ((last_x_le - prev_x_le) / d, (last_z_le - prev_z_le) / d)

    # -- station schedule --------------------------------------------------
    def _stations(self) -> Dict[str, List]:
        """y positions, chords, sections and wing indices for all stations."""
        wing = self.wing
        semi = wing.semi_span
        if self.mirror_full_span:
            return self._stations_full_span()
        out: Dict[str, List] = {
            "y": [], "chord": [], "airfoil": [], "x_le": [], "z_le": [],
            "twist": [],
        }

        # inboard plug: constant root section extended to the far field;
        # ordered farthest-from-root first so station y is monotonic
        plug_ys = (-semi * np.geomspace(0.05, self.far_radius / semi,
                                        self.root_plug))[::-1]
        root_sec = wing.get_interpolated_section(0.0)
        for y in plug_ys:
            out["y"].append(float(y))
            out["chord"].append(root_sec.chord)
            out["airfoil"].append(root_sec.airfoil)
            out["x_le"].append(root_sec.x_le)
            out["z_le"].append(root_sec.z_le)
            out["twist"].append(root_sec.twist_deg)
        n_plug = len(out["y"])

        # wing stations, cosine-clustered over the semi-span
        etas = _cosine_cluster(self.n_stations)
        prev_x_le, prev_z_le = None, None
        for eta in etas:
            sec = wing.get_interpolated_section(eta)
            y = semi * eta
            out["y"].append(float(y))
            out["chord"].append(sec.chord)
            out["airfoil"].append(sec.airfoil)
            out["x_le"].append(sec.x_le)
            out["z_le"].append(sec.z_le)
            out["twist"].append(sec.twist_deg)
            if eta >= 1.0 - 1e-12:
                prev_x_le, prev_z_le = sec.x_le, sec.z_le
        last_x_le = out["x_le"][-1]
        last_z_le = out["z_le"][-1]
        if prev_x_le is None:
            prev_x_le = last_x_le
            prev_z_le = last_z_le
        dx_le_dy = (last_x_le - prev_x_le) / (2.0 * (self.n_stations - 2))
        dz_le_dy = (last_z_le - prev_z_le) / (2.0 * (self.n_stations - 2))

        # outboard chain: chord shrinks to a point at the far field
        tip_sec = wing.get_interpolated_section(1.0)
        y_end = self.far_radius
        chain_ys = semi * np.geomspace(1.001, y_end / semi, self.tip_chain)
        for y in chain_ys:
            t = (y - semi) / (y_end - semi)
            chord = tip_sec.chord * max(0.02, (1.0 - t))
            out["y"].append(float(y))
            out["chord"].append(chord)
            out["airfoil"].append(tip_sec.airfoil)
            out["x_le"].append(tip_sec.x_le + dx_le_dy * (y - semi))
            out["z_le"].append(tip_sec.z_le + dz_le_dy * (y - semi))
            out["twist"].append(tip_sec.twist_deg)

        out["n_plug"] = n_plug
        out["wing_first"] = n_plug
        out["wing_last"] = n_plug + self.n_stations - 1
        return out

    # -- build -------------------------------------------------------------
    def build(self) -> VolumeMesh3D:
        st = self._stations()
        S = len(st["y"])
        Nc = self.ring_size
        R = self.n_layers
        g = self.growth
        y1 = self.first_cell_height

        # layer blend: geometric progression of offsets, t_k in [0, 1]
        offs = y1 * (g ** np.arange(R) - 1.0) / (g - 1.0)
        t_frac = offs / offs[-1]

        n_nodes = S * R * Nc
        pts = np.empty((n_nodes, 3), dtype=float)
        idx = lambda s, k, j: (s * R + k) * Nc + j

        for s in range(S):
            loop = self._section_loop(
                st["airfoil"][s], st["chord"][s], st["x_le"][s],
                st["z_le"][s], st["twist"][s])
            loop[:, 1] = st["y"][s]
            center = self._quarter_chord(
                st["airfoil"][s], st["chord"][s], st["x_le"][s],
                st["z_le"][s], st["twist"][s])
            center[1] = st["y"][s]
            d = loop - center
            nrm = np.linalg.norm(d, axis=1)
            nrm[nrm < 1e-300] = 1e-300
            outer = center + (d / nrm[:, None]) * self.far_radius
            for k in range(R):
                ring = loop + (outer - loop) * t_frac[k]
                base = (s * R + k) * Nc
                pts[base:base + Nc] = ring

        # hexa cells: j-1 winding (verified positive-Jacobian orientation)
        cells = []
        for s in range(S - 1):
            for k in range(R - 1):
                for j in range(Nc):
                    jm = (j - 1) % Nc
                    cells.append([
                        idx(s, k, j), idx(s, k, jm), idx(s + 1, k, jm),
                        idx(s + 1, k, j), idx(s, k + 1, j),
                        idx(s, k + 1, jm), idx(s + 1, k + 1, jm),
                        idx(s + 1, k + 1, j),
                    ])
        cells = np.asarray(cells, dtype=np.int64)

        iw0, iw1 = st["wing_first"], st["wing_last"]
        wall, far = [], []

        def ring_face(s, k, j):
            return [idx(s, k, j), idx(s, k, (j + 1) % Nc),
                    idx(s + 1, k, (j + 1) % Nc), idx(s + 1, k, j)]

        def axial_face(s, k, j):
            return [idx(s, k, j), idx(s, k, (j + 1) % Nc),
                    idx(s, k + 1, (j + 1) % Nc), idx(s, k + 1, j)]

        # wall: wing-skin k=0 faces over wing station pairs only
        for s in range(iw0, iw1):
            for j in range(Nc):
                wall.append(ring_face(s, 0, j))

        # far field: outer ring, axial caps, plug and tip-cone k=0 surfaces
        for s in range(S - 1):
            for j in range(Nc):
                far.append(ring_face(s, R - 1, j))
        for j in range(Nc):
            for k in range(R - 1):
                # caps face outward: -y at s=0 (reversed winding), +y at s=S-1
                far.append([idx(0, k, (j + 1) % Nc), idx(0, k, j),
                            idx(0, k + 1, j), idx(0, k + 1, (j + 1) % Nc)])
                far.append(axial_face(S - 1, k, j))
        for s in range(0, iw0):
            for j in range(Nc):
                far.append(ring_face(s, 0, j))
        for s in range(iw1, S - 1):
            for j in range(Nc):
                far.append(ring_face(s, 0, j))

        return VolumeMesh3D(
            wing=self.wing,
            points=pts,
            cells=cells,
            wall_faces=np.asarray(wall, dtype=np.int64),
            far_faces=np.asarray(far, dtype=np.int64),
            y_all=np.asarray(st["y"], dtype=float),
            n_loop=Nc,
            n_layers=R,
            first_cell_height=y1,
            far_radius=self.far_radius,
            meta={"y_plus": self.y_plus, "reynolds": self.reynolds},
        )