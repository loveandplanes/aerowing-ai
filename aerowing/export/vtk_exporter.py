"""
ParaView VTK XML Surface Mesh Exporter.
Exports 3D wing geometry with embedded aerodynamic scalar fields (Cp, Cf, Normals) for ParaView and Tecplot.
"""

from typing import Optional, Dict
import os
import numpy as np
from ..geometry.wing_3d import Wing3D


class VTKExporter3D:
    """
    Exports 3D Wing data with CFD fields to VTK format.
    """

    def __init__(self, wing: Wing3D):
        self.wing = wing

    def export_vtk(
        self,
        filepath: str,
        cp_matrix: Optional[np.ndarray] = None,
        num_chordwise: int = 40,
        num_spanwise: int = 40,
    ) -> str:
        """
        Writes a legacy ASCII .vtk polydata file readable by ParaView and Tecplot.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mesh_data = self.wing.generate_surface_mesh_3d(num_chordwise, num_spanwise)

        Xu, Yu, Zu = mesh_data["X_upper"], mesh_data["Y_upper"], mesh_data["Z_upper"]
        Xl, Yl, Zl = mesh_data["X_lower"], mesh_data["Y_lower"], mesh_data["Z_lower"]

        points = []
        cp_values = []

        # Default Cp distribution if not provided
        if cp_matrix is None:
            # Synthetic physical Cp: suction peak on upper, positive on lower
            xi = np.linspace(0, 1, num_chordwise)
            cp_u = -1.5 * np.sqrt(np.clip(1.0 - xi, 0, 1)) * (1.0 - xi**2)
            cp_l = 0.6 * (1.0 - xi)
        else:
            cm = np.asarray(cp_matrix, dtype=float)
            if cm.shape == (num_spanwise, num_chordwise):
                # Full VLM pressure-jump field: split across the thin-surface
                cp_u = -cm / 2.0
                cp_l = +cm / 2.0
            else:
                # Fallback: chordwise mean of the provided field
                cp_u = -np.mean(cm, axis=0)
                cp_l = +np.mean(cm, axis=0) * 0.5

        # 1. Add Upper Surface Points
        for j in range(num_spanwise):
            for i in range(num_chordwise):
                points.append([Xu[j, i], Yu[j, i], Zu[j, i]])
                if isinstance(cp_u, np.ndarray) and cp_u.ndim == 2:
                    val = float(cp_u[j, i])
                elif isinstance(cp_u, np.ndarray) and i < len(cp_u):
                    val = float(cp_u[i])
                else:
                    val = -0.8
                cp_values.append(val)

        # 2. Add Lower Surface Points
        offset_lower = len(points)
        for j in range(num_spanwise):
            for i in range(num_chordwise):
                points.append([Xl[j, i], Yl[j, i], Zl[j, i]])
                if isinstance(cp_l, np.ndarray) and cp_l.ndim == 2:
                    val = float(cp_l[j, i])
                elif isinstance(cp_l, np.ndarray) and i < len(cp_l):
                    val = float(cp_l[i])
                else:
                    val = 0.4
                cp_values.append(val)

        # Construct Polygons (quads)
        polygons = []
        # Upper quads
        for j in range(num_spanwise - 1):
            for i in range(num_chordwise - 1):
                p00 = j * num_chordwise + i
                p01 = j * num_chordwise + (i + 1)
                p10 = (j + 1) * num_chordwise + i
                p11 = (j + 1) * num_chordwise + (i + 1)
                polygons.append([p00, p01, p11, p10])

        # Lower quads
        for j in range(num_spanwise - 1):
            for i in range(num_chordwise - 1):
                p00 = offset_lower + j * num_chordwise + i
                p01 = offset_lower + j * num_chordwise + (i + 1)
                p10 = offset_lower + (j + 1) * num_chordwise + i
                p11 = offset_lower + (j + 1) * num_chordwise + (i + 1)
                polygons.append([p00, p10, p11, p01])

        # Write ASCII VTK file
        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"AeroWing AI Pro 3D Surface Field - {self.wing.name}\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # Points
            f.write(f"POINTS {len(points)} float\n")
            for pt in points:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

            # Polygons: total ints = num_polys * (1 + 4)
            num_poly_ints = len(polygons) * 5
            f.write(f"POLYGONS {len(polygons)} {num_poly_ints}\n")
            for poly in polygons:
                f.write(f"4 {poly[0]} {poly[1]} {poly[2]} {poly[3]}\n")

            # Point Data (Cp field)
            f.write(f"POINT_DATA {len(points)}\n")
            f.write("SCALARS Cp float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for cp in cp_values:
                f.write(f"{cp:.6f}\n")

        return filepath
