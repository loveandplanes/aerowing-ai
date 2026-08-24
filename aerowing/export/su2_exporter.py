"""
SU2 3D Native Unstructured Surface Mesh Exporter (.su2).
Generates ready-to-run 3D surface meshes with designated boundary markers for SU2 CFD.
"""

from typing import Optional
import os
import numpy as np
from ..geometry.wing_3d import Wing3D


class SU2MeshExporter3D:
    """
    Exports 3D wing surface triangulation in native SU2 mesh format (.su2).
    """

    def __init__(self, wing: Wing3D):
        self.wing = wing

    def export_su2(
        self,
        filepath: str,
        num_chordwise: int = 30,
        num_spanwise: int = 30,
    ) -> str:
        """
        Generates a 3D surface mesh in SU2 format.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mesh_data = self.wing.generate_surface_mesh_3d(num_chordwise, num_spanwise)

        Xu, Yu, Zu = mesh_data["X_upper"], mesh_data["Y_upper"], mesh_data["Z_upper"]
        Xl, Yl, Zl = mesh_data["X_lower"], mesh_data["Y_lower"], mesh_data["Z_lower"]

        nodes = []
        # Upper nodes
        for j in range(num_spanwise):
            for i in range(num_chordwise):
                nodes.append([Xu[j, i], Yu[j, i], Zu[j, i]])

        # Lower nodes
        offset_lower = len(nodes)
        for j in range(num_spanwise):
            for i in range(num_chordwise):
                nodes.append([Xl[j, i], Yl[j, i], Zl[j, i]])

        # Triangles on upper and lower
        elements = []
        # Upper triangles
        for j in range(num_spanwise - 1):
            for i in range(num_chordwise - 1):
                p00 = j * num_chordwise + i
                p01 = j * num_chordwise + (i + 1)
                p10 = (j + 1) * num_chordwise + i
                p11 = (j + 1) * num_chordwise + (i + 1)
                elements.append([5, p00, p01, p11])  # 5 = triangle in SU2
                elements.append([5, p00, p11, p10])

        # Lower triangles
        for j in range(num_spanwise - 1):
            for i in range(num_chordwise - 1):
                p00 = offset_lower + j * num_chordwise + i
                p01 = offset_lower + j * num_chordwise + (i + 1)
                p10 = offset_lower + (j + 1) * num_chordwise + i
                p11 = offset_lower + (j + 1) * num_chordwise + (i + 1)
                elements.append([5, p00, p10, p11])
                elements.append([5, p00, p11, p01])

        # Write SU2 mesh format
        with open(filepath, "w") as f:
            f.write("NDIME= 3\n")
            f.write(f"NELEM= {len(elements)}\n")
            for idx, elem in enumerate(elements):
                f.write(f"{elem[0]} {elem[1]} {elem[2]} {elem[3]} {idx}\n")

            f.write(f"NPOIN= {len(nodes)}\n")
            for idx, pt in enumerate(nodes):
                f.write(f"{pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f} {idx}\n")

            # Boundary markers
            f.write("NMARK= 1\n")
            f.write("MARKER_TAG= wing\n")
            f.write(f"MARKER_ELEMS= {len(elements)}\n")
            for elem in elements:
                f.write(f"5 {elem[1]} {elem[2]} {elem[3]}\n")

        return filepath
