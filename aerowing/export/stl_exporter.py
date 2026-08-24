"""
3D STL Triangular Mesh Exporter.
Produces watertight, manifold 3D meshes for CAD modeling, wind-tunnel prototyping, and 3D printing.
"""

from typing import Optional
import os
import struct
import numpy as np
from ..geometry.wing_3d import Wing3D


class STLExporter3D:
    """
    3D STL Mesh Exporter for Aerospace Wings.
    """

    def __init__(self, wing: Wing3D):
        self.wing = wing

    def export_stl(
        self,
        filepath: str,
        num_chordwise: int = 50,
        num_spanwise: int = 50,
        binary: bool = True,
        both_wings: bool = True,
    ) -> str:
        """
        Exports the 3D wing surface to an STL file.

        With both_wings=True (default) the full wing is built by mirroring the
        right semi-span over the y=0 plane, producing a watertight manifold
        (root is shared at the symmetry plane). With both_wings=False, the
        semi-span mesh is also closed with a root cap, keeping the export
        watertight.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mesh_data = self.wing.generate_surface_mesh_3d(num_chordwise, num_spanwise)

        # Right wing upper & lower
        Xu, Yu, Zu = mesh_data["X_upper"], mesh_data["Y_upper"], mesh_data["Z_upper"]
        Xl, Yl, Zl = mesh_data["X_lower"], mesh_data["Y_lower"], mesh_data["Z_lower"]

        triangles = []

        def add_quad(p1, p2, p3, p4):
            # Two triangles: (p1, p2, p3) and (p1, p3, p4)
            v1, v2, v3, v4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
            triangles.append((v1, v2, v3))
            triangles.append((v1, v3, v4))

        # 1. Right Wing Upper Surface
        for j in range(num_spanwise - 1):
            for i in range(num_chordwise - 1):
                p00 = [Xu[j, i], Yu[j, i], Zu[j, i]]
                p01 = [Xu[j, i + 1], Yu[j, i + 1], Zu[j, i + 1]]
                p10 = [Xu[j + 1, i], Yu[j + 1, i], Zu[j + 1, i]]
                p11 = [Xu[j + 1, i + 1], Yu[j + 1, i + 1], Zu[j + 1, i + 1]]
                add_quad(p00, p01, p11, p10)

        # 2. Right Wing Lower Surface
        for j in range(num_spanwise - 1):
            for i in range(num_chordwise - 1):
                p00 = [Xl[j, i], Yl[j, i], Zl[j, i]]
                p01 = [Xl[j, i + 1], Yl[j, i + 1], Zl[j, i + 1]]
                p10 = [Xl[j + 1, i], Yl[j + 1, i], Zl[j + 1, i]]
                p11 = [Xl[j + 1, i + 1], Yl[j + 1, i + 1], Zl[j + 1, i + 1]]
                add_quad(p00, p10, p11, p01)

        # 3. Tip Cap (closing right wing tip at j = num_spanwise - 1)
        j_tip = num_spanwise - 1
        for i in range(num_chordwise - 1):
            pu1 = [Xu[j_tip, i], Yu[j_tip, i], Zu[j_tip, i]]
            pu2 = [Xu[j_tip, i + 1], Yu[j_tip, i + 1], Zu[j_tip, i + 1]]
            pl1 = [Xl[j_tip, i], Yl[j_tip, i], Zl[j_tip, i]]
            pl2 = [Xl[j_tip, i + 1], Yl[j_tip, i + 1], Zl[j_tip, i + 1]]
            add_quad(pu1, pu2, pl2, pl1)

        # 4. Root Cap (closing the root at j = 0 — required for a watertight
        #    manifold when exporting a single semi-span with both_wings=False)
        if not both_wings:
            j_root = 0
            for i in range(num_chordwise - 1):
                pu1 = [Xu[j_root, i], Yu[j_root, i], Zu[j_root, i]]
                pu2 = [Xu[j_root, i + 1], Yu[j_root, i + 1], Zu[j_root, i + 1]]
                pl1 = [Xl[j_root, i], Yl[j_root, i], Zl[j_root, i]]
                pl2 = [Xl[j_root, i + 1], Yl[j_root, i + 1], Zl[j_root, i + 1]]
                # Reversed winding for an inward-facing root normal
                add_quad(pu1, pl1, pl2, pu2)

        # 5. If both wings, reflect symmetrically across y=0
        if both_wings:
            left_triangles = []
            for v1, v2, v3 in triangles:
                v1_sym = np.array([v1[0], -v1[1], v1[2]])
                v2_sym = np.array([v2[0], -v2[1], v2[2]])
                v3_sym = np.array([v3[0], -v3[1], v3[2]])
                # Invert winding order for outward normals
                left_triangles.append((v1_sym, v3_sym, v2_sym))
            triangles.extend(left_triangles)

        # Write file
        if binary:
            self._write_binary_stl(filepath, triangles)
        else:
            self._write_ascii_stl(filepath, triangles)

        return filepath

    def _write_binary_stl(self, filepath: str, triangles: list):
        with open(filepath, "wb") as f:
            # 80-byte header
            header = b"AeroWing AI Pro 3D Watertight Aerospace STL" + b"\x00" * 37
            f.write(header[:80])
            # Number of triangles (uint32)
            f.write(struct.pack("<I", len(triangles)))

            for v1, v2, v3 in triangles:
                # Normal vector
                n = np.cross(v2 - v1, v3 - v1)
                n_mag = np.linalg.norm(n)
                if n_mag > 1e-12:
                    n /= n_mag
                else:
                    n = np.array([0.0, 0.0, 1.0])

                # Normal (3 floats), Vertices (9 floats), Attribute byte count (uint16)
                data = struct.pack(
                    "<12fH",
                    n[0], n[1], n[2],
                    v1[0], v1[1], v1[2],
                    v2[0], v2[1], v2[2],
                    v3[0], v3[1], v3[2],
                    0,
                )
                f.write(data)

    def _write_ascii_stl(self, filepath: str, triangles: list):
        with open(filepath, "w") as f:
            f.write(f"solid {self.wing.name}\n")
            for v1, v2, v3 in triangles:
                n = np.cross(v2 - v1, v3 - v1)
                n_mag = np.linalg.norm(n)
                n = n / n_mag if n_mag > 1e-12 else [0.0, 0.0, 1.0]
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {self.wing.name}\n")
