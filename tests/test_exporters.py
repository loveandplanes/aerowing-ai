"""
Exporter unit tests: STL mesh validity/watertightness and VTK Cp field wiring.
"""

import os
import struct
import tempfile

from aerowing.geometry.wing_3d import Wing3D
from aerowing.export.stl_exporter import STLExporter3D
from aerowing.export.vtk_exporter import VTKExporter3D


def _stl_triangle_count(path: str) -> int:
    with open(path, "rb") as f:
        f.read(80)
        return struct.unpack("<I", f.read(4))[0]


def test_stl_export_validity():
    """Binary STL exports a valid file with a positive triangle count."""
    wing = Wing3D(span=10.0, aspect_ratio=6.0, taper_ratio=0.5, sweep_le_deg=20.0)
    path = os.path.join(tempfile.gettempdir(), "aerowing_test_wing.stl")
    STLExporter3D(wing).export_stl(path, num_chordwise=10, num_spanwise=10)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 80  # header + at least one triangle
    assert _stl_triangle_count(path) > 0


def test_stl_watertight_triangle_counts():
    """Semi-span exports include the root cap; mirrored exports do not."""
    wing = Wing3D(span=10.0, aspect_ratio=6.0, taper_ratio=0.5, sweep_le_deg=20.0)
    nx, ny = 8, 8

    path_semi = os.path.join(tempfile.gettempdir(), "aerowing_test_semi.stl")
    STLExporter3D(wing).export_stl(
        path_semi, num_chordwise=nx, num_spanwise=ny, both_wings=False
    )
    # upper + lower + tip cap + root cap: each quad = 2 triangles
    n_surfaces = (ny - 1) * (nx - 1)
    expected_semi = 4 * n_surfaces + 4 * (nx - 1)
    assert _stl_triangle_count(path_semi) == expected_semi

    path_full = os.path.join(tempfile.gettempdir(), "aerowing_test_full.stl")
    STLExporter3D(wing).export_stl(
        path_full, num_chordwise=nx, num_spanwise=ny, both_wings=True
    )
    # both wings: two mirrored semi-spans (upper+lower+tip cap), no root cap
    expected_full = 2 * (4 * n_surfaces + 2 * (nx - 1))
    assert _stl_triangle_count(path_full) == expected_full


def test_vtk_exporter_uses_cp_matrix():
    """VTK export honors a provided VLM delta-cp field on both surfaces."""
    import numpy as np

    wing = Wing3D(span=10.0, aspect_ratio=6.0, taper_ratio=0.5, sweep_le_deg=20.0)
    nx, ny = 6, 6
    cp_matrix = np.full((ny, nx), 0.8)  # uniform artificial jump

    path = os.path.join(tempfile.gettempdir(), "aerowing_test.vtk")
    VTKExporter3D(wing).export_vtk(path, cp_matrix=cp_matrix, num_chordwise=nx, num_spanwise=ny)

    with open(path, "r") as f:
        content = f.read()
    # Upper surface Cp = -jump/2 = -0.4, lower = +0.4: both must appear
    assert "-0.400000" in content
    assert "0.400000" in content
    # The legacy hardcoded 0.300000 placeholder must not be present
    assert "0.300000" not in content