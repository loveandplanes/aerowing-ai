# -*- coding: utf-8 -*-
"""Tests for the template volume mesher (structured hexahedral O-grid)."""

import os
import tempfile

import numpy as np
import pytest

from aerowing.mesher_3d import VolumeMesher3D, _signed_hexa_volume
from aerowing.geometry.wing_3d import Wing3D
from aerowing.geometry.benchmarks import (
    get_onera_m6_wing,
    get_nasa_crm_wing,
    get_naca0012_swept_wing,
    get_supersonic_arrow_wing,
)


def build(wing, **kw):
    defaults = dict(n_loop=48, n_layers=12, growth=1.2, far_field_mult=10.0,
                    y_plus=5.0, n_stations=8, root_plug=3, tip_chain=4)
    defaults.update(kw)
    return VolumeMesher3D(wing, **defaults).build()


@pytest.mark.parametrize("wing", [
    get_onera_m6_wing(),
    get_nasa_crm_wing(),
    get_naca0012_swept_wing(),
    get_supersonic_arrow_wing(),
    Wing3D(name="generic_default"),
])
def test_no_inverted_cells_all_benchmark_families(wing):
    mesh = build(wing)
    jacs = [_signed_hexa_volume(mesh.points, mesh.cells[i])
            for i in range(len(mesh.cells))]
    assert min(jacs) > 0.0
    assert mesh.validate()["inverted_cells"] == 0
    assert mesh.min_jacobian() > 0.0


@pytest.mark.parametrize("kw", [
    dict(n_loop=20, n_layers=5, n_stations=5, root_plug=2, tip_chain=3,
         growth=1.3, y_plus=10),
    dict(n_loop=96, n_layers=20, n_stations=12, root_plug=4, tip_chain=5,
         growth=1.15, y_plus=2),
])
def test_resolution_sweep_stays_valid(kw):
    wing = get_naca0012_swept_wing()
    mesh = build(wing, **kw)
    assert mesh.validate()["inverted_cells"] == 0


def test_deterministic_rebuild():
    wing = get_onera_m6_wing()
    a = build(wing)
    b = build(wing)
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.cells, b.cells)
    assert np.array_equal(a.wall_faces, b.wall_faces)
    assert np.array_equal(a.far_faces, b.far_faces)


def test_first_cell_height_scaling():
    wing = get_nasa_crm_wing()
    m1 = VolumeMesher3D(wing, y_plus=1.0, n_layers=10)
    m2 = VolumeMesher3D(wing, y_plus=4.0, n_layers=10)
    assert m2.first_cell_height == pytest.approx(4.0 * m1.first_cell_height,
                                                 rel=1e-6)
    m3 = VolumeMesher3D(wing, y_plus=1.0, n_layers=10,
                        reynolds=m1.reynolds * 1e2)
    assert m3.first_cell_height > m1.first_cell_height


def test_wall_spacing_matches_y1():
    wing = get_onera_m6_wing()
    m = VolumeMesher3D(wing, n_loop=48, growth=1.2,
                       y_plus=5.0, n_stations=8, root_plug=3, tip_chain=4)
    mesh = m.build()
    Nc, R = mesh.n_loop, mesh.n_layers
    y1 = mesh.first_cell_height
    st = m._stations()
    s_mid = st["wing_first"] + m.n_stations // 2
    d = []
    for j in range(Nc):
        p0 = mesh.points[(s_mid * R + 0) * Nc + j]
        p1 = mesh.points[(s_mid * R + 1) * Nc + j]
        d.append(float(np.linalg.norm(p1 - p0)))
    d = np.array(d)
    # first cell <= y1; auto layer count sizes the stack so the far ring
    # lands within one growth factor of the requested radius
    assert 0.75 * y1 < d.mean() < 1.05 * y1


def test_far_field_radius_respected():
    wing = get_onera_m6_wing()
    m = VolumeMesher3D(wing, n_loop=48, n_layers=12, growth=1.2,
                       far_field_mult=10.0, y_plus=5.0, n_stations=8,
                       root_plug=3, tip_chain=4)
    mesh = m.build()
    S, R, Nc = mesh.n_stations, mesh.n_layers, mesh.n_loop
    st = m._stations()
    s_mid = st["wing_first"] + m.n_stations // 2
    center = m._quarter_chord(st["airfoil"][s_mid], st["chord"][s_mid],
                              st["x_le"][s_mid], st["z_le"][s_mid],
                              st["twist"][s_mid])
    center[1] = st["y"][s_mid]
    radii = []
    for j in range(Nc):
        p = mesh.points[(s_mid * R + (R - 1)) * Nc + j]
        radii.append(float(np.linalg.norm(p - center)))
    radii = np.array(radii)
    assert np.allclose(radii, mesh.far_radius, rtol=1e-6)
    assert mesh.far_radius == pytest.approx(10.0 * wing.mac)


def test_marker_structure_and_counts():
    wing = get_naca0012_swept_wing()
    m = VolumeMesher3D(wing, n_loop=40, n_layers=10,
                       growth=1.2, y_plus=5.0, n_stations=7, root_plug=3,
                       tip_chain=4)
    mesh = m.build()
    S, R, Nc = mesh.n_stations, mesh.n_layers, mesh.n_loop
    # wall faces: k=0 only, over wing station pairs
    assert len(mesh.wall_faces) == Nc * (m.n_stations - 1)
    for face in mesh.wall_faces:
        assert all(n % Nc < Nc for n in face)  # sanity, loop indices closed
        for n in face:
            assert (n // Nc) % R == 0         # k == 0
    # far faces: outer ring + 2 axial caps + plug/tip-cone k=0 surfaces
    # (the plug block sits under iw0 == root_plug pairs including the
    # boundary pair; the tip cone covers tip_chain station pairs)
    assert len(mesh.far_faces) == (S - 1) * Nc + 2 * (R - 1) * Nc \
        + (m.root_plug + m.tip_chain) * Nc
    for face in mesh.far_faces:
        assert len(set(face.tolist())) == 4   # no degenerate quads


def test_te_blunt_clamp_bounds_te_heights():
    wing = get_onera_m6_wing()
    m = VolumeMesher3D(wing, n_loop=48, n_layers=12, growth=1.2,
                       y_plus=5.0, n_stations=8, root_plug=3, tip_chain=4)
    st = m._stations()
    for s in [st["wing_first"], st["wing_first"] + 2, st["wing_last"]]:
        loop = m._section_loop(st["airfoil"][s], st["chord"][s],
                               st["x_le"][s], st["z_le"][s], st["twist"][s])
        assert loop[0, 2] <= loop[1, 2] + 1e-15      # upper TE <= neighbor
        assert loop[-1, 2] >= loop[-2, 2] - 1e-15     # lower TE >= neighbor
        assert loop[0, 2] >= loop[-1, 2]              # slit never inverts


def test_su2_export_roundtrip():
    wing = get_supersonic_arrow_wing()
    mesh = build(wing)
    tmp = os.path.join(tempfile.gettempdir(), "mesher_test_vol.su2")
    mesh.export_su2(tmp)
    text = open(tmp, "r", encoding="utf-8").read()
    lines = [l for l in text.splitlines() if l.strip()]
    assert lines[0] == "NDIME= 3"
    n_elem = int(lines[1].split("=")[1])
    assert n_elem == len(mesh.cells)
    line_idx = 2 + n_elem
    assert lines[line_idx].startswith("NPOIN= ")
    n_poin = int(lines[line_idx].split("=")[1])
    assert n_poin == len(mesh.points)
    idx = line_idx + 1 + n_poin
    assert lines[idx] == "NMARK= 2"
    assert lines[idx + 1] == "MARKER_TAG= wing"
    n_wing = int(lines[idx + 2].split("=")[1])
    assert n_wing == len(mesh.wall_faces)
    for line in lines[idx + 3: idx + 3 + n_wing]:
        cols = line.split()
        assert cols[0] == "9"           # QUAD_4
        assert all(0 <= int(c) < n_poin for c in cols[1:])
    idx2 = idx + 3 + n_wing
    assert lines[idx2] == "MARKER_TAG= farfield"
    assert int(lines[idx2 + 1].split("=")[1]) == len(mesh.far_faces)
    for line in lines[idx2 + 2:]:
        assert line.split()[0] == "9"


def test_tip_chain_shrinks_to_point():
    wing = get_nasa_crm_wing()
    m = VolumeMesher3D(wing, n_loop=48, n_layers=12, growth=1.2,
                       y_plus=5.0, n_stations=8, root_plug=3, tip_chain=4)
    mesh = m.build()
    R, Nc = mesh.n_layers, mesh.n_loop
    s_end = mesh.n_stations - 1
    ring = mesh.points[(s_end * R + 0) * Nc: (s_end * R + 0) * Nc + Nc]
    center = ring.mean(axis=0)
    extent = float(np.max(np.linalg.norm(ring - center, axis=1)))
    tip_chord = wing.tip_chord
    assert extent < 0.05 * tip_chord
    # chain reaches the far-field radius in y
    assert mesh.y_all[-1] > 0.9 * mesh.far_radius