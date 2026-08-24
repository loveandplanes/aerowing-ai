"""
Validation Tests for 3D Non-Planar VLM Solver & Aerodynamic Physics.
"""

import pytest
import numpy as np
from aerowing.geometry.wing_3d import Wing3D
from aerowing.solvers.vlm_3d import VLMSolver3D
from aerowing.solvers.aero_engine import AeroEngine3D


def test_vlm_lift_curve_slope():
    """Verifies that 3D VLM produces physical lift-curve slope dCL/dalpha."""
    wing = Wing3D(span=20.0, aspect_ratio=8.0, taper_ratio=0.5, sweep_le_deg=15.0)
    vlm = VLMSolver3D(wing, num_chordwise=8, num_spanwise=16)

    res_alpha2 = vlm.solve(alpha_deg=2.0)
    res_alpha4 = vlm.solve(alpha_deg=4.0)

    # dCL/dalpha per radian should be around 2*pi * AR / (AR + 2) ~ 5.0 rad^-1
    dcl_dalpha_deg = (res_alpha4["cl"] - res_alpha2["cl"]) / 2.0
    dcl_dalpha_rad = dcl_dalpha_deg * (180.0 / np.pi)

    assert dcl_dalpha_rad > 3.0 and dcl_dalpha_rad < 6.0


def test_vlm_trefftz_drag_positive():
    """Verifies that Trefftz plane induced drag is strictly positive and span efficiency is bounded."""
    wing = Wing3D(span=25.0, aspect_ratio=9.0, taper_ratio=0.35, sweep_le_deg=25.0)
    vlm = VLMSolver3D(wing, num_chordwise=10, num_spanwise=18)
    res = vlm.solve(alpha_deg=3.0)

    assert res["cd_induced"] > 0.001
    assert 0.70 <= res["span_efficiency"] <= 1.05


def test_aero_engine_total_drag():
    """Verifies coupled drag decomposition CD = CDi + CDp + CDw."""
    wing = Wing3D(span=30.0, aspect_ratio=9.5, taper_ratio=0.28, sweep_le_deg=27.5)
    engine = AeroEngine3D(wing, num_chordwise=10, num_spanwise=18)
    res = engine.evaluate(alpha_deg=2.5, mach=0.82, reynolds=2.5e7)

    assert res.cd > res.cd_induced
    assert res.cd > res.cd_profile
    assert np.isclose(res.cd, res.cd_induced + res.cd_profile + res.cd_wave, atol=1e-6)
    assert res.l_over_d > 10.0
