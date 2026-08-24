"""
Unit Tests for AeroWing 3D Geometry and CST-3D Parameterization Engine.
"""

import pytest
import numpy as np
from aerowing.geometry.cst_3d import CSTAirfoil3D
from aerowing.geometry.wing_3d import Wing3D, WingSection


def test_cst_airfoil_naca():
    """Verifies that CST fitting accurately reconstructs a NACA 0012 section."""
    cst = CSTAirfoil3D.from_naca4("0012", order=6)
    tc = cst.get_max_thickness()
    # Check max thickness matches 0.12 within 1%
    assert np.isclose(tc, 0.12, atol=0.005)
    
    # Area must be positive and reasonable
    area = cst.get_cross_sectional_area()
    assert area > 0.05 and area < 0.15


def test_wing_3d_planform_metrics():
    """Verifies reference area, MAC, and aspect ratio geometric relationships."""
    wing = Wing3D(
        span=30.0,
        aspect_ratio=10.0,
        taper_ratio=0.3,
        sweep_le_deg=25.0,
        dihedral_deg=3.0,
    )
    # S = b^2 / AR = 900 / 10 = 90 m^2
    assert np.isclose(wing.s_ref, 90.0, atol=1e-5)
    assert np.isclose(wing.semi_span, 15.0, atol=1e-5)

    # MAC must be strictly positive and between root and tip chords
    assert wing.root_chord > wing.mac > wing.tip_chord

    # Fuel volume must be positive
    fuel_vol = wing.compute_internal_fuel_volume()
    assert fuel_vol > 1.0


def test_wing_3d_vector_roundtrip():
    """Verifies parameter vector serialization and deserialization."""
    wing = Wing3D(span=35.0, aspect_ratio=8.5, taper_ratio=0.4, sweep_le_deg=28.0)
    vec = wing.to_parameter_vector()
    
    reconstructed = Wing3D.from_parameter_vector(vec)
    assert np.isclose(reconstructed.span, wing.span, atol=1e-5)
    assert np.isclose(reconstructed.aspect_ratio, wing.aspect_ratio, atol=1e-5)
    assert np.isclose(reconstructed.taper_ratio, wing.taper_ratio, atol=1e-5)
    assert np.isclose(reconstructed.sweep_le_deg, wing.sweep_le_deg, atol=1e-5)
