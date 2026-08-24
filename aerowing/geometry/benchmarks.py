"""
Aerospace Industry Benchmark 3D Wings.
Contains standard geometries for CFD and experimental validation.
"""

import numpy as np
from .wing_3d import Wing3D
from .cst_3d import CSTAirfoil3D


def get_onera_m6_wing() -> Wing3D:
    """
    ONERA M6 Transonic Benchmark Wing (AGARD AR-138).
    
    Standard transonic validation case with lambda shock formation.
    - Span: 2.392 m (Semi-span: 1.196 m)
    - Aspect Ratio: 3.80
    - Taper Ratio: 0.562
    - LE Sweep: 30.0 deg
    - Root/Tip Profile: NACA 0010 stand-in for the ONERA "D" airfoil (t/c ~ 10%)
    - Untwisted (the experimental M6 has zero geometric twist)
    """
    # ONERA D symmetric airfoil fitted with 6th-order CST
    onera_d = CSTAirfoil3D.from_naca4("0010", order=6)
    
    return Wing3D(
        name="ONERA_M6",
        span=2.392,
        aspect_ratio=3.80,
        taper_ratio=0.562,
        sweep_le_deg=30.0,
        dihedral_deg=0.0,
        twist_root_deg=0.0,
        twist_tip_deg=0.0,
        root_airfoil=onera_d,
        tip_airfoil=onera_d,
    )


def get_nasa_crm_wing() -> Wing3D:
    """
    NASA Common Research Model (CRM) Transonic Transport Wing.
    
    AIAA Drag Prediction Workshop standard modern transport aircraft.
    - Full Span: 58.76 m
    - Aspect Ratio: 9.0
    - Taper Ratio: 0.275
    - LE Sweep: 35.0 deg
    - Dihedral: 5.0 deg
    - Supercritical-class profiles (Root t/c ~ 15%, Tip t/c ~ 9.5%, 1-2% camber)
    - Root incidence +1.0 deg with tip washout to -3.8 deg
    """
    # Supercritical-class stand-in profiles: modest camber, appropriate thickness.
    # (The previous synthetic fit carried ~5% camber, far beyond real supercritical
    # sections, and severely over-lifted the wing.)
    root_supercritical = CSTAirfoil3D.from_naca4("2415", order=6)
    tip_supercritical = CSTAirfoil3D.from_naca4("1409", order=6)

    return Wing3D(
        name="NASA_CRM",
        span=58.76,
        aspect_ratio=9.0,
        taper_ratio=0.275,
        sweep_le_deg=35.0,
        dihedral_deg=5.0,
        twist_root_deg=1.0,
        twist_tip_deg=-3.8,
        root_airfoil=root_supercritical,
        tip_airfoil=tip_supercritical,
    )


def get_naca0012_swept_wing() -> Wing3D:
    """Standard swept validation wing with NACA 0012 section."""
    naca0012 = CSTAirfoil3D.from_naca4("0012", order=6)
    return Wing3D(
        name="NACA0012_Swept",
        span=10.0,
        aspect_ratio=6.0,
        taper_ratio=0.5,
        sweep_le_deg=25.0,
        dihedral_deg=2.0,
        twist_root_deg=0.0,
        twist_tip_deg=-2.0,
        root_airfoil=naca0012,
        tip_airfoil=naca0012,
    )


def get_supersonic_arrow_wing() -> Wing3D:
    """Supersonic low-aspect-ratio cranked-arrow wing."""
    thin_diamond = CSTAirfoil3D.from_naca4("0004", order=6)
    return Wing3D(
        name="Supersonic_Arrow",
        span=12.0,
        aspect_ratio=2.2,
        taper_ratio=0.15,
        sweep_le_deg=65.0,
        dihedral_deg=0.0,
        twist_root_deg=0.0,
        twist_tip_deg=-1.5,
        root_airfoil=thin_diamond,
        tip_airfoil=thin_diamond,
    )
