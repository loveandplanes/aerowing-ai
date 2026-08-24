"""
Geometry module for AeroWing AI Pro.
"""

from .cst_3d import CSTAirfoil3D
from .wing_3d import Wing3D, WingSection
from .benchmarks import (
    get_onera_m6_wing,
    get_nasa_crm_wing,
    get_naca0012_swept_wing,
    get_supersonic_arrow_wing,
)

__all__ = [
    "CSTAirfoil3D",
    "Wing3D",
    "WingSection",
    "get_onera_m6_wing",
    "get_nasa_crm_wing",
    "get_naca0012_swept_wing",
    "get_supersonic_arrow_wing",
]
