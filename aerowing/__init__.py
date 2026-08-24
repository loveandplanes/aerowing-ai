"""
AeroWing AI Pro: Enterprise-Grade 3D Aerodynamic AI Design, CFD Surrogate & MDO Optimization Suite.
"""

__version__ = "1.0.0"
__author__ = "Aerospace AI Engineering Team"

from .geometry.wing_3d import Wing3D, WingSection
from .geometry.cst_3d import CSTAirfoil3D
from .geometry.benchmarks import (
    get_onera_m6_wing,
    get_nasa_crm_wing,
    get_naca0012_swept_wing,
    get_supersonic_arrow_wing,
)
from .solvers.aero_engine import AeroEngine3D, AeroResult3D
from .models.surrogate_3d import AeroSurrogate3D
from .models.generator_3d import GenerativeWingVAE3D
from .models.ensemble_3d import EnsembleSurrogate3D, train_ensemble_surrogate
from .mesher_3d import VolumeMesher3D, VolumeMesh3D

__all__ = [
    "Wing3D",
    "WingSection",
    "CSTAirfoil3D",
    "AeroEngine3D",
    "AeroResult3D",
    "AeroSurrogate3D",
    "GenerativeWingVAE3D",
    "EnsembleSurrogate3D",
    "train_ensemble_surrogate",
    "VolumeMesher3D",
    "VolumeMesh3D",
    "get_onera_m6_wing",
    "get_nasa_crm_wing",
    "get_naca0012_swept_wing",
    "get_supersonic_arrow_wing",
]
