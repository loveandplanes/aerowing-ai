"""
Solvers module for AeroWing AI Pro.
"""

from .vlm_3d import VLMSolver3D
from .viscous_3d import ViscousEngine3D
from .wave_drag import WaveDragEngine3D
from .su2_3d import SU2Driver3D
from .aero_engine import AeroEngine3D, AeroResult3D

__all__ = [
    "VLMSolver3D",
    "ViscousEngine3D",
    "WaveDragEngine3D",
    "SU2Driver3D",
    "AeroEngine3D",
    "AeroResult3D",
]
