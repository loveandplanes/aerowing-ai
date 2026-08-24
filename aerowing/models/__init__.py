"""
Models module for AeroWing AI Pro.
"""

from .surrogate_3d import AeroSurrogate3D
from .generator_3d import GenerativeWingVAE3D
from .dataset_3d import WingDataset3D, generate_synthetic_wing_dataset
from .trainer_3d import AeroTrainer3D

__all__ = [
    "AeroSurrogate3D",
    "GenerativeWingVAE3D",
    "WingDataset3D",
    "generate_synthetic_wing_dataset",
    "AeroTrainer3D",
]
