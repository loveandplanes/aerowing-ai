"""
3D Wing Inverse Aerodynamic Design Synthesizer.
Generates full 3D wing CAD parameters matching specified mission flight requirements.
"""

from typing import Dict, Any
import numpy as np
from ..geometry.wing_3d import Wing3D
from ..models.generator_3d import GenerativeWingVAE3D
from ..models.surrogate_3d import AeroSurrogate3D


class InverseWingSynthesizer3D:
    """
    Inverse Aerodynamic Synthesizer for 3D Wings.
    """

    def __init__(
        self,
        generator: GenerativeWingVAE3D,
        surrogate: AeroSurrogate3D,
    ):
        self.generator = generator
        self.surrogate = surrogate

    def synthesize(
        self,
        target_cl: float = 0.55,
        target_mach: float = 0.82,
        target_ar: float = 9.5,
        target_l_over_d: float = 19.5,
        target_reynolds: float = 2.5e7,
    ) -> Dict[str, Any]:
        """
        Synthesizes a 3D wing satisfying target aerodynamics.
        """
        # 1. Generative VAE sample
        synth_vec = self.generator.generate(
            target_cl=target_cl,
            target_mach=target_mach,
            target_ar=target_ar,
            target_l_over_d=target_l_over_d,
            target_reynolds=target_reynolds,
        )

        # 2. Build 3D Wing object
        wing = Wing3D.from_parameter_vector(synth_vec, name="Inverse_Synth_Wing")

        # 3. Predict aerodynamic performance
        telemetry = self.surrogate.predict_wing(
            synth_vec,
            alpha_deg=2.5,
            mach=target_mach,
            reynolds=target_reynolds,
        )

        return {
            "wing": wing,
            "parameter_vector": synth_vec,
            "telemetry": telemetry,
            "target_cl": target_cl,
            "achieved_cl": telemetry["cl"],
            "achieved_l_over_d": telemetry["l_over_d"],
        }
