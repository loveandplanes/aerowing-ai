"""
Class-Shape Transformation (CST) 3D Geometry Engine.
Provides high-order parametric representation of aerodynamic profiles and 3D lofts.
"""

from typing import Tuple, List, Optional, Union
import numpy as np
from math import comb


class CSTAirfoil3D:
    """
    Parametric airfoil generator using Class-Shape Transformation (CST).
    
    Standard class exponents:
      N1 = 0.5, N2 = 1.0 (Round leading edge, sharp/blunt trailing edge)
    """

    def __init__(
        self,
        weights_upper: np.ndarray,
        weights_lower: np.ndarray,
        dz_te: float = 0.001,
        n1: float = 0.5,
        n2: float = 1.0,
    ):
        self.weights_upper = np.asarray(weights_upper, dtype=float)
        self.weights_lower = np.asarray(weights_lower, dtype=float)
        self.dz_te = float(dz_te)
        self.n1 = float(n1)
        self.n2 = float(n2)
        self.order = len(self.weights_upper) - 1

    @classmethod
    def from_naca4(cls, code: str = "0012", order: int = 6) -> "CSTAirfoil3D":
        """Fit CST parameters to a standard NACA 4-digit airfoil."""
        m = float(code[0]) / 100.0
        p = float(code[1]) / 10.0
        t = float(code[2:]) / 100.0

        x = np.linspace(0.0, 1.0, 101)
        # Thickness distribution
        yt = 5.0 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * (x ** 2)
            + 0.2843 * (x ** 3)
            - 0.1015 * (x ** 4)
        )

        # Camber line
        yc = np.zeros_like(x)
        if p > 0:
            idx1 = x < p
            idx2 = ~idx1
            yc[idx1] = m / (p ** 2) * (2 * p * x[idx1] - x[idx1] ** 2)
            yc[idx2] = m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[idx2] - x[idx2] ** 2)

        zu = yc + yt
        zl = yc - yt

        return cls.fit_coordinates(x, zu, zl, order=order)

    @classmethod
    def fit_coordinates(
        cls,
        x: np.ndarray,
        zu: np.ndarray,
        zl: np.ndarray,
        order: int = 6,
        dz_te: float = 0.0,
    ) -> "CSTAirfoil3D":
        """Least-squares fit of CST weights to discrete coordinate curves."""
        x = np.asarray(x, dtype=float)
        zu = np.asarray(zu, dtype=float)
        zl = np.asarray(zl, dtype=float)

        # Avoid zero division at endpoints
        x_eps = np.clip(x, 1e-7, 1.0 - 1e-7)
        class_func = (x_eps ** 0.5) * (1.0 - x_eps)

        n = order
        bernstein_matrix = np.zeros((len(x), n + 1))
        for i in range(n + 1):
            k = comb(n, i)
            bernstein_matrix[:, i] = k * (x ** i) * ((1.0 - x) ** (n - i))

        # Design matrix: Class * Bernstein
        A = class_func[:, None] * bernstein_matrix

        # Upper weights
        b_u = zu - x * (dz_te / 2.0)
        wu, _, _, _ = np.linalg.lstsq(A, b_u, rcond=None)

        # Lower weights
        b_l = zl + x * (dz_te / 2.0)
        wl, _, _, _ = np.linalg.lstsq(A, b_l, rcond=None)

        return cls(weights_upper=wu, weights_lower=wl, dz_te=dz_te)

    def evaluate(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate (x, z_upper, z_lower) coordinate points using cosine spacing.
        """
        beta = np.linspace(0.0, np.pi, num_points)
        x = 0.5 * (1.0 - np.cos(beta))

        class_func = (x ** self.n1) * ((1.0 - x) ** self.n2)

        n = self.order
        shape_u = np.zeros_like(x)
        shape_l = np.zeros_like(x)

        for i in range(n + 1):
            k = comb(n, i)
            poly = k * (x ** i) * ((1.0 - x) ** (n - i))
            shape_u += self.weights_upper[i] * poly
            shape_l += self.weights_lower[i] * poly

        zu = class_func * shape_u + x * (self.dz_te / 2.0)
        zl = class_func * shape_l - x * (self.dz_te / 2.0)

        # Ensure nose point closes at (0, 0)
        zu[0] = 0.0
        zl[0] = 0.0

        return x, zu, zl

    def get_max_thickness(self, num_points: int = 200) -> float:
        """Returns the maximum thickness-to-chord ratio (t/c)."""
        _, zu, zl = self.evaluate(num_points=num_points)
        thickness = zu - zl
        return float(np.max(thickness))

    def get_max_camber(self, num_points: int = 200) -> Tuple[float, float]:
        """Returns (max_camber, x_camber) values."""
        x, zu, zl = self.evaluate(num_points=num_points)
        camber = 0.5 * (zu + zl)
        idx_max = np.argmax(np.abs(camber))
        return float(camber[idx_max]), float(x[idx_max])

    def get_cross_sectional_area(self, num_points: int = 200) -> float:
        """Returns non-dimensional cross-sectional area normalized by c^2."""
        x, zu, zl = self.evaluate(num_points=num_points)
        thickness = zu - zl
        # Trapezoidal integration across chord
        area = np.trapezoid(thickness, x)
        return float(area)

    def to_vector(self) -> np.ndarray:
        """Serializes upper and lower weights into a flat 1D vector."""
        return np.concatenate([self.weights_upper, self.weights_lower, [self.dz_te]])

    @classmethod
    def from_vector(cls, vec: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> "CSTAirfoil3D":
        """Deserializes from a 1D vector."""
        vec = np.asarray(vec, dtype=float)
        dz_te = vec[-1]
        weights = vec[:-1]
        half = len(weights) // 2
        wu = weights[:half]
        wl = weights[half:]
        return cls(weights_upper=wu, weights_lower=wl, dz_te=dz_te, n1=n1, n2=n2)
