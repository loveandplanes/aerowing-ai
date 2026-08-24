"""
CAD B-Spline Loft & STEP/IGES Cross-Section Exporter.
Exports parametric cross-sectional coordinate curves for SolidWorks, CATIA, and NX.
"""

from typing import Optional
import os
import json
import numpy as np
from ..geometry.wing_3d import Wing3D


class CADCurveExporter3D:
    """
    Exports multi-station 3D cross-sectional spline curves and CAD metadata.
    """

    def __init__(self, wing: Wing3D):
        self.wing = wing

    def export_cad_curves(
        self,
        filepath: str,
        num_stations: int = 10,
        num_points_per_curve: int = 60,
    ) -> str:
        """
        Exports sectional 3D spline curves in structured JSON and CSV coordinate tables.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        etas = np.linspace(0.0, 1.0, num_stations)

        cad_data = {
            "wing_name": self.wing.name,
            "span_m": self.wing.span,
            "aspect_ratio": self.wing.aspect_ratio,
            "s_ref_m2": self.wing.s_ref,
            "mac_m": self.wing.mac,
            "sweep_le_deg": self.wing.sweep_le_deg,
            "dihedral_deg": self.wing.dihedral_deg,
            "stations": [],
        }

        # Also write a combined CSV table
        csv_filepath = os.path.splitext(filepath)[0] + ".csv"
        csv_lines = ["station_idx,eta,y_m,chord_m,twist_deg,point_idx,x_m,y_m_coord,z_m\n"]

        for idx, eta in enumerate(etas):
            sec = self.wing.get_interpolated_section(eta)
            y_val = eta * self.wing.semi_span
            x, zu_norm, zl_norm = sec.airfoil.evaluate(num_points=num_points_per_curve)

            rad_twist = np.radians(sec.twist_deg)
            cos_t = np.cos(rad_twist)
            sin_t = np.sin(rad_twist)

            curve_3d = []
            # Upper curve (LE to TE)
            for i in range(num_points_per_curve):
                xc = x[i] * sec.chord
                zc = zu_norm[i] * sec.chord
                xr = 0.25 * sec.chord + ((xc - 0.25 * sec.chord) * cos_t + zc * sin_t)
                zr = -(xc - 0.25 * sec.chord) * sin_t + zc * cos_t
                gx, gy, gz = sec.x_le + xr, y_val, sec.z_le + zr
                curve_3d.append({"x": float(gx), "y": float(gy), "z": float(gz)})
                csv_lines.append(f"{idx},{eta:.4f},{y_val:.4f},{sec.chord:.4f},{sec.twist_deg:.2f},{i},{gx:.6f},{gy:.6f},{gz:.6f}\n")

            # Lower curve (TE back to LE)
            for i in reversed(range(num_points_per_curve)):
                xc = x[i] * sec.chord
                zc = zl_norm[i] * sec.chord
                xr = 0.25 * sec.chord + ((xc - 0.25 * sec.chord) * cos_t + zc * sin_t)
                zr = -(xc - 0.25 * sec.chord) * sin_t + zc * cos_t
                gx, gy, gz = sec.x_le + xr, y_val, sec.z_le + zr
                curve_3d.append({"x": float(gx), "y": float(gy), "z": float(gz)})
                csv_lines.append(f"{idx},{eta:.4f},{y_val:.4f},{sec.chord:.4f},{sec.twist_deg:.2f},{num_points_per_curve + i},{gx:.6f},{gy:.6f},{gz:.6f}\n")

            cad_data["stations"].append({
                "station_idx": idx,
                "eta": float(eta),
                "y_location_m": float(y_val),
                "chord_m": float(sec.chord),
                "twist_deg": float(sec.twist_deg),
                "curve_points": curve_3d,
            })

        # Save JSON
        with open(filepath, "w") as f:
            json.dump(cad_data, f, indent=2)

        # Save CSV
        with open(csv_filepath, "w") as f:
            f.writelines(csv_lines)

        return filepath
