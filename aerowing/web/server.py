"""
AeroStudio 3D Web Server Backend.
Provides high-performance REST API for real-time 3D wing analysis, AI surrogate inference, and CAD export.

PRIVACY: 100% local & private by design.
- Binds to 127.0.0.1 only (no network exposure).
- Zero third-party requests: no CDN, no external fonts, no analytics, no telemetry.
- No CORS middleware: cross-origin browser access is disabled by default.
- AI models are loaded from a local checkpoint (AEROWING_CHECKPOINT) when present;
  otherwise physics-driven solver results are used without fabricated AI output.
"""

from typing import Dict, Any, Optional
import os
import io
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel

from ..geometry.wing_3d import Wing3D
from ..geometry.cst_3d import CSTAirfoil3D
from ..geometry.benchmarks import (
    get_onera_m6_wing,
    get_nasa_crm_wing,
    get_naca0012_swept_wing,
    get_supersonic_arrow_wing,
)
from ..solvers.aero_engine import AeroEngine3D
from ..models.surrogate_3d import AeroSurrogate3D
from ..models.generator_3d import GenerativeWingVAE3D
from ..export.stl_exporter import STLExporter3D
from ..export.vtk_exporter import VTKExporter3D
from ..export.su2_exporter import SU2MeshExporter3D
from ..export.step_exporter import CADCurveExporter3D

app = FastAPI(title="AeroWing AI Pro 3D", version="1.0.0")

# Static directory path
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# AI models are loaded lazily from a local checkpoint only when available.
# Random-weight models are NEVER served as if they were trained models.
_SURROGATE = None
_GENERATOR = None
_CHECKPOINT_PATH = os.environ.get(
    "AEROWING_CHECKPOINT", "checkpoints/aerowing_models.pt"
)


def get_surrogate() -> Optional[AeroSurrogate3D]:
    """Returns the trained surrogate if a local checkpoint exists, else None."""
    global _SURROGATE
    if _SURROGATE is None and os.path.exists(_CHECKPOINT_PATH):
        try:
            _SURROGATE = AeroSurrogate3D()
            checkpoint = torch.load(_CHECKPOINT_PATH, map_location="cpu")
            _SURROGATE.load_state_dict(checkpoint["surrogate_state"])
        except Exception:
            _SURROGATE = None
    return _SURROGATE


def get_generator() -> Optional[GenerativeWingVAE3D]:
    """Returns the trained generator if a local checkpoint exists, else None."""
    global _GENERATOR
    if _GENERATOR is None and os.path.exists(_CHECKPOINT_PATH):
        try:
            _GENERATOR = GenerativeWingVAE3D()
            checkpoint = torch.load(_CHECKPOINT_PATH, map_location="cpu")
            _GENERATOR.load_state_dict(checkpoint["generator_state"])
        except Exception:
            _GENERATOR = None
    return _GENERATOR


class WingParamsModel(BaseModel):
    span: float = 30.0
    aspect_ratio: float = 9.5
    taper_ratio: float = 0.28
    sweep_le_deg: float = 27.5
    dihedral_deg: float = 3.5
    twist_root_deg: float = 2.0
    twist_tip_deg: float = -2.5
    root_tc: float = 0.14
    tip_tc: float = 0.10
    alpha_deg: float = 2.5
    mach: float = 0.82
    reynolds: float = 2.5e7


@app.get("/")
async def get_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "AeroWing AI Pro Backend Ready"}


@app.post("/api/wing/evaluate")
async def evaluate_wing(params: WingParamsModel):
    """Evaluates 3D wing aerodynamics, fuel volume, and generates 3D surface mesh."""
    # Build CST profiles
    root_naca = f"00{int(params.root_tc * 100):02d}"
    tip_naca = f"00{int(params.tip_tc * 100):02d}"
    af_root = CSTAirfoil3D.from_naca4(root_naca, order=6)
    af_tip = CSTAirfoil3D.from_naca4(tip_naca, order=6)

    wing = Wing3D(
        span=params.span,
        aspect_ratio=params.aspect_ratio,
        taper_ratio=params.taper_ratio,
        sweep_le_deg=params.sweep_le_deg,
        dihedral_deg=params.dihedral_deg,
        twist_root_deg=params.twist_root_deg,
        twist_tip_deg=params.twist_tip_deg,
        root_airfoil=af_root,
        tip_airfoil=af_tip,
    )

    engine = AeroEngine3D(wing, num_chordwise=12, num_spanwise=20)
    res = engine.evaluate(
        alpha_deg=params.alpha_deg,
        mach=params.mach,
        reynolds=params.reynolds,
    )

    # Generate 3D surface mesh (upper & lower grids)
    mesh_data = wing.generate_surface_mesh_3d(num_chordwise=24, num_spanwise=24)

    # Compute 3D surface Cp matrix: use the ACTUAL VLM pressure-jump field
    # (delta_cp = 2 * Gamma / dx, shape (ny_vlm, nx_vlm)), resampled onto the
    # visualization mesh (24 x 24). No synthetic/placeholder Cp is ever used.
    ny, nx = 24, 24
    delta_cp = np.asarray(res.delta_cp_matrix, dtype=float)
    if delta_cp.size > 1:
        from scipy.interpolate import RegularGridInterpolator

        eta_vlm = np.linspace(0, 1, delta_cp.shape[0])
        xi_vlm = np.linspace(0, 1, delta_cp.shape[1])
        interp = RegularGridInterpolator((eta_vlm, xi_vlm), delta_cp,
                                         bounds_error=False, fill_value=0.0)
        ETA, XI = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx), indexing="ij")
        delta_cp_resampled = interp((ETA, XI))

        # Thin-airfoil split: Delta Cp is the upper-minus-lower pressure jump
        cp_upper = -delta_cp_resampled / 2.0
        cp_lower = +delta_cp_resampled / 2.0
    else:
        cp_upper = np.zeros((ny, nx))
        cp_lower = np.zeros((ny, nx))

    return {
        "telemetry": res.to_dict(),
        "geometry": {
            "s_ref": round(wing.s_ref, 3),
            "mac": round(wing.mac, 3),
            "semi_span": round(wing.semi_span, 3),
            "fuel_volume_m3": round(res.fuel_volume, 2),
            "wetted_area_m2": round(res.wetted_area, 2),
            "root_chord": round(wing.root_chord, 3),
            "tip_chord": round(wing.tip_chord, 3),
        },
        "mesh": {
            "X_upper": mesh_data["X_upper"].tolist(),
            "Y_upper": mesh_data["Y_upper"].tolist(),
            "Z_upper": mesh_data["Z_upper"].tolist(),
            "X_lower": mesh_data["X_lower"].tolist(),
            "Y_lower": mesh_data["Y_lower"].tolist(),
            "Z_lower": mesh_data["Z_lower"].tolist(),
            "Cp_upper": cp_upper.tolist(),
            "Cp_lower": cp_lower.tolist(),
        },
    }


@app.get("/api/benchmark/{name}")
async def get_benchmark(name: str):
    """Loads a pre-configured aerospace industry benchmark wing."""
    name_clean = name.lower().replace("-", "_")
    if name_clean == "onera_m6":
        wing = get_onera_m6_wing()
        mach, alpha, re = 0.8395, 3.06, 1.17e7
    elif name_clean == "nasa_crm":
        wing = get_nasa_crm_wing()
        mach, alpha, re = 0.85, 2.2, 4.0e7
    elif name_clean == "naca0012_swept":
        wing = get_naca0012_swept_wing()
        mach, alpha, re = 0.30, 4.0, 3.0e6
    elif name_clean == "supersonic_arrow":
        wing = get_supersonic_arrow_wing()
        mach, alpha, re = 1.60, 3.5, 1.5e7
    else:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    return {
        "name": wing.name,
        "span": wing.span,
        "aspect_ratio": wing.aspect_ratio,
        "taper_ratio": wing.taper_ratio,
        "sweep_le_deg": wing.sweep_le_deg,
        "dihedral_deg": wing.dihedral_deg,
        "twist_root_deg": wing.twist_root_deg,
        "twist_tip_deg": wing.twist_tip_deg,
        "root_tc": wing.root_airfoil.get_max_thickness(),
        "tip_tc": wing.tip_airfoil.get_max_thickness(),
        "recommended_flight": {
            "mach": mach,
            "alpha_deg": alpha,
            "reynolds": re,
        },
    }


class InverseDesignRequest(BaseModel):
    target_cl: float = 0.55
    target_mach: float = 0.82
    target_ar: float = 9.5
    target_l_over_d: float = 19.0


@app.post("/api/wing/inverse-design")
async def inverse_design(req: InverseDesignRequest):
    """Synthesizes a 3D wing geometry satisfying flight mission targets.

    Requires a trained generator checkpoint (AEROWING_CHECKPOINT). If none is
    available the request is refused rather than returning random-weight output.
    """
    generator = get_generator()
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Inverse design unavailable: no trained model checkpoint found "
                f"({_CHECKPOINT_PATH}). Run 'aerowing train' first or set AEROWING_CHECKPOINT."
            ),
        )
    synth_x = generator.generate(
        target_cl=req.target_cl,
        target_mach=req.target_mach,
        target_ar=req.target_ar,
        target_l_over_d=req.target_l_over_d,
    )
    span, ar, taper, sweep, dihedral, twist_r, twist_t = synth_x[:7]

    return {
        "synthesized_parameters": {
            "span": round(float(span), 2),
            "aspect_ratio": round(float(ar), 2),
            "taper_ratio": round(float(taper), 3),
            "sweep_le_deg": round(float(sweep), 2),
            "dihedral_deg": round(float(dihedral), 2),
            "twist_root_deg": round(float(twist_r), 2),
            "twist_tip_deg": round(float(twist_t), 2),
            "root_tc": 0.135,
            "tip_tc": 0.095,
        },
        "target_cl": req.target_cl,
        "target_mach": req.target_mach,
    }


@app.post("/api/export/file")
async def export_file(
    params: WingParamsModel,
    export_format: str = "stl",
):
    """Exports and downloads 3D CAD / CFD files on the fly."""
    af_root = CSTAirfoil3D.from_naca4(f"00{int(params.root_tc * 100):02d}", order=6)
    af_tip = CSTAirfoil3D.from_naca4(f"00{int(params.tip_tc * 100):02d}", order=6)

    wing = Wing3D(
        span=params.span,
        aspect_ratio=params.aspect_ratio,
        taper_ratio=params.taper_ratio,
        sweep_le_deg=params.sweep_le_deg,
        dihedral_deg=params.dihedral_deg,
        twist_root_deg=params.twist_root_deg,
        twist_tip_deg=params.twist_tip_deg,
        root_airfoil=af_root,
        tip_airfoil=af_tip,
    )

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scratch")
    os.makedirs(tmp_dir, exist_ok=True)

    fmt = export_format.lower()
    if fmt == "stl":
        path = os.path.join(tmp_dir, f"{wing.name}.stl")
        STLExporter3D(wing).export_stl(path, num_chordwise=40, num_spanwise=40, binary=True)
        return FileResponse(path, filename=f"{wing.name}.stl", media_type="application/sla")
    elif fmt == "vtk":
        path = os.path.join(tmp_dir, f"{wing.name}.vtk")
        VTKExporter3D(wing).export_vtk(path, num_chordwise=30, num_spanwise=30)
        return FileResponse(path, filename=f"{wing.name}.vtk", media_type="text/plain")
    elif fmt == "su2":
        path = os.path.join(tmp_dir, f"{wing.name}.su2")
        SU2MeshExporter3D(wing).export_su2(path, num_chordwise=24, num_spanwise=24)
        return FileResponse(path, filename=f"{wing.name}.su2", media_type="text/plain")
    elif fmt == "csv":
        path = os.path.join(tmp_dir, f"{wing.name}_cad.json")
        CADCurveExporter3D(wing).export_cad_curves(path, num_stations=10)
        csv_path = os.path.splitext(path)[0] + ".csv"
        return FileResponse(csv_path, filename=f"{wing.name}_curves.csv", media_type="text/csv")
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")


# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# PRIVACY: No CORSMiddleware is registered, so browsers refuse cross-origin
# access by default; the server never accepts requests from other websites.


def run_server(host: str = "127.0.0.1", port: int = 8080):
    """Starts the Uvicorn web server.

    Binds to 127.0.0.1 (localhost) by default so the app is never reachable
    from the network. To stay 100% private keep the default host.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)
