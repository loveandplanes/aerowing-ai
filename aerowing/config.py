"""
YAML Configuration Loader for AeroWing AI Pro.
Provides geometry (and flight-condition defaults) loading from configs/*.yaml,
wiring the packaged benchmark configurations into the CLI and web tools.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str = "default_3d") -> Dict[str, Any]:
    """
    Loads a YAML configuration by base name (e.g. "onera_m6", "nasa_crm").
    Raises FileNotFoundError when the config does not exist.
    """
    config_path = _CONFIGS_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration '{name}' not found. Available: "
            + ", ".join(p.stem for p in sorted(_CONFIGS_DIR.glob("*.yaml")))
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def wing_from_config(config: Dict[str, Any]) -> "Wing3D":
    """
    Builds a Wing3D from the 'geometry' section of a loaded configuration.
    """
    from .geometry.wing_3d import Wing3D
    from .geometry.cst_3d import CSTAirfoil3D

    g = config.get("geometry", {})

    def _naca_from_tc(tc: float) -> str:
        thickness = int(round(float(tc) * 100.0))
        return f"00{thickness:02d}"

    root_af = CSTAirfoil3D.from_naca4(
        _naca_from_tc(g.get("root_tc", 0.14)), order=int(g.get("cst_order", 6))
    )
    tip_af = CSTAirfoil3D.from_naca4(
        _naca_from_tc(g.get("tip_tc", 0.10)), order=int(g.get("cst_order", 6))
    )

    return Wing3D(
        name=str(g.get("name", "Config_Wing")),
        span=float(g["span"]),
        aspect_ratio=float(g["aspect_ratio"]),
        taper_ratio=float(g["taper_ratio"]),
        sweep_le_deg=float(g["sweep_le_deg"]),
        dihedral_deg=float(g.get("dihedral_deg", 0.0)),
        twist_root_deg=float(g.get("twist_root_deg", 0.0)),
        twist_tip_deg=float(g.get("twist_tip_deg", 0.0)),
        root_airfoil=root_af,
        tip_airfoil=tip_af,
    )


def flight_condition_from_config(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Returns the flight_condition section of a configuration, if present."""
    fc = config.get("flight_condition")
    if not fc:
        return None
    return {
        "alpha_deg": float(fc.get("alpha_deg", 2.5)),
        "mach": float(fc.get("mach", 0.82)),
        "reynolds": float(fc.get("reynolds", 2.5e7)),
    }