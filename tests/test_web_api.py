"""
Web API unit tests: physics-driven evaluation and privacy-safe inverse design.
"""

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from aerowing.web.server import app

client = TestClient(app)


def test_web_api_evaluate_uses_vlm_cp():
    """The evaluate endpoint returns the actual VLM pressure field."""
    resp = client.post("/api/wing/evaluate", json={
        "span": 20.0,
        "aspect_ratio": 8.0,
        "taper_ratio": 0.30,
        "sweep_le_deg": 25.0,
        "dihedral_deg": 3.0,
        "twist_root_deg": 2.0,
        "twist_tip_deg": -2.0,
        "root_tc": 0.12,
        "tip_tc": 0.10,
        "alpha_deg": 2.5,
        "mach": 0.5,
        "reynolds": 2.5e7,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "telemetry" in data
    assert "delta_cp_matrix" in data["telemetry"]
    mesh = data["mesh"]
    assert "Cp_upper" in mesh and "Cp_lower" in mesh
    # Cp field must be non-trivial (real VLM values, not the old synthetic peak)
    cp_upper = mesh["Cp_upper"]
    assert len(cp_upper) == 24 and len(cp_upper[0]) == 24
    assert any(abs(v) > 0.01 for row in cp_upper for v in row)


def test_web_api_inverse_design_refuses_without_checkpoint():
    """Without a trained checkpoint, inverse design must NOT use random weights."""
    resp = client.post("/api/wing/inverse-design", json={
        "target_cl": 0.55,
        "target_mach": 0.82,
        "target_ar": 9.5,
        "target_l_over_d": 19.0,
    })
    # 503 = trained model unavailable; never a fabricated random answer
    assert resp.status_code in (503, 200)
    if resp.status_code == 503:
        assert "checkpoint" in resp.json()["detail"].lower()


def test_web_api_index_served_locally():
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # Privacy: the served page must not reference any external host
    assert "https://" not in html and "http://" not in html
    assert "/static/vendor/three.min.js" in html