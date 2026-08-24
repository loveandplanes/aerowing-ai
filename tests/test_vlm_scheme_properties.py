"""Regression guards for the VLM's *scheme personality* — the converged,
documented properties established by the tier-2 verification battery.

These are NOT correctness-of-physics claims against experiment; they pin the
discretization's signature so silent regressions (normalization, moment arm,
mirror assembly, wake handling) fail loudly:

  * elliptic planform -> span efficiency ~ 1 (Trefftz/CL joint normalization)
  * lift slope sits in the horseshoe-VLM band below lifting line, and
    approaches the 2*pi/rad thin-airfoil limit as AR grows
  * pitching-moment arm is nx-converged (swept AC ~ 0.30 MAC is physical)
  * LE<->TE swap asymmetry exists, is bounded, and is nx-INVARIANT
    (legs-from-quarter-chord breaks mirror symmetry by construction)
  * Trefftz downwash / CDi identities hold to machine precision
  * finite-wake door effect stays negligible
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from aerowing.geometry.wing_3d import Wing3D, WingSection
from aerowing.geometry.benchmarks import get_naca0012_swept_wing
from aerowing.solvers.vlm_3d import VLMSolver3D

AF = get_naca0012_swept_wing().root_airfoil


def _make(b, ar, lam, sweep, rev=False, n_sec=33):
    semi = b / 2.0
    c0 = (b / ar) * 2.0 / (1.0 + lam)
    secs = []
    for eta in np.linspace(0.0, 1.0, n_sec):
        c = c0 * (1.0 - (1.0 - lam) * eta)
        x_le = eta * semi * np.tan(np.radians(sweep)) + (c if rev else 0.0)
        secs.append(WingSection(float(eta), float(c), float(x_le), 0.0,
                                0.0, AF))
    return Wing3D(span=b, aspect_ratio=ar, taper_ratio=lam,
                  sweep_le_deg=sweep, dihedral_deg=0.0,
                  twist_root_deg=0.0, twist_tip_deg=0.0, sections=secs)


def _slope(wing, nx, ny):
    s = VLMSolver3D(wing, num_chordwise=nx, num_spanwise=ny)
    return (s.solve(alpha_deg=2.0)["cl"]
            - s.solve(alpha_deg=1.0)["cl"]) / np.radians(1.0)


def _elliptic_wing(b=30.0, ar=9.5, n_sec=33):
    c_root = 4.0 * b / (np.pi * ar)
    secs = []
    for eta in np.linspace(0.0, 1.0, n_sec):
        c = c_root * np.sqrt(max(1.0 - eta * eta, 1e-6))
        secs.append(WingSection(float(eta), float(c), 0.0, 0.0, 0.0, AF))
    return Wing3D(span=b, aspect_ratio=ar, taper_ratio=1.0, sweep_le_deg=0.0,
                  dihedral_deg=0.0, twist_root_deg=0.0, twist_tip_deg=0.0,
                  sections=secs)


def test_elliptic_span_efficiency_near_one():
    wing = _elliptic_wing()
    r = VLMSolver3D(wing, num_chordwise=12, num_spanwise=24).solve(alpha_deg=2.0)
    assert abs(r["span_efficiency"] - 1.0) < 0.02


def test_elliptic_lift_slope_in_horseshoe_band():
    ar = 9.5
    a = _slope(_elliptic_wing(), 12, 24)
    a_ll = 2.0 * np.pi * ar / (ar + 2.0)
    assert abs(a - a_ll) / a_ll < 0.06
    assert a < a_ll  # horseshoe schemes sit below lifting line


def test_rectangular_slope_approaches_2pi():
    a10 = _slope(_make(30.0, 10.0, 1.0, 0.0), 12, 32)
    a30 = _slope(_make(30.0, 30.0, 1.0, 0.0), 12, 48)
    ll10 = 2.0 * np.pi * 10.0 / 12.0
    ll30 = 2.0 * np.pi * 30.0 / 32.0
    assert abs(a10 - ll10) / ll10 < 0.09
    assert abs(a30 - ll30) / ll30 < 0.05
    assert a30 > a10  # slope grows toward 2*pi with AR


def test_cm_arm_nx_converged_swept_ac_physical():
    wing = _make(30.0, 9.5, 0.28, 27.5)

    def dcm_dcl(nx, ny):
        s = VLMSolver3D(wing, num_chordwise=nx, num_spanwise=ny)
        als = np.radians((1.0, 3.0, 5.0, 7.0))
        cls = [s.solve(alpha_deg=d)["cl"] for d in (1.0, 3.0, 5.0, 7.0)]
        cms = [s.solve(alpha_deg=d)["cm"] for d in (1.0, 3.0, 5.0, 7.0)]
        return np.polyfit(als, cms, 1)[0] / np.polyfit(als, cls, 1)[0]

    v1 = dcm_dcl(12, 24)
    v2 = dcm_dcl(24, 36)
    assert abs(v1 - v2) < 0.01            # arm is grid-converged
    assert -0.08 < v1 < -0.02             # swept AC aft of qc-MAC: physical


def test_reciprocity_gap_bounded_and_nx_invariant():
    orig = _make(30.0, 9.5, 0.28, 27.5)
    rev = _make(30.0, 9.5, 0.28, 27.5, rev=True)
    gaps = []
    for nx in (6, 12):
        c_o = VLMSolver3D(orig, num_chordwise=nx, num_spanwise=24).solve(
            alpha_deg=2.0)["cl"]
        c_r = VLMSolver3D(rev, num_chordwise=nx, num_spanwise=24).solve(
            alpha_deg=2.0)["cl"]
        gaps.append((c_r - c_o) / c_o)
    assert 0.01 < abs(gaps[0]) < 0.07                 # scheme-level, bounded
    assert abs(gaps[0] - gaps[1]) < 0.005             # exactly nx-invariant


def test_trefftz_identities_machine_exact():
    wing = Wing3D(span=30.0, aspect_ratio=9.5, taper_ratio=0.28,
                  sweep_le_deg=27.5)
    r = VLMSolver3D(wing, num_chordwise=12, num_spanwise=24).solve(alpha_deg=2.5)
    g, w_t = r["gamma_spanwise"], r["w_trefftz"]
    ny = len(g)
    semi = wing.semi_span
    y_edges = np.arange(ny + 1) / ny * semi
    dg = np.diff(np.concatenate([[g[0]], g, [0.0]]))
    y_c = (np.arange(ny) + 0.5) / ny * semi
    w_mine = np.array([(np.sum(dg / (yj - y_edges))
                        - np.sum(dg / (yj + y_edges))) / (4.0 * np.pi)
                       for yj in y_c])
    assert np.max(np.abs(w_mine - w_t)) < 1e-13
    cdi_mine = float((4.0 / wing.s_ref) * np.sum(g * w_mine * (semi / ny)))
    assert abs(cdi_mine - r["cd_induced"]) < 1e-14


def test_wake_length_insensitive():
    wing = Wing3D(span=30.0, aspect_ratio=9.5, taper_ratio=0.28,
                  sweep_le_deg=27.5)
    ra = VLMSolver3D(wing, num_chordwise=12, num_spanwise=24,
                     wake_length=100.0 * wing.mac).solve(alpha_deg=2.5)
    rb = VLMSolver3D(wing, num_chordwise=12, num_spanwise=24,
                     wake_length=400.0 * wing.mac).solve(alpha_deg=2.5)
    assert abs(rb["cl"] - ra["cl"]) / ra["cl"] < 5e-4
    assert abs(rb["cd_induced"] - ra["cd_induced"]) / ra["cd_induced"] < 5e-4
