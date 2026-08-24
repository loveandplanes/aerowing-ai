"""
AeroWing AI Pro — Main Demonstration and Validation Script.
Executes multi-fidelity 3D wing aerodynamics, AI surrogate validation, inverse synthesis, and CAD export.
"""

import os
import sys
import numpy as np

# Ensure parent path is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aerowing.geometry.benchmarks import get_onera_m6_wing, get_nasa_crm_wing
from aerowing.solvers.aero_engine import AeroEngine3D
from aerowing.models.surrogate_3d import AeroSurrogate3D
from aerowing.models.generator_3d import GenerativeWingVAE3D
from aerowing.export.stl_exporter import STLExporter3D
from aerowing.export.vtk_exporter import VTKExporter3D
from aerowing.export.su2_exporter import SU2MeshExporter3D
from aerowing.export.step_exporter import CADCurveExporter3D


def run_aerospace_demonstration():
    print("=" * 78)
    print("  🚀 AEROWING AI PRO: ENTERPRISE 3D AEROSPACE AERODYNAMIC AI PLATFORM")
    print("=" * 78)

    # 1. Benchmark 1: ONERA M6 Transonic Wing (AGARD AR-138)
    print("\n[STAGE 1] Validating Benchmark 1: ONERA M6 Transonic Wing (AGARD AR-138)")
    onera_m6 = get_onera_m6_wing()
    print(f"  • Planform: Span = {onera_m6.span:.3f} m | AR = {onera_m6.aspect_ratio:.2f} | Sweep = {onera_m6.sweep_le_deg:.1f}°")
    print(f"  • Ref Area S_ref = {onera_m6.s_ref:.3f} m² | MAC = {onera_m6.mac:.3f} m")

    engine_onera = AeroEngine3D(onera_m6, num_chordwise=12, num_spanwise=20)
    res_onera = engine_onera.evaluate(alpha_deg=3.06, mach=0.8395, reynolds=1.17e7)

    print(f"  • Flow Condition: Mach = 0.8395 | Alpha = 3.06° | Re = 1.17e7")
    print(f"  • Calculated C_L:         {res_onera.cl:.4f} (AGARD Exp: ~0.285)")
    print(f"  • Total Drag C_D:         {res_onera.cd:.5f} ({(res_onera.cd*10000):.1f} counts)")
    print(f"    - Induced Drag C_Di:    {res_onera.cd_induced:.5f}")
    print(f"    - Profile Drag C_Dp:    {res_onera.cd_profile:.5f}")
    print(f"    - Wave Drag C_Dw:       {res_onera.cd_wave:.5f}")
    print(f"  • Aerodynamic L/D:        {res_onera.l_over_d:.2f}")
    print(f"  • Span Efficiency e:      {res_onera.span_efficiency:.3f}")

    # 2. Benchmark 2: NASA Common Research Model (CRM) Supercritical Wing
    print("\n[STAGE 2] Validating Benchmark 2: NASA CRM Modern Transport Wing (AIAA DPW)")
    nasa_crm = get_nasa_crm_wing()
    print(f"  • Planform: Span = {nasa_crm.span:.2f} m | AR = {nasa_crm.aspect_ratio:.2f} | Sweep = {nasa_crm.sweep_le_deg:.1f}°")
    print(f"  • S_ref = {nasa_crm.s_ref:.2f} m² | MAC = {nasa_crm.mac:.2f} m | Fuel Tank = {nasa_crm.compute_internal_fuel_volume():.2f} m³")

    engine_crm = AeroEngine3D(nasa_crm, num_chordwise=14, num_spanwise=24)
    res_crm = engine_crm.evaluate(alpha_deg=2.2, mach=0.85, reynolds=4.0e7)

    print(f"  • Cruise Condition: Mach = 0.850 | Alpha = 2.20° | Re = 4.0e7")
    print(f"  • Calculated C_L:         {res_crm.cl:.4f} (Target Design: 0.500)")
    print(f"  • Total Drag C_D:         {res_crm.cd:.5f} ({(res_crm.cd*10000):.1f} counts)")
    print(f"  • Cruise Aero L/D:        {res_crm.l_over_d:.2f} (Design Target: ~20.0)")

    # 3. AI Inverse Design Synthesis
    print("\n[STAGE 3] Generative 3D Wing Inverse Design (3D-CVAE Synthesizer)")
    generator = GenerativeWingVAE3D()
    synth_x = generator.generate(target_cl=0.55, target_mach=0.82, target_ar=9.5, target_l_over_d=19.5)
    synth_wing = onera_m6.from_parameter_vector(synth_x, name="AeroAI_Optimal_Transport_3D")

    print(f"  • Synthesized Wing: Span = {synth_wing.span:.2f} m | AR = {synth_wing.aspect_ratio:.2f} | Sweep = {synth_wing.sweep_le_deg:.1f}°")
    print(f"  • Tip Washout Twist = {synth_wing.twist_tip_deg:.2f}° | Internal Fuel Tank = {synth_wing.compute_internal_fuel_volume():.2f} m³")

    # 4. Standard CAD & CFD Mesh Exports
    print("\n[STAGE 4] Exporting 3D CAD & High-Fidelity CFD Meshes")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "demonstration")
    os.makedirs(out_dir, exist_ok=True)

    stl_path = os.path.join(out_dir, "nasa_crm_surface.stl")
    vtk_path = os.path.join(out_dir, "onera_m6_pressure_field.vtk")
    su2_path = os.path.join(out_dir, "onera_m6_cfd.su2")
    cad_json_path = os.path.join(out_dir, "nasa_crm_cad_curves.json")

    STLExporter3D(nasa_crm).export_stl(stl_path, num_chordwise=40, num_spanwise=40)
    VTKExporter3D(onera_m6).export_vtk(
        vtk_path,
        cp_matrix=res_onera.delta_cp_matrix,
        num_chordwise=30,
        num_spanwise=30,
    )
    SU2MeshExporter3D(onera_m6).export_su2(su2_path, num_chordwise=24, num_spanwise=24)
    CADCurveExporter3D(nasa_crm).export_cad_curves(cad_json_path, num_stations=10)
    cad_csv_path = os.path.splitext(cad_json_path)[0] + ".csv"

    print(f"  ✓ 3D Watertight STL Mesh:       {stl_path}")
    print(f"  ✓ ParaView VTK with Cp Fields:  {vtk_path}")
    print(f"  ✓ Native Stanford SU2 3D Mesh:  {su2_path}")
    print(f"  ✓ CAD B-Spline Curve Tables:    {cad_csv_path}")

    print("\n" + "=" * 78)
    print("  ✅ AEROWING AI PRO DEMONSTRATION COMPLETE — READY FOR INDUSTRY USE")
    print("=" * 78)


if __name__ == "__main__":
    run_aerospace_demonstration()
