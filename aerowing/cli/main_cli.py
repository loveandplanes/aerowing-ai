"""
AeroWing AI Pro Unified Aerospace CLI.
Provides command-line interfaces for 3D aerodynamic analysis, AI training, inverse design, and export.
"""

import argparse
import sys
import os
import json
import numpy as np

from ..config import load_config, wing_from_config, flight_condition_from_config

from ..geometry.wing_3d import Wing3D
from ..geometry.benchmarks import (
    get_onera_m6_wing,
    get_nasa_crm_wing,
    get_naca0012_swept_wing,
    get_supersonic_arrow_wing,
)
from ..solvers.aero_engine import AeroEngine3D
from ..models.surrogate_3d import AeroSurrogate3D
from ..models.generator_3d import GenerativeWingVAE3D
from ..models.dataset_3d import WingDataset3D, generate_synthetic_wing_dataset
from ..models.trainer_3d import AeroTrainer3D
from ..export.stl_exporter import STLExporter3D
from ..export.vtk_exporter import VTKExporter3D
from ..export.su2_exporter import SU2MeshExporter3D
from ..export.step_exporter import CADCurveExporter3D
from ..web.server import run_server
from ..mesher_3d import VolumeMesher3D


def get_wing_from_name(name: str) -> Wing3D:
    name_clean = name.lower().replace("-", "_")
    if name_clean == "onera_m6":
        return get_onera_m6_wing()
    elif name_clean == "nasa_crm":
        return get_nasa_crm_wing()
    elif name_clean == "naca0012_swept":
        return get_naca0012_swept_wing()
    elif name_clean == "supersonic_arrow":
        return get_supersonic_arrow_wing()
    else:
        return Wing3D(name=name)


def main():
    parser = argparse.ArgumentParser(
        description="AeroWing AI Pro — 3D Aerospace Aerodynamic AI Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info
    parser_info = subparsers.add_parser("info", help="Print system information and capabilities")

    # Analyze
    parser_analyze = subparsers.add_parser("analyze", help="Perform 3D aerodynamic evaluation")
    parser_analyze.add_argument("--wing", type=str, default="onera_m6", help="Wing name or benchmark")
    parser_analyze.add_argument("--config", type=str, default=None,
                                help="YAML config from configs/ (overrides --wing; e.g. onera_m6, nasa_crm)")
    parser_analyze.add_argument("--alpha", type=float, default=None, help="Angle of attack in degrees")
    parser_analyze.add_argument("--mach", type=float, default=None, help="Flight Mach number")
    parser_analyze.add_argument("--reynolds", type=float, default=None, help="Flight Reynolds number")
    parser_analyze.add_argument("--uncertainty", action="store_true",
                                help="Also show per-prediction UQ bands from the ensemble "
                                     "checkpoint (checkpoints/aerowing_ensemble.pt)")

    # Train
    parser_train = subparsers.add_parser("train", help="Train 3D AI surrogate and generative models")
    parser_train.add_argument("--samples", type=int, default=80, help="Number of 3D wing dataset samples")
    parser_train.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser_train.add_argument("--checkpoint", type=str, default="checkpoints/aerowing_models.pt")
    parser_train.add_argument("--ensemble", type=int, default=0, metavar="K",
                              help="Also train a K-member deep ensemble (UQ bands) "
                                   "and save it to checkpoints/aerowing_ensemble.pt "
                                   "(K >= 2; 0 = off)")

    # Inverse Design
    parser_inverse = subparsers.add_parser("inverse-design", help="Synthesize 3D wing for target aerodynamics")
    parser_inverse.add_argument("--target-cl", type=float, default=0.55, help="Target cruise lift coefficient")
    parser_inverse.add_argument("--mach", type=float, default=0.82, help="Target Mach number")
    parser_inverse.add_argument("--target-ar", type=float, default=9.5, help="Target aspect ratio")
    parser_inverse.add_argument("--target-ld", type=float, default=19.5, help="Target L/D efficiency")

    # Export
    parser_export = subparsers.add_parser("export", help="Export 3D wing to CAD/CFD formats")
    parser_export.add_argument("--wing", type=str, default="onera_m6", help="Wing name or benchmark")
    parser_export.add_argument("--format", type=str, default="stl", help="Export formats (stl, vtk, su2, csv)")
    parser_export.add_argument("--outdir", type=str, default="outputs/cad_export")

    # Mesh generator
    parser_mesher = subparsers.add_parser(
        "mesher", help="Generate structured hexahedral O-grid volume meshes for CFD")
    mesher_sub = parser_mesher.add_subparsers(dest="mesher_command",
                                              help="Mesher action")
    p_volume = mesher_sub.add_parser(
        "volume",
        help="Build a y+ -resolved O-grid around a parametric wing and write SU2")
    p_volume.add_argument("--wing", type=str, default="onera_m6",
                          help="Wing name or benchmark")
    p_volume.add_argument("--out", type=str, default=None,
                          help="SU2 output file (default: outputs/mesh/<wing>_o_grid.su2)")
    p_volume.add_argument("--coarse", action="store_true",
                          help="Fast verification mesh (~0.1M cells, y+ = 5)")
    p_volume.add_argument("--y-plus", type=float, default=None,
                          help="Wall spacing target; auto layer count (default 1, coarse 5)")
    p_volume.add_argument("--growth", type=float, default=1.2,
                          help="Geometric layer growth ratio")
    p_volume.add_argument("--far-mult", type=float, default=15.0,
                          help="Far-field radius in MACs (minimum)")

    # Serve Web UI
    parser_serve = subparsers.add_parser("serve", help="Launch interactive 3D Web UI")
    parser_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser_serve.add_argument("--port", type=int, default=8080, help="Port number")

    # Continual learning
    parser_cont = subparsers.add_parser(
        "continual",
        help="Continuous learning: ingest CFD results, fine-tune the surrogate, "
             "let the tool keep improving from runs you do anyway")
    cont_sub = parser_cont.add_subparsers(dest="cont_command", help="Continual action")

    p_status = cont_sub.add_parser("status", help="Show data lake statistics")
    p_status.add_argument("--lake", type=str, default="data_lake/aero.sqlite",
                          help="SQLite data lake path")

    p_ingest = cont_sub.add_parser(
        "ingest",
        help="Ingest one CFD run as a training label (quality-gated)")
    p_ingest.add_argument("--lake", type=str, default="data_lake/aero.sqlite")
    p_ingest.add_argument("--forces", type=str, required=True,
                          help="SU2 forces/log/forces_breakdown text file")
    p_ingest.add_argument("--convergence", type=str, default=None,
                          help="Optional SU2 convergence history file")
    p_ingest.add_argument("--design-json", type=str, required=True,
                          help="JSON with {\"x\": [40]} or {\"design\": [37], "
                               "\"flight\": {...}} or {\"wing\": {...}, \"flight\": {...}}")
    p_ingest.add_argument("--source", type=str, default="su2",
                          help="Solver label, e.g. su2, cfx, windtunnel")

    p_update = cont_sub.add_parser(
        "update",
        help="Fine-tune the surrogate from newly accepted lake data and "
             "promote the checkpoint only if the holdout did not regress")
    p_update.add_argument("--lake", type=str, default="data_lake/aero.sqlite")
    p_update.add_argument("--checkpoint", type=str, default="checkpoints/aerowing_models.pt")
    p_update.add_argument("--epochs", type=int, default=6)
    p_update.add_argument("--min-new", type=int, default=8,
                          help="Minimum new accepted samples per update")
    p_update.add_argument("--auto-grow", action="store_true",
                          help="Grow model capacity when holdout improvement plateaus")

    p_collect = cont_sub.add_parser(
        "collect",
        help="Batch-ingest every quality-gated CFD run under a directory "
             "(zero manual steps, idempotent, safe on a schedule)")
    p_collect.add_argument("--dir", type=str, required=True,
                           help="Root directory of CFD run outputs")
    p_collect.add_argument("--lake", type=str, default="data_lake/aero.sqlite")
    p_collect.add_argument("--design-json", type=str, default=None,
                           help="Design spec applied to runs without their own design.json")
    p_collect.add_argument("--source", type=str, default="su2")
    p_collect.add_argument("--dry-run", action="store_true",
                           help="Only report what would be ingested")
    p_collect.add_argument("--update", action="store_true",
                           help="Run continual update after ingestion")
    p_collect.add_argument("--update-min-new", type=int, default=8)
    p_collect.add_argument("--no-recurse", action="store_true",
                           help="Only scan the given directory, not subdirectories")

    p_mf = cont_sub.add_parser(
        "mf-study",
        help="Multi-fidelity 'money slide': measure how fast the learning "
             "loop descends toward high-fidelity truth as expensive labels "
             "accrue, vs the flat error of VLM alone")
    p_mf.add_argument("--seed", type=int, default=1337,
                      help="Determinism seed (same seed -> same curve)")
    p_mf.add_argument("--designs", type=int, default=160,
                      help="Candidate designs, all labeled at VLM fidelity")
    p_mf.add_argument("--holdout", type=int, default=48,
                      help="Unseen designs whose high-fidelity truth fixes "
                           "the error metric (never trained on)")
    p_mf.add_argument("--budgets", type=str, default="0,16,32,64,128",
                      help="Comma-separated CFD label budgets to sweep")
    p_mf.add_argument("--warm-epochs", type=int, default=18,
                      help="Epochs for the VLM warm-start training")
    p_mf.add_argument("--epochs", type=int, default=6,
                      help="Fine-tune epochs per ContinualTrainer update")
    p_mf.add_argument("--out", type=str, default=None,
                      help="Optional JSON file to receive the full study")
    p_mf.add_argument("--workdir", type=str, default=None,
                      help="Keep lake/checkpoint artifacts here (default: "
                           "OS temp, cleaned up)")
    p_mf.add_argument("--truth", type=str, default="engine",
                      choices=("engine", "lake"),
                      help="High-fidelity truth source: 'engine' = synthetic "
                           "AeroEngine3D stand-in (mechanism demo only); "
                           "'lake' = real accepted CFD rows from --lake-path")
    p_mf.add_argument("--lake-path", type=str, default=None,
                      help="Data-lake SQLite file for --truth lake "
                           "(default: data_lake/aero.sqlite)")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        print("=" * 70)
        print("  AEROWING AI PRO — 3D AERODYNAMIC DESIGN & CFD PLATFORM v1.0.0")
        print("=" * 70)
        print("  • 3D Parametric CAD: Multi-station CST-3D lofting, planform schedules")
        print("  • Physics Solvers: 3D VLM, Trefftz induced drag, compressible boundary layer, wave drag")
        print("  • Neural Models: AeroSurrogate3D (<5ms), GenerativeWingVAE3D, deep-ensemble UQ bands")
        print("  • Optimization: NSGA-II Multi-Objective Pareto, SQP gradient MDO")
        print("  • Continuous Learning: CFD feedback loop + quality gates + auto growth")
        print("  • Uncertainty: per-prediction ±bands that tighten as CFD labels accrue")
        print("  • Exporters: STL (3D watertight), VTK (ParaView), SU2 (CFD mesh), CSV (CAD curves)")
        print("  • Mesher: y+ -resolved structured O-grid volume meshes for CFD")
        print("  • Web Studio: Real-time Three.js 3D WebGL HUD Dashboard")
        print("=" * 70)
        print("Usage: aerowing [info|analyze|train|inverse-design|export|mesher|serve|continual]")
        print("       continual [status|ingest|update|collect|mf-study]  -- learn from CFD runs you already do")
        print("       mesher [volume]  -- y+ -resolved O-grid volume mesh for CFD")
        print("       train --ensemble 4  /  analyze --uncertainty  -- per-prediction UQ bands")

    elif args.command == "analyze":
        alpha = args.alpha
        mach = args.mach
        reynolds = args.reynolds
        if args.config is not None:
            # YAML config wins over --wing defaults and supplies flight conditions
            config = load_config(args.config)
            wing = wing_from_config(config)
            fc = flight_condition_from_config(config)
            if fc is not None:
                alpha = alpha if alpha is not None else fc["alpha_deg"]
                mach = mach if mach is not None else fc["mach"]
                reynolds = reynolds if reynolds is not None else fc["reynolds"]
            print(f"[config] Loaded geometry from configs/{args.config}.yaml")
        else:
            wing = get_wing_from_name(args.wing)
            alpha = alpha if alpha is not None else 2.5
            mach = mach if mach is not None else 0.82
            reynolds = reynolds if reynolds is not None else 2.5e7

        print(f"\n--- Analyzing 3D Wing: {wing.name} ---")
        print(f"Planform: Span = {wing.span:.2f} m | AR = {wing.aspect_ratio:.2f} | Taper = {wing.taper_ratio:.3f} | Sweep = {wing.sweep_le_deg:.1f}°")
        print(f"S_ref = {wing.s_ref:.2f} m² | MAC = {wing.mac:.2f} m | Fuel Volume = {wing.compute_internal_fuel_volume():.2f} m³")

        engine = AeroEngine3D(wing)
        res = engine.evaluate(alpha_deg=alpha, mach=mach, reynolds=reynolds)

        print(f"\nFlight Condition: Mach = {mach:.3f} | Alpha = {alpha:.2f}° | Re = {reynolds:.2e}")
        print(f"Results:")
        print(f"  Lift Coefficient (C_L):         {res.cl:.4f}")
        print(f"  Total Drag (C_D):               {res.cd:.5f} ({(res.cd * 10000):.1f} drag counts)")
        print(f"    - Induced Drag (C_Di):        {res.cd_induced:.5f}")
        print(f"    - Profile Drag (C_Dp):        {res.cd_profile:.5f}")
        print(f"    - Wave Drag (C_Dw):           {res.cd_wave:.5f}")
        print(f"  Aero Efficiency (L/D):          {res.l_over_d:.2f}")
        print(f"  Span Efficiency (e):            {res.span_efficiency:.3f}")
        print(f"  Pitching Moment (C_M):          {res.cm:.4f}")

        if args.uncertainty:
            from ..models.ensemble_3d import (
                EnsembleSurrogate3D, OUTPUT_NAMES, uncertainty_label)
            ens_path = os.path.join(
                "checkpoints", "aerowing_ensemble.pt")
            if not os.path.exists(ens_path):
                print("\n[uncertainty] no ensemble checkpoint found - run "
                      "`aerowing train --ensemble 4` first")
            else:
                ens = EnsembleSurrogate3D.load(ens_path)
                out = ens.predict_wing(
                    wing.to_parameter_vector(),
                    alpha_deg=alpha, mach=mach, reynolds=reynolds or 2.5e7)
                mean = np.array([out[n] for n in OUTPUT_NAMES])
                std = np.array([out[n + "_uncertainty"] for n in OUTPUT_NAMES])
                print(f"\nUncertainty bands ({ens.n_members}-member ensemble, 2-sigma):")
                for name in OUTPUT_NAMES:
                    print(f"  {uncertainty_label(mean, std, name, width=2.0)}")
                print("  (bands shrink as the learning loop accrues CFD labels)")

    elif args.command == "train":
        print(f"\n--- Generating {args.samples} 3D Wing Samples and Training Neural Models ---")
        x_data, y_data = generate_synthetic_wing_dataset(num_samples=args.samples, verbose=True)
        dataset = WingDataset3D(x_data, y_data)
        trainer = AeroTrainer3D()
        print(f"\n--- Training AeroSurrogate3D with Physics Regularization ---")
        trainer.train_surrogate(dataset, epochs=args.epochs, verbose=True)
        print("\n--- Training GenerativeWingVAE3D with KL Annealing ---")
        trainer.train_generator(dataset, epochs=args.epochs, verbose=True)
        trainer.save_checkpoint(args.checkpoint)
        print(f"\n[SUCCESS] Model weights saved to {args.checkpoint}")
        if args.ensemble >= 2:
            from ..models.ensemble_3d import train_ensemble_surrogate
            print(f"\n--- Training {args.ensemble}-member Deep Ensemble (UQ bands) ---")
            ens = train_ensemble_surrogate(
                dataset, n_members=args.ensemble, epochs=args.epochs)
            ens_path = os.path.join(
                os.path.dirname(os.path.abspath(args.checkpoint)),
                "aerowing_ensemble.pt")
            ens.save(ens_path)
            print(f"[SUCCESS] Ensemble ({args.ensemble} members) saved to {ens_path}")
        elif args.ensemble != 0:
            print("\n[WARN] --ensemble must be >= 2 to train a deep ensemble; skipped")

    elif args.command == "inverse-design":
        print(f"\n--- AI Inverse Design for Target C_L={args.target_cl}, Mach={args.mach}, AR={args.target_ar} ---")
        generator = GenerativeWingVAE3D()
        surrogate = AeroSurrogate3D()
        synth_x = generator.generate(
            target_cl=args.target_cl,
            target_mach=args.mach,
            target_ar=args.target_ar,
            target_l_over_d=args.target_ld,
        )
        synth_wing = Wing3D.from_parameter_vector(synth_x, name="Synthesized_AeroWing")
        tele = surrogate.predict_wing(synth_x, alpha_deg=2.5, mach=args.mach)

        print("Synthesized 3D Wing Configuration:")
        print(f"  Span:         {synth_wing.span:.2f} m")
        print(f"  Aspect Ratio: {synth_wing.aspect_ratio:.2f}")
        print(f"  Taper Ratio:  {synth_wing.taper_ratio:.3f}")
        print(f"  LE Sweep:     {synth_wing.sweep_le_deg:.2f}°")
        print(f"  Dihedral:     {synth_wing.dihedral_deg:.2f}°")
        print(f"  Tip Washout:  {synth_wing.twist_tip_deg:.2f}°")
        print(f"Predicted Performance: C_L = {tele['cl']:.3f} | C_D = {tele['cd']:.4f} | L/D = {tele['l_over_d']:.2f}")

    elif args.command == "export":
        wing = get_wing_from_name(args.wing)
        os.makedirs(args.outdir, exist_ok=True)
        formats = [f.strip().lower() for f in args.format.split(",")]

        print(f"\n--- Exporting 3D Wing: {wing.name} to {args.outdir} ---")
        for fmt in formats:
            if fmt == "stl":
                p = os.path.join(args.outdir, f"{wing.name}.stl")
                STLExporter3D(wing).export_stl(p)
                print(f"  ✓ Exported 3D STL Mesh: {p}")
            elif fmt == "vtk":
                p = os.path.join(args.outdir, f"{wing.name}.vtk")
                VTKExporter3D(wing).export_vtk(p)
                print(f"  ✓ Exported ParaView VTK: {p}")
            elif fmt == "su2":
                p = os.path.join(args.outdir, f"{wing.name}.su2")
                SU2MeshExporter3D(wing).export_su2(p)
                print(f"  ✓ Exported SU2 3D Mesh: {p}")
            elif fmt == "csv":
                p = os.path.join(args.outdir, f"{wing.name}_cad.json")
                CADCurveExporter3D(wing).export_cad_curves(p)
                print(f"  ✓ Exported CAD Cross-Sections: {os.path.splitext(p)[0]}.csv")

    elif args.command == "mesher":
        if args.mesher_command == "volume" or args.mesher_command is None:
            wing = get_wing_from_name(args.wing)
            kwargs = dict(
                n_loop=48 if args.coarse else 96,
                n_stations=16 if args.coarse else 64,
                root_plug=4 if args.coarse else 8,
                tip_chain=6 if args.coarse else 12,
                growth=args.growth,
                far_field_mult=args.far_mult,
                y_plus=5.0 if args.coarse else 1.0,
            )
            if args.y_plus is not None:
                kwargs["y_plus"] = args.y_plus
            if args.out:
                out = args.out
            else:
                out = os.path.join("outputs", "mesh", f"{wing.name}_o_grid.su2")
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

            print(f"\n--- Volume O-Grid Mesher: {wing.name} ---")
            print(f"  Span {wing.span:.2f} m | MAC {wing.mac:.2f} m | "
                  f"S_ref {wing.s_ref:.2f} m2")
            m = VolumeMesher3D(wing, **kwargs)
            print(f"  Ring {m.ring_size} pts | stations {m.n_stations} | "
                  f"layers {m.n_layers} | y1 {m.first_cell_height:.3e} m | "
                  f"far field {m.far_radius:.2f} m")
            print(f"  Building...")
            mesh = m.build()
            stats = mesh.validate()
            mesh.export_su2(out)
            print(f"  Nodes {stats['n_nodes']:,} | cells {stats['n_cells']:,} | "
                  f"wall {stats['n_wall_faces']:,} | far {stats['n_far_faces']:,}")
            print(f"  Min jacobian {stats['min_jacobian']:.4e} | inverted "
                  f"{stats['inverted_cells']}")
            if stats["inverted_cells"]:
                print("  WARNING: inverted cells present - mesh unusable")
            print(f"  [OK] SU2 volume mesh written to {out}")
        else:
            print("Usage: aerowing mesher [volume]")

    elif args.command == "serve":
        print(f"\n🚀 Launching AeroStudio 3D Web UI on http://{args.host}:{args.port}")
        print("   PRIVACY: localhost-only, zero external requests, no telemetry.")
        run_server(host=args.host, port=args.port)

    elif args.command == "continual":
        _run_continual(args)


def _run_continual(args):
    from ..continual import (
        AeroDataLake,
        CfdQualityGate,
        ContinualTrainer,
        parse_su2_forces,
        parse_su2_residuals,
    )
    from ..collector import (
        SU2BatchCollector,
        design_spec_to_input,
        label_from_forces,
    )

    if args.cont_command == "status" or args.cont_command is None:
        lake = AeroDataLake(args.lake)
        stats = lake.stats()
        print(f"\n--- Continual Learning Data Lake: {args.lake} ---")
        print(f"  Samples:            {stats['samples']}")
        print(f"  Accepted labels:    {stats['accepted']}")
        print(f"  Exploration rows:   {stats['exploration']}")
        print(f"  Holdout rows:       {stats['holdout']}")
        baseline = lake.get_meta("holdout_mse_baseline")
        print(f"  Holdout MSE baseline: {baseline if baseline is not None else 'not set'}")
        print(f"  Last processed id:  {lake.get_meta('last_processed_id', 0)}")
        promotions = lake.history_vals("promote", "holdout_mse")
        if promotions:
            best = min(promotions)
            print(f"  Promotions:         {len(promotions)} (best holdout MSE {best:.6f})")
        lake.close()
        return

    if args.cont_command == "ingest":
        lake = AeroDataLake(args.lake)
        with open(args.forces, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if args.convergence:
            with open(args.convergence, "r", encoding="utf-8", errors="replace") as f:
                text += "\n" + f.read()

        gate = CfdQualityGate()
        result = gate.gate_su2_text(text) if args.convergence is None else gate.gate(
            parse_su2_forces(text), residuals=parse_su2_residuals(text))
        forces = parse_su2_forces(text)

        spec = json.load(open(args.design_json, "r", encoding="utf-8"))
        try:
            x_row = design_spec_to_input(spec)
        except ValueError as exc:
            print(f"\n[ingest] rejected: {exc}")
            lake.close()
            return
        y_row, mask = label_from_forces(forces)

        row_id = lake.append(
            x_row, y_row, source=f"cfd:{args.source}", mask=mask,
            accepted=result.accepted,
            gate_reason="; ".join(result.reasons),
            y_vlm=spec.get("y_vlm"))
        status = "ACCEPTED" if result.accepted else "REJECTED"
        print(f"\n[ingest] sample #{row_id} {status}"
              f" | source cfd:{args.source} | CL={forces.get('cl', float('nan')):.4f} "
              f"CD={forces.get('cd', float('nan')):.4f}")
        if not result.accepted:
            print("  rejected reasons: " + "; ".join(result.reasons))
        lake.close()
        return

    if args.cont_command == "collect":
        lake = AeroDataLake(args.lake)
        default_design = None
        if args.design_json:
            default_design = json.load(open(args.design_json, "r", encoding="utf-8"))
        collector = SU2BatchCollector(lake, source=args.source,
                                      default_design=default_design)
        summary = collector.collect(
            args.dir, recurse=not args.no_recurse, dry_run=args.dry_run,
            auto_update=args.update, update_min_new=args.update_min_new)
        if args.dry_run:
            print(f"\n[collect --dry-run] {summary['new']} new runs would be ingested"
                  f" (of {summary['runs_found']} found, "
                  f"{summary['skipped_duplicates']} already known)")
        else:
            print(f"\n[collect] {summary['new']} new runs | "
                  f"{summary['accepted']} accepted | {summary['rejected']} rejected")
            for reason, count in summary["rejected_reasons"].items():
                print(f"    rejected x{count}: {reason}")
            if summary.get("update"):
                u = summary["update"]
                if u["updated"]:
                    action = ("promoted" if u["promoted"] else "refused (holdout regressed)")
                    print(f"  [update] {action} | holdout MSE {u['holdout_mse']:.6f}")
                else:
                    print(f"  [update] skipped: {u.get('reason')}")
        lake.close()
        return

    if args.cont_command == "mf-study":
        from ..multi_fidelity import MultiFidelityStudy
        budgets = [int(b.strip()) for b in args.budgets.split(",")
                   if b.strip()]
        if not budgets:
            print("--budgets must contain at least one budget")
            return
        lake_path = args.lake_path
        if args.truth == "lake" and not lake_path:
            lake_path = os.path.join("data_lake", "aero.sqlite")
            if not os.path.exists(lake_path):
                print("--truth lake requires --lake-path (no data_lake/"
                      "aero.sqlite in the current directory)")
                return
        study = MultiFidelityStudy(seed=args.seed)
        try:
            result = study.run(
                n_designs=args.designs, n_holdout=args.holdout, budgets=budgets,
                warm_epochs=args.warm_epochs, finetune_epochs=args.epochs,
                workdir=args.workdir, verbose=True,
                truth=args.truth, lake_path=lake_path)
        except ValueError as exc:
            print(f"[mf-study] {exc}")
            return
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                        exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\n[mf-study] JSON written to {args.out}")
        print(result.table(key="cd"))
        print(result.ascii_curve(key="cd"))
        return

    if args.cont_command == "update":
        lake = AeroDataLake(args.lake)
        trainer = ContinualTrainer(lake, checkpoint_path=args.checkpoint)
        summary = trainer.update(epochs=args.epochs, min_new_samples=args.min_new,
                                 auto_grow=args.auto_grow)
        if not summary["updated"]:
            print(f"\n[continual] nothing to do: {summary.get('reason')}")
        elif summary["promoted"]:
            print(f"\n[continual] checkpoint promoted -> {args.checkpoint} "
                  f"(holdout MSE {summary['holdout_mse']:.6f})")
        else:
            print(f"\n[continual] update refused: holdout regressed "
                  f"({summary['holdout_mse']:.6f} vs baseline "
                  f"{summary['holdout_baseline']:.6f}) - keeping previous checkpoint")
        lake.close()
        return

    print("Usage: aerowing continual [status|ingest|update|collect]")


if __name__ == "__main__":
    main()
