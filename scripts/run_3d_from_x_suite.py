from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the inference-only EagleVision 3D-from-X evaluation suite.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dataset-format", default="scannet")
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--frames-per-scene", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands/status without executing.")
    return parser


def _cmd_to_str(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run(cmd: list[str], dry_run: bool) -> tuple[str, str]:
    print(_cmd_to_str(cmd))
    if dry_run:
        return "planned", ""
    completed = subprocess.run(cmd, check=False)
    return ("ok" if completed.returncode == 0 else "failed", str(completed.returncode))


def main() -> int:
    args = build_argparser().parse_args()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    manifests = out_root / "manifests"
    predictions = out_root / "predictions"
    raw = out_root / "raw"
    aggregated = out_root / "aggregated"

    frozen_pred = predictions / "da2s_frozen"
    adapted_pred = predictions / "da2s_eaglevision_adapted"
    commands: list[tuple[str, list[str]]] = [
        (
            "build_manifests",
            [
                sys.executable,
                "-m",
                "eaglevision.cli.build_rgbd_manifests",
                "--root",
                str(args.root),
                "--out-dir",
                str(manifests),
                "--dataset-format",
                args.dataset_format,
                "--seed",
                str(args.seed),
                "--max-frames-per-scene",
                str(args.frames_per_scene),
            ],
        ),
        (
            "dump_frozen",
            [
                sys.executable,
                "-m",
                "eaglevision.cli.dump_depth_predictions",
                "--config",
                "configs/3d_eval/da2s_frozen.yaml",
                "--frame-manifest",
                str(manifests / "frames_val.jsonl"),
                "--method",
                "frozen",
                "--out-dir",
                str(frozen_pred),
            ],
        ),
        (
            "dump_adapted",
            [
                sys.executable,
                "-m",
                "eaglevision.cli.dump_depth_predictions",
                "--config",
                "configs/3d_eval/da2s_eaglevision.yaml",
                "--frame-manifest",
                str(manifests / "frames_val.jsonl"),
                "--method",
                "adapted",
                "--checkpoint",
                str(args.checkpoint),
                "--out-dir",
                str(adapted_pred),
            ],
        ),
    ]

    task_specs = [
        ("single_frame_depth", "eaglevision.cli.eval_single_frame_depth", ["--pred-manifest"], []),
        ("cycle", "eaglevision.cli.eval_cycle_consistency", ["--pair-manifest", str(manifests / "pairs_val.jsonl"), "--pred-manifest"], []),
        ("reprojection", "eaglevision.cli.eval_rgbd_reprojection", ["--pair-manifest", str(manifests / "pairs_val.jsonl"), "--pred-manifest"], []),
        ("view_warping", "eaglevision.cli.eval_view_warping", ["--pair-manifest", str(manifests / "pairs_val.jsonl"), "--pred-manifest"], []),
        ("normals", "eaglevision.cli.eval_surface_normals", ["--pred-manifest"], []),
        (
            "pointcloud",
            "eaglevision.cli.eval_pointcloud_fusion",
            ["--frame-manifest", str(manifests / "frames_val.jsonl"), "--pred-manifest"],
            ["--frames-per-scene", str(args.frames_per_scene), "--max-scenes", str(args.max_scenes)],
        ),
    ]
    for task, module, prefix, suffix in task_specs:
        for label, pred_manifest in (("frozen", frozen_pred / "manifest.jsonl"), ("adapted", adapted_pred / "manifest.jsonl")):
            commands.append(
                (
                    f"{task}_{label}",
                    [
                        sys.executable,
                        "-m",
                        module,
                        *prefix,
                        str(pred_manifest),
                        "--out",
                        str(raw / f"{task}_{label}.csv"),
                        *suffix,
                    ],
                )
            )
        commands.append(
            (
                f"aggregate_{task}",
                [
                    sys.executable,
                    "scripts/aggregate_3d_eval_results.py",
                    "--task",
                    task,
                    "--inputs",
                    str(raw / f"{task}_frozen.csv"),
                    str(raw / f"{task}_adapted.csv"),
                    "--baseline",
                    "frozen",
                    "--out-dir",
                    str(aggregated / task),
                ],
            )
        )

    scaffold_specs = [
        (
            "tsdf",
            "eaglevision.cli.eval_tsdf_fusion",
            ["--frame-manifest", str(manifests / "frames_val.jsonl"), "--pred-manifest"],
            ["--frames-per-scene", str(args.frames_per_scene)],
        ),
        ("mesh", "eaglevision.cli.eval_depth_to_mesh", ["--pred-manifest"], []),
        ("pose_alignment", "eaglevision.cli.eval_pose_alignment_diagnostic", ["--pair-manifest", str(manifests / "pairs_val.jsonl"), "--pred-manifest"], []),
        ("stereo", "eaglevision.cli.eval_stereo_depth", ["--pred-manifest"], []),
    ]
    for task, module, prefix, suffix in scaffold_specs:
        for label, pred_manifest in (("frozen", frozen_pred / "manifest.jsonl"), ("adapted", adapted_pred / "manifest.jsonl")):
            commands.append(
                (
                    f"{task}_{label}",
                    [
                        sys.executable,
                        "-m",
                        module,
                        *prefix,
                        str(pred_manifest),
                        "--out",
                        str(raw / f"{task}_{label}.csv"),
                        *suffix,
                    ],
                )
            )

    status_rows = []
    for name, cmd in commands:
        status, code = _run(cmd, args.dry_run)
        status_rows.append({"step": name, "status": status, "returncode": code, "command": _cmd_to_str(cmd)})
        if status == "failed":
            break

    with (out_root / "run_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "status", "returncode", "command"])
        writer.writeheader()
        writer.writerows(status_rows)
    summary = ["# 3D-from-X Suite Run", "", f"- Root: `{args.root}`", f"- Checkpoint: `{args.checkpoint}`", f"- Dry run: `{args.dry_run}`", ""]
    summary += [f"- `{row['step']}`: {row['status']}" for row in status_rows]
    (out_root / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0 if all(row["status"] in {"ok", "planned"} for row in status_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
