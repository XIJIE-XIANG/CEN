#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DATASET_URL = "https://drive.google.com/file/d/1a0eDgwk-6tjjZQCigShkX_S3cgHybZHT/view?usp=drive_link"

try:
    from autofocus_cen import CENConfig
    from autofocus_cen.generated import (
        DEFAULT_GT_IMAGE_MAP,
        build_default_scene_list,
        build_scene_list_from_generated_root,
        compare_one_scene,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from autofocus_cen import CENConfig
    from autofocus_cen.generated import (
        DEFAULT_GT_IMAGE_MAP,
        build_default_scene_list,
        build_scene_list_from_generated_root,
        compare_one_scene,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CEN and baselines on generated spikes")
    parser.add_argument(
        "--scene_dirs",
        nargs="*",
        default=None,
        help="Scene directories. If omitted, read simu* scenes under --generated_root unless --base_dir is set.",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Optional base directory containing simu01/simu02/simu03.",
    )
    parser.add_argument(
        "--generated_root",
        default="./simulate_moderate_light",
        help="Root directory of generated spikes.",
    )
    parser.add_argument("--dt", type=int, default=10, help="Spike frames accumulated into one block.")
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Optional output directory. Default: ./results_dt{dt}",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_mean_rows_long(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    methods = []
    for row in rows:
        method = row.get("method")
        if method not in methods:
            methods.append(method)

    out = list(rows)
    numeric_cols = ["abs_err", "image_abs_err"]
    for method in methods:
        sub = [row for row in rows if row.get("method") == method]
        mean_row = {"scene": "MEAN", "method": method}
        for col in numeric_cols:
            values = [float(row[col]) for row in sub if col in row]
            if values:
                mean_row[col] = sum(values) / len(values)
        out.append(mean_row)
    return out


def main() -> None:
    args = parse_args()
    if args.dt <= 0:
        raise ValueError("dt must be positive")

    generated_root = Path(args.generated_root)
    if not generated_root.exists():
        raise FileNotFoundError(
            f"generated_root not found: {generated_root}\n"
            f"Download the public dataset from {DATASET_URL} and pass the extracted "
            "simulate_moderate_light or simulate_low_light directory via --generated_root."
        )

    if args.scene_dirs:
        scene_dirs = [Path(p) for p in args.scene_dirs]
    elif args.base_dir:
        scene_dirs = build_default_scene_list(args.base_dir)
    else:
        scene_dirs = build_scene_list_from_generated_root(generated_root)
    save_dir = Path(args.save_dir or f"./results_dt{args.dt}")
    config = CENConfig()

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    print("========== COMPARE SPIKE-DOMAIN METHODS ON GENERATED SPIKES ==========")
    print(f"generated_root = {args.generated_root}")
    print(f"save_dir       = {save_dir}")

    for scene_dir in scene_dirs:
        scene_name = Path(scene_dir).resolve().name
        gt_image_name = DEFAULT_GT_IMAGE_MAP.get(scene_name)
        if gt_image_name is None:
            print(f"[warn] no default GT image for {scene_name}; skipped")
            continue

        try:
            wide_row, scene_long_rows = compare_one_scene(
                scene_dir=scene_dir,
                generated_root=args.generated_root,
                dt=args.dt,
                gt_image_name=gt_image_name,
                config=config,
            )
        except Exception as exc:
            print(f"[ERROR] scene={scene_name}, dt={args.dt}, error={exc}")
            wide_row = {"scene": scene_name, "dt": args.dt, "error": str(exc)}
            scene_long_rows = [{"scene": scene_name, "dt": args.dt, "method": "ERROR", "error": str(exc)}]

        wide_rows.append(wide_row)
        long_rows.extend(scene_long_rows)
        if "OURS_CEN_pred_block" in wide_row:
            print(
                f"{scene_name}: OURS_CEN pred_block={wide_row['OURS_CEN_pred_block']} "
                f"pred_focus={wide_row['OURS_CEN_pred_focus']:.8f} "
                f"gt_focus={wide_row['gt_focus']:.8f} "
                f"abs_err={wide_row['OURS_CEN_abs_err']:.8f} "
                f"r2={wide_row['OURS_CEN_r2_hat']:.4f}"
            )

    summary_csv = save_dir / "summary.csv"
    methods_csv = save_dir / "methods.csv"
    write_csv(summary_csv, wide_rows)
    write_csv(methods_csv, append_mean_rows_long(long_rows))
    print(f"[save] {summary_csv}")
    print(f"[save] {methods_csv}")
    print("========== DONE ==========")


if __name__ == "__main__":
    main()
