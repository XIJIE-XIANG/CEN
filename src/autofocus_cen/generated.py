from __future__ import annotations

import glob
import csv
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .cen import CENConfig, estimate_focus_from_npy_files, normalize_01, peak_top_centroid
from .io import stream_blocks_from_npy_files


DEFAULT_GT_IMAGE_MAP = {
    "simu01": "Im37",
    "simu02": "Im37",
    "simu03": "Im13",
}


def natural_key(path: str | Path) -> list[object]:
    name = Path(path).name
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def build_default_scene_list(base_dir: str | Path) -> list[Path]:
    base = Path(base_dir)
    return [base / "simu01", base / "simu02", base / "simu03"]


def build_scene_list_from_generated_root(generated_root: str | Path) -> list[Path]:
    root = Path(generated_root)
    return sorted([path for path in root.glob("simu*") if path.is_dir()], key=natural_key)


def gt_name_to_index(gt_name: str) -> int:
    match = re.search(r"Im(\d+)", gt_name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot parse GT image name: {gt_name}")
    return int(match.group(1)) - 1


def find_generated_spike_files(scene_dir: str | Path, generated_root: str | Path) -> tuple[list[Path], Path]:
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.resolve().name
    candidates = [
        Path(generated_root) / scene_name / "spikes_npy",
        scene_dir / "spikes_npy",
        scene_dir,
    ]

    for directory in candidates:
        if not directory.is_dir():
            continue
        files = sorted(glob.glob(str(directory / "Im*_spikes.npy")), key=natural_key)
        valid = [
            Path(path)
            for path in files
            if re.search(r"Im\d+_spikes\.npy$", Path(path).name, re.IGNORECASE)
        ]
        if valid:
            return valid, directory

    raise FileNotFoundError(f"cannot find Im*_spikes.npy for scene {scene_name}")


def load_focus_distances(scene_dir: str | Path, expected_len: int | None = None) -> tuple[np.ndarray, Path]:
    scene_dir = Path(scene_dir)
    csv_path = scene_dir / "focus_distances.csv"
    json_path = scene_dir / "focus_distances.json"

    if csv_path.exists():
        values = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values.append(float(row["focus_distance"]))
        arr = np.asarray(values, dtype=np.float64)
        source = csv_path
    elif json_path.exists():
        import json

        data = json.loads(json_path.read_text())
        arr = np.asarray(data["focus_distances"], dtype=np.float64)
        source = json_path
    else:
        raise FileNotFoundError(f"focus_distances.csv not found under {scene_dir}")

    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"{source} has {arr.size} focus distances, expected {expected_len}")
    return arr, source


def radial_grid(height: int, width: int) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32) - (height / 2.0)
    xx = np.arange(width, dtype=np.float32) - (width / 2.0)
    y, x = np.meshgrid(yy, xx, indexing="ij")
    y = y / float(height)
    x = x / float(width)
    return np.sqrt(x * x + y * y).astype(np.float32)


def make_radial_bins(r: np.ndarray, r_max: float = 0.35, k_bins: int = 256) -> np.ndarray:
    mask = r <= r_max
    rr = r.copy()
    rr[~mask] = 0.0
    rr_norm = np.clip(rr / r_max, 0.0, 1.0 - 1e-12)
    idx = (rr_norm * k_bins).astype(np.int32)
    idx = np.clip(idx, 0, k_bins - 1)
    idx_map = idx.copy()
    idx_map[~mask] = -1
    return idx_map


def grad_score_fast(block: np.ndarray) -> float:
    dx = block[:, 1:] - block[:, :-1]
    dy = block[1:, :] - block[:-1, :]
    return float(np.sum(dx * dx) + np.sum(dy * dy))


def laplacian_energy_fast(block: np.ndarray, mean_value: float) -> float:
    padded = np.pad(block, ((1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return float(np.mean(np.abs(up + down + left + right - 4.0 * center))) / (mean_value + 1e-6)


def mfdct_filter_energy(block: np.ndarray) -> float:
    padded = np.pad(block, ((1, 2), (1, 2)), mode="symmetric")
    response = (
        padded[:-3, :-3]
        + padded[:-3, 1:-2]
        - padded[:-3, 2:-1]
        - padded[:-3, 3:]
        + padded[1:-2, :-3]
        + padded[1:-2, 1:-2]
        - padded[1:-2, 2:-1]
        - padded[1:-2, 3:]
        - padded[2:-1, :-3]
        - padded[2:-1, 1:-2]
        + padded[2:-1, 2:-1]
        + padded[2:-1, 3:]
        - padded[3:, :-3]
        - padded[3:, 1:-2]
        + padded[3:, 2:-1]
        + padded[3:, 3:]
    )
    return float(np.sum(response * response, dtype=np.float64))


def run_baselines_stream(
    npy_files: Sequence[str | Path],
    dt: int,
    r_max: float = 0.35,
    k_bins: int = 256,
    hf_r1: float = 0.05,
    hf_r2: float = 0.30,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    arr0 = np.load(npy_files[0], mmap_mode="r")
    _, height, width = arr0.shape
    del arr0

    r = radial_grid(height, width)
    idx_map = make_radial_bins(r, r_max=r_max, k_bins=k_bins)
    valid_mask = idx_map >= 0
    idx_flat = idx_map[valid_mask].reshape(-1).astype(np.int32)

    b1 = int(np.clip(int(np.floor((hf_r1 / r_max) * k_bins)), 0, k_bins - 1))
    b2 = int(np.clip(int(np.floor((hf_r2 / r_max) * k_bins)), 0, k_bins - 1))
    lo_b, hi_b = min(b1, b2), max(b1, b2)

    curves = {
        "SD": [],
        "SPIKE_COUNT": [],
        "GRAD": [],
        "MFDCT_FILTER": [],
        "LAPLACIAN_ENERGY": [],
        "HF_ENERGY": [],
    }

    block_ids = []
    for block_id, block_sum in stream_blocks_from_npy_files(npy_files, dt):
        block = block_sum.astype(np.float32, copy=False)
        centered = block - float(block.mean())
        power = np.fft.fftshift((np.abs(np.fft.fft2(centered)) ** 2).astype(np.float64))
        p_flat = power[valid_mask].reshape(-1).astype(np.float64)
        energy = np.bincount(idx_flat, weights=p_flat, minlength=k_bins).astype(np.float64)

        mean_value = float(block.mean())
        var_value = float(((block - mean_value) ** 2).mean())
        total_energy = float(np.sum(energy)) + 1e-12
        hf_energy = float(np.sum(energy[lo_b : hi_b + 1]))

        curves["SD"].append(var_value / (mean_value * mean_value + 1e-12))
        curves["SPIKE_COUNT"].append(float(block.sum()))
        curves["GRAD"].append(grad_score_fast(block))
        curves["MFDCT_FILTER"].append(mfdct_filter_energy(block))
        curves["LAPLACIAN_ENERGY"].append(laplacian_energy_fast(block, mean_value))
        curves["HF_ENERGY"].append(hf_energy / total_energy)
        block_ids.append(block_id)

    return np.asarray(block_ids, dtype=np.int32), {
        name: np.asarray(values, dtype=np.float64) for name, values in curves.items()
    }


def pred_by_peak(curve: Sequence[float], edge_frac: float = 0.05, tau: float = 0.012) -> tuple[int, np.ndarray]:
    y = normalize_01(curve)
    block_count = int(y.size)
    margin = max(1, int(edge_frac * block_count))
    pred = peak_top_centroid(y, tau=tau)
    pred = int(np.clip(pred, margin, max(margin, block_count - margin - 1)))
    return pred, y


def compare_one_scene(
    scene_dir: str | Path,
    generated_root: str | Path,
    dt: int,
    gt_image_name: str,
    config: CENConfig = CENConfig(),
) -> list[dict[str, object]]:
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.resolve().name
    npy_files, spikes_dir = find_generated_spike_files(scene_dir, generated_root)

    num_input_images = len(npy_files)
    focus_distances, focus_path = load_focus_distances(Path(spikes_dir).parent, expected_len=num_input_images)
    gt_image_idx = gt_name_to_index(gt_image_name)
    if gt_image_idx < 0 or gt_image_idx >= num_input_images:
        raise ValueError(f"GT image {gt_image_name} is out of range for {scene_name}")

    spikes_per_image = int(np.load(npy_files[0], mmap_mode="r").shape[0])
    gt_block_center = int(
        np.clip(int(np.round((gt_image_idx * spikes_per_image) / float(dt))), 0, 10**9)
    )
    gt_focus = float(focus_distances[gt_image_idx])

    cen_result = estimate_focus_from_npy_files(npy_files, dt=dt, config=config)
    _block_ids, base_curves = run_baselines_stream(npy_files, dt=dt)

    methods = {
        "OURS_CEN": (int(cen_result.focus_block), cen_result.normalized_curve),
    }
    for name, curve in base_curves.items():
        methods[name] = pred_by_peak(curve, edge_frac=config.edge_ratio, tau=0.012)

    method_rows: list[dict[str, object]] = []

    for method_name, (pred_block, y_norm) in methods.items():
        y_norm = np.asarray(y_norm, dtype=np.float64)
        pred_spike_frame_center = float(pred_block) * float(dt) + 0.5 * float(max(dt - 1, 0))
        pred_img_idx = int(
            np.clip(int(np.round(pred_spike_frame_center / float(spikes_per_image))), 0, num_input_images - 1)
        )
        pred_img_name = f"Im{pred_img_idx + 1:02d}"
        image_abs_err = int(abs(pred_img_idx - gt_image_idx))
        pred_focus = float(focus_distances[pred_img_idx])
        focus_signed_err = float(pred_focus - gt_focus)
        abs_err = float(abs(focus_signed_err))

        method_row = {
            "scene": scene_name,
            "dt": dt,
            "method": method_name,
            "gt_block_center": gt_block_center,
            "pred_block": int(pred_block),
            "gt_image": gt_image_name,
            "pred_image": pred_img_name,
            "pred_image_idx_0based": pred_img_idx,
            "image_abs_err": image_abs_err,
            "gt_focus": gt_focus,
            "pred_focus": pred_focus,
            "focus_signed_err": focus_signed_err,
            "abs_err": abs_err,
        }
        if method_name == "OURS_CEN":
            method_row["r2_hat"] = float(cen_result.r2)
            method_row.update(cen_result.r2_score_detail)
        method_rows.append(method_row)

    return method_rows
