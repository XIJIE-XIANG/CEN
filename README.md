# CEN Autofocus

This repository provides the official code for our ICML 2026 paper, "Autofocus for Spike Cameras via Spectral Centroid in the Frequency Domain".

The code runs CEN-based autofocus on generated spike-camera simulation datasets and reports focus-distance errors for moderate-light and low-light settings.

## Install

```bash
python3 -m pip install -e .
```

## Dataset

Download the simulation dataset used in the paper from Google Drive:

```text
https://drive.google.com/file/d/1a0eDgwk-6tjjZQCigShkX_S3cgHybZHT/view?usp=drive_link
```

The downloaded file is `simulate_dataset.zip`. After extraction, it contains two dataset directories:

```text
simulate_moderate_light/
simulate_low_light/
```

Place either directory at the repository root, or pass its full path with `--generated_root`.

## Generated Simulation Layout

The comparison script expects generated spikes in this layout:

```text
simulate_moderate_light/
  dataset_manifest.json
  simu01/focus_distances.csv
  simu01/focus_distances.json
  simu01/spikes_npy/Im1_spikes.npy
  simu01/spikes_npy/Im2_spikes.npy
  ...
  simu02/spikes_npy/Im1_spikes.npy
  ...
  simu03/spikes_npy/Im1_spikes.npy
  ...
simulate_low_light/
  dataset_manifest.json
  simu01/focus_distances.csv
  simu01/spikes_npy/Im1_spikes.npy
  ...
```

Each `Im*_spikes.npy` file is a `[T, H, W]` array. Values are binarized with `>0` before block accumulation, matching the experiment script.

Each scene also contains `focus_distances.csv`, exported from the original `data.mat` focus-distance vector. After export, the code only needs one simulation-light directory.

The default public path placeholder is:

```text
--generated_root ./simulate_moderate_light
```

## Compare Generated Spikes

```bash
python3 scripts/spike_autofocus_CEN.py \
  --generated_root ./simulate_moderate_light \
  --dt 10
```

For the low-light simulation:

```bash
python3 scripts/spike_autofocus_CEN.py \
  --generated_root ./simulate_low_light \
  --dt 10
```

By default the script uses the same GT image mapping as the internal experiment script:

```text
simu01 -> Im37
simu02 -> Im37
simu03 -> Im13
```

Outputs are written to `results_spike_autofocus_CEN_dt{dt}/summary_wide_scene_compare.csv` and `summary_long_method_compare.csv`. The main error column is `abs_err`, the absolute focus-distance error.

## Manifest Mode

For custom datasets, `scripts/run_cen.py` also supports a small manifest JSON:

```bash
python3 scripts/run_cen.py --manifest examples/manifest.example.json --output results.csv
```

## Algorithm Summary

For each temporal block:

1. Sum `dt` spike frames.
2. Remove the DC component by subtracting the block mean.
3. Compute the 2D FFT power spectrum.
4. Compute the cumulative radial energy centroid up to each candidate radius `r2`.
5. Select `r2` without ground truth using the same curve-shape scoring as `syn_cmp_generated_spikes_keepHW.py`.
6. Normalize the selected CEN curve and localize the focus block with top-centroid peak estimation.

## Citation

If you use this code or dataset, please cite our ICML 2026 paper:

```bibtex
@inproceedings{xiang2026cen,
  title     = {Autofocus for Spike Cameras via Spectral Centroid in the Frequency Domain},
  author    = {Xiang, Xijie and Zhu, Lin and Tian, Yonghong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
