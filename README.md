# CEN Autofocus

This is the minimal public implementation of the CEN autofocus algorithm used in our spike-camera experiments.

The repository contains only the code needed to run the algorithm on generated simulated spike datasets. It does not include private datasets, cached experiment outputs, paper figures, local absolute paths, or the spike simulator.

## Install

```bash
python3 -m pip install -e .
```

## Dataset

Download the public generated-spike dataset from Google Drive:

```text
https://drive.google.com/file/d/1a0eDgwk-6tjjZQCigShkX_S3cgHybZHT/view?usp=drive_link
```

After downloading, extract it so the repository can see one of these directories:

```text
simulations01/simulate_moderate_light
simulations01/simulate_low_light
```

You can also place the dataset elsewhere and pass the full path with `--generated_root`.

## Generated Simulation Layout

The comparison script expects generated spikes in this layout:

```text
simulations01/
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
--generated_root ./simulations01/simulate_moderate_light
```

## Compare Generated Spikes

```bash
python3 scripts/spike_autofocus_CEN.py \
  --generated_root ./simulations01/simulate_moderate_light \
  --dt 10
```

For the low-light simulation:

```bash
python3 scripts/spike_autofocus_CEN.py \
  --generated_root ./simulations01/simulate_low_light \
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
