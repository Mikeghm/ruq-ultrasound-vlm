# M3D Preprocessing Pipeline (No SAM)

This repository provides a unified, end‑to‑end preprocessing pipeline for ultrasound image datasets without using SAM. The pipeline converts DICOMs packed in ZIP files to JPG, crops the ultrasound fan area via a geometric mask, removes black images, builds M3D inputs (`img.npy` + `text.txt`), and generates a dataset JSON.

Key script: `pipeline_no_sam.py`

## Features
- DICOM → JPG with top black bar (de‑identification)
- Fan crop using a fixed geometric mask (no SAM dependency)
- Black image removal (mean grayscale threshold)
- M3D inputs builder: stacks 32 frames and outputs `img.npy` with shape `(1, 32, 256, 256)`
- Dataset JSON generation for training/evaluation
- CLI with per‑step skipping for incremental runs

## Requirements
- Python 3.9+
- Packages: `pydicom`, `Pillow`, `numpy`, `opencv-python`, `monai`, `pandas`, `tqdm`

Install dependencies:

```bash
pip install pydicom Pillow numpy opencv-python monai pandas tqdm
```

> Note: MONAI may install PyTorch if not present.

## Input Expectations
- `--zips_dir`: Directory containing `.zip` bundles. Each ZIP represents one session. The session folder name is the ZIP filename without extension (e.g., `S12345.zip` → session `S12345`).
- `--reports_csv`: CSV containing at least two columns:
  - `XNATSessionID` (or custom via `--session_col`): must exactly match session folder names.
  - `FINDINGS` (or custom via `--text_col`): the free‑text clinical report.

## Outputs (by stage)
- JPG extraction: `--jpg_out/<session>/*.jpg`
- Fan crop (no SAM): `--cropped_out/<session>/cropped_*.jpg`
- M3D inputs: `--preproc_out/<session>/img.npy` and `--preproc_out/<session>/text.txt`
- Dataset JSON: `--preproc_out/ultrasound_dataset.json`

## Quick Start
Example paths adapted from existing usage:

```bash
python pipeline_no_sam.py \
  --zips_dir /media/user1/myHD20TB/Ultrasound_XNAT \
  --jpg_out /media/user1/myHD20TB/preprocessed/original_images/Ultrasound_XNAT \
  --cropped_out /media/user1/HD8TB/External_Validation/Ultrasound_Ext_Fan_NoSAM \
  --reports_csv /media/user1/HD8TB/External_Validation/Ext_Val.csv \
  --preproc_out /media/user1/HD8TB/External_Validation/Ext_Preprocessed \
  --dataset_mode val-only \
  --workers 6
```

This will run all stages: ZIP→JPG → crop → black removal → M3D → JSON.

## Incremental/Resumable Runs
You can skip completed stages using flags:

- `--skip_dicom` — skip DICOM→JPG
- `--skip_crop` — skip fan crop
- `--skip_clean` — skip black image removal
- `--skip_m3d` — skip M3D NPY+text
- `--skip_json` — skip dataset JSON

Example (JPGs and crops already done):

```bash
python pipeline_no_sam.py \
  --zips_dir /media/user1/myHD20TB/Ultrasound_XNAT \
  --jpg_out /media/user1/myHD20TB/preprocessed/original_images/Ultrasound_XNAT \
  --cropped_out /media/user1/HD8TB/External_Validation/Ultrasound_Ext_Fan_NoSAM \
  --reports_csv /media/user1/HD8TB/External_Validation/Ext_Val.csv \
  --preproc_out /media/user1/HD8TB/External_Validation/Ext_Preprocessed \
  --dataset_mode val-only \
  --skip_dicom --skip_crop \
  --workers 6
```

## CLI Reference
```text
usage: pipeline_no_sam.py [-h] --zips_dir ZIPS_DIR --jpg_out JPG_OUT --cropped_out CROPPED_OUT --reports_csv REPORTS_CSV --preproc_out PREPROC_OUT [--dataset_mode {full,val-only}] [--session_col SESSION_COL] [--text_col TEXT_COL] [--workers WORKERS] [--skip_dicom] [--skip_crop] [--skip_clean] [--skip_m3d] [--skip_json]
```

- `--zips_dir`: Input directory with ZIPs containing DICOM files.
- `--jpg_out`: Output root for per‑session JPGs.
- `--cropped_out`: Output root for per‑session cropped JPGs.
- `--reports_csv`: Reports CSV filepath.
- `--preproc_out`: Output root for per‑session `img.npy` and `text.txt`.
- `--dataset_mode`: `val-only` (default) or `full` (9:1 split for train vs val/test).
- `--session_col`: Column name of session IDs in the CSV (default `XNATSessionID`).
- `--text_col`: Column name of report text in the CSV (default `FINDINGS`).
- `--workers`: Number of parallel processes for M3D step (default 6).
- `--skip_*`: Flags to skip completed stages.

## Details by Stage
1) DICOM→JPG
- Converts all `.dcm` within each ZIP to JPG under `--jpg_out/<session>`.
- Applies a black bar over the top 10% of the image to cover identifiers.

2) Fan Crop (no SAM)
- Uses a fixed geometric fan mask to keep the ultrasound fan area.
- Cropped files are saved as `cropped_*.jpg` in `--cropped_out/<session>`.

3) Black Image Removal
- Deletes images with mean grayscale below threshold (default 20).
- If removal is too aggressive or too lax, modify `is_black_image(..., threshold=20)` in `pipeline_no_sam.py`.

4) M3D Inputs
- For each session, samples 32 frames uniformly; if fewer than 32 frames, cycles sequentially.
- Processing: grayscale → normalize to [0, 1] → MONAI `CropForeground` → `Resize(256, 256)`.
- Stacks to `(32, 256, 256)` then expands to `(1, 32, 256, 256)` and saves as `img.npy`.
- Writes `text.txt` with the report text from the CSV.
- Records skipped sessions and reasons in `preproc_out/skipped_patients.csv`.

5) Dataset JSON
- Creates `ultrasound_dataset.json` under `--preproc_out` with relative paths to `img.npy` and `text.txt`.
- Modes: `val-only` uses all sessions for validation/test; `full` uses a 90/10 train/val split.

## Verifying Outputs
- Inspect a session folder:
  ```bash
  ls -1 /media/user1/HD8TB/External_Validation/Ext_Preprocessed/<session>
  ```
- Check NPY shape:
  ```bash
  python -c "import numpy as np, sys; d=np.load(sys.argv[1]); print(d.shape)" \
    /media/user1/HD8TB/External_Validation/Ext_Preprocessed/<session>/img.npy
  ```
- Preview JSON:
  ```bash
  head -n 40 /media/user1/HD8TB/External_Validation/Ext_Preprocessed/ultrasound_dataset.json
  ```

## Troubleshooting
- No JPGs created for a session: ensure the ZIP contains `.dcm` files; nested folders are supported.
- Sessions marked `no_report`: check that ZIP basenames equal values in `--session_col`.
- Too many images deleted as black: lower the threshold (e.g., 10–15). Too few deleted: raise it (e.g., 30–40).
- M3D step `insufficient_images`: the session has <2 usable images after trimming; review the cropped images.
- Missing dependencies: reinstall packages with pip; for MONAI, ensure compatible Python/PyTorch.

## Notes
- This pipeline replaces the previous SAM‑based cropping step with a geometry‑based fan crop.
- The legacy step list in `Readme.txt` is superseded by this unified script.

