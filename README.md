# Multitask RUQ Ultrasound Interpretation (Codebase for Paper)

This repository contains the codebase accompanying the paper:

"A multitask framework for automated interpretation of multi-frame right upper quadrant ultrasound in clinical decision support."

It provides scripts and references for running multi-frame RUQ ultrasound inference (report-style generation from sequences of frames) and includes related multi‑modal model components.

## Repository Structure

- `Qwen/` — Batch inference over multi-frame ultrasound studies using Qwen2.5‑VL. Includes a small test dataset, a JSON manifest, and a multi‑GPU inference script.
- `M3D/` — Upstream M3D resources for multi‑modal medical imaging (vision encoder, training, benchmark). See `M3D/README.md` for detailed usage of those components.

## Environment Setup

Requirements (typical versions, adjust to your CUDA/driver):

- Python 3.9+
- PyTorch with CUDA (`torch`, `torchvision`, `torchaudio`)
- `transformers`, `accelerate`, `tqdm`
- Qwen VL utilities (`qwen-vl-utils` if using the packaged utils)

Quick start with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the right CUDA wheel
pip install transformers accelerate tqdm qwen-vl-utils

# (Optional) If you plan to use M3D training/benchmarks, also install:
pip install -r M3D/requirements.txt
```

## Data Format (Qwen Inference)

The demo uses a small RUQ ultrasound test set under `Qwen/Test_Dataset/` and a JSON manifest `Qwen/qwen_test_dataset.json` with the following structure:

- A list of samples; each sample has:
  - `images`: list of image paths (relative to `Qwen/`) representing frames from a study.
  - `messages`: a conversation array. The first `user` turn contains repeated `<image>` tokens followed by an instruction (e.g., “Describe findings …”). The subsequent `assistant` turn may contain a reference report (ground truth) for comparison.

You can create your own dataset by mirroring this structure and updating paths accordingly.

## Inference

The script `Qwen/batch_inference_test.py` performs distributed, multi‑GPU batched inference with Qwen2.5‑VL.

1) Prepare a locally available merged/converted Qwen2.5‑VL model directory (e.g., `Merged-Qwen2.5-VL-7B`). Update the path if yours differs.

2) Run distributed inference (defaults to 4 GPUs; change `world_size` in the script if needed):

```bash
cd Qwen
python batch_inference_test.py
```

Outputs:

- A `test_inference_results.json` file with per‑sample predictions and basic statistics. Temporary shard files `.gpu_{rank}` are combined and removed at the end.

Notes:

- The script sets `MASTER_ADDR`/`MASTER_PORT` internally. Ensure the number of visible GPUs matches the `world_size` used by the script.
- To limit the number of samples or enable random sampling, adjust the arguments in `main_distributed(...)` at the bottom of the script.

## Training and Benchmarks (Optional)

If you intend to train or evaluate multi‑modal components referenced in the framework, see `M3D/README.md` for detailed instructions on:

- Preparing data for M3D‑Data
- Pretraining the vision encoder and instruction tuning (LaMed)
- Running the M3D‑Bench evaluation tasks

## Citation

If you use this repository, please cite the paper (update with your final citation):

```
@article{yourkey2025ruq,
  title   = {A Multitask Framework for Automated Interpretation of Multi-Frame Right Upper Quadrant Ultrasound in Clinical Decision Support},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2025}
}
```

## License

Specify the license for this repository here (e.g., MIT, Apache-2.0, or institutional). If using external models or datasets, please follow their respective licenses and terms.

## Acknowledgements

- Qwen2.5‑VL and related tooling.
- M3D: multi‑modal medical imaging components and benchmarks (see `M3D/README.md`).
