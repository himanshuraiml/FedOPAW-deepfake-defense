# FedOPAW: federated open-set deepfake image detection with provenance-aware watermarking

Code, per-run results, and data manifest for the study of federated training (FedAvg, FedProx, Fed-EWC, GEM, proxy-set distillation), open-set scoring (CLIP zero-shot, frequency sensitivity), synthetic C2PA trust levels, and watermark-detector co-design on public face images.

## Layout
- `code/` training and evaluation (`fedopaw_pilot.py`, `wm_experiment.py`, `analysis_models.py`, `extras.py`, `aggregate.py`), data builder (`fetch_dff.py`), and run scripts (`run_full.sh`, `run_sweeps.sh`, `finish_full.sh`). Run the scripts from inside `code/` with the data file in `code/data/`.
- `data_manifest/dff_n6000_identity_stems.txt` the 6,000 identities used (file names in the DeepFakeFace archives).
- `results/full/` per-run JSON for 3 seeds x 3 held-out generators x 9 methods, plus the watermark studies.
- `results/sensitivity_sweeps/` EWC-strength and client-skew sweeps.
- `results/generated_tables_figures/` all tables, figures, and numbers used in the paper.
- `logs/` run logs.

## Data
- Raw: DeepFakeFace (Song et al., 2023), OpenRAIL license, https://huggingface.co/datasets/OpenRL/DeepFakeFace (not re-hosted here).
- Processed: `dff_n6000_s128.npz` (real image + three fakes per identity, 128x128 center crop, uint8) and `dff_n6000_s128_clip.npy` (CLIP ViT-B/32 image features) are attached to the GitHub Release `v1.0-data`. To rebuild from the raw archives: `python code/fetch_dff.py --n 6000 --size 128 --out data/dff_n6000_s128.npz`.
- Use of the data is subject to the OpenRAIL terms of the original dataset.

## Requirements
Python 3, PyTorch, numpy, scikit-learn, scipy, matplotlib, transformers (CLIP), Pillow, remotezip. The study ran on one Apple-silicon GPU (MPS).
