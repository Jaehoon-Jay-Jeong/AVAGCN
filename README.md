# AVAGCN: Adaptive Variational Autoencoder with Graph Convolutional Network

Official PyTorch implementation of **"Adaptive Variational Autoencoder with Graph Convolutional Network for Semi-Supervised Few-Shot Hyperspectral Image Classification"**

AVAGCN is a semi-supervised few-shot framework for hyperspectral image (HSI) classification that uses **only target-domain data** (no external source dataset). It learns a **per-patch discrete undirected adjacency matrix** end-to-end from a variational latent representation via straight-through Gumbel-Softmax, and combines a unified objective (prototype-contrastive + consistency + self-supervised + regularization losses) with **R-round iterative training** driven by curriculum-based pseudo-labeling.

## Repository Structure

```
.
├── main.py                  # Entry point (Hydra)
├── configs/
│   ├── IP.yaml              # Indian Pines
│   ├── PU.yaml              # Pavia University
│   └── SA.yaml              # Salinas
├── dataset/
│   └── dataset.py           # Data loading, splits, patch extraction
├── model/
│   └── AVAGCN.py            # AVAGCN model (embedding, VAE, decoder, GCN, classifier)
├── training/
│   ├── trainer.py           # R-round iterative training / evaluation
│   ├── aug.py               # Hyperspectral patch augmentation
│   └── criterion.py         # Prototype-contrastive & regularization losses
├── util/
│   └── utils.py             # Deterministic seed utilities
├── datas/                   # Place datasets here (see below)
├── requirements.txt         # Direct dependencies
├── requirements-lock.txt    # Exact environment used for the reported results
└── Dockerfile               # Optional containerized environment
```

## Environment

Tested with: Ubuntu 26.04 LTS, Python 3.12.13, PyTorch 2.12.0, CUDA 13.0, cuDNN 9.20, NVIDIA GeForce RTX 5080 (16 GB VRAM).

All experiments were run in a **conda environment with packages installed via pip**:

```bash
conda create -n avagcn python=3.12.13 -y
conda activate avagcn
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The exact package set used for the reported results is recorded in `requirements-lock.txt` (Linux x86_64, CUDA 13.0):

```bash
pip install -r requirements-lock.txt   # exact reproduction of the experiment environment
```

A plain `venv` + pip installation works equivalently, since all packages are installed from PyPI wheels. Alternatively, build the container:

```bash
docker build -t avagcn .
docker run --gpus all -it -v $(pwd)/datas:/workspace/datas avagcn
```

> **Note (GPU access):** If your host's NVIDIA Container Toolkit is configured in CDI mode, `--gpus all` alone fails with an "NVIDIA Container Runtime Hook … not supported" error. In that case add `--runtime=nvidia`:
>
> ```bash
> docker run --runtime=nvidia --gpus all -it -v $(pwd)/datas:/workspace/datas avagcn
> ```
>
> (or use the CDI device syntax `--device nvidia.com/gpu=all`).

## Datasets

Download the corrected versions of Indian Pines, Pavia University, and Salinas from the
[GIC Hyperspectral Remote Sensing Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) page, and place them as:

```
datas/
├── IndianPines/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PaviaU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
└── Salinas/
    ├── Salinas_corrected.mat
    └── Salinas_gt.mat
```

## Usage

```bash
python main.py --config-name IP   # Indian Pines
python main.py --config-name PU   # Pavia University
python main.py --config-name SA   # Salinas
```

Each run trains and evaluates over **10 deterministic seeds** (5 labeled samples per class) and writes per-seed and aggregate results (`full_seed_results.json`, `full_seed_test_only_results.json`, classification maps) under `outputs/`.

Hydra overrides can be used for quick checks, e.g. a single short round:

```bash
python main.py --config-name IP num_round=1 round0_epochs=1
```

### Reproducibility notes

- The 10 seeds are generated deterministically in `util/utils.py::seed_generate` from the fixed key `"JAYHSICAVAGCN2026"`. **Do not modify this key** — the reported results depend on it.
- `set_seed` enables deterministic algorithms (`torch.use_deterministic_algorithms`, fixed cuDNN settings, `CUBLAS_WORKSPACE_CONFIG`).
- `full_seed_results.json` corresponds to the main-text metrics (Tables 7–9 of the paper; evaluation over the entire test split including validation samples), and `full_seed_test_only_results.json` corresponds to Appendix B (validation samples excluded).

### Expected results (OA %, mean ± std over 10 seeds, 5 labels/class)

| Dataset | OA | AA | Kappa |
|---|---|---|---|
| Indian Pines | 80.18 ± 2.41 | 87.42 ± 1.33 | 77.51 ± 2.66 |
| Pavia University | 89.31 ± 1.93 | 88.99 ± 2.16 | 85.89 ± 2.52 |
| Salinas | 92.44 ± 1.16 | 96.11 ± 0.74 | 91.60 ± 1.29 |

## License

This project is released under the MIT License (see `LICENSE`).