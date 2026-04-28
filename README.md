# SynthInspect: Generative Augmentation for Steel Defect Detection

PyTorch implementation of **SynthInspect** — a conditional WGAN-GP pipeline that generates synthetic steel surface defects and integrates them into a curriculum-based training loop to improve rare-class detection on the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) benchmark.

> **IEEE Research Project**: Conditional DG-GAN + discriminator-score quality filtering + dynamic curriculum mixing for industrial defect detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Data Setup](#data-setup)
5. [Project Structure](#project-structure)
6. [Full Pipeline (A-Z)](#full-pipeline-a-z)
   - [Step 1: Baseline Detector](#step-1-train-the-baseline-detector)
   - [Step 2: GAN Training](#step-2-train-the-conditional-wgan-gp)
   - [Step 3: Generate Synthetic Data](#step-3-generate--filter-synthetic-defects)
   - [Step 4: Fixed-Ratio Detector](#step-4-train-detector-with-fixed-synthetic-ratio)
   - [Step 5: Curriculum Detector](#step-5-train-detector-with-curriculum-scheduler)
   - [Step 6: Evaluation & Comparison](#step-6-evaluation--comparison)
7. [Experiment Configs](#experiment-configs)
8. [Monitoring & Logging](#monitoring--logging)
9. [Troubleshooting](#troubleshooting)
10. [Research Citation](#research-citation)
11. [License](#license)

---

## Overview

The Severstal dataset contains 12,568 steel strip images (1600×256) with pixel-level masks for 4 defect classes. The dataset is heavily imbalanced — more than 60% of images are defect-free, and the rarest class has only ~300 annotated instances.

**SynthInspect** addresses this by:

1. **Conditional WGAN-GP**: A generator conditioned on defect class, size, and severity learns to synthesize realistic defect patches from real Severstal data.
2. **Quality Filtering**: Synthetic samples are filtered using discriminator scores (keep 30th–80th percentile) to remove low-quality or too-easy fakes.
3. **Curriculum Mixing**: A validation-driven scheduler dynamically adjusts the real:synthetic mixing ratio, starting low and increasing only when rare-class metrics improve.

---

## Prerequisites

- **Python**: 3.10 or newer (tested on 3.12+)
- **Data**: Kaggle Severstal dataset (`train_images/`, `test_images/`, `train.csv`)
- **Hardware**:
  - **macOS**: Apple Silicon (MPS backend)
  - **Windows/Linux**: NVIDIA GPU with CUDA 11.8+ or 12.x
  - **Fallback**: CPU-only (slow but works)
- **Disk**: ~5 GB for dataset + checkpoints + synthetic data

---

## Installation

### 1. Clone and enter the repo

```bash
git clone https://github.com/khornlund/severstal-steel-defect-detection.git
cd severstal-steel-defect-detection
```

### 2. Create and activate virtual environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch (pick your platform)

```bash
# macOS (MPS is included)
pip install torch torchvision torchaudio

# Linux / Windows with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Linux / Windows with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install remaining dependencies

```bash
pip install segmentation-models-pytorch efficientnet_pytorch
pip install albumentations opencv-python pandas numpy scipy matplotlib
pip install tensorboard tqdm pyyaml click scikit-learn jupyter pillow
pip install -e .
```

> **Note**: On macOS with Apple Silicon, PyTorch automatically uses MPS. The `pin_memory` warning is harmless.

---

## Data Setup

Download the [Severstal competition data](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) and organize it as:

```
project-root/
├── data/
│   ├── train_images/          # 12,568 JPG files
│   ├── test_images/           # 5,506 JPG files
│   └── train.csv              # ImageId_ClassId, EncodedPixels
```

If your data lives elsewhere, update `data_dir` in the experiment configs (see [Experiment Configs](#experiment-configs)).

### Quick data check

```bash
python -m gan.inspect_patches
# Should print patch counts and save gan_samples/debug_patches.png
```

---

## Project Structure

```
severstal-steel-defect-detection/
├── data/                           # Dataset (not in git)
│   ├── train_images/
│   ├── test_images/
│   └── train.csv
│
├── sever/                          # Baseline detector pipeline
│   ├── cli.py                      # CLI entry point
│   ├── main.py                     # Runner orchestration
│   ├── base/
│   │   ├── base_model.py
│   │   └── base_trainer.py
│   ├── data_loader/
│   │   ├── data_loaders.py         # DataLoader classes (with synthetic support)
│   │   ├── datasets.py             # PyTorch Datasets (real + synthetic + mixed)
│   │   ├── augmentation.py         # Albumentations transforms
│   │   ├── process.py              # RLE encode/decode
│   │   └── sampling.py             # Balanced samplers
│   ├── model/
│   │   ├── model.py                # SMP model wrappers
│   │   ├── loss.py                 # BCEDiceLoss, etc.
│   │   ├── metric.py               # Dice metrics per class
│   │   ├── optimizer.py            # RAdam, etc.
│   │   └── scheduler.py            # Cosine annealing
│   ├── trainer/
│   │   └── trainer.py              # Epoch-level train/val + curriculum logic
│   └── utils/
│       ├── logger.py
│       ├── saving.py
│       ├── visualization.py
│       └── upload.py
│
├── gan/                            # Generative augmentation module
│   ├── __init__.py
│   ├── dataset.py                  # DefectPatchDataset for GAN training
│   ├── models.py                   # ConditionalGenerator, PatchGANCritic
│   ├── utils.py                    # Gradient penalty, image grid, seed
│   ├── train_wgan.py               # WGAN-GP training loop
│   ├── generate_synthetic.py       # Synthetic generation + filtering
│   ├── inspect_patches.py          # Visual inspection of defect patches
│   └── config.yml                  # GAN hyperparameters
│
├── experiments/                    # Detector experiment configs
│   ├── unet-b5.yml                 # Baseline (real only)
│   ├── unet-b5-synthetic.yml       # Fixed 30% synthetic mixing
│   ├── unet-b5-curriculum.yml      # Dynamic curriculum mixing
│   ├── fpn-b5.yml
│   ├── unet-se_resnext.yml
│   └── fpn-se_resnext.yml
│
├── saved/                          # Checkpoints + logs (created at runtime)
│   ├── sever-*/                    # Baseline checkpoints
│   ├── gan/                        # GAN checkpoints + samples
│   └── ...
│
├── synthetic/                      # Generated synthetic dataset (created at runtime)
│   ├── images/cls_*/               # Synthetic defect images
│   ├── masks/cls_*/                # Corresponding masks
│   ├── metadata_raw.csv            # All generated samples with scores
│   └── metadata_filtered.csv       # Quality-filtered samples
│
├── gan_samples/                    # Debug visualizations
│   └── debug_patches.png
│
├── notebook/                       # Jupyter notebooks (EDA, etc.)
├── docs/                           # Project documentation
├── tests/                          # Unit tests
├── setup.py
├── logging.yml
└── README.md                       # This file
```

---

## Full Pipeline (A-Z)

### Step 1: Train the Baseline Detector

Train a U-Net with EfficientNet-B5 encoder on real data only. This is your comparison baseline.

```bash
# Quick smoke test (5 epochs)
python -c "
import yaml
c = yaml.safe_load(open('experiments/unet-b5.yml'))
c['training']['epochs'] = 5
c['training']['start_val_epoch'] = 3
yaml.dump(c, open('experiments/unet-b5-smoke.yml','w'), default_flow_style=False)
"
sever train -c experiments/unet-b5-smoke.yml

# Full baseline training (250 epochs)
sever train -c experiments/unet-b5.yml
```

> On macOS, training automatically uses MPS. On Windows, reduce `nworkers` to `0` or `2` if you see `BrokenPipeError`.

**Output**: Checkpoints saved to `saved/sever-Unet-efficientnet-b5-.../`

---

### Step 2: Train the Conditional WGAN-GP

Train the generator + critic on real defect patches extracted from Severstal.

```bash
python -m gan.train_wgan --config gan/config.yml
```

**What this does:**

- Extracts ~6,000+ defect patches from `data/train_images`
- Trains a conditional generator (class + size conditioned)
- Trains a PatchGAN critic with WGAN-GP loss
- Saves sample grids every 10 epochs to `saved/gan/samples/`
- Saves checkpoints every 20 epochs to `saved/gan/checkpoints/`

**Key hyperparameters** (edit `gan/config.yml`):

| Parameter    | Default    | Description                           |
| ------------ | ---------- | ------------------------------------- |
| `batch_size` | 16         | Reduce to 8 for MPS memory limits     |
| `n_epochs`   | 200        | Increase if samples still look blurry |
| `n_critic`   | 5          | Critic updates per generator update   |
| `lambda_gp`  | 10.0       | Gradient penalty coefficient          |
| `z_dim`      | 100        | Latent noise dimension                |
| `patch_size` | [256, 256] | Size of extracted defect patches      |

**Output**:

- `saved/gan/G_final.pth` — trained generator
- `saved/gan/D_final.pth` — trained critic
- `saved/gan/samples/epoch_XXXX.png` — visual samples

---

### Step 3: Generate & Filter Synthetic Defects

Generate a large pool of synthetic defects and keep only the high-quality ones.

```bash
python -m gan.generate_synthetic \
    --generator saved/gan/G_final.pth \
    --critic saved/gan/D_final.pth \
    --output synthetic/ \
    --n_samples 2000 \
    --score_lower_pct 30 \
    --score_upper_pct 80
```

**What this does:**

- Generates 500 samples per class (adjustable via `--n_samples`)
- Scores each sample with the critic
- Filters samples to the 30th–80th percentile score band per class
- Saves images to `synthetic/images/cls_X/`
- Saves masks to `synthetic/masks/cls_X/`
- Saves metadata to `synthetic/metadata_filtered.csv`

**Output structure**:

```
synthetic/
├── images/
│   ├── cls_1/img_000000.png
│   ├── cls_2/img_000500.png
│   ├── cls_3/img_001000.png
│   └── cls_4/img_001500.png
├── masks/
│   ├── cls_1/img_000000.png
│   └── ...
├── metadata_raw.csv
└── metadata_filtered.csv
```

---

### Step 4: Train Detector with Fixed Synthetic Ratio

Train the U-Net baseline with a fixed 30% synthetic data mixing ratio.

```bash
sever train -c experiments/unet-b5-synthetic.yml
```

**What this does:**

- Loads real training data from `data/train_images/`
- Loads synthetic data from `synthetic/images/`
- Wraps both in `MixedSeverstalDataset` with `synth_ratio=0.3`
- Trains for 250 epochs with the same optimizer and scheduler as baseline

**To experiment with different ratios**, edit `experiments/unet-b5-synthetic.yml`:

```yaml
data_loader:
  args:
    synthetic_ratio: 0.5 # Try 0.1, 0.3, 0.5
```

**Output**: Checkpoints saved to `saved/sever-synthetic-Unet-efficientnet-b5-.../`

---

### Step 5: Train Detector with Curriculum Scheduler

Train the U-Net with a dynamic curriculum that adjusts the synthetic ratio based on validation performance.

```bash
sever train -c experiments/unet-b5-curriculum.yml
```

**What this does:**

- Starts with `synthetic_ratio=0.1`
- After each validation epoch, monitors `dice_2` (rare-class metric)
- If `dice_2` improves by >0.005: increase ratio by 0.05 (up to max 0.5)
- If `dice_2` degrades by >0.005: decrease ratio by 0.05 (down to min 0.0)
- Logs curriculum transitions to `logs/info.log`

**Curriculum config** (in `experiments/unet-b5-curriculum.yml`):

```yaml
training:
  curriculum: true
  curriculum_step: 0.05
  max_synth_ratio: 0.5
  min_synth_ratio: 0.0
  synth_metric_idx: 2 # Index of dice_2 in metrics list
```

**Output**: Checkpoints saved to `saved/sever-curriculum-Unet-efficientnet-b5-.../`

---

### Step 6: Evaluation & Comparison

Compare all experiments using TensorBoard:

```bash
tensorboard --logdir saved/
```

Key metrics to compare:

| Metric            | Meaning                                        |
| ----------------- | ---------------------------------------------- |
| `epoch/dice_mean` | Average Dice across all 4 classes              |
| `epoch/dice_0`    | Dice for class 1 (rolled-in scale)             |
| `epoch/dice_1`    | Dice for class 2 (pitted surface) — **rarest** |
| `epoch/dice_2`    | Dice for class 3 (scratches)                   |
| `epoch/dice_3`    | Dice for class 4 (dirty spot)                  |
| `val_dice_mean`   | Validation Dice (used for early stopping)      |

**Expected results** (hypothesized):

| Experiment      | dice_mean | dice_1 (rare) | Notes                      |
| --------------- | --------- | ------------- | -------------------------- |
| Baseline only   | ~0.60     | ~0.25         | Limited rare-class data    |
| + 30% synthetic | ~0.65     | ~0.35         | Better rare-class coverage |
| + Curriculum    | ~0.67     | ~0.40         | Optimal mixing over time   |

---

## Experiment Configs

### Baseline configs

| Config                | Architecture | Encoder         | Batch Size | Data      |
| --------------------- | ------------ | --------------- | ---------- | --------- |
| `unet-b5.yml`         | UNet         | EfficientNet-B5 | 18         | Real only |
| `fpn-b5.yml`          | FPN          | EfficientNet-B5 | 18         | Real only |
| `unet-se_resnext.yml` | UNet         | SE-ResNeXt50    | 32         | Real only |
| `fpn-se_resnext.yml`  | FPN          | SE-ResNeXt50    | 36         | Real only |

### Synthetic configs

| Config                   | Synthetic Ratio | Curriculum | Purpose                     |
| ------------------------ | --------------- | ---------- | --------------------------- |
| `unet-b5-synthetic.yml`  | 0.3 (fixed)     | No         | Fixed-ratio ablation        |
| `unet-b5-curriculum.yml` | 0.1 → 0.5       | Yes        | Dynamic curriculum ablation |

### How to create your own config

Copy `experiments/unet-b5.yml` and modify:

```yaml
short_name: my-experiment # Determines checkpoint folder name
data_loader:
  args:
    data_dir: /path/to/data # Your data directory
    batch_size: 12 # Adjust for your GPU memory
    use_synthetic: true # Enable synthetic mixing
    synthetic_ratio: 0.3 # Fixed ratio (ignored if curriculum is on)
    synthetic_root: ./synthetic # Path to generated synthetic data
training:
  epochs: 250
  curriculum: false # Set true for dynamic scheduling
  curriculum_step: 0.05
  max_synth_ratio: 0.5
  min_synth_ratio: 0.0
  synth_metric_idx: 2 # Monitor dice_2 for curriculum decisions
```

---

## Monitoring & Logging

### TensorBoard

```bash
tensorboard --logdir saved/
```

Open `http://localhost:6006` to view:

- Training/validation loss curves
- Per-class Dice scores over epochs
- Learning rate schedules
- Batch-level metrics

### Log files

```bash
# Training logs
tail -f saved/<experiment>/logs/info.log

# Debug logs
tail -f saved/<experiment>/logs/debug.log
```

### GAN training logs

The GAN script prints to stdout:

```
[Epoch 10/200] [Batch 50/400] [D loss: -0.2341] [G loss: 0.5678] [GP: 0.8912]
Saved samples to saved/gan/samples/epoch_0010.png
```

---

## Troubleshooting

### macOS / MPS

| Issue                                        | Solution                                          |
| -------------------------------------------- | ------------------------------------------------- |
| `MPS out of memory` during GAN training      | Reduce `batch_size` to 8 or 4 in `gan/config.yml` |
| `MPS out of memory` during detector training | Reduce `batch_size` to 8 in experiment config     |
| `pin_memory` warning                         | Harmless — ignore it                              |
| Training is slow                             | Expected on MPS; reduce `nworkers` to 0           |

### Windows

| Issue                           | Solution                                           |
| ------------------------------- | -------------------------------------------------- |
| `BrokenPipeError` in DataLoader | Set `nworkers: 0` in config                        |
| CUDA not found                  | Reinstall PyTorch with correct `--index-url`       |
| `sed` not found                 | Use Python one-liner or PowerShell to edit configs |

### General

| Issue                                      | Solution                                                                        |
| ------------------------------------------ | ------------------------------------------------------------------------------- |
| GAN samples are pure noise after 50 epochs | Reduce learning rate (try `lr_g: 0.0001`, `lr_d: 0.0001`)                       |
| GAN samples are all identical              | Increase `z_dim` to 128 or add more diversity in conditions                     |
| Synthetic data hurts detector performance  | Reduce `synthetic_ratio` or tighten quality filter (e.g., 40th–70th percentile) |
| Curriculum never increases ratio           | Check that `synth_metric_idx` matches the rare-class metric index               |
| `ModuleNotFoundError: gan`                 | Run `pip install -e .` from project root                                        |

---

## Research Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{synthinspect2026,
  title={SynthInspect: Generative Augmentation with Curriculum Learning for Steel Defect Detection},
  author={Your Name},
  booktitle={IEEE Conference on Industrial Electronics},
  year={2026}
}
```

### Related Work

- **Baseline**: [khornlund/severstal-steel-defect-detection](https://github.com/khornlund/severstal-steel-defect-detection)
- **SMP**: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- **WGAN-GP**: Gulrajani et al., "Improved Training of Wasserstein GANs," NeurIPS 2017
- **Dataset**: [Severstal Kaggle Competition](https://www.kaggle.com/c/severstal-steel-defect-detection)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about this research project, open an issue or reach out via the repository.
