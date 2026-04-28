# Severstal Steel Defect Detection

PyTorch implementation for the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) Kaggle competition.

Uses **segmentation-models-pytorch** with EfficientNet-B5 and SE-ResNeXt encoders.

---

## Prerequisites

- **Python**: 3.12 or newer (tested on 3.14)
- **Data**: Kaggle competition dataset with `train_images/`, `test_images/`, `train.csv`
- **Hardware**: CUDA GPU (Linux/Windows) or Apple Silicon MPS (macOS) or CPU

---

## Installation

### 1. Clone and create virtual environment

```bash
git clone https://github.com/khornlund/severstal-steel-defect-detection.git
cd severstal-steel-defect-detection
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch efficientnet_pytorch
pip install albumentations opencv-python pandas numpy scipy matplotlib
pip install tensorboard tqdm pyyaml click scikit-learn jupyter
pip install -e .
```

> **Note**: On macOS with Apple Silicon, PyTorch will use the MPS backend automatically. On CUDA systems, install the CUDA-enabled PyTorch wheels from [pytorch.org](https://pytorch.org).

---

## Data Setup

Download the competition data from Kaggle and organize it as:

```
/data/path/
├── train_images/
│   ├── 0002cc93b.jpg
│   ├── 0007a71bf.jpg
│   └── ...
├── test_images/
│   └── ...
└── train.csv
```

Update the `data_dir` in the experiment config:

```bash
# Edit experiments/unet-b5.yml (or any config)
# Change:
#   data_dir: /your/actual/data/path
```

---

## Training

### Quick start (5 epochs)

```bash
# Edit config to reduce epochs
sed -i '' 's/epochs: 250/epochs: 5/' experiments/unet-b5.yml  # macOS
# sed -i 's/epochs: 250/epochs: 5/' experiments/unet-b5.yml    # Linux

# Run training
sever train -c experiments/unet-b5.yml
```

### Full training

```bash
sever train -c experiments/unet-b5.yml
```

### Available experiment configs

| Config | Architecture | Encoder | Batch Size |
|--------|-------------|---------|------------|
| `experiments/unet-b5.yml` | UNet | EfficientNet-B5 | 18 |
| `experiments/fpn-b5.yml` | FPN | EfficientNet-B5 | 18 |
| `experiments/unet-se_resnext.yml` | UNet | SE-ResNeXt50 | 32 |
| `experiments/fpn-se_resnext.yml` | FPN | SE-ResNeXt50 | 36 |

### Resume from checkpoint

```bash
sever train -c experiments/unet-b5.yml -r saved/sever-Unet-efficientnet-b5-BCEDiceLoss-RAdam/checkpoint-epoch10.pth
```

---

## Monitoring

Training logs and TensorBoard summaries are saved to:

```
saved/<experiment-name>/
├── logs/
│   ├── debug.log
│   └── info.log
└── tensorboard/
```

View TensorBoard:

```bash
tensorboard --logdir saved/
```

---

## Project Structure

```
severstal-steel-defect-detection/
├── sever/
│   ├── cli.py              # CLI entry point
│   ├── main.py             # Runner (train loop orchestration)
│   ├── trainer/
│   │   └── trainer.py      # Epoch-level train/val logic
│   ├── data_loader/
│   │   ├── data_loaders.py # DataLoader classes
│   │   ├── datasets.py     # PyTorch Dataset classes
│   │   ├── augmentation.py # Albumentations transforms
│   │   └── process.py      # RLE encoding/decoding
│   ├── model/
│   │   ├── model.py        # Model architectures
│   │   ├── loss.py         # Loss functions
│   │   ├── metric.py       # Metrics
│   │   ├── optimizer.py    # Custom optimizers
│   │   └── scheduler.py    # LR schedulers
│   └── utils.py
├── experiments/            # YAML experiment configs
│   ├── unet-b5.yml
│   ├── fpn-b5.yml
│   ├── unet-se_resnext.yml
│   └── fpn-se_resnext.yml
├── environment.yml         # Conda environment (legacy)
└── setup.py
```

---

## Key Dependencies

| Package | Version |
|---------|---------|
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| segmentation-models-pytorch | 0.5.0 |
| albumentations | 2.0.8 |
| efficientnet_pytorch | 0.7.1 |
| opencv-python | 4.13.0 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| tensorboard | 2.20.0 |

---

## Notes

- The original code targeted **albumentations 0.3.3** and **PyTorch 1.2**. This fork has been updated for modern versions.
- `in_channels` is set to `1` for grayscale steel images.
- On macOS, training uses MPS (Metal Performance Shaders). `pin_memory` warnings are harmless.
- The `addcmul_` deprecation warning from the custom RAdam optimizer is harmless.

---

## License

MIT
