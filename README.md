# Severstal Steel Defect Detection

PyTorch implementation for the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) Kaggle competition.

Uses **segmentation-models-pytorch** with EfficientNet-B5 and SE-ResNeXt encoders.

---

## Prerequisites

- **Python**: 3.12 or newer (tested on 3.14)
- **Data**: Kaggle competition dataset with `train_images/`, `test_images/`, `train.csv`
- **Hardware**:
  - **Windows/Linux**: NVIDIA GPU with CUDA 11.8+ or 12.x
  - **macOS**: Apple Silicon (MPS) or Intel (CPU)
  - **Fallback**: CPU-only mode works on all platforms

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

## Windows + CUDA Setup

### 1. Install CUDA-enabled PyTorch

```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check your CUDA version with `nvidia-smi` in Command Prompt.

### 2. Install remaining dependencies

```bash
pip install segmentation-models-pytorch efficientnet_pytorch
pip install albumentations opencv-python pandas numpy scipy matplotlib
pip install tensorboard tqdm pyyaml click scikit-learn jupyter
pip install -e .
```

### 3. Activate virtual environment (Windows syntax)

```bash
.venv\Scripts\activate
```

### 4. Update config for Windows paths

Edit `experiments/unet-b5.yml` (or any config). Change the data path:

```yaml
data_loader:
  args:
    data_dir: C:/Users/YourName/Code/IEEE/data
```

Forward slashes work in YAML. If you use backslashes, quote the path:

```yaml
    data_dir: "C:\\Users\\YourName\\Code\\IEEE\\data"
```

### 5. Adjust epochs (Windows commands)

The `sed` command doesn't exist on Windows. Use one of these instead:

**PowerShell:**
```powershell
(Get-Content experiments/unet-b5.yml) -replace 'epochs: 250', 'epochs: 5' | Set-Content experiments/unet-b5.yml
```

**Python (cross-platform):**
```bash
python -c "import yaml; c=yaml.safe_load(open('experiments/unet-b5.yml')); c['training']['epochs']=5; yaml.dump(c, open('experiments/unet-b5.yml','w'), default_flow_style=False)"
```

### 6. Run training

```bash
sever train -c experiments/unet-b5.yml
```

### Windows-specific tips

| Issue | Solution |
|-------|----------|
| `BrokenPipeError` or DataLoader crashes | Reduce `nworkers` to `0` or `2` in the config |
| Out of memory | Reduce `batch_size` (try `8` or `4`) |
| CUDA not found | Reinstall PyTorch with the correct `--index-url` for your CUDA version |
| Very slow on first epoch | Normal — CUDA kernel compilation happens once |

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
