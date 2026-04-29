# Running the Baseline on Google Colab

> Complete step-by-step guide to train the U-Net baseline on Google Colab's free T4 GPU.

---

## What You Need Before Starting

1. **A Google account** (for Google Drive + Colab)
2. **The Severstal dataset** downloaded from Kaggle
3. **Your code pushed to GitHub** (already done: `github.com/MohitGoyal09/severstal-steel-defect-detection`)

---

## Step 1: Upload Data to Google Drive

Colab disk is **not persistent** — if your session disconnects, uploaded files vanish. Google Drive persists.

### Option A: Upload via Browser (Easiest)

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder: `My Drive/ColabData/severstal/`
3. Upload these files/folders into it:
   ```
   ColabData/severstal/
   ├── train_images/       # 12,568 JPG files
   ├── test_images/        # 5,506 JPG files
   └── train.csv           # annotations
   ```
4. **Wait for upload to complete** (2–3 GB, ~10–20 minutes)

### Option B: Kaggle API Download (Inside Colab)

If you haven't downloaded the data yet, do it directly in Colab:

```python
# In a Colab cell
!pip install -q kaggle
!mkdir -p ~/.kaggle
# Upload your kaggle.json API token first:
from google.colab import files
files.upload()  # select kaggle.json
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle competitions download -c severstal-steel-defect-detection
!mkdir -p /content/data
!unzip -q severstal-steel-defect-detection.zip -d /content/data/
!mv /content/data/train_images/* /content/data/train_images/ 2>/dev/null
```

---

## Step 2: Open Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. **File → New notebook**
3. **Runtime → Change runtime type → Select GPU → Save**
4. Verify GPU is available:

```python
# Cell 1: GPU check
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

Expected output:

```
GPU available: True
GPU name: Tesla T4
```

---

## Step 3: Mount Google Drive & Setup

```python
# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Create data directory and symlink from Drive
!mkdir -p /content/data
!ln -s /content/drive/MyDrive/ColabData/severstal/train_images /content/data/train_images
!ln -s /content/drive/MyDrive/ColabData/severstal/test_images /content/data/test_images
!ln -s /content/drive/MyDrive/ColabData/severstal/train.csv /content/data/train.csv

# Verify
!ls -la /content/data/
!ls /content/data/train_images/ | head -5
!wc -l /content/data/train.csv
```

You should see:

```
train.csv  test_images  train_images
0002cc93b.jpg
0007a71bf.jpg
...
12569 train.csv
```

---

## Step 4: Clone Repository & Install

```python
# Cell 4: Clone your repo
%cd /content
!git clone https://github.com/MohitGoyal09/severstal-steel-defect-detection.git
%cd /content/severstal-steel-defect-detection

# Cell 5: Install dependencies
!pip install -q torch torchvision torchaudio
!pip install -q segmentation-models-pytorch efficientnet_pytorch
!pip install -q albumentations opencv-python pandas numpy scipy matplotlib
!pip install -q tensorboard tqdm pyyaml click scikit-learn pillow
!pip install -q -e .

# Cell 6: Verify installation
!sever --help
```

---

## Step 5: Quick Smoke Test (5 Epochs)

Before running the full 250 epochs, verify everything works with a short run:

```python
# Cell 7: Smoke test — 5 epochs, validation starts at epoch 3
!sed -i 's/epochs: 50/epochs: 5/' experiments/unet-b5-colab.yml
!sed -i 's/start_val_epoch: 20/start_val_epoch: 3/' experiments/unet-b5-colab.yml

!sever train -c experiments/unet-b5-colab.yml
```

This should run in ~10–15 minutes. You should see:

- Loss decreasing each batch
- Validation metrics appearing after epoch 3
- Checkpoint saved to `saved/sever-colab-.../`

**If this works, move to Step 6.**

**If it fails**, check the [Troubleshooting](#troubleshooting) section.

---

## Step 6: Full Baseline Training (50 Epochs)

```python
# Cell 8: Reset config to full training
# First, restore the original config
!git checkout experiments/unet-b5-colab.yml

# Cell 9: Start full training
!sever train -c experiments/unet-b5-colab.yml
```

**What happens:**

- Trains U-Net + EfficientNet-B5 for 50 epochs
- Validates every 10 epochs starting at epoch 20
- Saves checkpoints to `saved/sever-colab-Unet-efficientnet-b5-.../`
- TensorBoard logs to `saved/.../tensorboard/`

**Expected time:** ~2–3 hours on T4

---

## Step 7: Monitor Training

### Option A: TensorBoard in Colab

```python
# Cell 10: Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/severstal-steel-defect-detection/saved
```

A TensorBoard panel will appear below the cell. Watch:

- `epoch/loss` — should decrease
- `epoch/dice_mean` — should increase
- `epoch/dice_2` — rare class metric

### Option B: Print Log Files

```python
# Cell 11: Check latest logs
!ls -lt saved/*/logs/ | head -10
!cat saved/sever-colab-*/logs/info.log | tail -30
```

---

## Step 8: Save Results to Drive (IMPORTANT)

Colab's `/content/` folder is **deleted when the session ends**. Copy checkpoints to Google Drive:

```python
# Cell 12: Copy checkpoints to Drive
!mkdir -p /content/drive/MyDrive/ColabResults/severstal-baseline
!cp -r /content/severstal-steel-defect-detection/saved/* /content/drive/MyDrive/ColabResults/severstal-baseline/
!cp -r /content/severstal-steel-defect-detection/logs/* /content/drive/MyDrive/ColabResults/severstal-baseline/ 2>/dev/null || true

print("Saved to Google Drive!")
```

---

## Step 9: Resume If Disconnected

If Colab disconnects, restart from the last checkpoint:

```python
# Cell 13: Restore from Drive + resume
from google.colab import drive
drive.mount('/content/drive')

# Recreate symlink
!mkdir -p /content/data
!ln -sf /content/drive/MyDrive/ColabData/severstal/train_images /content/data/train_images
!ln -sf /content/drive/MyDrive/ColabData/severstal/test_images /content/data/test_images
!ln -sf /content/drive/MyDrive/ColabData/severstal/train.csv /content/data/train.csv

# Re-clone and install
%cd /content
!rm -rf severstal-steel-defect-detection
!git clone https://github.com/MohitGoyal09/severstal-steel-defect-detection.git
%cd /content/severstal-steel-defect-detection
!pip install -q -e .

# Find latest checkpoint
import glob
checkpoints = glob.glob('/content/drive/MyDrive/ColabResults/severstal-baseline/sever-colab-*/checkpoint-epoch*.pth')
if checkpoints:
    latest = sorted(checkpoints)[-1]
    print(f"Resuming from: {latest}")
    !sever train -c experiments/unet-b5-colab.yml -r "{latest}"
else:
    print("No checkpoint found, starting fresh")
    !sever train -c experiments/unet-b5-colab.yml
```

---

## Complete Notebook (All Cells in Order)

Copy-paste this entire block into a fresh Colab notebook:

```python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 1: GPU Check                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
import torch
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 2: Mount Google Drive                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
from google.colab import drive
drive.mount('/content/drive')

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 3: Setup Data (symlink from Drive)                       ║
# ╚══════════════════════════════════════════════════════════════════╝
!mkdir -p /content/data
!ln -sf /content/drive/MyDrive/ColabData/severstal/train_images /content/data/train_images
!ln -sf /content/drive/MyDrive/ColabData/severstal/test_images /content/data/test_images
!ln -sf /content/drive/MyDrive/ColabData/severstal/train.csv /content/data/train.csv
!ls -la /content/data/

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 4: Clone Repo & Install                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
%cd /content
!rm -rf severstal-steel-defect-detection
!git clone https://github.com/MohitGoyal09/severstal-steel-defect-detection.git
%cd /content/severstal-steel-defect-detection
!pip install -q torch torchvision torchaudio
!pip install -q segmentation-models-pytorch efficientnet_pytorch
!pip install -q albumentations opencv-python pandas numpy scipy matplotlib
!pip install -q tensorboard tqdm pyyaml click scikit-learn pillow
!pip install -q -e .

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 5: Smoke Test (5 epochs)                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
!sed -i 's/epochs: 50/epochs: 5/' experiments/unet-b5-colab.yml
!sed -i 's/start_val_epoch: 20/start_val_epoch: 3/' experiments/unet-b5-colab.yml
!sever train -c experiments/unet-b5-colab.yml

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 6: Full Training (restore config first)                   ║
# ╚══════════════════════════════════════════════════════════════════╝
!git checkout experiments/unet-b5-colab.yml
!sever train -c experiments/unet-b5-colab.yml

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 7: Save Results to Drive                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
!mkdir -p /content/drive/MyDrive/ColabResults/severstal-baseline
!cp -r /content/severstal-steel-defect-detection/saved/* /content/drive/MyDrive/ColabResults/severstal-baseline/
print("Done! Checkpoints saved to Drive.")
```

---

## Colab Config Details (`experiments/unet-b5-colab.yml`)

Changes from the original `unet-b5.yml`:

| Parameter         | Original             | Colab           | Reason                       |
| ----------------- | -------------------- | --------------- | ---------------------------- |
| `batch_size`      | 18                   | 8               | T4 has 16 GB VRAM            |
| `nworkers`        | 8                    | 2               | Colab has 2 CPU cores        |
| `data_dir`        | hardcoded local path | `/content/data` | Colab filesystem             |
| `epochs`          | 250                  | 50              | Faster iteration for testing |
| `start_val_epoch` | 100                  | 20              | Earlier validation feedback  |
| `early_stop`      | 50                   | 15              | Stop if not improving        |

**To run the full 250 epochs**, edit the config or create a new one:

```bash
!cp experiments/unet-b5-colab.yml experiments/unet-b5-colab-full.yml
!sed -i 's/epochs: 50/epochs: 250/' experiments/unet-b5-colab-full.yml
!sed -i 's/start_val_epoch: 20/start_val_epoch: 100/' experiments/unet-b5-colab-full.yml
!sed -i 's/early_stop: 15/early_stop: 50/' experiments/unet-b5-colab-full.yml
!sever train -c experiments/unet-b5-colab-full.yml
```

---

## Troubleshooting

### "No module named 'sever'"

```python
# Reinstall
%cd /content/severstal-steel-defect-detection
!pip install -q -e .
```

### "CUDA out of memory"

```python
# Reduce batch size further
!sed -i 's/batch_size: 8/batch_size: 4/' experiments/unet-b5-colab.yml
# Or use a smaller encoder
!sed -i 's/efficientnet-b5/efficientnet-b0/' experiments/unet-b5-colab.yml
```

### "FileNotFoundError: train_images"

Your Drive symlink is broken. Check:

```python
!ls -la /content/drive/MyDrive/ColabData/severstal/
!ls /content/data/train_images/ | head -3
```

If empty, re-upload the data to Drive.

### Training is very slow

```python
# Check GPU is actually being used
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# If CPU is being used, restart runtime:
# Runtime → Restart runtime → re-run all cells
```

### Session disconnected after 12 hours

This is expected on free Colab. Your checkpoints are saved to Drive if you ran Step 8. Resume with:

```bash
!sever train -c experiments/unet-b5-colab.yml -r /content/drive/MyDrive/ColabResults/.../checkpoint-epochXX.pth
```

---

## Next Steps After Baseline

Once baseline completes:

1. **Record baseline metrics** from TensorBoard (`val_dice_mean`, `val_dice_2`)
2. **Train the GAN** on Colab:
   ```python
   !python -m gan.train_wgan --config gan/config.yml
   ```
3. **Generate synthetic data**:
   ```python
   !python -m gan.generate_synthetic --generator saved/gan/G_final.pth --critic saved/gan/D_final.pth --output synthetic/ --n_samples 2000
   ```
4. **Run synthetic experiment**:
   ```python
   !sever train -c experiments/unet-b5-synthetic.yml
   ```

See `docs/PROJECT_GUIDE.md` for the full pipeline details.
