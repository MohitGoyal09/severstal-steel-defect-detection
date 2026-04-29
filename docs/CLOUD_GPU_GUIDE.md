> Complete guide to running SynthInspect on cloud GPUs (Vast.ai, RunPod, Lambda) with cost estimates for the full project.

---

## Table of Contents

1. [Why Cloud GPU?](#why-cloud-gpu)
2. [Cost Summary (Full Project)](#cost-summary-full-project)
3. [Vast.ai (Recommended: Best Price)](#vastai-recommended-best-price)
4. [RunPod (Recommended: Best UX)](#runpod-recommended-best-ux)
5. [Lambda Cloud (Simplest)](#lambda-cloud-simplest)
6. [Colab Pro / Pro+](#colab-pro--pro)
7. [Cost Comparison Table](#cost-comparison-table)
8. [Step-by-Step: Vast.ai Deployment](#step-by-step-vastai-deployment)
9. [Step-by-Step: RunPod Deployment](#step-by-step-runpod-deployment)
10. [One-Click Setup Script](#one-click-setup-script)
11. [Data Upload Options](#data-upload-options)
12. [Monitoring & Saving Results](#monitoring--saving-results)
13. [Troubleshooting](#troubleshooting)

---

## Why Cloud GPU?

| Problem                                 | Cloud Solution                |
| --------------------------------------- | ----------------------------- |
| Colab 12-hour limit kills long training | Rent GPU for days             |
| Mac MPS is 2× slower than CUDA          | Get real NVIDIA GPU           |
| No local GPU                            | Access RTX 3090/4090/A100     |
| Kaggle 30 hrs/week cap                  | Unlimited hours               |
| Need results in 2–3 days                | Finish full pipeline in 1 day |

---

## Cost Summary (Full Project)

### What's in "Full Project"?

| Stage                    | Epochs | GPU Hours         | Description                    |
| ------------------------ | ------ | ----------------- | ------------------------------ |
| **Baseline detector**    | 250    | ~10 hr            | U-Net on real data only        |
| **GAN training**         | 200    | ~6 hr             | Conditional WGAN-GP            |
| **Synthetic generation** | —      | ~0.5 hr           | Generate + filter 2000 samples |
| **Fixed-ratio detector** | 250    | ~10 hr            | 30% synthetic mixing           |
| **Curriculum detector**  | 250    | ~10 hr            | Dynamic curriculum             |
| **Extra experiments**    | —      | ~5 hr             | Ablations, different ratios    |
| **Buffer/debug**         | —      | ~3 hr             | Restarts, failed runs          |
| **TOTAL**                |        | **~45 GPU hours** |                                |

### Total Cost by Platform

| Platform         | GPU Type  | Hourly Rate       | Total Cost | Time to Complete |
| ---------------- | --------- | ----------------- | ---------- | ---------------- |
| **Vast.ai**      | RTX 3090  | **$0.35–0.55/hr** | **$16–25** | ~2 days          |
| **Vast.ai**      | RTX 4090  | **$0.60–0.90/hr** | **$27–40** | ~1.5 days        |
| **RunPod**       | RTX 3090  | $0.44–0.65/hr     | $20–29     | ~2 days          |
| **RunPod**       | RTX 4090  | $0.74–1.10/hr     | $33–50     | ~1.5 days        |
| **Lambda Cloud** | A10       | $0.60/hr          | $27        | ~2 days          |
| **Lambda Cloud** | A100      | $1.99/hr          | $90        | ~1 day           |
| **Colab Pro**    | V100/P100 | $10/month flat    | **$10**    | ~4–5 days\*      |
| **Colab Pro+**   | A100/V100 | $50/month flat    | **$50**    | ~2–3 days\*      |

\*Colab has usage limits; actual speed varies based on availability.

### 💰 Cheapest Option for Your Budget

| Budget     | Best Choice                      | What You Get              |
| ---------- | -------------------------------- | ------------------------- |
| **$0**     | Kaggle + Colab Free              | Finish in ~2 weeks        |
| **$10–15** | **Colab Pro (1 month)**          | V100/P100, ~4 days        |
| **$15–25** | **Vast.ai RTX 3090 (~2 days)**   | Fastest single-GPU value  |
| **$25–40** | **Vast.ai RTX 4090 (~1.5 days)** | Fastest consumer GPU      |
| **$50**    | **Colab Pro+ (1 month)**         | A100, most reliable       |
| **$100+**  | Lambda A100 or multi-GPU         | Overkill for this project |

---

## Vast.ai (Recommended: Best Price)

**Website:** [vast.ai](https://vast.ai)

### Why Vast.ai?

- **Cheapest RTX 3090/4090** rentals on the market
- **Spot pricing** — bid lower than asking price
- Pay by the hour, no subscription
- SSH access (full control)

### How to Rent

1. **Create account** at [vast.ai](https://vast.ai) (email + password)
2. **Add payment** (credit card or crypto)
3. **Search for GPU:**
   - Template: Pytorch
   - GPU: RTX 3090 or RTX 4090
   - Sort by: Lowest price
   - Image: `pytorch/pytorch:latest`
4. **Click "Rent"** on an instance
5. **Connect via SSH** (copy the SSH command from the instance page)

### Example SSH Command

```bash
ssh -p 12345 root@123.45.67.89 -L 6006:localhost:6006
# The -L flag forwards port 6006 for TensorBoard
```

---

## RunPod (Recommended: Best UX)

**Website:** [runpod.io](https://runpod.io)

### Why RunPod?

- **Best web UI** — JupyterLab built-in
- **Serverless + persistent storage**
- **Template marketplace** — one-click PyTorch setup
- Better for beginners than Vast.ai

### How to Rent

1. **Create account** at [runpod.io](https://runpod.io)
2. **Add credits** (minimum $10)
3. **Go to "Pods" → "Deploy"**
4. **Select GPU:**
   - RTX 3090 (24 GB) — ~$0.44/hr
   - RTX 4090 (24 GB) — ~$0.74/hr
5. **Select Template:** PyTorch (or Jupyter Lab PyTorch)
6. **Set Disk:** 50 GB (for dataset + checkpoints)
7. **Click "Deploy"**
8. **Open JupyterLab** (click "Connect" on your pod)

---

## Lambda Cloud (Simplest)

**Website:** [lambdalabs.com](https://lambdalabs.com)

### Why Lambda?

- **No setup** — pre-configured ML instances
- Reliable, enterprise-grade
- Good for longer runs (no eviction)

### Pricing

- 1× A10 (24 GB): $0.60/hr
- 1× A100 (40 GB): $1.99/hr
- 1× A100 (80 GB): $2.49/hr

---

## Colab Pro / Pro+

| Plan           | Price     | GPU       | Best For                          |
| -------------- | --------- | --------- | --------------------------------- |
| **Colab Pro**  | $10/month | V100/P100 | Budget users, slower but reliable |
| **Colab Pro+** | $50/month | A100/V100 | Faster, background execution      |

**Pros:** Flat rate, easy to use, integrated with Drive
**Cons:** Still has usage limits, not as fast as dedicated cloud

---

## Cost Comparison Table

| Task                   | T4 (Colab Free) | V100 (Colab Pro) | RTX 3090 (Vast.ai) | RTX 4090 (Vast.ai) | A100 (Lambda) |
| ---------------------- | --------------- | ---------------- | ------------------ | ------------------ | ------------- |
| Baseline (250 ep)      | ~45 hr          | ~15 hr           | ~10 hr             | ~6 hr              | ~5 hr         |
| GAN (200 ep)           | ~10 hr          | ~3.5 hr          | ~2.5 hr            | ~1.5 hr            | ~1 hr         |
| Synthetic gen          | ~0.5 hr         | ~0.2 hr          | ~0.15 hr           | ~0.1 hr            | ~0.1 hr       |
| Fixed-ratio (250 ep)   | ~45 hr          | ~15 hr           | ~10 hr             | ~6 hr              | ~5 hr         |
| Curriculum (250 ep)    | ~45 hr          | ~15 hr           | ~10 hr             | ~6 hr              | ~5 hr         |
| **Total time**         | ~150 hr         | ~50 hr           | ~35 hr             | ~20 hr             | ~17 hr        |
| **Total cost**         | $0              | $10/mo           | ~$18               | ~$30               | ~$90          |
| **Real calendar days** | ~2 weeks        | ~5–7 days        | ~2 days            | ~1 day             | ~1 day        |

---

## Step-by-Step: Vast.ai Deployment

### 1. Find and Rent an Instance

Go to [vast.ai/console/create](https://vast.ai/console/create/):

```
Filters:
  GPU: RTX 3090 or RTX 4090
  CUDA: >= 11.8
  Image: pytorch/pytorch:latest
  On-Demand (not interruptible)
  Sort by: lowest price
```

Click **Rent** on the cheapest reliable instance.

### 2. Connect via SSH

Copy the SSH command from the instance page:

```bash
ssh -p 12345 root@123.45.67.89 -L 6006:localhost:6006
```

### 3. Run the Setup Script

```bash
# Download and run the setup script
curl -fsSL https://raw.githubusercontent.com/MohitGoyal09/severstal-steel-defect-detection/master/scripts/setup_cloud.sh -o setup_cloud.sh
chmod +x setup_cloud.sh
./setup_cloud.sh
```

This will:

- Install system dependencies
- Install PyTorch with CUDA
- Clone your repo
- Install all Python packages
- Create helper scripts (`~/run_baseline.sh`, `~/run_gan.sh`, etc.)

### 4. Upload Data

**Option A: Download from Kaggle**

```bash
# Inside the SSH session
pip install kaggle
mkdir -p ~/.kaggle
# Upload kaggle.json via SCP or paste content
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_USERNAME","key":"YOUR_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c severstal-steel-defect-detection -p ~/data
unzip -q ~/data/severstal-steel-defect-detection.zip -d ~/data/
```

**Option B: Upload via SCP**

```bash
# From your local machine
scp -P 12345 -r /path/to/severstal/data/* root@123.45.67.89:~/data/
```

**Option C: Download from Google Drive**

```bash
pip install gdown
# Get file ID from Drive share link
gdown --id YOUR_FILE_ID -O ~/data/severstal.zip
unzip -q ~/data/severstal.zip -d ~/data/
```

### 5. Run Training

```bash
# Baseline
~/run_baseline.sh

# Or manually:
cd ~/severstal-steel-defect-detection
source ~/.venvs/severstal/bin/activate
 sever train -c experiments/unet-b5-colab.yml
```

### 6. Monitor with TensorBoard

The SSH command already forwarded port 6006. On your local machine:

```bash
# Open browser to:
http://localhost:6006
```

---

## Step-by-Step: RunPod Deployment

### 1. Create Pod

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods)
2. Click **+ Deploy**
3. Select **RTX 3090** or **RTX 4090**
4. Select **Template:** `RunPod Pytorch 2.1`
5. Set **Container Disk:** 50 GB
6. Set **Volume Disk:** 100 GB (for persistent data)
7. Click **Deploy**

### 2. Open JupyterLab

Click **Connect** → **JupyterLab** on your pod.

### 3. Run Setup in a Terminal

Inside JupyterLab, open a terminal and run:

```bash
# Run the setup script
curl -fsSL https://raw.githubusercontent.com/MohitGoyal09/severstal-steel-defect-detection/master/scripts/setup_cloud.sh | bash
```

### 4. Upload Data

In JupyterLab file browser:

1. Create `~/data/` folder
2. Drag-and-drop your `train_images/`, `test_images/`, `train.csv`

Or use the terminal:

```bash
# From Kaggle
pip install kaggle
# ... (same as Vast.ai instructions)
```

### 5. Run Training

Open a new terminal in JupyterLab:

```bash
~/run_baseline.sh
```

### 6. Monitor

RunPod has a built-in TensorBoard proxy. Or run:

```bash
~/run_tensorboard.sh
```

---

## One-Click Setup Script

The script at `scripts/setup_cloud.sh` automates everything:

```bash
# On any fresh Ubuntu GPU instance (Vast.ai, RunPod, Lambda, etc.)
curl -fsSL https://raw.githubusercontent.com/MohitGoyal09/severstal-steel-defect-detection/master/scripts/setup_cloud.sh | bash
```

### What it does:

1. Detects GPU and CUDA version
2. Installs system packages (git, opencv dependencies, etc.)
3. Creates Python virtual environment
4. Installs PyTorch with matching CUDA
5. Clones your repo
6. Installs all Python dependencies
7. Creates helper scripts (`run_baseline.sh`, `run_gan.sh`, etc.)
8. Verifies installation

### Created helper scripts:

| Script                    | Command                             |
| ------------------------- | ----------------------------------- |
| `~/run_baseline.sh`       | Train baseline detector             |
| `~/run_gan.sh`            | Train conditional WGAN-GP           |
| `~/generate_synthetic.sh` | Generate + filter synthetic defects |
| `~/run_tensorboard.sh`    | Launch TensorBoard on port 6006     |

---

## Data Upload Options

| Method                   | Speed  | Best For                         |
| ------------------------ | ------ | -------------------------------- |
| **Kaggle API**           | Fast   | If you have Kaggle account       |
| **SCP/SFTP**             | Medium | If data is on your local machine |
| **Google Drive + gdown** | Medium | If data is on Drive              |
| **AWS S3 / R2**          | Fast   | If data is in cloud storage      |
| **rsync**                | Fast   | Large transfers, resumable       |

### Quick Kaggle Download (inside cloud instance)

```bash
pip install kaggle
mkdir -p ~/.kaggle
# Paste your API key
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c severstal-steel-defect-detection
unzip -q severstal-steel-defect-detection.zip -d ~/data/
```

---

## Monitoring & Saving Results

### TensorBoard

```bash
# On the cloud instance
~/run_tensorboard.sh

# On your local machine (with SSH port forwarding)
ssh -p 12345 root@123.45.67.89 -L 6006:localhost:6006
# Then open: http://localhost:6006
```

### Save Checkpoints Before Stopping

Cloud instances are ephemeral. Always copy results before stopping:

```bash
# Option 1: Rsync to your local machine
rsync -avz -e "ssh -p 12345" root@123.45.67.89:~/severstal-steel-defect-detection/saved/ ./local-saved/

# Option 2: Upload to Google Drive
pip install gdown google-api-python-client
# Use Drive API or rclone

# Option 3: Upload to HuggingFace Hub
pip install huggingface-hub
huggingface-cli upload YOUR_USERNAME/severstal-checkpoints ./saved/ .
```

### Resume from Checkpoint

```bash
# Find latest checkpoint
ls -lt ~/severstal-steel-defect-detection/saved/*/checkpoint-epoch*.pth | head -1

# Resume baseline
 sever train -c experiments/unet-b5-colab.yml -r ~/severstal-steel-defect-detection/saved/.../checkpoint-epoch50.pth

# Resume GAN
python -m gan.train_wgan --config gan/config.yml --resume latest
```

---

## Troubleshooting

### "CUDA out of memory"

```bash
# Reduce batch size
sed -i 's/batch_size: 8/batch_size: 4/' ~/severstal-steel-defect-detection/experiments/unet-b5-colab.yml
# Or use a smaller encoder
sed -i 's/efficientnet-b5/efficientnet-b0/' ~/severstal-steel-defect-detection/experiments/unet-b5-colab.yml
```

### "nvcc not found" / PyTorch CPU mode

```bash
# Reinstall PyTorch with CUDA
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Instance gets preempted (Vast.ai spot)

Use **On-Demand** instances instead of spot. Or save checkpoints frequently:

```bash
# In a separate terminal, sync every 10 minutes
while true; do
  rsync -avz ~/severstal-steel-defect-detection/saved/ ./backup/
  sleep 600
done
```

### SSH connection drops

Use `tmux` or `screen` to keep training alive:

```bash
# Start a tmux session
tmux new -s training

# Run training
~/run_baseline.sh

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## Quick Reference: Platform Selection

```
Budget $0?       → Kaggle (30 hrs/week) + Colab Free
Budget $10–15?   → Colab Pro (1 month) → easiest
Budget $15–25?   → Vast.ai RTX 3090 → best value
Budget $25–40?   → Vast.ai RTX 4090 → fastest consumer
Budget $50?      → Colab Pro+ (1 month) → most reliable
Budget $100+?    → Lambda A100 → overkill, not needed
```

**My top recommendation for your project:** **Vast.ai RTX 3090 at ~$0.40/hr**.

- Total cost: **~$18–20**
- Finish full pipeline in **2 days**
- Best price/performance ratio

---

_For questions or issues, refer to the main README.md or docs/PROJECT_GUIDE.md._
