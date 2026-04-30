# SynthInspect: Task Parallelization & Optimal Scheduling

> How to minimize total project time by running tasks in parallel or sequentially.

---

## 1. Task Dependency Map

Every task in the pipeline has dependencies. Some can run simultaneously on different GPUs; others must wait.

```
TASK                     INPUTS                    DEPENDENCY     CAN PARALLELIZE?
──────────────────────────────────────────────────────────────────────────────────
Baseline detector    → data/train_images          NONE           ✅ YES (alone)
GAN training        → data/train_images          NONE           ✅ YES (alone)
Synthetic gen      → saved/gan/G_final.pth      GAN            ❌ NO
Fixed-ratio        → synthetic/                SYNTHETIC      ❌ NO
Curriculum         → synthetic/                SYNTHETIC      ❌ NO
```

### Dependency Tree

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
          ┌───────────────┼───────────────┐
          ▼               ▼               │
    ┌───────────┐   ┌───────────┐       │
    │  BASELINE │   │    GAN    │       │
    │  (250 ep)│   │  (200 ep)│       │
    └─────┬─────┘   └─────┬─────┘       │
          │               │               │
          │           ┌───┴───┐           │
          │           │ DONE  │           │
          │           └───┬───┘           │
          │               ▼               │
          │        ┌─────────────┐       │
          │        │  SYNTHETIC │       │
          │        │   GENERATE  │       │
          │        └──────┬──────┘       │
          │               │               │
          │         ┌────┴────┐         │
          │         ▼         ▼         │
          │    ┌─────────┐ ┌─────────┐  │
          │    │ FIXED-  │ │CURRICU-│  │
          │    │ RATIO   │ │  LUM   │  │
          │    │(250 ep)│ │(250 ep)│  │
          │    └────┬────┘ └────┬────┘  │
          │         │         │         │
          │         └────┬────┘         │
          │              ▼              │
          │         ┌────────┐         │
          │         │RESULTS │         │
          │         └────────┘         │
          │                              │
          ▼                              │
    ┌─────────────┐                    │
    │ BASELINE    │                    │
    │  RESULTS    │                    │
    └─────────────┘                    │
                                    ▼
                             ┌─────────────┐
                             │  ALL DONE  │
                             └─────────────┘
```

---

## 2. What Can Run in Parallel

### CAN Run in Parallel (Different GPUs)

| Task A                | Task B                     | Why                                |
| --------------------- | -------------------------- | ---------------------------------- |
| **Baseline detector** | **GAN training**           | Both only need `data/train_images` |
| **Baseline detector** | Synthetic generation       | Baseline doesn't need GAN output   |
| Any independent task  | Any other independent task | If they don't share outputs        |

### MUST Be Sequential

| Task                     | Must Wait For        | Why                         |
| ------------------------ | -------------------- | --------------------------- |
| Synthetic generation     | GAN training         | Needs trained `G_final.pth` |
| Fixed-ratio detector     | Synthetic generation | Needs `synthetic/` folder   |
| Curriculum detector      | Synthetic generation | Needs `synthetic/` folder   |
| Any detector (synthetic) | Synthetic generation | Needs synthetic data        |

---

## 3. Task Time Reference

All estimates for **A100 40GB** (your Vast.ai instance @ $0.42/hr).

| Task                   | Epochs | Batch Size | Time (A100)   | Cost (A100) | GPU Needed |
| ---------------------- | ------ | ---------- | ------------- | ----------- | ---------- |
| Baseline detector      | 250    | 32         | ~5–6 hr       | ~$2.50      | Any        |
| GAN training           | 200    | 16         | ~3–4 hr       | ~$1.50      | Any        |
| Synthetic generation   | —      | —          | ~0.3 hr       | ~$0.15      | Any        |
| Fixed-ratio detector   | 250    | 32         | ~5–6 hr       | ~$2.50      | Any        |
| Curriculum detector    | 250    | 32         | ~5–6 hr       | ~$2.50      | Any        |
| **TOTAL (sequential)** |        |            | **~19–22 hr** | **~$9**     | 1 GPU      |

---

## 4. Optimal Schedules

### 1 GPU — Sequential (~19–22 hours)

```
HOUR  0-5   │ Baseline (250 ep)                ─────────────────────────►
HOUR  5-8   │ GAN (200 ep)                    ──────────────────────►
HOUR  8-9   │ Synthetic generation            ────►
HOUR  9-14  │ Fixed-ratio (250 ep)            ──────────────────►
HOUR 14-19  │ Curriculum (250 ep)             ──────────────────►
────────────────────────────────────────────────────────────────►
Total: ~19 hours, Cost: ~$9
```

---

### 2 GPUs — Parallel Start (~12 hours) ⭐

```
GPU 1:
HOUR  0-5   │ Baseline (250 ep)                ──────────────────►
                              HOUR  5-10  │ Fixed-ratio (250 ep)  ────────────►

GPU 2:
HOUR  0-3   │ GAN (200 ep)                    ─────────────►
HOUR  3-4   │ Synthetic generation            ────►
                              HOUR  4-9   │ Curriculum (250 ep)  ────────────►
──────────────────────────────────────────────────────────────────►
Total: ~10 hours on GPU 2, Cost: ~$9
```

**Time saved: ~7–9 hours**

---

### 3 GPUs — Optimal (~8 hours)

```
GPU 1:
HOUR  0-5   │ Baseline (250 ep)          ─────────────►
                              HOUR  5-10  │ Fixed-ratio (250 ep)  ────────────►

GPU 2:
HOUR  0-3   │ GAN (200 ep)              ─────────────►
HOUR  3-4   │ Synthetic generation      ────►
                              HOUR  4-9   │ Curriculum (250 ep)  ────────────►

GPU 3:
(Idle — use for ablation studies, FPN experiments, etc.)
─────────────────────────────────────────────────────────────►
```

**Time saved: ~11 hours**

---

## 5. Your A100: Optimal 2-GPU Schedule

Since you have **1 A100** but can run 2 tmux sessions:

```
HOUR  0 → Start BASELINE in tmux session 1
HOUR  0 → Start GAN in tmux session 2

HOUR  5 → BASELINE done
         → Start FIXED-RATIO in tmux session 1
         → GAN still running...

HOUR  8 → GAN done
         → Start SYNTHETIC GENERATION in tmux session 2

HOUR  9 → SYNTHETIC done
         → Start CURRICULUM in tmux session 2

HOUR 14 → CURRICULUM done → ALL DONE!
─────────────────────────────────────────
Total: ~14 hours (sequential on A100)
Cost: ~$6
```

But with proper parallelization:

```
HOUR  0 → BASELINE (tmux session 1)
HOUR  0 → GAN (tmux session 2)

HOUR  5 → BASELINE done → FIXED-RATIO starts
HOUR  8 → GAN done → SYNTHETIC starts

HOUR  9 → SYNTHETIC done → CURRICULUM starts

HOUR 14 → CURRICULUM done
─────────────────────────────────────────
Total: ~14 hours (on 1 GPU, but GAN runs in parallel with baseline)
```

**Key insight: Run BASELINE and GAN together from the start. This saves ~3-4 hours.**

---

## 6. tmux Commands for Parallel Execution

### On Your A100 Instance

```bash
# ─── Install tmux ───
apt-get update -qq && apt-get install -y -qq tmux

# ─── Start Session 1: Baseline ───
tmux new-session -d -s baseline -c ~/severstal-steel-defect-detection
tmux send-keys -t baseline ' sever train -c experiments/unet-b5-a100.yml' C-m

# ─── Start Session 2: GAN ───
tmux new-session -d -s gan -c ~/severstal-steel-defect-detection
tmux send-keys -t gan 'python -m gan.train_wgan --config gan/config.yml' C-m

# ─── Check Both Running ───
tmux list-sessions
# Output should show: baseline running, gan running

# ─── View Baseline ───
tmux attach -t baseline

# ─── Switch to GAN ───
# Press Ctrl+B, then D to detach
tmux attach -t gan

# ─── Switch Between Sessions ───
# Ctrl+B, then (previous window)
# Ctrl+B, then ) (next window)
```

### Quick Reference

| Command                     | Action               |
| --------------------------- | -------------------- |
| `tmux new -s name`          | Create named session |
| `tmux attach -t name`       | View session         |
| `Ctrl+B, then D`            | Detach (minimize)    |
| `Ctrl+B, then %`            | Split horizontally   |
| `Ctrl+B, then "`            | Split vertically     |
| `Ctrl+B, then )`            | Next session         |
| `Ctrl+B, then (`            | Previous session     |
| `tmux kill-session -t name` | Kill session         |
| `tmux list-sessions`        | List all sessions    |

### When BASELINE Finishes (Hour ~5)

```bash
# Detach from baseline
# Ctrl+B, then D

# Start Fixed-Ratio
tmux new-session -d -s fixed-ratio -c ~/severstal-steel-defect-detection
tmux send-keys -t fixed-ratio 'sever train -c experiments/unet-b5-synthetic.yml' C-m
```

### When GAN Finishes (Hour ~8)

```bash
# Detach from gan
# Ctrl+B, then D

# Start Synthetic Generation
tmux new-session -d -s synthetic -c ~/severstal-steel-defect-detection
tmux send-keys -t synthetic 'python -m gan.generate_synthetic \
    --generator saved/gan/G_final.pth \
    --critic saved/gan/D_final.pth \
    --output synthetic/ \
    --n_samples 2000' C-m
```

### When SYNTHETIC Finishes (Hour ~9)

```bash
# Detach from synthetic

# Start Curriculum
tmux new-session -d -s curriculum -c ~/severstal-steel-defect-detection
tmux send-keys -t curriculum 'sever train -c experiments/unet-b5-curriculum.yml' C-m
```

---

## 7. Summary Table

| GPUs  | Parallel Strategy                                            | Total Time | Cost | Time Saved    |
| ----- | ------------------------------------------------------------ | ---------- | ---- | ------------- |
| **1** | Sequential (baseline → GAN → synthetic → fixed → curriculum) | ~19 hr     | ~$9  | —             |
| **1** | Partial parallel (baseline + GAN together, then sequential)  | ~14 hr     | ~$6  | **~5 hours**  |
| **2** | Baseline+GAN parallel, then detectors sequential             | ~10 hr     | ~$9  | **~9 hours**  |
| **3** | Full parallel (3 simultaneous tasks)                         | ~8 hr      | ~$9  | **~11 hours** |

---

## 8. Decision Guide

```
How many GPUs do you have?
│
├── 1 GPU ($0.42/hr A100)
│   └── Run BASELINE + GAN together in tmux sessions
│       → Saves ~5 hours
│
├── 2 GPUs
│   └── GPU 1: Baseline → Fixed-ratio
│       GPU 2: GAN → Synthetic → Curriculum
│       → Saves ~9 hours
│
└── 3+ GPUs
    └── GPU 1: Baseline → Fixed-ratio
        GPU 2: GAN → Synthetic → Curriculum
        GPU 3: Ablation studies, FPN experiments
        → Saves ~11 hours
```

---

## 9. One-Command: Start Everything

Paste this entire block on your A100 instance to start the optimal schedule:

```bash
# ══════════════════════════════════════════════════════════════
#  OPTIMAL PARALLEL SCHEDULE — A100 SINGLE GPU
# ══════════════════════════════════════════════════════════════

# 1. Setup tmux
apt-get update -qq && apt-get install -y -qq tmux

# 2. Create A100-optimized config (if not exists)
cd ~/severstal-steel-defect-detection
if [ ! -f experiments/unet-b5-a100.yml ]; then
  cp experiments/unet-b5-colab.yml experiments/unet-b5-a100.yml
  sed -i 's|batch_size: 8|batch_size: 32|' experiments/unet-b5-a100.yml
  sed -i 's|nworkers: 2|nworkers: 8|' experiments/unet-b5-a100.yml
  sed -i 's|start_val_epoch: 20|start_val_epoch: 100|' experiments/unet-b5-a100.yml
  sed -i 's|epochs: 50|epochs: 250|' experiments/unet-b5-a100.yml
fi

# 3. Start BASELINE (Session 1)
tmux new-session -d -s baseline -c ~/severstal-steel-defect-detection
tmux send-keys -t baseline ' echo "=== BASELINE STARTED $(date) ===" && sever train -c experiments/unet-b5-a100.yml && echo "=== BASELINE DONE $(date) ==="' C-m

# 4. Start GAN (Session 2)
tmux new-session -d -s gan -c ~/severstal-steel-defect-detection
tmux send-keys -t gan ' echo "=== GAN STARTED $(date) ===" && python -m gan.train_wgan --config gan/config.yml && echo "=== GAN DONE $(date) ==="' C-m

# 5. Print schedule
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PARALLEL SCHEDULE STARTED                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Session 1 (baseline): BASELINE + FIXED-RATIO          ║"
echo "║  Session 2 (gan):       GAN + SYNTHETIC + CURRICULUM  ║"
echo "║                                                          ║"
echo "║  TO MONITOR:                                             ║"
echo "║    tmux attach -t baseline   (view baseline)          ║"
echo "║    tmux attach -t gan        (view GAN)                ║"
echo "║    tmux list-sessions        (see both)                 ║"
echo "║    Ctrl+B, then D          (detach from session)       ║"
echo "║                                                          ║"
echo "║  ESTIMATED TIMELINE:                                    ║"
echo "║    Hour  0-5:  Baseline running                       ║"
echo "║    Hour  5-10: Fixed-ratio running                     ║"
echo "║    Hour  0-3:  GAN running (parallel with baseline)   ║"
echo "║    Hour  3-4:  Synthetic generation                   ║"
echo "║    Hour  4-9:  Curriculum running                    ║"
echo "║                                                          ║"
echo "║  TOTAL TIME: ~14 hours                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
```

---

## 10. Checkpoints Before Stopping

**IMPORTANT**: Before stopping the instance, save all results:

```bash
# Create results directory
mkdir -p ~/results

# Copy everything
rsync -avz ~/severstal-steel-defect-detection/saved/ ~/results/
rsync -avz ~/severstal-steel-defect-detection/synthetic/ ~/results/synthetic/

# Download to your local machine (run on LOCAL terminal):
rsync -avz -e "ssh -p 40065" root@209.146.116.50:~/results/ ./severstal-results/

# THEN stop the instance on Vast.ai dashboard
```

---

_See also: docs/COST_BREAKDOWN.md (cost estimates), docs/CLOUD_GPU_GUIDE.md (cloud setup)_
