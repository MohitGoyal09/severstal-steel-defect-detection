# SynthInspect: Complete Cost & Time Breakdown

> One-file reference for GPU costs, time estimates, and platform selection for the full project pipeline.

---

## 1. Project Scope: What Counts as "Full Pipeline"

| #   | Stage                    | Description                             | Epochs | GPU Hours         |
| --- | ------------------------ | --------------------------------------- | ------ | ----------------- |
| 1   | **Baseline detector**    | Train U-Net on real data only           | 250    | ~10               |
| 2   | **GAN training**         | Conditional WGAN-GP on defect patches   | 200    | ~6                |
| 3   | **Synthetic generation** | Generate + quality-filter 2,000 samples | —      | ~0.5              |
| 4   | **Fixed-ratio detector** | Train with 30% synthetic mixing         | 250    | ~10               |
| 5   | **Curriculum detector**  | Train with dynamic curriculum mixing    | 250    | ~10               |
| 6   | **Extra experiments**    | Ablations (10%, 50%, no-filter, etc.)   | —      | ~5                |
| 7   | **Buffer / debug**       | Restarts, failed runs, tuning           | —      | ~3                |
|     | **TOTAL**                |                                         |        | **~45 GPU hours** |

> All estimates assume a **single GPU** running one experiment at a time.

---

## 2. Time per Epoch by GPU

| GPU                    | VRAM    | Speed vs T4 | Time per Epoch (batch=8) | 250 Epochs |
| ---------------------- | ------- | ----------- | ------------------------ | ---------- |
| **RTX 4090**           | 24 GB   | **~5×**     | ~2.5 min                 | ~6–7 hr    |
| **A100 (80 GB)**       | 80 GB   | **~4–5×**   | ~3 min                   | ~7–8 hr    |
| **RTX 3090**           | 24 GB   | **~2.5×**   | ~4 min                   | ~10–12 hr  |
| **V100**               | 16 GB   | **~3×**     | ~3.5 min                 | ~9–10 hr   |
| **P100**               | 16 GB   | **~1.5×**   | ~6 min                   | ~15–18 hr  |
| **T4 (Colab/Kaggle)**  | 16 GB   | **1×**      | ~9 min                   | ~22–25 hr  |
| **MPS (Mac M-series)** | Unified | **~0.5×**   | ~18 min                  | ~45–50 hr  |

---

## 3. Stage-by-Stage Time & Cost

### On **RTX 3090** (~$0.40/hr on Vast.ai)

| Stage                | Time       | Cost        |
| -------------------- | ---------- | ----------- |
| Baseline (250 ep)    | ~10 hr     | ~$4.00      |
| GAN (200 ep)         | ~6 hr      | ~$2.40      |
| Synthetic gen        | ~0.5 hr    | ~$0.20      |
| Fixed-ratio (250 ep) | ~10 hr     | ~$4.00      |
| Curriculum (250 ep)  | ~10 hr     | ~$4.00      |
| Extras + buffer      | ~8 hr      | ~$3.20      |
| **TOTAL**            | **~35 hr** | **~$18.00** |

### On **RTX 4090** (~$0.70/hr on Vast.ai)

| Stage                | Time       | Cost        |
| -------------------- | ---------- | ----------- |
| Baseline (250 ep)    | ~6 hr      | ~$4.20      |
| GAN (200 ep)         | ~3.5 hr    | ~$2.45      |
| Synthetic gen        | ~0.25 hr   | ~$0.18      |
| Fixed-ratio (250 ep) | ~6 hr      | ~$4.20      |
| Curriculum (250 ep)  | ~6 hr      | ~$4.20      |
| Extras + buffer      | ~5 hr      | ~$3.50      |
| **TOTAL**            | **~20 hr** | **~$14.00** |

> Wait — RTX 4090 costs more per hour but is so much faster that total cost is similar or even lower! The main difference is **calendar time**: 1 day vs 2 days.

### On **T4 (Colab Free / Kaggle)**

| Stage                | Time        | Cost   |
| -------------------- | ----------- | ------ |
| Baseline (250 ep)    | ~45 hr      | **$0** |
| GAN (200 ep)         | ~10 hr      | **$0** |
| Synthetic gen        | ~0.5 hr     | **$0** |
| Fixed-ratio (250 ep) | ~45 hr      | **$0** |
| Curriculum (250 ep)  | ~45 hr      | **$0** |
| Extras + buffer      | ~20 hr      | **$0** |
| **TOTAL**            | **~165 hr** | **$0** |

> Free platforms have session limits (12 hr Colab, 9 hr Kaggle). You'll need **10+ sessions** spread over **2–3 weeks**.

---

## 4. Platform Comparison

| Platform         | GPU Options   | Hourly Rate          | Total Cost\* | Calendar Days | Session Limit | Best For           |
| ---------------- | ------------- | -------------------- | ------------ | ------------- | ------------- | ------------------ |
| **Kaggle**       | T4            | **FREE** (30 hrs/wk) | **$0**       | ~14 days      | 9 hr          | Zero budget        |
| **Colab Free**   | T4            | **FREE**             | **$0**       | ~14 days      | 12 hr         | Zero budget        |
| **Colab Pro**    | V100/P100     | **$10/mo flat**      | **$10**      | ~5–7 days     | 24 hr         | Low budget, simple |
| **Colab Pro+**   | A100/V100     | **$50/mo flat**      | **$50**      | ~2–3 days     | Background    | Reliable fast      |
| **Vast.ai** ⭐   | RTX 3090      | **~$0.35–0.55**      | **~$16–25**  | ~2 days       | None          | **Best value**     |
| **Vast.ai**      | RTX 4090      | **~$0.60–0.90**      | **~$12–18**  | ~1 day        | None          | **Fastest cheap**  |
| **RunPod**       | RTX 3090      | ~$0.44–0.65          | ~$20–29      | ~2 days       | None          | Good UX            |
| **RunPod**       | RTX 4090      | ~$0.74–1.10          | ~$15–22      | ~1 day        | None          | Fast + UX          |
| **Lambda Cloud** | A10           | $0.60/hr             | ~$27         | ~2 days       | None          | Enterprise         |
| **Lambda Cloud** | A100          | $1.99/hr             | ~$90         | ~1 day        | None          | Overkill           |
| **AWS EC2**      | g4dn (T4)     | ~$0.50/hr            | ~$23         | ~3 days       | None          | AWS users          |
| **AWS EC2**      | p3.2xl (V100) | ~$3.00/hr            | ~$135        | ~2 days       | None          | Expensive          |

\*Total cost = ~45 GPU hours × hourly rate (except flat-rate subscriptions)

---

## 5. Budget-Based Recommendations

| Budget     | Best Option              | GPU      | Total Cost | Finish In | Commands                               |
| ---------- | ------------------------ | -------- | ---------- | --------- | -------------------------------------- |
| **$0**     | Kaggle + Colab Free      | T4       | **$0**     | ~2 weeks  | See COLAB_BASELINE_GUIDE.md            |
| **$10**    | Colab Pro (1 month)      | V100     | **$10**    | ~5–7 days | Subscribe at colab.research.google.com |
| **$15–20** | **Vast.ai RTX 3090** ⭐  | RTX 3090 | **~$18**   | ~2 days   | See CLOUD_GPU_GUIDE.md                 |
| **$25–35** | **Vast.ai RTX 4090**     | RTX 4090 | **~$15**   | ~1 day    | See CLOUD_GPU_GUIDE.md                 |
| **$50**    | Colab Pro+ (1 month)     | A100     | **$50**    | ~2–3 days | Subscribe at colab.research.google.com |
| **$100+**  | Lambda A100 or multi-GPU | A100     | ~$90+      | ~1 day    | Overkill for this project              |

### 🏆 Top Pick: Vast.ai RTX 3090

- **Cost**: ~$18 for the entire project
- **Speed**: Finish in 2 days
- **Why**: Best price/performance. RTX 3090 (24 GB) handles batch_size=16 comfortably. $0.35–0.50/hr is the cheapest reliable option.

### 🚀 Speed Pick: Vast.ai RTX 4090

- **Cost**: ~$15 (fewer hours offset higher rate)
- **Speed**: Finish in ~1 day
- **Why**: 2× faster than 3090. If your deadline is tight, spend $10 more and finish in half the time.

---

## 6. Hidden Costs to Know

| Cost                                 | Amount           | When                                                       |
| ------------------------------------ | ---------------- | ---------------------------------------------------------- |
| **Data transfer (cloud → instance)** | Usually $0       | Most providers include it                                  |
| **Storage (persistent disk)**        | ~$0.10/GB/month  | If you keep data on cloud storage                          |
| **Idle GPU time**                    | Full hourly rate | You pay even if not training — **stop instance when done** |
| **Overage (Colab Pro)**              | None             | Flat rate, but may get slower GPUs if overused             |
| **Setup time**                       | ~30 min once     | One-time per platform                                      |

**Money-saving tip**: Always stop/delete your cloud instance when not training. On Vast.ai/RunPod, you pay by the second. A forgotten instance running overnight costs $5–10 for nothing.

---

## 7. Colab Pro vs Cloud Rental: Detailed Comparison

| Factor            | Colab Pro ($10/mo)                       | Vast.ai RTX 3090 (~$18 total)   |
| ----------------- | ---------------------------------------- | ------------------------------- |
| **Total cost**    | $10/month (can reuse for other projects) | ~$18 one-time                   |
| **GPU**           | V100 or P100 (random assignment)         | RTX 3090 (guaranteed)           |
| **Speed**         | ~3× T4                                   | ~2.5× T4                        |
| **Session limit** | ~24 hours                                | None                            |
| **Availability**  | May get downgraded to T4                 | Guaranteed GPU type             |
| **Setup**         | Zero (familiar)                          | SSH + setup script (~10 min)    |
| **Persistence**   | Google Drive integration                 | Manual rsync/SCP                |
| **Best if**       | You do ML regularly                      | You want this project done fast |

**Verdict**: If this is your only ML project this month, Vast.ai is better value. If you do ML year-round, Colab Pro is a good subscription.

---

## 8. Example: Full Project on Vast.ai RTX 3090

### Day 1 (~$9 spent, ~18 hours GPU time)

```bash
# Morning: Setup (30 min)
ssh -p 12345 root@123.45.67.89
curl -fsSL https://raw.githubusercontent.com/MohitGoyal09/severstal-steel-defect-detection/master/scripts/setup_cloud.sh | bash

# Morning: Upload data (~1 hr)
kaggle competitions download -c severstal-steel-defect-detection
unzip -q severstal-steel-defect-detection.zip -d ~/data/

# Afternoon: Baseline training (~10 hr)
~/run_baseline.sh   # 250 epochs

# Evening: GAN training (~6 hr)
~/run_gan.sh        # 200 epochs
```

### Day 2 (~$9 spent, ~17 hours GPU time)

```bash
# Morning: Generate synthetic data (~0.5 hr)
~/generate_synthetic.sh

# Morning–Evening: Fixed-ratio detector (~10 hr)
 sever train -c experiments/unet-b5-synthetic.yml

# Evening–Night: Curriculum detector (~10 hr)
 sever train -c experiments/unet-b5-curriculum.yml
```

### Day 3 (~$0 spent)

```bash
# Morning: Download all results
rsync -avz root@123.45.67.89:~/severstal-steel-defect-detection/saved/ ./results/

# Stop instance on Vast.ai dashboard (important!)
```

**Total: ~2 days, ~$18, full pipeline complete.**

---

## 9. Example: Free Path (Kaggle + Colab)

### Week 1 (~15 Kaggle hours used)

| Day | Task                                 | Hours |
| --- | ------------------------------------ | ----- |
| Mon | Baseline epochs 0–50                 | 5 hr  |
| Tue | Baseline epochs 50–100 (resume)      | 5 hr  |
| Wed | Baseline epochs 100–150 (resume)     | 5 hr  |
| Thu | Baseline epochs 150–200 (resume)     | 5 hr  |
| Fri | Baseline epochs 200–250 (resume)     | 5 hr  |
| Sat | GAN training epochs 0–100            | 5 hr  |
| Sun | GAN training epochs 100–200 (resume) | 5 hr  |

### Week 2 (~15 Kaggle hours used)

| Day | Task                                  | Hours |
| --- | ------------------------------------- | ----- |
| Mon | Generate synthetic data               | 1 hr  |
| Tue | Fixed-ratio detector 0–100            | 5 hr  |
| Wed | Fixed-ratio detector 100–250 (resume) | 8 hr  |
| Thu | Curriculum detector 0–100             | 5 hr  |
| Fri | Curriculum detector 100–250 (resume)  | 8 hr  |
| Sat | Extra experiments                     | 5 hr  |
| Sun | Analyze results, write paper          | —     |

**Total: ~2 weeks, $0, full pipeline complete.**

> Kaggle gives 30 GPU hours/week. If you need more, switch to Colab Free for the overflow.

---

## 10. Checkpoint Strategy (Saves Money)

Saving checkpoints lets you resume instead of restarting — critical for limited budgets.

| Platform    | Checkpoint Location                         | How to Resume                                     |
| ----------- | ------------------------------------------- | ------------------------------------------------- |
| **Colab**   | `saved/` then copy to Drive                 | `!cp -r saved/ /content/drive/...`                |
| **Kaggle**  | `saved/` (persistent in output)             | Attach output dataset to new notebook             |
| **Vast.ai** | `~/severstal-steel-defect-detection/saved/` | Download via `rsync` or keep on persistent volume |
| **RunPod**  | Network volume (persistent)                 | Automatically persists across pod restarts        |

### Auto-save script (run in background)

```bash
# On Vast.ai / any cloud instance
while true; do
  rsync -avz --delete ~/severstal-steel-defect-detection/saved/ ~/backup-checkpoints/
  echo "Checkpoint backup: $(date)"
  sleep 1800  # every 30 minutes
done &
```

---

## 11. Summary Table: One Glance

| Your Situation          | Best Platform          | Cost          | Time     |
| ----------------------- | ---------------------- | ------------- | -------- |
| Student, $0 budget      | Kaggle + Colab Free    | **$0**        | ~2 weeks |
| Casual, $10 budget      | Colab Pro (1 month)    | **$10**       | ~1 week  |
| Deadline in 3 days      | Vast.ai RTX 3090       | **~$18**      | ~2 days  |
| Deadline in 1 day       | Vast.ai RTX 4090       | **~$15**      | ~1 day   |
| Corporate / grant money | Lambda Cloud A100      | **~$90**      | ~1 day   |
| Do ML all year          | Colab Pro subscription | **$10–50/mo** | Ongoing  |

---

## 12. Quick Start Commands

### Vast.ai (cheapest)

```bash
# 1. Rent RTX 3090 at vast.ai
# 2. SSH in
# 3. Run setup
curl -fsSL https://raw.githubusercontent.com/MohitGoyal09/severstal-steel-defect-detection/master/scripts/setup_cloud.sh | bash
# 4. Start training
~/run_baseline.sh
```

### Colab Pro (simplest)

```bash
# 1. Subscribe at colab.research.google.com/signup
# 2. Open notebook
# 3. Runtime → Change runtime → GPU
# 4. Run cells from docs/COLAB_BASELINE_GUIDE.md
```

### Kaggle (free)

```bash
# 1. Create notebook at kaggle.com
# 2. Add Severstal dataset
# 3. Turn on GPU in Settings
# 4. Run training cells
```

---

_Last updated: 2026-04-29. Prices are approximate and subject to change. Check provider websites for current rates._
