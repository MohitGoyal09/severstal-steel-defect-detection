Here’s a **clean product/context spec** you can paste into an “AI agent” (like a code assistant or builder) so it understands exactly what to build for this project.

***

## Project Name

**SynthInspect: Generative Augmentation for Steel Defect Detection (Severstal)**

***

## High‑Level Goal

Build an end‑to‑end PyTorch project that:

1. Trains a strong **baseline defect detector** on the **Severstal: Steel Defect Detection** Kaggle dataset.  
2. Trains a **conditional GAN (DG‑GAN‑style)** on defect patches from the same dataset.  
3. Uses the GAN to generate synthetic defect images + masks.  
4. Mixes real + synthetic data in a **curriculum‑based training loop** to improve rare‑class detection performance, with proper experiments and metrics.

The final system should be fully reproducible on a **MacBook M‑series (MPS)** and optionally portable to a CUDA GPU.

***

## Data / Dataset Context

- Main dataset: **Severstal: Steel Defect Detection** from Kaggle. [kaggle](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data)
- Format:
  - Images: `train_images/*.jpg`, size 1600×256.  
  - Annotations: `train.csv` with column `ImageId_ClassId` and `EncodedPixels` (RLE masks) for 4 defect classes.  
- Project repository already cloned: **`khornlund/severstal-steel-defect-detection`**. [github](https://github.com/khornlund/severstal-steel-defect-detection)
- Data directory layout (must be respected):
  ```text
  repo_root/
    data/
      train_images/
      test_images/      # optional
      train.csv
  ```

***

## Tech Stack and Environment

- Language: **Python 3.10+** (NOT 3.6 / old).  
- Framework: **PyTorch** with **MPS (Apple Silicon) support**.  
- Key libraries:
  - `segmentation_models_pytorch` (SMP) for U‑Net/FPN baseline.  
  - `albumentations` for augmentations.  
  - `opencv-python`, `pandas`, `numpy`, `scikit-learn`.  
- Run model training using the existing CLI in the repo: `sever train -c <config>`.

Do **not** try to reproduce the old `environment.yml` exactly (it’s CUDA 10 + PyTorch 1.2 + TF1). Use a modern stack and patch minor API issues (already done: `Sequence` import, `ToTensorV2`, removal of `Flip`).

***

## Existing Baseline (Must Keep Working)

The repo already provides:

- A **U‑Net/FPN segmentation baseline** implemented via SMP.  
- Training/validation loop accessible via:
  ```bash
  sever train -c experiments/unet-b5.yml
  ```
- Data loaders that:
  - Read `train.csv`, decode RLE to masks,  
  - Load images from `data/train_images`.

**Requirements for the agent:**

- **Do not break** existing baseline training.  
- All new code (GAN, mixed datasets) should be **added** in a clean way (new modules, new configs), not hacking the baseline to death.

***

## New Components to Build

### 1. GAN Data Module: DefectPatchDataset

Purpose: Provide training data for the GAN.

Implementation:

- Add a new package `gan/` in repo root:
  ```text
  gan/
    __init__.py
    dataset.py
    models.py
    train_wgan.py
    generate_synthetic.py
  ```
- In `gan/dataset.py`, implement a class like:

  ```python
  class DefectPatchDataset(Dataset):
      def __init__(self, csv_path, image_root, patch_size=(256, 256), min_defect_area=some_threshold):
          # read train.csv, group by ImageId
          # decode RLE to masks
          # for each defect region, extract patch around it
          # save (image_patch, mask_patch, condition_vector)

      def __getitem__(self, idx):
          # returns image_patch (C,H,W), condition (e.g., one-hot class + size bucket), mask_patch

      def __len__(self):
          ...
  ```

- Condition vector at minimum includes **defect class (1–4)**. Later may include size/severity buckets.

### 2. GAN Models (DG‑GAN‑style, initial simple version)

In `gan/models.py`:

- Implement a **conditional generator**:
  - Start from a DCGAN‑like generator or simple U‑Net generator that outputs a defect patch of fixed size.
  - Input: noise `z` + condition vector (class, etc.).
  - Output: generated **image patch** (and optionally a mask).

- Implement a **PatchGAN critic (discriminator)**:
  - CNN mapping patch → scalar score (for WGAN‑GP).
  - Receives both real and synthetic patches.

Design for extensibility:

- Start simple (no ASPP/attention).  
- Leave TODOs or hooks for later adding:
  - ASPP block in generator bottleneck.  
  - Attention modules.

### 3. GAN Training Script (WGAN‑GP)

In `gan/train_wgan.py`:

- Implement training loop:

  - Use `DefectPatchDataset` + `DataLoader` to get real patches and conditions.
  - For each batch:

    - Sample noise, generate synthetic patches: `x_fake = G(z, c)`.  
    - Compute critic loss (WGAN‑GP):
      \[
      L_D = E[D(x_{fake})] - E[D(x_{real})] + \lambda_{gp} \cdot \text{GP}
      \]
    - Update D multiple times per G update.
    - Generator loss:
      \[
      L_G = -E[D(x_{fake})]
      \]
      (optionally extend later with reconstruction/perceptual losses).

- Support training on MPS:
  ```python
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  ```

- Log losses and periodically save sample grids to a folder (`gan_samples/`) for visual inspection.

### 4. Synthetic Data Generation + Quality Filtering

In `gan/generate_synthetic.py`:

- Load trained G and D.
- For a given number of samples:
  - Sample random conditions (`class`, maybe size).  
  - Generate synthetic patches (`x_fake`).  
  - Compute D scores for each (`score = D(x_fake)`).

- Save:
  - Images: `synthetic/images/cls_X/img_XXXXX.png`.  
  - Masks (if G outputs them, or approximate via threshold).  
  - CSV: `synthetic/metadata.csv` with columns: `filename, class_id, score`.

- Implement quality filtering:
  - Keep only samples where score lies in a chosen band, e.g. between 30th and 80th percentile.

### 5. Mixed Dataset for Detector Training

Extend baseline data loader:

- Add a **synthetic dataset class** that reads from `synthetic/images` + masks + metadata CSV.
- Add a `MixedDataset`:

  ```python
  class MixedSeverstalDataset(Dataset):
      def __init__(self, real_ds, synth_ds, synth_ratio):
          self.real_ds = real_ds
          self.synth_ds = synth_ds
          self.synth_ratio = synth_ratio

      def __len__(self):
          return len(self.real_ds)

      def __getitem__(self, idx):
          if random.random() < self.synth_ratio and len(self.synth_ds) > 0:
              return self.synth_ds[random.randint(0, len(self.synth_ds)-1)]
          else:
              return self.real_ds[idx]
  ```

- Add config flags for detector experiments:

  ```yaml
  use_synthetic: true
  synthetic_ratio: 0.3
  synthetic_root: ./synthetic
  ```

- In detector data loader creation, if `use_synthetic` is true:
  - Build `real_ds` as usual.  
  - Build `synth_ds` from synthetic folder.  
  - Wrap with `MixedSeverstalDataset`.

### 6. Curriculum Scheduler

Inside the detector training (Trainer):

- Track `synthetic_ratio` as a state variable.
- After every `N` epochs:
  - Run validation.  
  - If rare‑class AP improved by more than threshold → increase `synthetic_ratio` (up to a max).  
  - If it degrades → decrease `synthetic_ratio` (down to 0 min).

Implement this as a simple function that updates `trainer.synth_ratio`, and make `MixedSeverstalDataset` read from that field each epoch (or rebuild dataset with updated ratio).

***

## Experiments & Outputs (What the Agent Should Enable)

The final codebase should make it easy to run:

1. **Baseline**  
   - `use_synthetic: false`.  
   - Output: baseline mAP / per‑class AP on held‑out real test set.

2. **Fixed‑ratio experiments**  
   - `use_synthetic: true`, `synthetic_ratio ∈ {0.1, 0.3, 0.5}`.  
   - Compare metrics vs baseline.

3. **Curriculum experiments**  
   - Enable scheduler, start at 0.1, allow auto‑adjust in [0.0, 0.5].  
   - Compare vs best fixed ratio.

4. **GAN metrics**  
   - Scripts to compute FID between real vs synthetic defect patches (optional but nice). [mdpi](https://www.mdpi.com/2073-8994/13/7/1176)

Outputs:

- Clean folders for:
  - Baseline detector checkpoints + logs.  
  - GAN checkpoints + sample images.  
  - Synthetic dataset.  
  - Experiment results (JSON/CSV with metrics per run).

***

## Quality / Design Constraints

- New code must be modular (separate `gan/` package, minimal changes to existing core).  
- Config‑driven: all hyperparameters (ratios, thresholds, patch size, etc.) configurable via YAML or command‑line.  
- Target hardware: MacBook M‑series (MPS) – keep batch sizes and models reasonable.  
- Clear docstrings and at least one `README` section explaining:
  - How to train baseline,  
  - How to train GAN,  
  - How to generate synthetic data,  
  - How to run mixed/curriculum experiments.

***

You can give this entire specification to your agent as **project context**. Then ask it for specific tasks like:

- “Implement `gan/dataset.py` following the spec.”  
- “Implement WGAN‑GP training in `gan/train_wgan.py`.”  
- “Extend the baseline dataloader to support `MixedSeverstalDataset` with a `synthetic_ratio` parameter.”