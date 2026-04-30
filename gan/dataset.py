"""
DefectPatchDataset: extracts defect patches from Severstal images for GAN training.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DefectPatchDataset(Dataset):
    """
    Dataset that extracts patches around defect regions from Severstal steel images.

    Supports two CSV formats:
      Format A: ImageId_ClassId, EncodedPixels [, ImageId]
                (e.g. rows like "00bbcd9af.jpg_3", "155553 65 ...")
      Format B: ImageId, ClassId, EncodedPixels
                (long format with separate columns)

    Args:
        csv_path: Path to train.csv
        image_root: Path to train_images folder
        patch_size: (H, W) for extracted patches
        min_defect_area: Minimum number of pixels in defect mask to keep
        return_mask: If True, also return the binary mask patch
    """

    def __init__(
        self,
        csv_path,
        image_root,
        patch_size=(256, 256),
        min_defect_area=50,
        return_mask=True,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.patch_h, self.patch_w = patch_size
        self.min_defect_area = min_defect_area
        self.return_mask = return_mask

        self.samples = self._build_samples()
        self.mean_val = 0.3439

    def _build_samples(self):
        """Parse CSV and build list of defect patches."""
        df = pd.read_csv(self.csv_path)

        # ── Format A: ImageId_ClassId column present ──────────────────
        if "ImageId_ClassId" in df.columns:
            if "ImageId" in df.columns:
                # Format A1: ImageId_ClassId, EncodedPixels, ImageId
                # Use existing ImageId column as index (filename string)
                class_ids = df["ImageId_ClassId"].str.split("_").str[-1].astype(int)
                df["ClassId"] = class_ids
                df["ImageId"] = df["ImageId"].astype(str)
                df = df.set_index("ImageId")
                df = df.pivot(columns="ClassId", values="EncodedPixels")
                df.columns = [f"rle{int(c)}" for c in df.columns]
            else:
                # Format A2: ImageId_ClassId, EncodedPixels (no ImageId column)
                # Extract both filename and class from ImageId_ClassId
                parts = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
                df["ImageId"] = parts[0].astype(str)
                df["ClassId"] = parts[1].astype(int)
                df = df.set_index("ImageId")
                df = df.pivot(columns="ClassId", values="EncodedPixels")
                df.columns = [f"rle{int(c)}" for c in df.columns]

        # ── Format B: ClassId column present (long format) ─────────────
        elif "ClassId" in df.columns:
            df["ImageId"] = df["ImageId"].astype(str)
            df["ClassId"] = df["ClassId"].astype(int)
            df = df.set_index("ImageId")
            df = df.pivot(columns="ClassId", values="EncodedPixels")
            df.columns = [f"rle{int(c)}" for c in df.columns]

        # ── Fallback ───────────────────────────────────────────────────
        else:
            df = df.copy()
            df.index = df.index.astype(str)

        # ── Build sample list ───────────────────────────────────────────
        samples = []
        for img_name in df.index:
            img_path = self.image_root / str(img_name)
            if not img_path.exists():
                continue

            for c in range(4):
                col = f"rle{c + 1}"
                if col not in df.columns:
                    continue
                rle = df.loc[img_name, col]
                if pd.isna(rle):
                    continue

                mask = self._rle_to_mask(rle)
                area = mask.sum()
                if area < self.min_defect_area:
                    continue

                size_bucket = self._size_bucket(area)
                samples.append(
                    {
                        "image_id": str(img_name),
                        "class_id": c,  # 0-indexed: 0,1,2,3
                        "rle": rle,
                        "area": int(area),
                        "size_bucket": size_bucket,
                    }
                )

        return samples

    def _rle_to_mask(self, rle):
        """Decode RLE string to binary mask (256x1600)."""
        mask = np.zeros(256 * 1600, dtype=np.uint8)
        if pd.isna(rle) or rle == "":
            return mask.reshape(256, 1600)

        s = rle.split()
        starts = np.asarray(s[0::2], dtype=int)
        lengths = np.asarray(s[1::2], dtype=int)
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1
        return mask.reshape(256, 1600)

    def _size_bucket(self, area):
        """Categorize defect size into 3 buckets."""
        if area < 500:
            return 0  # small
        elif area < 5000:
            return 1  # medium
        else:
            return 2  # large

    def _extract_patch(self, img, mask, bbox):
        """Extract patch centered on bbox, padding if necessary."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        jitter_x = random.randint(-self.patch_w // 5, self.patch_w // 5)
        jitter_y = random.randint(-self.patch_h // 5, self.patch_h // 5)
        cx += jitter_x
        cy += jitter_y

        x1_p = max(0, cx - self.patch_w // 2)
        y1_p = max(0, cy - self.patch_h // 2)
        x2_p = min(img.shape[1], x1_p + self.patch_w)
        y2_p = min(img.shape[0], y1_p + self.patch_h)

        if x2_p - x1_p < self.patch_w:
            x1_p = max(0, x2_p - self.patch_w)
        if y2_p - y1_p < self.patch_h:
            y1_p = max(0, y2_p - self.patch_h)

        patch = img[y1_p:y2_p, x1_p:x2_p]
        mask_patch = mask[y1_p:y2_p, x1_p:x2_p]

        if patch.shape[0] < self.patch_h or patch.shape[1] < self.patch_w:
            padded = np.full(
                (self.patch_h, self.patch_w), self.mean_val * 255, dtype=np.float32
            )
            padded_mask = np.zeros((self.patch_h, self.patch_w), dtype=np.float32)
            h, w = patch.shape
            padded[:h, :w] = patch
            padded_mask[:h, :w] = mask_patch
            patch = padded
            mask_patch = padded_mask

        return patch, mask_patch

    def _get_bbox(self, mask):
        """Get bounding box of defect region."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = self.image_root / sample["image_id"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        mask = self._rle_to_mask(sample["rle"])
        bbox = self._get_bbox(mask)
        patch, mask_patch = self._extract_patch(img, mask, bbox)

        patch = patch / 255.0

        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
            mask_patch = np.fliplr(mask_patch).copy()

        patch_t = torch.from_numpy(patch).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_patch).unsqueeze(0).float()

        condition = np.zeros(7, dtype=np.float32)
        condition[sample["class_id"]] = 1.0
        condition[4 + sample["size_bucket"]] = 1.0
        condition_t = torch.from_numpy(condition)

        if self.return_mask:
            return patch_t, mask_t, condition_t
        return patch_t, condition_t


class SyntheticDefectDataset(Dataset):
    """
    Dataset that reads synthetic defect images and masks generated by the GAN.
    """

    def __init__(
        self, synthetic_root, metadata_csv="metadata_filtered.csv", transforms=None
    ):
        self.root = Path(synthetic_root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.transforms = transforms

        meta_path = self.root / metadata_csv
        if not meta_path.exists():
            self.samples = []
            for cls_dir in sorted(self.img_dir.glob("cls_*")):
                cls_id = int(cls_dir.name.split("_")[1]) - 1
                for img_path in sorted(cls_dir.glob("*.png")):
                    self.samples.append(
                        {
                            "filename": f"{cls_dir.name}/{img_path.name}",
                            "class_id": cls_id,
                            "score": 0.0,
                        }
                    )
        else:
            df = pd.read_csv(meta_path)
            self.samples = df.to_dict("records")
            for s in self.samples:
                s["class_id"] = int(s.get("class_id", 1)) - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fname = sample["filename"]

        img_path = self.img_dir / fname
        mask_path = self.mask_dir / fname.replace(".png", ".png")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((256, 256), dtype=np.uint8)
        img = img.astype(np.float32) / 255.0

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros_like(img)

        img = img[:, :, np.newaxis]
        mask = mask[:, :, np.newaxis]

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            mask = torch.from_numpy(np.array(mask)).permute(2, 0, 1).float()

        return img, mask
