import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MedSegPNGDataset(Dataset):
    """Dataset per immagini/mask PNG esportate da MedSegBench.

    Struttura attesa:
        root/
          <dataset_name>/
            <split>/
              images/*.png
              masks/*.png

    Converte le maschere in label 0..C-1.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        image_size: Tuple[int, int] | None = None,
        normalize_imagenet: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.normalize_imagenet = normalize_imagenet

        images_dir = os.path.join(root, dataset_name, split, "images")
        masks_dir = os.path.join(root, dataset_name, split, "masks")

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
        )
        mask_files = sorted(
            [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")]
        )
        common = sorted(set(image_files) & set(mask_files))
        if not common:
            raise RuntimeError(
                f"No matching image/mask PNG pairs found in {images_dir} and {masks_dir}"
            )

        self.images = [os.path.join(images_dir, f) for f in common]
        self.masks = [os.path.join(masks_dir, f) for f in common]

        # Scansiona le mask per costruire la mappa di label -> indice
        all_labels: set[int] = set()
        for mask_path in self.masks:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            vals = np.unique(mask)
            all_labels.update(int(v) for v in vals)

        if not all_labels:
            raise RuntimeError(f"No labels found in masks under {masks_dir}")

        self.raw_labels: List[int] = sorted(all_labels)
        self.label_to_index: Dict[int, int] = {
            raw: idx for idx, raw in enumerate(self.raw_labels)
        }
        self.num_classes: int = len(self.raw_labels)

        print(
            f"[{dataset_name} {split}] found labels: {self.raw_labels} -> num_classes={self.num_classes}"
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            h, w = self.image_size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        if self.normalize_imagenet:
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
            img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))  # [3,H,W]
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if self.image_size is not None:
            h, w = self.image_size
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)
        index_mask = np.zeros_like(mask, dtype=np.int64)
        for raw, idx in self.label_to_index.items():
            index_mask[mask == raw] = idx
        return index_mask

    def __getitem__(self, idx: int):  # type: ignore[override]
        import torch

        img = self._load_image(self.images[idx])
        mask = self._load_mask(self.masks[idx])
        return torch.from_numpy(img), torch.from_numpy(mask)
