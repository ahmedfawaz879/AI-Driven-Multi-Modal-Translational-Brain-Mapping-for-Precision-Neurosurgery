"""NIfTI I/O, intensity normalization, 3D augmentation, and the dataset class.

Extracted from the original script (load_nifti, save_nifti,
IntensityNormalization, Augmentation3D, EnhancedBrainDataset,
center_crop_or_pad). Logic is unchanged; only imports and logger wiring
were adjusted for the package layout.
"""

import os
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import rotate, zoom
from torch.utils.data import Dataset

from .utils import logger

# ======================== NIfTI I/O ========================


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load NIfTI image with error handling"""
    try:
        img = nib.load(path)
        return img.get_fdata(), img.affine, img.header
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str):
    """Save data as NIfTI"""
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, path)
    logger.info(f"Saved NIfTI to {path}")


# ======================== Preprocessing ========================


class IntensityNormalization:
    """Advanced intensity normalization with multiple methods"""

    @staticmethod
    def zscore(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Z-score normalization"""
        if mask is not None:
            mean = vol[mask > 0].mean()
            std = vol[mask > 0].std()
        else:
            mean, std = vol.mean(), vol.std()
        return (vol - mean) / (std + 1e-8)

    @staticmethod
    def percentile_clip(vol: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
        """Percentile-based clipping and normalization"""
        pmin, pmax = np.percentile(vol, [lower, upper])
        vol = np.clip(vol, pmin, pmax)
        return (vol - pmin) / (pmax - pmin + 1e-8)

    @staticmethod
    def nyul_normalization(vol: np.ndarray) -> np.ndarray:
        """Nyul histogram matching normalization"""
        # Simplified version - full implementation would use reference histograms
        percentiles = np.percentile(vol, [1, 10, 25, 50, 75, 90, 99])
        vol_norm = np.interp(vol, percentiles, np.linspace(0, 1, len(percentiles)))
        return vol_norm


# ======================== Data Augmentation ========================


class Augmentation3D:
    """3D data augmentation for brain MRI"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def random_flip(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random flip along axes"""
        if np.random.random() < self.p:
            axis = np.random.choice([1, 2, 3])
            img = np.flip(img, axis)
            mask = np.flip(mask, axis)
        return img.copy(), mask.copy()

    def random_rotate(
        self, img: np.ndarray, mask: np.ndarray, max_angle: float = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random rotation"""
        if np.random.random() < self.p:
            angle = np.random.uniform(-max_angle, max_angle)
            axes = np.random.choice([(1, 2), (1, 3), (2, 3)])
            for c in range(img.shape[0]):
                img[c] = rotate(img[c], angle, axes=axes, reshape=False, order=1)
            mask = rotate(mask, angle, axes=axes, reshape=False, order=0)
        return img, mask

    def random_scale(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        scale_range: Tuple[float, float] = (0.9, 1.1),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random scaling"""
        if np.random.random() < self.p:
            scale = np.random.uniform(*scale_range)
            for c in range(img.shape[0]):
                img[c] = zoom(img[c], scale, order=1)
            mask = zoom(mask, scale, order=0)
        return img, mask

    def random_noise(self, img: np.ndarray, std: float = 0.05) -> np.ndarray:
        """Add random Gaussian noise"""
        if np.random.random() < self.p:
            noise = np.random.normal(0, std, img.shape)
            img = img + noise
        return img

    def random_gamma(
        self, img: np.ndarray, gamma_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """Random gamma correction"""
        if np.random.random() < self.p:
            gamma = np.random.uniform(*gamma_range)
            img = np.power(np.abs(img), gamma) * np.sign(img)
        return img


# ======================== Dataset ========================


class EnhancedBrainDataset(Dataset):
    """Enhanced multi-modal brain dataset with augmentation"""

    def __init__(
        self,
        cases: List[str],
        channels: List[str] = ["T1", "T2", "FLAIR"],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        normalization: str = "zscore",
    ):
        self.cases = cases
        self.channels = channels
        self.patch_size = patch_size
        self.augment = augment
        self.normalization = normalization
        self.augmentor = Augmentation3D(p=0.5) if augment else None

    def __len__(self):
        return len(self.cases)

    def normalize(self, vol: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        if self.normalization == "zscore":
            return IntensityNormalization.zscore(vol)
        elif self.normalization == "percentile":
            return IntensityNormalization.percentile_clip(vol)
        elif self.normalization == "nyul":
            return IntensityNormalization.nyul_normalization(vol)
        return vol

    def __getitem__(self, idx):
        case = self.cases[idx]
        imgs = []

        # Load all channels
        for ch in self.channels:
            path = os.path.join(case, f"{ch}.nii.gz")
            if not os.path.exists(path):
                logger.warning(f"Missing {path}, using zeros")
                data = np.zeros((128, 128, 128))
            else:
                data, _, _ = load_nifti(path)
                data = self.normalize(data)
            imgs.append(data)

        img = np.stack(imgs, axis=0).astype(np.float32)

        # Load mask
        mask_path = os.path.join(case, "mask.nii.gz")
        if os.path.exists(mask_path):
            mask, _, _ = load_nifti(mask_path)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros(img.shape[1:], dtype=np.uint8)

        # Apply augmentation
        if self.augment and self.augmentor:
            img, mask = self.augmentor.random_flip(img, mask)
            img, mask = self.augmentor.random_rotate(img, mask)
            img = self.augmentor.random_noise(img)
            img = self.augmentor.random_gamma(img)

        # Crop/pad to patch size
        img = center_crop_or_pad(img, self.patch_size)
        mask = center_crop_or_pad(mask[None], self.patch_size)[0]

        return torch.from_numpy(img), torch.from_numpy(mask).long(), case


def center_crop_or_pad(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or pad array to target shape"""
    if arr.ndim == 4:
        C = arr.shape[0]
        out = np.zeros((C,) + tuple(target_shape), dtype=arr.dtype)
        for c in range(C):
            out[c] = center_crop_or_pad(arr[c], target_shape)
        return out

    assert arr.ndim == 3
    out = np.zeros(tuple(target_shape), dtype=arr.dtype)
    in_shape = arr.shape

    # Calculate crop/pad indices
    starts = [max(0, (in_s - tar_s) // 2) for in_s, tar_s in zip(in_shape, target_shape)]
    ends = [start + tar for start, tar in zip(starts, target_shape)]

    # Crop
    cropped = arr[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]]

    # Pad if necessary
    pad_starts = [max(0, (tar - cr) // 2) for tar, cr in zip(target_shape, cropped.shape)]
    out[
        pad_starts[0] : pad_starts[0] + cropped.shape[0],
        pad_starts[1] : pad_starts[1] + cropped.shape[1],
        pad_starts[2] : pad_starts[2] + cropped.shape[2],
    ] = cropped

    return out
