"""Tests for brain_mapping.data: cropping/padding and normalization.

NOTE: brain_mapping.data imports torch (EnhancedBrainDataset subclasses
torch.utils.data.Dataset), so importing this test module requires torch
to be installed, even though center_crop_or_pad/IntensityNormalization
themselves are pure-numpy. See README/CONTRIBUTING for environment setup.
"""

import numpy as np
import pytest

from brain_mapping.data import IntensityNormalization, center_crop_or_pad


class TestCenterCropOrPad:
    def test_crop_3d_to_smaller_shape(self):
        arr = np.arange(10 * 10 * 10, dtype=np.float32).reshape(10, 10, 10)
        out = center_crop_or_pad(arr, (4, 4, 4))
        assert out.shape == (4, 4, 4)
        # Center crop of a 10^3 cube to 4^3: start index = (10-4)//2 = 3
        expected = arr[3:7, 3:7, 3:7]
        np.testing.assert_array_equal(out, expected)

    def test_pad_3d_to_larger_shape_is_centered(self):
        arr = np.ones((4, 4, 4), dtype=np.float32)
        out = center_crop_or_pad(arr, (8, 8, 8))
        assert out.shape == (8, 8, 8)
        # Original 4^3 block of ones should sit centered: start = (8-4)//2 = 2
        assert np.all(out[2:6, 2:6, 2:6] == 1)
        # Everything outside that block should be zero-padded
        total_ones = out.sum()
        assert total_ones == 4 * 4 * 4

    def test_4d_applies_per_channel(self):
        arr = np.stack(
            [np.full((10, 10, 10), c, dtype=np.float32) for c in range(3)], axis=0
        )
        out = center_crop_or_pad(arr, (6, 6, 6))
        assert out.shape == (3, 6, 6, 6)
        for c in range(3):
            assert np.all(out[c] == c)

    def test_identity_when_shape_matches(self):
        arr = np.random.rand(5, 5, 5).astype(np.float32)
        out = center_crop_or_pad(arr, (5, 5, 5))
        np.testing.assert_array_equal(out, arr)

    def test_dtype_preserved(self):
        arr = np.ones((6, 6, 6), dtype=np.uint8)
        out = center_crop_or_pad(arr, (4, 4, 4))
        assert out.dtype == np.uint8


class TestIntensityNormalization:
    def test_zscore_zero_mean_unit_std(self):
        rng = np.random.RandomState(0)
        vol = rng.normal(loc=100, scale=20, size=(20, 20, 20)).astype(np.float32)
        out = IntensityNormalization.zscore(vol)
        assert out.mean() == pytest.approx(0.0, abs=1e-4)
        assert out.std() == pytest.approx(1.0, abs=1e-4)

    def test_zscore_with_mask_only_uses_masked_voxels(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        vol[0, 0, 0] = 10.0
        vol[1, 1, 1] = 20.0
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[1, 1, 1] = 1

        out = IntensityNormalization.zscore(vol, mask=mask)
        # mean/std computed from {10, 20} only -> mean=15, std=5
        assert out[0, 0, 0] == pytest.approx((10 - 15) / (5 + 1e-8), abs=1e-3)
        assert out[1, 1, 1] == pytest.approx((20 - 15) / (5 + 1e-8), abs=1e-3)

    def test_percentile_clip_bounds_output_zero_to_one(self):
        rng = np.random.RandomState(1)
        vol = rng.uniform(0, 1000, size=(15, 15, 15)).astype(np.float32)
        out = IntensityNormalization.percentile_clip(vol, lower=0.5, upper=99.5)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6
