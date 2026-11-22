# -*- coding: utf-8 -*-
"""
Unit tests for the preprocessing module.
Compatible with both pytest and python -m unittest.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (
    DISEASE_INFO,
    LESION_TYPE_DICT,
    custom_preprocessing_generator,
    hair_removal,
    preprocess_for_prediction,
)


class TestHairRemoval(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.sample_rgb_image = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        self.black_image = np.zeros((64, 64, 3), dtype=np.uint8)

    def test_returns_same_shape(self):
        result = hair_removal(self.sample_rgb_image)
        self.assertEqual(result.shape, self.sample_rgb_image.shape)

    def test_returns_uint8(self):
        result = hair_removal(self.sample_rgb_image)
        self.assertEqual(result.dtype, np.uint8)

    def test_return_mask_flag(self):
        result, mask = hair_removal(self.sample_rgb_image, return_mask=True)
        self.assertEqual(result.shape, self.sample_rgb_image.shape)
        self.assertEqual(mask.shape, self.sample_rgb_image.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)

    def test_handles_float_input(self):
        float_img = self.sample_rgb_image.astype(np.float64)
        result = hair_removal(float_img)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, self.sample_rgb_image.shape)

    def test_black_image_unchanged(self):
        result = hair_removal(self.black_image)
        self.assertTrue(np.allclose(result, self.black_image, atol=1))

    def test_does_not_modify_input(self):
        original = self.sample_rgb_image.copy()
        hair_removal(self.sample_rgb_image)
        np.testing.assert_array_equal(self.sample_rgb_image, original)


class TestPreprocessForPrediction(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.sample_rgb_image = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    def test_output_shape_default(self):
        """Default target_size in the function signature is (28, 28) matching the CNN model."""
        result = preprocess_for_prediction(self.sample_rgb_image)
        self.assertEqual(result.shape, (1, 28, 28, 3))

    def test_output_shape_custom(self):
        result = preprocess_for_prediction(self.sample_rgb_image, target_size=(64, 64))
        self.assertEqual(result.shape, (1, 64, 64, 3))

    def test_output_normalized(self):
        result = preprocess_for_prediction(self.sample_rgb_image, target_size=(28, 28))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_accepts_pil_image(self):
        from PIL import Image
        pil_img = Image.fromarray(self.sample_rgb_image)
        result = preprocess_for_prediction(pil_img, target_size=(28, 28))
        self.assertEqual(result.shape, (1, 28, 28, 3))


class TestCustomPreprocessingGenerator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.sample_rgb_image = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    def test_returns_float32(self):
        result = custom_preprocessing_generator(self.sample_rgb_image)
        self.assertEqual(result.dtype, np.float32)

    def test_preserves_shape(self):
        result = custom_preprocessing_generator(self.sample_rgb_image)
        self.assertEqual(result.shape, self.sample_rgb_image.shape)


class TestDataDictionaries(unittest.TestCase):
    def test_disease_info_keys(self):
        expected = {
            "Melanocytic nevi",
            "Melanoma",
            "Benign keratosis-like lesions",
            "Basal cell carcinoma",
            "Actinic keratoses",
            "Vascular lesions",
            "Dermatofibroma",
            "Healthy skin",
        }
        self.assertEqual(set(DISEASE_INFO.keys()), expected)

    def test_disease_info_fields(self):
        for name, info in DISEASE_INFO.items():
            self.assertIn("description", info, f"Missing description for {name}")
            self.assertIn("severity", info, f"Missing severity for {name}")
            self.assertIn("recommendation", info, f"Missing recommendation for {name}")

    def test_lesion_type_dict_values_match_disease_info(self):
        for code, full_name in LESION_TYPE_DICT.items():
            self.assertIn(
                full_name,
                DISEASE_INFO,
                f"LESION_TYPE_DICT value '{full_name}' (code={code}) not found in DISEASE_INFO",
            )


if __name__ == "__main__":
    unittest.main()
