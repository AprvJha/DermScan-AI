# -*- coding: utf-8 -*-
"""
Unit and integration tests for CNN model loading, validation, and inference.
Compatible with both pytest and python -m unittest.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Suppress TF logs during tests
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    DEFAULT_CLASS_INDICES,
    IMG_SIZE,
    MODEL_PATH,
    get_model,
    load_class_indices,
    predict_disease,
)


class TestModelArtifacts(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}")

    def test_class_indices_file_exists(self):
        indices_path = PROJECT_ROOT / "models" / "class_indices.json"
        self.assertTrue(indices_path.exists(), f"class_indices.json not found at {indices_path}")

    def test_class_indices_contents(self):
        indices = load_class_indices()
        self.assertEqual(len(indices), 7, f"Expected 7 classes, got {len(indices)}")
        self.assertIn("Melanoma", indices, "Melanoma not found in class indices")
        self.assertEqual(sorted(indices.values()), list(range(len(indices))))


class TestModelArchitectureAndInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.path, cls.err = get_model()
        if cls.err:
            raise RuntimeError(f"get_model returned error: {cls.err}")
        if cls.model is None:
            raise RuntimeError("Loaded model is None")

    def test_input_shape(self):
        expected_shape = (None, IMG_SIZE[0], IMG_SIZE[1], 3)
        self.assertEqual(
            self.model.input_shape,
            expected_shape,
            f"Expected input shape {expected_shape}, got {self.model.input_shape}",
        )

    def test_output_shape(self):
        expected_shape = (None, len(load_class_indices()))
        self.assertEqual(
            self.model.output_shape,
            expected_shape,
            f"Expected output shape {expected_shape}, got {self.model.output_shape}",
        )

    def test_predict_disease_output_structure(self):
        rng = np.random.default_rng(42)
        dummy_image = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        indices = load_class_indices()

        result = predict_disease(self.model, dummy_image, indices)

        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("all_probabilities", result)

        self.assertIn(result["predicted_class"], indices)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        probs = result["all_probabilities"]
        self.assertEqual(len(probs), len(indices))
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_predict_disease_reproducible(self):
        rng = np.random.default_rng(123)
        dummy_image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        indices = load_class_indices()

        res1 = predict_disease(self.model, dummy_image, indices)
        res2 = predict_disease(self.model, dummy_image, indices)

        self.assertEqual(res1["predicted_class"], res2["predicted_class"])
        self.assertAlmostEqual(res1["confidence"], res2["confidence"], places=6)


if __name__ == "__main__":
    unittest.main()
