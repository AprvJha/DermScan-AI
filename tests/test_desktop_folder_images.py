# -*- coding: utf-8 -*-
"""
Full backend test suite executing against all images in:
C:\\Users\\DELL\\Desktop\\New folder (4)
"""

import os
import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import get_model, load_class_indices, predict_disease
from preprocessing import (
    DISEASE_INFO,
    hair_removal,
    preprocess_for_prediction,
)

IMAGE_FOLDER = Path(r"C:\Users\DELL\Desktop\New folder (4)")


class TestDesktopFolderImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.model_path, cls.err = get_model()
        if cls.err:
            raise RuntimeError(f"Failed to load model: {cls.err}")
        cls.indices = load_class_indices()
        cls.images = sorted(list(IMAGE_FOLDER.glob("*.jpg")) + list(IMAGE_FOLDER.glob("*.png")))
        if not cls.images:
            raise FileNotFoundError(f"No images found in {IMAGE_FOLDER}")

    def test_image_folder_contains_expected_files(self):
        self.assertGreaterEqual(len(self.images), 8, f"Expected at least 8 images, found {len(self.images)}")
        print(f"\n[Test] Found {len(self.images)} test images in {IMAGE_FOLDER}")

    def test_all_desktop_images_inference(self):
        results = []
        print("\n" + "=" * 80)
        print(f"{'Image Filename':<22} | {'Predicted Condition':<28} | {'Conf':<7} | {'Time (ms)':<9} | {'Mask Area'}")
        print("-" * 80)

        for img_path in self.images:
            # 1. Read image
            t0 = time.perf_counter()
            img_bgr = cv2.imread(str(img_path))
            self.assertIsNotNone(img_bgr, f"cv2.imread failed for {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 2. Hair removal + mask
            cleaned, mask = hair_removal(img_rgb, return_mask=True)
            self.assertEqual(cleaned.shape, img_rgb.shape)
            self.assertEqual(mask.shape, img_rgb.shape[:2])
            mask_ratio = float(np.mean(mask > 0)) * 100.0

            # 3. Preprocessing
            tensor = preprocess_for_prediction(img_rgb)
            self.assertEqual(tensor.shape, (1, 28, 28, 3))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertGreaterEqual(tensor.min(), 0.0)
            self.assertLessEqual(tensor.max(), 1.0)

            # 4. Predict
            pred = predict_disease(self.model, img_rgb, self.indices)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # 5. Assertions
            self.assertIn("predicted_class", pred)
            self.assertIn("confidence", pred)
            self.assertIn("all_probabilities", pred)
            self.assertIn(pred["predicted_class"], self.indices)
            self.assertGreaterEqual(pred["confidence"], 0.0)
            self.assertLessEqual(pred["confidence"], 1.0)

            probs = pred["all_probabilities"]
            self.assertEqual(len(probs), 8)
            prob_sum = sum(probs.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=2)

            predicted_name = pred["predicted_class"]
            conf = pred["confidence"] * 100.0
            print(f"{img_path.name:<22} | {predicted_name:<28} | {conf:>5.1f}% | {elapsed_ms:>7.1f}ms | {mask_ratio:>5.1f}%")

            results.append({
                "filename": img_path.name,
                "predicted_class": predicted_name,
                "confidence": conf,
                "elapsed_ms": elapsed_ms,
                "mask_ratio": mask_ratio,
                "all_probabilities": probs,
            })

        print("=" * 80)
        self.assertEqual(len(results), len(self.images))


if __name__ == "__main__":
    unittest.main()
