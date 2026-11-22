# -*- coding: utf-8 -*-
"""
Full Frontend End-to-End Test Suite using Streamlit AppTest.
Simulates real browser user sessions: file uploads, disclaimer gates,
inference runs, theme toggles, search filtering, and report downloads.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

IMAGE_FOLDER = Path(r"C:\Users\DELL\Desktop\New folder (4)")
APP_PATH = str(PROJECT_ROOT / "app.py")
TIMEOUT = 45


class TestFrontendStreamlitApp(unittest.TestCase):
    def test_initial_page_load(self):
        """Test initial render: no errors, correct status and elements."""
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        at.run(timeout=TIMEOUT)
        self.assertFalse(at.exception, f"App raised exceptions on load: {at.exception}")

        # Check title & section elements exist
        self.assertIn("latest_result", at.session_state)
        self.assertIsNone(at.session_state["latest_result"])
        self.assertEqual(at.session_state["theme"], "light")

    def test_theme_toggle(self):
        """Test toggling dark mode via checkbox in UI."""
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        at.run(timeout=TIMEOUT)

        # Find the theme toggle checkbox
        theme_boxes = [box for box in at.checkbox if box.key == "theme_toggle"]
        self.assertTrue(len(theme_boxes) > 0, "Theme toggle checkbox not found")

        # Toggle to dark
        theme_boxes[0].check().run(timeout=TIMEOUT)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["theme"], "dark")

        # Toggle back to light
        theme_boxes = [box for box in at.checkbox if box.key == "theme_toggle"]
        theme_boxes[0].uncheck().run(timeout=TIMEOUT)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["theme"], "light")

    def test_disclaimer_gate_blocks_prediction_without_ack(self):
        """Test that Run Prediction is disabled until disclaimer is checked."""
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        at.run(timeout=TIMEOUT)

        # Find disclaimer checkbox
        ack_boxes = [b for b in at.checkbox if b.key == "disclaimer_ack"]
        self.assertTrue(len(ack_boxes) > 0, "Disclaimer checkbox not found")
        self.assertFalse(ack_boxes[0].value, "Disclaimer should default to unchecked")

        # Find desktop run button
        run_buttons = [b for b in at.button if b.key == "desktop_run_prediction"]
        self.assertTrue(len(run_buttons) > 0, "Run Prediction button not found")
        self.assertTrue(run_buttons[0].disabled, "Run Prediction must be disabled initially")

    def test_upload_and_predict_each_desktop_image(self):
        """Upload each of the 8 images from Desktop folder and execute prediction."""
        images = sorted(list(IMAGE_FOLDER.glob("*.jpg")) + list(IMAGE_FOLDER.glob("*.png")))
        self.assertGreaterEqual(len(images), 8, f"Expected 8 images in {IMAGE_FOLDER}")

        for img_path in images:
            with self.subTest(image=img_path.name):
                at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
                at.run(timeout=TIMEOUT)

                # 1. Upload image to file uploader
                file_uploader = at.file_uploader[0]
                with open(img_path, "rb") as f:
                    file_bytes = f.read()
                file_uploader.upload(img_path.name, file_bytes, "image/jpeg")
                at.run(timeout=TIMEOUT)
                self.assertFalse(at.exception, f"Exception after upload of {img_path.name}: {at.exception}")

                # 2. Check disclaimer checkbox
                ack_box = [b for b in at.checkbox if b.key == "disclaimer_ack"][0]
                ack_box.check()
                at.run(timeout=TIMEOUT)
                self.assertFalse(at.exception)

                # 3. Click Run Prediction
                run_btn = [b for b in at.button if b.key == "desktop_run_prediction"][0]
                self.assertFalse(run_btn.disabled, "Run prediction button should now be enabled")
                run_btn.click()
                at.run(timeout=TIMEOUT)
                self.assertFalse(at.exception, f"Exception during prediction for {img_path.name}: {at.exception}")

                # 4. Verify results in session state and UI
                result = at.session_state["latest_result"]
                self.assertIsNotNone(result, f"latest_result is None for {img_path.name}")
                self.assertIn("predicted_class", result)
                self.assertIn("confidence", result)
                self.assertIn("all_probabilities", result)
                self.assertGreater(result["confidence"], 0.5)

                # 5. Check history was recorded
                history = at.session_state["history"]
                self.assertEqual(history[0]["filename"], img_path.name)
                self.assertEqual(history[0]["predicted_class"], result["predicted_class"])

                # 6. Verify Export Report download button is rendered
                dls = [el for el in at if getattr(el, "type", None) == "download_button"]
                self.assertTrue(len(dls) > 0, "Export Report download button not found")
                self.assertEqual(dls[0].type, "download_button")

                print(f"[Frontend E2E] Tested upload & inference for {img_path.name} -> {result['predicted_class']} ({result['confidence']*100:.1f}%)")

    def test_condition_library_search_filter(self):
        """Test searching and filtering condition cards in the Library section."""
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        at.run(timeout=TIMEOUT)

        # Find search input
        search_inputs = at.text_input
        self.assertTrue(len(search_inputs) > 0, "Search text input not found")
        search_inputs[0].input("Melanoma").run(timeout=TIMEOUT)
        self.assertFalse(at.exception)

    def test_condition_library_severity_filter(self):
        """Test the severity radio filter options."""
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        at.run(timeout=TIMEOUT)

        # Find severity filter radio
        radio = at.radio[0]
        self.assertEqual(radio.options, ["All", "High Risk", "Medium Risk", "Low Risk", "Normal / Healthy"])

        # Select each filter option and ensure clean rerun
        for option in ["High Risk", "Medium Risk", "Low Risk", "Normal / Healthy", "All"]:
            radio.set_value(option).run(timeout=TIMEOUT)
            self.assertFalse(at.exception, f"Exception when filtering by {option}: {at.exception}")


if __name__ == "__main__":
    unittest.main()
