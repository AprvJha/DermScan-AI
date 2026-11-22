# Skin Disease Detection System Using Deep Learning

**Final Year Project**  

## Overview

This project implements a deep learning-based skin-lesion classification system for the **7 lesion types in the HAM10000 dataset**. The system features a custom CNN architecture with morphological hair-removal preprocessing and a user-friendly, responsive Streamlit web interface with real-time predictions, light/dark theming, clinical note recommendations, and case consultation export.

## Objectives

1. **Design and develop** a deep learning-based model capable of detecting and classifying common skin diseases and normal skin from image data.
2. **Evaluate** the performance and accuracy of deep learning-based models for skin disease detection.
3. **Implement** a web-based skin disease detection interface that enables users to upload skin images, view preprocessing comparisons, and receive real-time prediction results with clinical guidance.

## Detectable Skin Conditions

| Condition | Code | Severity | Type | Description |
|-----------|------|----------|------|-------------|
| Melanoma | `mel` | 🔴 High | Malignant | Serious skin cancer originating in melanocytes |
| Basal cell carcinoma | `bcc` | 🟠 Medium-High | Malignant | Common, slow-growing skin cancer |
| Actinic keratoses | `akiec` | 🟡 Medium | Pre-cancerous | Rough, scaly sun-induced patches |
| Melanocytic nevi | `nv` | 🟢 Low | Benign | Common non-cancerous moles |
| Benign keratosis-like lesions | `bkl` | 🟢 Low | Benign | Seborrheic keratoses, solar lentigo |
| Vascular lesions | `vasc` | 🟢 Low | Benign | Angiomas, pyogenic granulomas, hemorrhage |
| Dermatofibroma | `df` | 🟢 Low | Benign | Firm, harmless fibrous skin growths |

## Project Structure

```
DermScan-AI/
├── assets/                    # UI branding, condition sample images, demo video
├── models/                    # Trained model files and class mappings
│   ├── skin_disease_model.h5  # Shipped CNN model weights (28x28x3 input)
│   ├── best_model.h5          # Best checkpoint saved during training
│   └── class_indices.json     # Class name → index mapping (7 HAM10000 classes)
├── results/                   # Training metrics and visualization plots
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── dataset_distribution.png
│   └── classification_report.txt
├── styles/                    # Custom CSS styling (light/dark mode, responsive)
│   └── custom.css
├── tests/                     # Unit and integration test suite
│   ├── test_preprocessing.py  # Tests for hair removal and normalization
│   └── test_model.py          # Tests for model loading, shapes, and inference
├── train.py                   # Model training script
├── preprocessing.py           # Image preprocessing pipeline (hair removal, resize)
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Installation

### Prerequisites
- Python 3.9 – 3.11
- pip package manager

### Setup

1. **Navigate to the project directory:**
   ```bash
   cd DermScan-AI-main
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Application Features
- **Upload lesion images:** Accepts JPG, PNG, and WEBP images.
- **Mandatory Clinical Disclaimer:** Ensures explicit user acknowledgment prior to running predictions.
- **Hair Removal Preprocessing:** Side-by-side original and Telea-inpainted image comparison with lightbox zoom.
- **Multi-Class Probabilities:** Softmax distribution across all 8 classes with highlighted top prediction.
- **Condition Context & Recommendations:** Clinical summaries and next-step consultation advice.
- **Export & Share:** Download full text prediction reports and generate clinical consultation emails.
- **Condition Library:** Searchable and filterable reference guide categorized by risk severity.
- **Theme Support:** One-click light and dark mode switching with persistent local state.

## Methodology

### 1. Data Preprocessing (`preprocessing.py`)

1. **Hair Removal (Inpainting):**
   - Convert RGB image to grayscale.
   - Apply morphological **Black-Hat transform** with a **9×9 rectangular kernel** (specifically tuned to isolate thin hair strands without eroding lesion border texture or clinical features).
   - Create a binary mask using thresholding (threshold = 15).
   - Inpaint hair regions using the **Telea algorithm** (`cv2.INPAINT_TELEA`, radius = 6).
2. **Standardization & Resizing:**
   - Standardize input resolution to **28×28 pixels** with 3 RGB color channels.
   - Scale pixel values to the range `[0.0, 1.0]`.

### 2. CNN Architecture

```
Input Layer: 28 × 28 × 3 RGB
    ↓
Conv Block 1:
    Conv2D(16 filters, 3×3, ReLU)
    Conv2D(32 filters, 3×3, padding='same', ReLU)
    MaxPooling2D(pool_size=2×2)
    ↓
Conv Block 2:
    Conv2D(32 filters, 3×3, padding='same', ReLU)
    Conv2D(64 filters, 3×3, padding='same', ReLU)
    MaxPooling2D(pool_size=2×2)
    ↓
Regularization:
    Dropout(rate = 0.40)
    ↓
Classifier Head:
    Flatten
    Dense(128, ReLU)
    Dense(64, ReLU)
    Dense(32, ReLU)
    Dense(8, Softmax)  → 8 Class Probabilities
```

### 3. Training the Model (`train.py`)

To train the CNN model from scratch:
1. Download the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and extract `HAM10000_images_part_1/`, `HAM10000_images_part_2/`, and `HAM10000_metadata.csv` into the project root.
2. Run the training script:
   ```bash
   python train.py
   ```

**Training Configuration:**
- `IMG_SIZE`: 28×28 pixels
- `BATCH_SIZE`: 32
- `EPOCHS`: 25 (with Early Stopping, patience=5)
- `LEARNING_RATE`: 0.0001 (Adam optimizer with ReduceLROnPlateau)
- `CLASS_BALANCING`: Random oversampling of minority classes
- `DATA_AUGMENTATION`: Rotation (±20°), zoom (20%), horizontal flip, width/height shifts (20%)

## Running Tests

Run the unit test suite using `pytest`:

```bash
pytest tests/
```

Or run tests with Python's built-in `unittest` runner (no additional dependencies required):

```bash
python -m unittest discover tests
```

The test suite validates:
- Hair removal preprocessing, mask generation, and inpainting integrity.
- Image normalization and tensor dimensions (`(1, 28, 28, 3)`).
- Pretrained model file existence, layer architecture, and weight loading.
- Model input/output shape consistency with `class_indices.json`.
- End-to-end inference output formatting and probability constraints.

## API Reference

### `preprocessing.py`

```python
from preprocessing import hair_removal, preprocess_for_prediction, DISEASE_INFO

# Remove hair artifacts from an image
cleaned = hair_removal("sample.jpg")

# Preprocess image for model input (returns float32 array of shape (1, 28, 28, 3))
tensor = preprocess_for_prediction("sample.jpg")

# Retrieve clinical metadata
melanoma_info = DISEASE_INFO["Melanoma"]
print(melanoma_info["severity"])        # "High"
print(melanoma_info["recommendation"])  # Clinical guidance
```

## Medical Disclaimer

⚠️ **Important Notice:** DermScan AI is developed for educational, screening demonstration, and research purposes only. It is **not** a diagnostic medical device and does **not** provide final clinical diagnoses. Users must always seek the advice of a qualified dermatologist or physician for evaluation of skin concerns.

## License & Attribution

- Developed as a **Final Year Project**.
- Uses the **HAM10000 Dataset**:
  > Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018).
