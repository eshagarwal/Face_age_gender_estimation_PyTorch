# 👤 AI Face Analyzer (Multi-Task Learning)

An end-to-end computer vision application that detects human faces in real-time and predicts **Age**, **Gender**, and **Ethnicity**. The system utilizes a customized **MobileNetV3** architecture and is deployed via a **Streamlit** web interface.

## 🚀 Features
* **Real-time Detection:** Uses OpenCV Haar Cascades to locate faces in a video stream or uploaded image.
* **Multi-Task Prediction:** A single model pass predicts three distinct attributes simultaneously, optimizing computational efficiency.
* **Dual-Mode UI:** 
    * 📸 **Live Camera:** Capture a selfie via webcam and get instant analysis.
    * 📤 **Image Upload:** Upload high-resolution photos (JPG, PNG) for deep analysis.

---

## 🏗️ System Architecture

The project follows a modular pipeline that transforms raw pixels into structured data.

### 1. The Model (MobileNetV3 Backbone)
The core of the system is a **Multi-Task CNN**. Instead of maintaining three separate models, we use a shared "feature extractor" (MobileNetV3) that branches out into three specialized "heads":
* **Age Head:** A regression layer (Linear) outputting a continuous numerical value.
* **Gender Head:** A classification layer for binary prediction (Male/Female).
* **Ethnicity Head:** A multi-class classification layer for 5 distinct categories.

### 2. The Inference Pipeline
1.  **Face Detection:** OpenCV scans the input image and identifies the bounding box coordinates (x, y, w, h).
2.  **Preprocessing:** The detected face is cropped, converted to grayscale, and resized to **48x48** pixels to match the training input.
3.  **Normalization:** Pixel values are scaled to a range of [0, 1].
4.  **Forward Pass:** The processed tensor is fed into the model to generate the three predictions.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV
* **Frontend:** Streamlit
* **Backbone:** MobileNetV3 (Transfer Learning)
* **Data Handling:** NumPy, PIL, Matplotlib

---

## 📁 Project Structure

### 📂 Root Directory Files
```text
├── app.py                              # Streamlit web UI for real-time face analysis
│                                       # - Live webcam capture & face prediction
│                                       # - Image upload & batch processing
│                                       # - Interactive visualization of results
│
├── face_dataset.py                     # Custom PyTorch Dataset class
│                                       # - process_pixels(): Converts pixel strings to 48×48 images
│                                       # - FaceDataset: Handles data loading, normalization & label mapping
│                                       # - Used in training pipeline with DataLoaders
│
├── model.py                            # Multi-task learning model architecture
│                                       # - FaceAnalysisModel: MobileNetV3 backbone + 3 prediction heads
│                                       # - Age Head: Regression layer (continuous value)
│                                       # - Gender Head: Binary classification (Male/Female)
│                                       # - Ethnicity Head: Multi-class classification (5 groups)
│
├── face_cnn.pth                        # Pre-trained model weights
│                                       # - Saved checkpoint from training
│                                       # - Loaded for inference (app, notebooks)
│                                       # - Contains optimized parameters for all 3 tasks
│
└── requirements.txt                    # Python dependencies
```

### 📓 Jupyter Notebooks
```text
├── face_estimator.ipynb                # Complete training & validation pipeline
│                                       # 1. Data loading & preprocessing
│                                       # 2. Train/val/test split (80/10/10)
│                                       # 3. Model training with loss weighting
│                                       # 4. Learning rate scheduling
│                                       # 5. Visualization & results analysis
│
├── inference.ipynb                     # Inference module for single predictions
│                                       # 1. Load pre-trained model
│                                       # 2. Define preprocessing pipeline
│                                       # 3. Predict on individual face images
│                                       # 4. Display age, gender, ethnicity predictions
│
└── Face_Detection_System.ipynb         # Complete end-to-end detection pipeline
                                        # 1. Face detection using OpenCV Haar Cascade
                                        # 2. Multi-face handling in single image
                                        # 3. Attribute prediction for each face
                                        # 4. Visualization with bounding boxes & labels
```

### 📁 Data Directories
```text
├── data/                               # Training dataset directory
│   └── age_gender.csv                  # UTKFace dataset (CSV format)
│                                       # - Columns: age, gender, ethnicity, pixels
│                                       # - ~20,000 face images encoded as pixel strings
│
└── testing_data/                       # Test images for inference
```
---

## 🚦 Getting Started

### ⚡ Prerequisites
- **Python 3.12+** (specified in `.python-version`)
- **uv** package manager (faster than pip, recommended)
  - Install: `brew install uv` (macOS) or `pip install uv`

### Setup Instructions

1. **Clone the repository**

```Bash
git clone https://github.com/eshagarwal/face-analyzer.git
cd face-analyzer
```

2. **Install Dependencies**

#### Option A: Using `uv sync` (Recommended)
Installs all dependencies from `pyproject.toml`:
```Bash
uv sync
```

#### Option B: Using `uv add` (Manual)
Add individual packages:
```Bash
uv add streamlit torch torchvision opencv-python pillow numpy matplotlib scikit-learn tqdm
```

3. **Activate Virtual Environment** (if using `uv sync`)

```Bash
source .venv/bin/activate
```

4. **Run the Application**

```Bash
streamlit run app.py
```

---

## 📈 Model Insights
The model was trained on the UTKFace Dataset, containing over 20,000 face images. By using a shared backbone, the model footprint is significantly smaller and faster than running three independent models, making it ideal for deployment on standard laptops and CPU-based servers.

---

## 💡 Future Improvements
- Integrate MediaPipe or MTCNN for more robust face detection in challenging lighting or angles.
- Add Confidence Scores (Softmax probabilities) to show how "sure" the model is.
- Implement Face Tracking for smoother real-time video performance.