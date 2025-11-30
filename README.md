# Ocean-Debris-Detecting-AI
This repository is for the CAP4630 Intro to Artificial Intelligence Project. Our goal is to create a presentation showcasing an AI solution to address the concerns of Oceanic debris.

#### Team Members  
- Felipe Lima  
- Jeremy Thyn  
- Emiley Coad  
- Nishat Shaneen  

**Resources:**  
- [PowerPoint](https://fau-my.sharepoint.com/:p:/r/personal/nshaneen2022_fau_edu1/Documents/Presentation.pptx?d=wf66b67e0404f46aaad6ccdcb56bd1485&csf=1&web=1&e=tZeo04)  
- [Dataset](https://www.kaggle.com/datasets/mexwell/trashcan-1-0)
- [Code Demo](https://www.youtube.com/watch?v=ZZe5SIoOgGo)

## Project Overview

This project implements a trash detection system that:
- Downloads and processes the TrashCan-1.0 dataset from Kaggle
- Converts bitmap annotations to YOLO format bounding boxes
- Trains a YOLOv8m (medium) model for trash detection
- Evaluates model performance with precision, recall, and mAP metrics

## Dependencies

### Required Python Packages

Install the following packages using pip:

```bash
pip install ultralytics kagglehub pillow opencv-python-headless numpy tqdm matplotlib pandas
```

### Core Dependencies

- **ultralytics**: YOLOv8 implementation for object detection
- **kagglehub**: Download datasets from Kaggle
- **pillow**: Image processing
- **opencv-python-headless**: Computer vision operations
- **numpy**: Numerical computations
- **tqdm**: Progress bars
- **matplotlib**: Visualization
- **pandas**: Data analysis

### Automatic Dependencies

The following packages are automatically installed as dependencies:
- **torch** and **torchvision**: PyTorch for deep learning
- **scipy**: Scientific computing
- **pyyaml**: YAML file parsing
- **requests**: HTTP requests

## Project Setup

### 1. Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training, but not required)
- Kaggle account (for dataset access)

### 2. Installation

1. Clone or download this project to your local machine.

2. Install the required dependencies:

```bash
pip install ultralytics kagglehub pillow opencv-python-headless numpy tqdm matplotlib pandas
```

Or install all at once:

```bash
pip install ultralytics kagglehub --upgrade
pip install pillow opencv-python-headless numpy tqdm matplotlib pandas
```

### 3. Kaggle Setup (for dataset download)

If you plan to download the dataset programmatically, you'll need to set up Kaggle API credentials:

1. Go to your Kaggle account settings
2. Create a new API token (download `kaggle.json`)
3. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

Alternatively, the notebook uses `kagglehub` which may handle authentication automatically in some environments.

## How to Run the Code

### Option 1: Run the Complete Notebook

1. Open the Jupyter notebook `trashDetection_Model_finalVersion.ipynb`

2. Execute cells in order:
   - **Cell 0**: Install dependencies
   - **Cell 1**: Download TrashCan-1.0 dataset from Kaggle
   - **Cells 2-5**: Explore and understand the dataset structure
   - **Cell 6**: Install additional image processing libraries
   - **Cell 7**: Create output directories for processed data
   - **Cell 8**: Define bitmap decoding function
   - **Cell 9**: Convert annotations from bitmap format to YOLO format
   - **Cell 10**: Verify conversion
   - **Cell 11**: Visualize sample images with bounding boxes
   - **Cell 12**: Create YAML configuration file for YOLO
   - **Cell 13**: Train the YOLOv8 model
   - **Cell 14**: Validate the trained model
   - **Cells 15-24**: Visualize results and metrics

### Option 2: Run Individual Components

You can also extract and run specific parts of the code:

#### Data Preprocessing

```python
# Download dataset
import kagglehub
path = kagglehub.dataset_download("mexwell/trashcan-1-0")

# Convert annotations (see notebook Cell 8-9 for full code)
# This converts bitmap masks to YOLO bounding box format
```

#### Model Training

```python
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8m.pt")

# Train on your dataset
model.train(
    data="/path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="/path/to/project",
    name="trashcan_model"
)
```

#### Model Validation

```python
# Load trained model
model = YOLO("/path/to/best.pt")

# Validate
model.val()
```

## Dataset Information

- **Dataset**: TrashCan-1.0 from Kaggle (`mexwell/trashcan-1-0`)
- **Total Images**: 7,212
- **Annotation Format**: Bitmap masks (converted to YOLO format)
- **Classes**: 1 class (trash)

## Model Configuration

- **Model Architecture**: YOLOv8m (medium)
- **Input Size**: 640x640 pixels
- **Batch Size**: 16
- **Training Epochs**: 50 ( We however ran it for 22 epochs due to runtime limitations on Colab)
- **Number of Classes**: 1 (trash)

## Expected Results

After training, the model achieves the following metrics:
- **Precision**: ~0.915
- **Recall**: ~0.844
- **mAP50**: ~0.923
- **mAP50-95**: ~0.701

## Output Files

The training process generates:
- **Model weights**: `best.pt` (best model) and `last.pt` (final epoch)
- **Training metrics**: `results.csv`
- **Validation results**: Confusion matrix, PR curves, F1 curves
- **Prediction samples**: Visualized validation batches

## Project Structure

```
.
├── trashDetection_Model_finalVersion.ipynb  # Main notebook
├── README.md                                # This file
├── dataset/                                 # Created during execution
│   ├── images/                             # Processed images
│   └── labels/                             # YOLO format labels
└── data.yaml                               # YOLO dataset configuration
```

## Notes

- The dataset is automatically downloaded from Kaggle when you run the notebook
- The conversion process transforms bitmap masks to YOLO bounding box format
- Training requires significant computational resources; GPU is highly recommended
- The model is saved to the specified project directory during training

## Troubleshooting

### Dataset Download Issues
- Ensure you have a Kaggle account and proper API credentials
- Check your internet connection
- Verify the dataset name is correct: `mexwell/trashcan-1-0`

### Training Issues
- Reduce batch size if you encounter out-of-memory errors
- Ensure sufficient disk space for model checkpoints
- Check that the data.yaml file points to correct image paths

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt` (if available)
- Verify Python version is 3.8 or higher
