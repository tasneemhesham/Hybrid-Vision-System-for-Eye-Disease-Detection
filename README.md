# Hybrid Vision System for Eye Disease Detection

This project demonstrates how a **hybrid YOLOv8 + Vision Transformer (ViT) model** can be applied to classify **eye diseases** from retinal images.
It combines **localized feature extraction**, **global attention mechanisms**, and **efficient training strategies** to achieve state-of-the-art results.

---

## Project Overview

The goal of this project is to automatically classify images of **cataract, diabetic retinopathy, glaucoma, and normal eyes**.
Unlike standard CNN-based classifiers, YOLO-ViT leverages:

* **YOLOv8** for precise local feature extraction
* **Vision Transformer (ViT)** for capturing global dependencies
* Efficient training that achieves high performance in fewer epochs

The notebook implements the hybrid model for end-to-end training and evaluation.

---

## How It Works

1. **Input retinal images** are passed through the pretrained YOLOv8 backbone for local feature extraction.
2. **Feature maps** are converted into patch embeddings for the Vision Transformer.
3. **ViT processes global relationships** across the image to generate a comprehensive representation.
4. The **classification head** outputs the predicted disease category.
5. The system reports **accuracy, F1-score, and confusion matrix** for evaluation.

---

## Technologies Used

* **PyTorch** for deep learning
* **Hugging Face Transformers** for Vision Transformer
* **YOLOv8 (Ultralytics)** for convolutional feature extraction
* **NumPy** and **Matplotlib** for data handling and visualization

---
## Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/tasneemhesham/Hybrid-Vision-System-for-Eye-Disease-Detection.git
cd Hybrid-Vision-System-for-Eye-Disease-Detection
```

### 2. Install dependencies

   Install all required packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```
   If you prefer, you can install them manually using `pip install` for each package.

### 3. Download Dataset

    [Download the dataset here](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification?resource=download)
    
### 4. Run the Notebook

```bash
jupyter notebook eye_disease_classification.ipynb
```
---

## Sample Results

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1R-WsM4SOQbzIkQ-GcIUsn9-TvNqW42FJ" height="200"/>
</p>

---

## Evaluation Metrics

| Metric            | Description                                         |
| ----------------- | --------------------------------------------------- |
| **Accuracy**      | Overall proportion of correct predictions          |
| **F1-Score**      | Harmonic mean of precision and recall              |
| **Confusion Matrix** | Detailed class-wise prediction analysis          |

---

## Key Results

| Epochs | Accuracy | F1-Score |
|--------|---------|----------|
| 40     | 96.9%   | 0.97     |

- Glaucoma detection shows strong performance compared to conventional CNNs.  
- High efficiency: fewer epochs required for state-of-the-art results.  

---

## Publication

This project is part of a paper accepted at **AMLTA 2025**, to be published in **Springer Conference Proceedings** on 17th November.

---
