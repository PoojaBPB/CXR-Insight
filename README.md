<div align="center">

# 🫁 CXR Insight

### Explainable Tuberculosis Detection from Chest X-rays

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

**[🌐 Open the Live App](https://cxr-insight-tb-screening.streamlit.app/)**

</div>

---

## About the Project

## About the Project

**CXR Insight** is a machine learning and deep learning project for automated tuberculosis (TB) detection from chest X-ray images using the **TBX11K dataset**.

The project compares HOG-based SVM and Random Forest models with ResNet50 and MobileNetV2, evaluates model performance at a screening-focused operating threshold, tests generalisation on the independent **Shenzhen chest X-ray dataset**, and uses Grad-CAM++ to explore the image regions influencing ResNet50 predictions.

---

## Models

| Approach | Model |
|---|---|
| Traditional Machine Learning | HOG + Support Vector Machine |
| Traditional Machine Learning | HOG + Random Forest |
| Deep Learning | ResNet50 |
| Deep Learning | MobileNetV2 |

ResNet50 and MobileNetV2 were developed using ImageNet transfer learning and fine-tuning.

---

## Project Workflow

```text
TBX11K Chest X-rays
        ↓
Data Inspection & EDA
        ↓
Preprocessing & Duplicate Removal
        ↓
 ┌──────────────────────┐
 │                      │
 ▼                      ▼
HOG Features       Deep Learning
 │                      │
SVM / RF        ResNet50 / MobileNetV2
 │                      │
 └──────────┬───────────┘
            ↓
     Model Evaluation
            ↓
Threshold Selection
 (≥90% Sensitivity)
            ↓
   External Validation
       on Shenzhen
            ↓
Grad-CAM++ Explainability
            ↓
     Streamlit App
```

---

## Internal Validation Results

The four models were evaluated on the cleaned **TBX11K validation set** and compared at an operating point requiring at least **90% sensitivity**.

| Model | Sensitivity | Specificity | F1-score | AUROC |
|---|---:|---:|---:|---:|
| **SVM** | 0.900 | **0.9705** | **0.8431** | **0.9884** |
| Random Forest | 0.900 | 0.9385 | 0.7531 | 0.9719 |
| ResNet50 | 0.900 | 0.9674 | 0.8333 | 0.9873 |
| MobileNetV2 | 0.900 | 0.9379 | 0.7516 | 0.9667 |

SVM achieved the strongest internal screening-focused performance on **TBX11K**, while ResNet50 was the strongest deep-learning model.

---

## External Validation

SVM and ResNet50 were evaluated on the independent **Shenzhen Hospital chest X-ray dataset** without retraining or threshold adjustment.

| Model | Sensitivity | Specificity | F1-score | AUROC |
|---|---:|---:|---:|---:|
| SVM | 0.6190 | 0.5276 | 0.5960 | 0.5907 |
| ResNet50 | 0.2411 | 0.9233 | 0.3665 | 0.6400 |

Both models showed a substantial reduction in performance compared with internal validation, highlighting the importance of evaluating medical-imaging models on independent data.

---

## Explainability

**Grad-CAM++** was applied to the fine-tuned ResNet50 model to visualise image regions influencing its predictions.

For some correctly classified TB cases, activation appeared within thoracic and lung regions. In other cases, attention extended towards peripheral or non-pulmonary areas.

These visualisations are used to explore model behaviour rather than to identify clinically confirmed TB lesions.

---

## CXR Insight Web App

The Streamlit application provides two main features.

### Case Explorer

Explore selected chest X-ray cases with known outcomes and view:

- TB / Non-TB prediction
- TB prediction score
- True result
- Original chest X-ray
- Grad-CAM++ visualisation
- Available TB annotation

### Upload & Analyse

Upload a frontal chest X-ray to view:

- TB / Non-TB prediction
- TB prediction score
- Grad-CAM++ visual explanation

**Live App:**  
https://cxr-insight-tb-screening.streamlit.app/

---

## Technologies

`Python` · `TensorFlow/Keras` · `scikit-learn` · `scikit-image` · `pandas` · `NumPy` · `Matplotlib` · `Pillow` · `Jupyter Notebook` · `Streamlit`

---


## Disclaimer

> **For research and educational use only.**
>
> CXR Insight is not a medical device and should not be used for clinical diagnosis, treatment decisions, or patient management.
