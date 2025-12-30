# ml-on-fpga

# Semiconductor Wafer Defect Detection (Multi-Label CNN)

## Overview
This project presents a **CNN-based automated system** for detecting **multiple semiconductor wafer defect types simultaneously**. It replaces slow and subjective manual inspection with a **fast, scalable, and accurate** deep learning solution suitable for high-volume semiconductor manufacturing.

## Motivation
- Manual wafer inspection is slow and inconsistent  
- Mixed-type defects are difficult to classify  
- High-volume fabs require automated inspection  
- Early defect detection reduces cost and improves yield  

## Dataset
- **MixedWM38 Dataset**
- **38,015** wafer maps (52×52 binary images)
- **8 basic defect types** and **29 mixed-type combinations**
- **38-dimensional one-hot encoded labels**
- **GAN-based data augmentation** to handle class imbalance

## Preprocessing
- Normalize pixel values to [0, 1]
- Reshape images to (52, 52, 1)
- Data split: **75% Train / 12.5% Validation / 12.5% Test**
- Augmentation: rotation, shift, and zoom

## Model Architecture
- 4 CNN blocks:
  - Conv2D (3×3) → Batch Normalization → Max Pooling
- Dense layer with 256 neurons
- Dropout (0.4) for regularization
- Sigmoid output layer for **multi-label classification**

## Training
- Trained up to 30 epochs
- **Early Stopping** used to prevent overfitting

## Results
- **Test Accuracy:** 98.9%
- **Test Loss:** 0.033
- **Macro F1-Score:** 0.97
- High precision and recall across all defect types

## Conclusion
The proposed CNN model delivers **robust multi-label wafer defect detection**, effectively handling mixed and rare defect patterns. This approach is well-suited for real-world semiconductor fabrication and can be extended to larger datasets or deployed on edge devices for real-time inspection.
