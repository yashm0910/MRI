# Tumor Detection Using Transfer Learning with MobileNetV2

---

## Overview

Brain tumor detection represents a critical challenge in medical diagnostics, where early identification can significantly improve patient outcomes. This project employs **deep learning** and **transfer learning** to classify MRI brain scans into **tumor** and **no tumor** categories. By leveraging **MobileNetV2**—a lightweight convolutional neural network pretrained on ImageNet—we achieve robust performance through fine-tuning and data augmentation. The resulting model balances accuracy and efficiency, making it ideal for deployment in resource-constrained environments.

---

## Dataset

- **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: `Yes (Tumor)` and `No (Healthy)`
- **Preprocessing and Augmentation**:
  - Dataset structured into train/validation/test splits (70/15/15 ratio)
  - Applied real-time augmentation: rotation, horizontal/vertical shifting, shearing, zoom, and horizontal flipping
  - Effective dataset expansion: ~2× to mitigate overfitting and enhance generalization

---

## Methodology

### 1. Data Preprocessing
- Loaded and normalized MRI images to 224×224 pixels (MobileNetV2 input size)
- Implemented stratified splitting to maintain class balance
- Integrated Keras `ImageDataGenerator` for on-the-fly augmentation during training

### 2. Model Architecture
- **Base Model**: MobileNetV2 (pretrained on ImageNet, weights frozen initially)
- **Custom Classifier Head**:
  - Global Average Pooling → Flatten → Dense (128 units, ReLU) → Dropout (0.5) → Dense (1 unit, Sigmoid)
- **Fine-Tuning**: Unfroze the last 20 layers for adaptation to the medical imaging domain, with a reduced learning rate

### 3. Training Strategy
- **Optimizer**: Adam (initial learning rate: 1e-4, with decay)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Regularization**: Early stopping (patience: 10 epochs) and model checkpointing for best validation performance
- **Hyperparameters**: Batch size: 32; Epochs: 50 (typically converges in 20–30)

---

## Results

| Metric              | Value   |
|---------------------|---------|
| **Test Accuracy**   | 84%    |
| **Validation Accuracy (Peak)** | 85% |
| **Precision**       | 83%    |
| **Recall**          | 84%    |
| **F1-Score**        | 83.5%  |

*Note*: Metrics may vary slightly across runs due to stochastic augmentation and random splits. Confusion matrix analysis confirms balanced performance across classes.

---

## Key Highlights

- **Efficiency**: MobileNetV2's depthwise separable convolutions enable ~3.5M parameters, facilitating edge deployment
- **Robustness**: Augmentation and fine-tuning yield strong generalization (minimal drop from validation to test)
- **Extensibility**: Model supports integration with explainability tools like activation heatmaps for clinical interpretability

---

## Future Enhancements

- Implement **Grad-CAM** for visualizing tumor-specific activations
- Benchmark against **EfficientNet** variants for potential accuracy gains
- Develop a **Streamlit-based web application** for real-time MRI predictions
- Expand to multi-class tumor subtyping (e.g., glioma vs. meningioma)

---

## Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow 2.x / Keras
- **Data Handling**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook / Google Colab

---

## Impact

This project exemplifies the transformative potential of **transfer learning** in healthcare AI. By combining open-source datasets with efficient CNN architectures, it democratizes access to advanced diagnostic tools—paving the way for scalable, cost-effective solutions in global health systems. Contributions to reproducibility: Full code available in the repository; trained model weights provided for inference.
