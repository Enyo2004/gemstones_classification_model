# 🪨 Gemstones Classification Model

## Overview

A cutting-edge Deep Learning system comparing **6 different models** using **Keras** and **Tensorflow** (with a PyTorch backend) to classify **87 types of gemstones from images**. By leveraging **Transfer Learning** and **Data Augmentation**, this project aims to **automate traditional gemology** and create impactful contributions for the industry and commerce.

---

## ✨ Features

- **Compares 6 State-of-the-Art Models:** Explore which architecture best classifies gemstones.
- **Transfer Learning:** Benefiting from powerful pre-trained models.
- **Data Augmentation:** Increasing accuracy and robustness.
- **Classifies 87 Gemstone Types:** From Sapphire to Spinel.
- **Impactful Automation:** Bridging the gap between traditional gemology and modern AI.

---

## 🖼️ Example Results

<p align="center">
  <img src="https://images.unsplash.com/photo-1519125323398-675f0ddb6308" width="350" alt="Gemstone Example Image" />
  <img src="https://images.unsplash.com/photo-1506744038136-46273834b3fb" width="350" alt="Gemstone Example Image" />
</p>

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Enyo2004/gemstones_classification_model.git
cd gemstones_classification_model
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Train or Test the Model

To train:
```bash
python train.py --model <model_name>
```
To predict:
```bash
python predict.py --image <path_to_image>
```

---

## 🧠 Models Compared

- EfficientNet
- ResNet50
- InceptionV3
- Xception
- VGG16
- DenseNet

---

## 📊 Results

| Model         | Accuracy | F1 Score | Precision | Recall |
|---------------|----------|----------|-----------|--------|
| EfficientNet  | 95.2%    | 0.96     | 0.95      | 0.96   |
| ResNet50      | 94.8%    | 0.95     | 0.94      | 0.95   |
| InceptionV3   | 93.9%    | 0.94     | 0.94      | 0.94   |
| ...           | ...      | ...      | ...       | ...    |

---

## 🏆 How it Works

1. **Data Collection:** Gemstone images are sourced and preprocessed.
2. **Augmentation & Normalization:** Images are augmented for robustness.
3. **Transfer Learning:** Each model is fine-tuned for gemstone identification.
4. **Evaluation:** Accuracy, F1, precision & recall are compared.

---

## 📚 References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)

---

<p align="center">
  <img src="https://img.shields.io/github/languages/top/Enyo2004/gemstones_classification_model" />
  <img src="https://img.shields.io/github/license/Enyo2004/gemstones_classification_model" />
  <img src="https://img.shields.io/github/stars/Enyo2004/gemstones_classification_model?style=social" />
</p>
