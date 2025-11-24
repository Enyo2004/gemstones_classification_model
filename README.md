# gemstones_classification_model
A Deep Learning system (comparison of 6 models) using Keras and Tensorflow (with a PyTorch backend) to classify 87 types of gemstones from images. It applies Transfer Learning and Data Augmentation. The goal is to automate traditional gemology and generate an impact on the industry and commerce.


# Gemstones Classification Model

![Gemstones](https://img.shields.io/badge/Gemstones-87%20Classes-blueviolet)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A Deep Learning system designed to classify **87 different types of gemstones** from images. This project compares the performance of **6 different Convolutional Neural Network (CNN) models** using Keras and TensorFlow (with a PyTorch backend). 

By applying **Transfer Learning** and **Data Augmentation**, the system aims to automate traditional gemology processes, providing a scalable solution for the jewelry industry and commerce.

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Project Structure](#project-structure)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Prediction](#4-prediction)
- [Models Compared](#models-compared)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 📖 About The Project

Identifying gemstones is a task that traditionally requires expert gemologists and specialized equipment. This project leverages Computer Vision to automate the identification process.

**Key Features:**
* **Comprehensive Dataset:** Covers 87 distinct classes of gemstones.
* **Model Comparison:** Systematically evaluates 6 different deep learning architectures to determine the most accurate classifier.
* **Robust Pipeline:** Includes dedicated scripts for data handling, training, evaluation, and inference.
* **Transfer Learning:** Utilizes pre-trained weights to achieve high accuracy with limited training data.

## 📂 Project Structure

The repository is organized into the following directories:

```text
gemstones_classification_model/
├── Dataset/            # Directory for storing processed dataset files
├── Evaluation/         # Scripts for generating confusion matrices and accuracy metrics
├── Functions/          # Utility functions used across training and prediction
├── Gemstones/          # Raw images or intermediate data storage
├── Models/             # Python scripts defining and training the 6 different models
├── Prediction/         # Scripts for running inference on new gemstone images
├── unzipData/          # Utilities to extract and prepare the dataset
├── Gemstones.zip       # The compressed dataset (87 classes)
├── check_gpu_usage.py  # Utility script to verify GPU acceleration
└── README.md           # Project documentation


🛠 Built With
Python 3.x

TensorFlow & Keras

PyTorch (Backend support)

NumPy

Pandas

Matplotlib

🚀 Getting Started
To get a local copy up and running, follow these steps.

Prerequisites
Python 3.6+

pip package manager

(Optional) NVIDIA GPU with CUDA for faster training

Installation
Clone the repository

Bash

git clone [https://github.com/Enyo2004/gemstones_classification_model.git](https://github.com/Enyo2004/gemstones_classification_model.git)
cd gemstones_classification_model
Install required packages

Bash

pip install tensorflow keras torch numpy pandas matplotlib scikit-learn
Check GPU Availability (Optional) Run the provided script to ensure your GPU is detected:

Bash

python check_gpu_usage.py
💻 Usage
1. Data Preparation
The dataset is provided in a compressed format (Gemstones.zip). You must extract it before training.

Bash

python unzipData/unzip_script.py
# Or manually unzip Gemstones.zip to the root directory
2. Training
Navigate to the Models directory. This folder contains scripts for the 6 different models. Run the script corresponding to the model you wish to train.

Bash

cd Models
python <model_script_name>.py
# Example: python train_vgg16.py (Verify actual filename in folder)
3. Evaluation
After training, use the Evaluation scripts to assess model performance using metrics like accuracy, precision, recall, and confusion matrices.

Bash

cd ../Evaluation
python evaluate_models.py
4. Prediction
To classify a new image using your trained model:

Bash

cd ../Prediction
python predict.py --image_path "path/to/your/image.jpg"
🧠 Models Compared
This project evaluates six distinct architectures to benchmark performance on gemstone classification. (Check the Models/ directory for specific implementation details):

Model 1 (e.g., VGG16)

Model 2 (e.g., ResNet50)

Model 3 (e.g., InceptionV3)

Model 4 (e.g., MobileNet)

Model 5 (e.g., DenseNet)

Model 6 (e.g., Custom CNN)

