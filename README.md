# 🌸 Flower Classification AI: Deep Learning Project

![Flower AI Banner](assets/banner.png)

Welcome to the **Flower Classification AI**! This project uses a state-of-the-art **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras** to classify images of different flowers (daisies, roses, sunflowers, tulips, and dandelions). 

It is designed to be lightweight, efficient, and ready for deployment using **TensorFlow Lite (TFLite)**.

## 🚀 Key Features

- **🌐 Deep Learning Architecture**: Uses a multi-layer CNN with Dropout for high accuracy.
- **🌀 Data Augmentation**: Real-time image flipping, rotation, and zooming to prevent overfitting.
- **⚡ TFLite Optimization**: High-performance model export for mobile and edge devices.
- **📸 Flexible Inference**: Easily classify any local images with a simple script.
- **📊 Training Visualizations**: Monitor your model's accuracy and loss through Matplotlib charts.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- Pip (Python Package Manager)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/algorithnicmind/image_processing.git
cd image_processing
pip install -r requirements.txt
```

### 3. Training the Model
To start the training process, run:
```bash
python train_model.py
```
This will:
1. Download the flower dataset (~230 MB).
2. Train the CNN for 15 epochs.
3. Save the final "brain" as `model.tflite` and `classes.txt`.

### 4. Classification & Inference
Once the model is trained, use `classify_image.py` to identify any flower:
```bash
python classify_image.py path/to/your/image.jpg
```
*(By default, it will classify a sample sunflower image if no path is provided.)*

---

## 📐 Architecture Overview

| Layer | Type | Filter/Units | Description |
| :--- | :--- | :--- | :--- |
| **Augmentation** | Sequential | - | Random Flip, Rotation, Zoom |
| **Rescaling** | Layer | - | Normalizes pixels to [0,1] |
| **Conv_1** | 2D Conv | 16 (3x3) | Edge & Texture extraction |
| **Max_Pool** | 2D MaxPool | - | Spatial dimension reduction |
| **Dropout** | Dropout | 0.2 | Prevents memorization (overfitting) |
| **Dense** | ReLu | 128 | High-level pattern recognition |
| **Output** | Softmax | 5 | Class probabilities |

---

## 🤝 Contributing

This project is a work in progress! We welcome all kinds of contributions:
1. **Find a Bug?** Open an Issue.
2. **Have a Feature?** Fork the repo and submit a Pull Request.
3. **Better Models?** Feel free to experiment with deeper architectures or transfer learning!

**How to contribute:**
- Fork the project.
- Create your feature branch (`git checkout -b feature/AmazingFeature`).
- Commit your changes (`git commit -m 'Add some AmazingFeature'`).
- Push to the branch (`git push origin feature/AmazingFeature`).
- Open a Pull Request.

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for more information.

## 🙌 Attributions
Dataset provided by **Tensorflow/Keras** examples.

---

*Made with ❤️ by [ankit](https://github.com/algorithnicmind)*
