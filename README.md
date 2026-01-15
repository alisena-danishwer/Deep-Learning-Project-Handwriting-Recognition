

# ✍️ Handwriting Recognition using OCR & CNN

## 📘 Project Overview

This project presents a **Handwriting Recognition System** based on **Optical Character Recognition (OCR)** combined with a **Convolutional Neural Network (CNN)**.
The system processes handwritten document images, segments them into **lines, words, and characters**, and performs **character-level classification**. Final predictions are obtained using **majority voting** to produce **image-level classification results**.

This project was developed as part of the **CCS3113 Deep Learning** course at **Albukhary International University**.

---

## 👥 Project Members

* **Ali Sena Danishwer** 
* **Nouhan Doumboya** 
* **Amadou Oury Diallo** 

**Instructor:** Dr. Mozaherul Hoque
**Academic Year:** 2025 / 2026

---

## 🧠 Key Features

* OCR-based handwritten character recognition
* CNN-based deep learning classification
* Robust image preprocessing and segmentation
* Data augmentation for improved generalization
* Majority voting for image-level prediction
* Pre-trained model included for inference

---

## 📂 Repository Structure

```
├── train/              # Training dataset (character images)
├── test/               # Test dataset
├── README.md           # Project documentation
├── labels.json         # Class labels mapping
├── model.h5            # Trained Keras model (HDF5)
├── model.keras         # Trained Keras model (new format)
├── result.csv          # Evaluation results
├── train.py            # Model training script
└── run.py              # Inference / prediction script
```

---

## ⚙️ Preprocessing Pipeline

1. **Grayscale Conversion**
2. **Otsu Thresholding** (Binary Inversion)
3. **Segmentation**

   * Line segmentation (horizontal projection)
   * Word segmentation (morphological dilation)
   * Character segmentation (vertical projection)
4. **Resizing**

   * Characters resized to **64 × 64 pixels**
   * Aspect ratio preserved with white padding
5. **Normalization**

   * Pixel values scaled to **[0, 1]**
6. **Data Augmentation**

   * Rotation
   * Translation
   * Zoom
   * Contrast adjustment
   * Gaussian noise

---

## 🏗️ CNN Architecture

* **5 Convolutional Blocks**

  * Filters: 32 → 64 → 128 → 256 → 256
* Batch Normalization
* Max Pooling
* Dropout (to prevent overfitting)
* Global Average Pooling
* Fully Connected Layer (1024 neurons, ReLU)
* Output Layer: Softmax

**Training Configuration**

* Optimizer: **Adam**
* Loss Function: **Categorical Cross-Entropy**
* Techniques:

  * Early Stopping
  * Learning Rate Reduction
  * Class Weighting

---

## 📊 Model Performance

Evaluation on the test dataset produced the following results:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 90.00% |
| Precision | 90.71% |
| Recall    | 90.00% |
| F1-Score  | 88.86% |

These results indicate **balanced and reliable performance across all classes**.

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn
```

### 2️⃣ Train the Model (Optional)

```bash
python train.py
```

### 3️⃣ Run Inference

```bash
python run.py
```

The script will load the trained model and perform predictions on test images.

---

## 📹 Project Resources

* **Video Presentation:**
  [https://drive.google.com/file/d/1tQTrZeqf5x-FU8ga6uZ3-x1gJ7MKR9ZL/view](https://drive.google.com/file/d/1tQTrZeqf5x-FU8ga6uZ3-x1gJ7MKR9ZL/view)


## 🧩 Conclusion

This project demonstrates an effective integration of **classical image processing techniques** with **deep learning-based CNN models** for handwritten text recognition.
The system achieves strong generalization performance and can be **scaled to larger datasets or more complex OCR tasks** in future work.

---

## 📌 License

This project is developed for **academic purposes** under Albukhary International University.


