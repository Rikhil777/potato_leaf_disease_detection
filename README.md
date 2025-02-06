# Potato Leaf Disease Detection using CNN & Streamlit

## Project Overview
This project is a deep learning-based web application for detecting potato leaf diseases using a **Convolutional Neural Network (CNN)**. The model classifies potato leaves into three categories:
- **Healthy**
- **Early Blight**
- **Late Blight**

The trained model is integrated into a **Streamlit** web application with an interactive dark-mode UI, image transformations, and prediction features.

---

## Technologies Used
- **Python**
- **TensorFlow/Keras** (for building and training the CNN model)
- **Streamlit** (for the web-based user interface)
- **OpenCV** (for image processing)
- **Matplotlib & Seaborn** (for visualization and evaluation)
- **Scikit-learn** (for model evaluation metrics)

---

## Dataset Structure
The dataset consists of three categories of potato leaf images:
```
Dataset/
│── Train/
│   ├── Potato_Early_blight/
│   ├── Potato_healthy/
│   ├── Potato_Late_blight/
│── Valid/
│   ├── Potato_Early_blight/
│   ├── Potato_healthy/
│   ├── Potato_Late_blight/
│── Test/
│   ├── Potato_Early_blight/
│   ├── Potato_healthy/
│   ├── Potato_Late_blight/
```
---

## Installation & Setup
### **1. Create a Virtual Environment**
```sh
python -m venv env
```
Activate the environment:
- **Windows**: `env\Scripts\activate`
- **Linux/Mac**: `source env/bin/activate`

### **2. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3. Train the Model** *(Only if needed)*
```sh
python train_model.py
```
This will train a CNN model and save it as `models/potato_leaf_cnn.h5`.

### **4. Run the Streamlit App**
```sh
streamlit run app.py
```

---

## Features
✔ **Deep Learning-based Classification** (Healthy, Early Blight, Late Blight)  
✔ **Interactive Streamlit UI** (Dark Mode, File Upload)  
✔ **Data Augmentation** (Rotation, Zoom, Flip, Shift)  
✔ **Model Performance Evaluation** (Confusion Matrix, Classification Report)  
✔ **Real-time Image Processing & Prediction**  

---

## Model Architecture
The model consists of:
- **4 Convolutional Layers** (with ReLU activation & MaxPooling)
- **Fully Connected Dense Layers** (with Dropout for regularization)
- **Softmax Output Layer** (for multi-class classification)
- **Adam Optimizer** (learning rate: 0.0001)
- **Categorical Cross-Entropy Loss**

---

## Results & Evaluation
After training, the model achieved the following performance:
- **Accuracy:** 85% (Validation Set)
- **Precision & Recall:** Improved using class balancing
- **Confusion Matrix & Classification Report:** Displayed in `train_model.py`

---

## Future Improvements
🔹 **Deploy on Cloud (AWS/GCP)**  
🔹 **Increase Dataset Size for Better Accuracy**  
🔹 **Use Transfer Learning (MobileNetV2, ResNet50)**  

---

## License
This project is open-source and available under the **MIT License**.

---

## Author
**K.Rikhil**  
Feel free to contribute or raise issues!

