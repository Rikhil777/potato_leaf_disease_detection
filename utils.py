import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

MODEL_PATH = "models/potato_leaf_cnn.keras"  # Changed to .keras
CLASS_NAMES = ["Potato_Early_blight", "Potato_healthy", "Potato_Late_blight"]

def load_trained_model():
    return load_model(MODEL_PATH)

def predict_image(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    
    return predicted_class, confidence

def visualize_prediction(image_path, model):
    img = load_img(image_path)
    predicted_class, confidence = predict_image(image_path, model)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.tight_layout()
    plt.show()