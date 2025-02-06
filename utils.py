import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


MODEL_PATH = "models/potato_leaf_cnn.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)


CLASS_NAMES = list(model.class_names) if hasattr(model, 'class_names') else [
    "Potato_Early_blight", "Potato_healthy", "Potato_Late_blight"
]

def predict_image(image_path):
    """Load image, preprocess it, and make a prediction."""
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = round(float(prediction[0][predicted_class_index]) * 100, 2)
        
        return predicted_class, confidence
    except Exception as e:
        return f"Error processing image: {str(e)}", 0

def visualize_prediction(image_path):
    """Display the image with the prediction and confidence score."""
    try:
        img = load_img(image_path)
        predicted_class, confidence = predict_image(image_path)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")
