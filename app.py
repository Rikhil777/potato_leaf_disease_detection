import streamlit as st
from PIL import Image
from utils import load_trained_model, predict_image

# Load the trained model
model = load_trained_model()

# Streamlit app
st.set_page_config(page_title="Potato Leaf Disease Detection", layout="centered")

st.title("🌿 Potato Leaf Disease Detection")
st.markdown("Upload an image of a potato leaf to identify its condition (Early Blight, Healthy, or Late Blight).")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated here
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            result = predict_image(uploaded_file, model)
        st.success(f"Prediction: {result}")
