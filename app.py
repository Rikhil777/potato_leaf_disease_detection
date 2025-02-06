import streamlit as st
import numpy as np
from PIL import Image
from utils import load_trained_model, predict_image


st.set_page_config(
    page_title="Potato Leaf Disease Detection", 
    page_icon="🥔", 
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stTitle {
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
    }
    .stMarkdown {
        text-align: center;
        color: #34495e;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-size: 18px;
        display: block;
        margin: 20px auto;
    }
    .stButton>button:hover {
        background-color: #2ecc71;
    }
    .uploadbox {
        background-color: white;
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .uploaded-image {
        max-width: 400px;
        max-height: 400px;
        margin: 0 auto;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)


def format_disease_name(disease_name):
    
    formatted_name = disease_name.replace('Potato_', '').replace('_', ' ')
    return formatted_name

# Main app
def main():
    st.title("🥔 Potato Leaf Disease Detector")
    st.markdown("Upload an image of a potato leaf to identify its health condition")

   
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "png", "jpeg"], 
        help="Upload a clear, close-up image of a potato leaf"
    )

    if uploaded_file:
        
        st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=False, width=400, caption="Uploaded Leaf")
        st.markdown('</div>', unsafe_allow_html=True)
        
       
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🔬 Analyze Leaf", key="analyze_button"):
                with st.spinner("AI is examining the leaf..."):
                    result = predict_image(uploaded_file, load_trained_model())
                
                
                disease_name, confidence = result
                
                
                formatted_disease = format_disease_name(disease_name)
                confidence_percentage = float(confidence) * 100
                
               
                st.success(f"Prediction: {formatted_disease}")
                st.info(f"Confidence: {confidence_percentage:.2f}%")
                
                
                if 'Healthy' in formatted_disease:
                    st.info("Your potato leaf looks healthy! 🌱")
                elif 'Early Blight' in formatted_disease:
                    st.warning("Early Blight detected. Consider crop management strategies.")
                elif 'Late Blight' in formatted_disease:
                    st.error("Late Blight detected. Immediate action recommended.")

if __name__ == "__main__":
    main()