import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Load your pre-trained model
try:
    model = load_model('BrainTumorDectection.h5')  # Make sure this path is correct
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()

# Title of the Streamlit app
st.title("Tumor Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Final Prediction...")

        # Convert image to numpy array
        image_np = np.array(image)

        # Preprocess the image for the modela
        def preprocess_image(image):
            # Resize image to match model input size (128x128)
            image = cv2.resize(image, (128, 128))
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Normalize image
            image = image.astype(np.float32) / 255.0
            # Add batch dimension (1, 128, 128, 3)
            image = np.expand_dims(image, axis=0)
            return image
        
        # Predict using the model
        preprocessed_image = preprocess_image(image_np)
        prediction = model.predict(preprocessed_image)
        
        # Ensure prediction is in the correct format
        if prediction.shape[-1] == 1:
            tumor_probability = prediction[0][0]
        else:
            tumor_probability = prediction[0][1]  # Assuming binary classification

        
        if tumor_probability >0.5:
            st.write(f"Prediction: Tumor Detected with a probability of {tumor_probability:.2f}")
        else:  
            st.write(f"Prediction: No Tumor Detected with a probability of {tumor_probability:.2f}")
       
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")


