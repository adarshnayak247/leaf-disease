import os
import json
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Google Drive file ID from environment variable
gdrive_model_id = os.getenv('GDRIVE_MODEL_ID')
model_url = f'https://drive.google.com/uc?id={gdrive_model_id}'

# Set up paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'plant_disease_prediction_model.h5')
class_indices_path = os.path.join(working_dir, 'class_indices.json')

# Download the model file if not already present
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names
with open(class_indices_path) as f:
    class_indices = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    # Load the image
    img = Image.open(image)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Ensure the image has 3 channels
    if img_array.ndim == 2:  # grayscale image
        img_array = np.stack([img_array] * 3, axis=-1)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown class")
    return predicted_class_name

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        # Display uploaded image with resizing for preview
        image = Image.open(uploaded_image)
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption='Uploaded Image')

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
