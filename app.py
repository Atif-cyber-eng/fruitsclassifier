import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests # Required for downloading from the URL

# --- Configuration ---
# Google Drive File ID from the shared link: 1ZW2iX4mN6v6v7_IvyBq3VRsAkfPAkoAd
GDRIVE_FILE_ID = "1ZW2iX4mN6v6v7_IvyBq3VRsAkfPAkoAd"
LOCAL_MODEL_PATH = "fruit_cnn_model.keras" 
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Define the class names
CLASS_NAMES = [
    'Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2',
    'Apple Golden 3', 'Apple Granny Smith', 'Apple Red 1', 'Apple Red 2',
    'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
    'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger',
    'Banana Red', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula',
    'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red',
    'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Dates',
    'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2',
    'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White',
    'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kumquats', 'Lemon',
    'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red',
    'Mangostan', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat',
    'Nut Forest', 'Nut Pecan', 'Olive Green', 'Olive Yellow', 'Orange',
    'Papaya', 'Passion Fruit', 'Peach', 'Peach Flat', 'Pear', 'Pear Abate',
    'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Williams', 'Pepino',
    'Pepper Green', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis Fresh',
    'Pineapple', 'Pineapple Extra Sweet', 'Plum', 'Plum 2', 'Plum 3',
    'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed',
    'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry',
    'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo',
    'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red',
    'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut'
]

# --- Model Download Function ---
@st.cache_resource
def download_gdrive_file(file_id, destination):
    """Downloads a file from Google Drive using its file ID."""
    if os.path.exists(destination):
        st.success(f"Model already downloaded to '{destination}'. Skipping download.")
        return

    st.info(f"Downloading model from Google Drive (ID: {file_id}). This may take a moment...")
    
    # Construct the direct download URL (handle large file warning)
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Check for the Google Drive warning page for large files
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Save the file locally
    try:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        st.success(f"Model successfully downloaded to '{destination}'.")
    except Exception as e:
        st.error(f"Failed to save file locally: {e}")
        raise

def get_confirm_token(response):
    """Extracts the confirmation token for large Google Drive files."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# --- Model Loading ---
@st.cache_resource
def load_cnn_model():
    """Downloads the file and loads the Keras model."""
    
    # 1. Ensure the model file is present (download if necessary)
    try:
        download_gdrive_file(GDRIVE_FILE_ID, LOCAL_MODEL_PATH)
    except Exception as e:
        st.error(f"Could not prepare model file: {e}")
        return None
        
    # 2. Load the model from the local path
    try:
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model from local path '{LOCAL_MODEL_PATH}': {e}")
        st.warning("Please verify the Google Drive ID is correct and the file is publicly shared.")
        return None

model = load_cnn_model()

# --- Prediction Function ---
def predict_image(image_file, model):
    """
    Preprocesses the uploaded image and makes a prediction using the model.
    """
    try:
        # Open and convert to RGB
        img_pil = Image.open(image_file).convert('RGB')
        
        # Resize the image
        img_pil = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to numpy array and normalize
        img_array = np.array(img_pil)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # Get result
        predicted_class_index = np.argmax(score)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = 100 * np.max(score)

        return predicted_class_name, confidence, img_pil

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Fruit Classifier", layout="centered")

    st.title("🍎 Fruit Image Classifier")
    st.markdown("Upload an image of a fruit to classify its type using a pre-trained CNN model.")

    if model is None:
        # If model failed to load, stop the app execution
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.subheader("Classifying...")
        
        # Perform prediction
        predicted_class_name, confidence, _ = predict_image(uploaded_file, model)
        
        if predicted_class_name:
            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence:.2f}%")

if __name__ == "__main__":
    main()
