import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

UPLOAD_FOLDER = 'user_uploads'
EMBEDDINGS_FILE = 'user_embeddings.pkl'
FILENAMES_FILE = 'user_filenames.pkl'

@st.cache(allow_output_mutation=True)
def load_user_data():
    feature_list = np.array(pickle.load(open(EMBEDDINGS_FILE, 'rb')))
    filenames = pickle.load(open(FILENAMES_FILE, 'rb'))
    return feature_list, filenames

user_feature_list, user_filenames = load_user_data()

@st.cache(allow_output_mutation=True)
def initialize_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    return model

user_model = initialize_model()

st.title('Fashion Recommendation Engine')

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_features_from_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

def find_similar_images(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)
        image_features = extract_features_from_image(file_path, user_model)
        similar_image_indices = find_similar_images(image_features, user_feature_list)
        st.subheader("Recommended Images:")
        columns = st.columns(5)
        for i, col in enumerate(columns):
            with col:
                st.image(user_filenames[similar_image_indices[0][i]])
    else:
        st.error("Error uploading file. Please try again.")
