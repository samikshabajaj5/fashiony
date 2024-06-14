import os
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm

IMAGE_DIR = 'images'
EMBEDDINGS_FILE = 'embeddings.pkl'
FILENAMES_FILE = 'filenames.pkl'

def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    return model

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def get_image_filenames(directory):
    filenames = []
    for file in os.listdir(directory):
        filenames.append(os.path.join(directory, file))
    return filenames

def main():
    model = load_model()
    filenames = get_image_filenames(IMAGE_DIR)
    feature_list = []
    for file in tqdm(filenames, desc="Extracting Features"):
        feature_list.append(extract_features(file, model))
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(feature_list, f)
    with open(FILENAMES_FILE, 'wb') as f:
        pickle.dump(filenames, f)
    print(f"Features extracted from {len(filenames)} images and saved successfully.")

if __name__ == "__main__":
    main()
