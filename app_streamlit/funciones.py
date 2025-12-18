import pickle
import os
import numpy as np 
from PIL import Image
from tensorflow import keras

BASE_DIR = os.path.dirname(__file__)
MODELOS_DIR = os.path.join(BASE_DIR, "../modelos")
img_height, img_width = 64, 64

emociones = ["angry", "happy", "relaxed", "sad"]

def procesar_imagen(imagen_perro):
    img = Image.open(imagen_perro).convert("RGB")
    img_resized = img.resize((img_height, img_width))
    img_3d = np.array(img_resized)
    img_flat = img_3d.flatten()
    return img_3d, img_flat

def rf_model(imagen_perro):
    modelo_rf_path = os.path.join(MODELOS_DIR, "modelo_randomforest.pkl")
    modelo_rf = pickle.load(open(modelo_rf_path, "rb"))
    imagen_perro = imagen_perro.reshape(1, -1)
    probs = modelo_rf.predict_proba(imagen_perro)[0]
    return emociones[np.argmax(probs)], probs

def knn_model(imagen_perro):
    modelo_knn_path = os.path.join(MODELOS_DIR, "modelo_knn.pkl")
    modelo_knn = pickle.load(open(modelo_knn_path, "rb"))
    imagen_perro = imagen_perro.reshape(1, -1)
    probs = modelo_knn.predict_proba(imagen_perro)[0]
    return emociones[np.argmax(probs)], probs

def rn_model(imagen_perro):
    modelo_rn_path = os.path.join(MODELOS_DIR, "modelo_rn.keras")
    modelo_rn = keras.models.load_model(modelo_rn_path)
    input_data = np.expand_dims(imagen_perro, axis=0).astype("float32") / 255.0
    probs = modelo_rn.predict(input_data)[0]
    return emociones[np.argmax(probs)], probs

def conv_rn_model(imagen_perro):
    modelo_conv_rn_path = os.path.join(MODELOS_DIR, "modelo_conv_rn.keras")
    modelo_conv_rn = keras.models.load_model(modelo_conv_rn_path)
    input_data = np.expand_dims(imagen_perro, axis=0).astype("float32") / 255.0
    probs = modelo_conv_rn.predict(input_data)[0]
    return emociones[np.argmax(probs)], probs

def conv_rn_da_model(imagen_perro):
    best_model_path = os.path.join(MODELOS_DIR, "best_model.keras")
    best_model = keras.models.load_model(best_model_path)
    input_data = np.expand_dims(imagen_perro, axis=0).astype("float32") / 255.0
    probs = best_model.predict(input_data)[0]
    return emociones[np.argmax(probs)], probs
