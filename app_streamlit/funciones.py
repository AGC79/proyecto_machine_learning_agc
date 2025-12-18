import pickle
import os
import numpy as np 

from PIL import Image
from keras.models import load_model
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras


BASE_DIR = os.path.dirname(__file__)
MODELOS_DIR = os.path.join(BASE_DIR, "../modelos")
img_height, img_width = 64, 64

# Función para procesar imágenes
def procesar_imagen(imagen_perro):

    img = Image.open(imagen_perro).convert("RGB")
    img_resized = img.resize((img_height, img_width))
    img_3d = np.array(img_resized)
    img_flat = img_3d.flatten()

    print("Shape X_3d:", img_3d.shape)
    print("Shape X_flat:", img_flat.shape)

    return img_3d, img_flat

# Función modelo Random Forest
def rf_model(imagen_perro):
    modelo_rf_path = os.path.join(MODELOS_DIR, "modelo_randomforest.pkl")
    modelo_rf = pickle.load(open(modelo_rf_path, "rb"))
    imagen_perro = imagen_perro.reshape(1, -1)
    pred_rf = modelo_rf.predict(imagen_perro)
    return pred_rf

# Función modelo KNN
def knn_model(imagen_perro):
    modelo_knn_path = os.path.join(MODELOS_DIR, "modelo_knn.pkl")
    modelo_knn = pickle.load(open(modelo_knn_path, "rb"))

    pred_rf = modelo_knn.predict(imagen_perro)

# Función modelo Red Neuronal
def rn_model(imagen_perro):
    modelo_rn_path = os.path.join(MODELOS_DIR, "modelo_rn.keras")
    modelo_rn = keras.models.load_model(modelo_rn_path)
    
    pred_rn = modelo_rn.predict(imagen_perro)

# Función modelo Red Convolucional + Red Neuronal
def conv_rn_model(imagen_perro):
    modelo_conv_rn_path = os.path.join(MODELOS_DIR, "modelo_conv_rn.keras")
    modelo_conv_rn = keras.models.load_model(modelo_conv_rn_path)

    pred_conv_rn = modelo_conv_rn.predict(imagen_perro)

# Función modelo Red Convolucional + Red Neuronal + Data Augmentation
def conv_rn_da_model(imagen_perro):
    best_model_path = os.path.join(MODELOS_DIR, "best_model.keras")
    best_model = keras.models.load_model(best_model_path)

    pred_conv_rn_da = best_model.predict(imagen_perro)

