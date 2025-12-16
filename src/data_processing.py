# Importaciones
import pandas as pd
import numpy as np
import os

from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def procesar_rutas_fotos(base_ruta, fotos_ruta):

    lista_rutas = []
    target = []

    for ruta in fotos_ruta:
        ruta_total = os.path.join(base_ruta, ruta)

        for foto in os.listdir(ruta_total):

            if foto.lower().endswith((".jpeg", ".JPEG", ".png", ".bmp", ".tiff", ".jpg")):
                ruta_original = os.path.join(ruta_total, foto)
                nombre_sin_ext = os.path.splitext(foto)[0]
                ruta_nueva = os.path.join(ruta_total, nombre_sin_ext + ".jpg")

                if not ruta_original.lower().endswith(".jpg"):
                    img = Image.open(ruta_original).convert("RGB")
                    img.save(ruta_nueva, "JPEG")
                    os.remove(ruta_original)
                    print(f"Convertido: {ruta_original} → {ruta_nueva}")
                else:
                    ruta_nueva = ruta_original  

                lista_rutas.append(ruta_nueva)
                target.append(os.path.basename(ruta_total))

    return lista_rutas, target


def crear_dataframe_rutas_target(rutas_fotos, target):
    df = pd.DataFrame({"path_fotos": rutas_fotos, "target": target})
    return df


def crear_lista_fotos(df, img_height, img_width):

    lista_fotos_3d = []

    for foto in df["path_fotos"]:
        if os.path.exists(foto):
            img = Image.open(foto).convert("RGB")
            img_resized = img.resize((img_height, img_width))
            img_array = np.array(img_resized)
            lista_fotos_3d.append(img_array)
        else:
            print(f"Imagen no encontrada: {foto}")

    return lista_fotos_3d


def crear_x_y(lista_fotos_3d, df):

    X_rgb = np.array(lista_fotos_3d)
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(df[["target"]]).toarray()

    return X_rgb, y_encoded


def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.1, 
        random_state=11)

    print("Shape de X_train: ", X_train.shape)
    print("Shape de y_train: ", y_train.shape)
    print("Shape de X_test: ", X_test.shape)
    print("Shape de y_test: ", y_test.shape)

    return X_train, X_test, y_train, y_test


def normalizar_train_test(X_tr, X_ts, y_tr, y_ts):
    X_train_nor = X_tr.astype("float32")/255
    X_test_nor = X_ts.astype("float32")/255

    y_train_nor = y_tr.astype("float32")
    y_test_nor = y_ts.astype("float32")

    return X_train_nor, X_test_nor, y_train_nor, y_test_nor


def split_train_val(X_tr_nor, y_tr_nor, X_ts_nor):
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_nor, y_tr_nor,
        test_size=0.1,         
        random_state=11
    )

    print("Tamaño X_train:", X_train.shape)
    print("Tamaño X_val:", X_val.shape)
    print("Tamaño X_test:", X_ts_nor.shape)

    return X_train, X_val, y_train, y_val

def desordenar_train(X_train_m, y_train_m):

    X_train, y_train = shuffle(X_train_m, y_train_m, random_state=11)
    
    return X_train, y_train
