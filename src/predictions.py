import pandas as pd
import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


def obtener_predicciones(modelo, X_test, y_test):
    y_true = np.argmax(y_test, axis=1)
    y_pred_probs = modelo.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return y_true, y_pred, y_pred_probs


def matriz_confusion(y_true_classes, y_pred_classes, clases):
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    fig, ax = plt.subplots(figsize=(8,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(cmap='Blues', values_format='d', ax=ax)

    ax.set_title("Matriz de Confusión")

    plt.savefig("confusion_matrix.png")
    plt.show()
    

def exportar_modelo(modelo):
    modelo.save("best_model.keras")

 