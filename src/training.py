import pandas as pd
import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def generar_imagenes(X_train_m, y_train_m, X_val_m, y_val_m):

    train_datagen = ImageDataGenerator(
        rescale=1./1,
        rotation_range=10,      
        width_shift_range=0.1,  
        height_shift_range=0.1, 
        zoom_range=0.1,         
        horizontal_flip=True,   
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train_m, y_train_m, batch_size=16)
    val_generator   = val_datagen.flow(X_val_m, y_val_m, batch_size=16)

    return train_generator, val_generator


def crear_capas():
    layers_final = [
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ]

    return layers_final

def crear_modelo_secuencial(layers_f):

    model_conv_neu_final = keras.Sequential(layers_f)

    model_conv_neu_final.summary()

    return model_conv_neu_final


def compilar_modelo(modelo):
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    
def crear_history(modelo, train_gen, val_gen):
    history_conv_neu_final = modelo.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50
    )

    return history_conv_neu_final


def crear_df_history(history_f):

    df_history_conv_neu_final = pd.DataFrame(history_f.history)

    return df_history_conv_neu_final


def grafico_loss(df_hist_f):
    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.5)  # rango vertical

    plt.plot(df_hist_f['loss'], label='train_loss')
    plt.plot(df_hist_f['val_loss'], label='val_loss')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Loss history')

    plt.savefig("loss_history.png")
    plt.show()


def grafico_train(df_hist_f):

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.5) 

    plt.plot(df_hist_f['categorical_accuracy'], label='train_accuracy')
    plt.plot(df_hist_f['val_categorical_accuracy'], label='val_accuracy')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training history')

    plt.savefig("train_history.png")
    plt.show()