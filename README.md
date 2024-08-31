**Colorized Autoencoder for CIFAR-10 Dataset**
This repository contains an implementation of an autoencoder model for colorizing grayscale images using the CIFAR-10 dataset. The autoencoder consists of an encoder and decoder network built with Keras and TensorFlow.

Project Overview
Dataset: CIFAR-10
Model: Autoencoder
Objective: Convert color images to grayscale and then reconstruct the original color images using an autoencoder model.
Requirements
Python 3.x
TensorFlow
Keras
NumPy
OpenCV
Matplotlib

**Installation**

Clone the repository:

bash
git clone https://github.com/yourusername/colorized-autoencoder.git
cd colorized-autoencoder

Install the required packages:
bash
pip install tensorflow keras numpy opencv-python matplotlib

**Usage**
Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K

**Data Preprocessing**
Load CIFAR-10 Dataset:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

Convert RGB Images to Grayscale:
def rgb_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x_train_Gray = np.array([rgb_gray(img) for img in x_train])
x_test_Gray = np.array([rgb_gray(img) for img in x_test])

Normalize and Reshape Data:
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_Gray = x_train_Gray.astype('float32') / 255
x_test_Gray = x_test_Gray.astype('float32') / 255

img_dim = x_train.shape[1]
x_train = x_train.reshape(x_train.shape[0], img_dim, img_dim, 3)
x_test = x_test.reshape(x_test.shape[0], img_dim, img_dim, 3)
x_train_Gray = x_train_Gray.reshape(x_train_Gray.shape[0], img_dim, img_dim, 1)
x_test_Gray = x_test_Gray.reshape(x_test_Gray.shape[0], img_dim, img_dim, 1)

**Model Architecture**
Encoder Model:
inputs = Input(shape=(img_dim, img_dim, 1), name='encoder_input')
x = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(inputs)
x = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), strides=2, activation='relu', padding='same')(x)
x = Flatten()(x)
latent = Dense(256, name='latent_vector')(x)
encoder = Model(inputs, latent, name='encoder_model')

Decoder Model:
latent_inputs = Input(shape=(256,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)
decoder = Model(latent_inputs, outputs, name='decoder_model')

Autoencoder Model:
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

**Training**
Compile and Train the Model:
autoencoder.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, verbose=1, min_lr=0.5e-6)
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_ae_model.keras'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoints = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True)

callbacks = [lr_reducer, checkpoints]

autoencoder.fit(x_train_Gray, x_train, validation_data=(x_test_Gray, x_test), epochs=30, batch_size=32, callbacks=callbacks)

**Results**
The trained model will be saved in the saved_models directory with the filename colorized_ae_model.keras. You can use this model to colorize grayscale images by loading it and using it for inference.
