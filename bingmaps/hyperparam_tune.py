import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle as pickle
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
import glob
import random
import datetime
from pathlib import Path
from PIL import Image


def read_pil_image(img_path, height, width):
        with open(img_path, 'rb') as f:
            return np.array(Image.open(f).convert('RGB').resize((width, height)))

def load_all_images(dataset_path, height, width, img_ext='tif'):
    return np.array([read_pil_image(str(p), height, width) for p in 
                                    glob.glob(dataset_path+"*."+img_ext)[::200]]) 


directory = "/gws/nopw/j04/aopp/manshausen/bing_dl/patches/"
print(tf.config.list_physical_devices())

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    # rescale = 1./255,
                                shear_range=0.2,
                                zoom_range=[0.8,1.],
                                horizontal_flip=True,
                                vertical_flip=True,
                                featurewise_center=True,
                                featurewise_std_normalization=True,
                                validation_split=0.20,
                                preprocessing_function= tf.keras.applications.resnet_v2.preprocess_input
)
             
height = width = 256 
train_datagen.fit(load_all_images( "/gws/nopw/j04/aopp/manshausen/bing_dl/patches/*/", height, width))

train_image_generator = train_datagen.flow_from_directory(
    directory,
    batch_size=10,
    # color_mode="grayscale",
    # target_size=(400, 400),
    class_mode='categorical',
    subset='training',
    seed=42,
    shuffle=True,
)
val_image_generator = train_datagen.flow_from_directory(
    directory,
    batch_size=10,
    # color_mode="grayscale",
    # target_size=(400, 400),
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=True,
)

import keras_tuner
from keras_tuner.applications import HyperResNet

hypermodel = HyperResNet(input_shape=(256, 256, 3), classes=3)

tuner = keras_tuner.RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=2,
    overwrite=True,
    directory="hyper",
    project_name="built_in_hypermodel",
)

tuner.search(
    train_image_generator, epochs=1, validation_data=val_image_generator
)