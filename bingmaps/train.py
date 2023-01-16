import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle as pickle
from tensorflow import keras
import tensorflow.keras.layers as layers
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
                                rescale = 1./255,
                                shear_range=0.2,
                                zoom_range=[0.8,1.],
                                horizontal_flip=True,
                                vertical_flip=True,
                                featurewise_center=True,
                                featurewise_std_normalization=True,
                                validation_split=0.20,
                                # preprocessing_function= tf.keras.applications.resnet_v2.preprocess_input
)
             
height = width = 1400
train_datagen.fit(load_all_images( "/gws/nopw/j04/aopp/manshausen/bing_dl/patches/*/", height, width))

train_image_generator = train_datagen.flow_from_directory(
    directory,
    batch_size=10,
    # color_mode="grayscale",
    target_size=(height, width),
    class_mode='categorical',
    subset='training',
    seed=42,
    shuffle=True,
)
val_image_generator = train_datagen.flow_from_directory(
    directory,
    batch_size=10,
    # color_mode="grayscale",
    target_size=(height, width),
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=True,
)

model = tf.keras.Sequential([
    tf.keras.applications.resnet_v2.ResNet50V2(
    #
    # resize_rescale_augment,
    # layers.Conv2D(3, 5, padding='same', activation='tanh'), # this seems like a bit of a brute force approach to handing a 3 channel image to resnet, 
                                                            # maybe try changing the source so it accepts 4 channels?
    
        include_top=True,
        weights=None, # if I don't use the pre trained weights from image net, does it matter that i don't use the preprocessing step which reorders RGB to BGR and zero-centers wrt imagenet?
        input_shape=(height, width, 3),
        # pooling=max ,
        classes=3,),
    # layers.Flatten(), # does this make sense? or is there another way to get down to just three output dimensions?
    # layers.Dense(3)
])

log_dir = '/home/users/pete_nut/sentinel_industry/bingmaps/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '120,140')

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.02)
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50,verbose=1, restore_best_weights=True)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.build(input_shape=(height, width, 3)) #????
model.summary()

# now with normalisation
model.fit(
        train_image_generator,
        workers=4, 
        use_multiprocessing=True,
        steps_per_epoch=200,
        epochs=600,
        validation_data=val_image_generator,
        validation_steps=100,
        callbacks=[tensorboard_callback, es_callback, lr_callback]
)
model.save_weights('/gws/nopw/j04/aopp/manshausen/saved_bing_models/1400res.h5')  # always save your weights after training or during training