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
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
                                featurewise_center=True,
                                featurewise_std_normalization=True,
                                # validation_split=0.0,
                                # preprocessing_function= tf.keras.applications.resnet_v2.preprocess_input
)

height = width = 4*256
def read_pil_image(img_path, height, width):
        with open(img_path, 'rb') as f:
            return np.array(Image.open(f).convert('RGB').resize((width, height)))

def load_all_images(dataset_path, height, width, img_ext='tif'):
    return np.array([read_pil_image(str(p), height, width) for p in 
                                    glob.glob(dataset_path+"*."+img_ext)[::200]]) 

train_datagen.fit(load_all_images( "/gws/nopw/j04/aopp/manshausen/bing_dl/train/patches/*/", height, width))

test_image_generator = train_datagen.flow_from_directory(
    '/gws/nopw/j04/aopp/manshausen/bing_dl/deploy_cl/',
    batch_size=1,
    # color_mode="grayscale",
    target_size=(height, width),
    class_mode=None,
    seed=42,
    shuffle=False,
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

model.load_weights('/gws/nopw/j04/aopp/manshausen/saved_bing_models/4_256res.h5')

pred = model.predict(test_image_generator, verbose = 1, workers=4,
    use_multiprocessing=True)


# Get filenames (set shuffle=false in generator is important)
filenames=test_image_generator.filenames
# Data frame
results = pd.DataFrame({"file":filenames,"coal":pred[:,0], "steel":pred[:,1], "other": pred[:,2]})

results.to_csv('pred_results.csv')