# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:20:20 2021

@author: gianm
"""

#%% Import Library

#Untuk Visualisasi dll
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import string
import random

#Library Tensorflow (CPU)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#%% Download data

import pathlib
#dataset_url = "https://doc-0o-9g-docs.googleusercontent.com/docs/securesc/58l87q89spfospdd7kv85jr8ck01c9dg/0tt116vv95mcmelgrn80oebfandnrmdh/1618201725000/07848392253171536891/07848392253171536891/1nshMe0x3Sk5XL5TZYqfZnGsq90hOx22Z?e=download&authuser=0"
#data_dir = tf.keras.utils.get_file('vehicle_photos', origin=dataset_url, untar=True)
#data_dir = pathlib.Path(data_dir)
data_dir = pathlib.Path(r'C:\Users\gianm\.keras\datasets\vehicle_photos2')

#%% Buat dataset

batch_size = 32
img_height = 180
img_width = 180

#bagi data menjadi 'training' dan 'validation'
#80% untuk training dan 20% untuk validation

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

#%% Optimize performance

AUTOTUNE = tf.data.AUTOTUNE

#keeps the images in memory after they're loaded off disk during the first epoch
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

#overlaps data preprocessing and model execution while training
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#%% Data Augmentation

data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

#%% Standardize the data

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

#%% Layers

num_classes = 3

#buat model dengan 20% dropout
model = Sequential([
  data_augmentation,
  normalization_layer,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), #20% dropout
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#%% Compile and train model (2)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#%% Random String generator

def id_generator(size=12, chars=string.ascii_letters + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

#%% Display image

def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

#%% Save model

model.save(r'C:\Users\gianm\.keras\saved_models\{}'.format('img_2'))

#%% Load model

model = keras.models.load_model(r'C:\Users\gianm\.keras\saved_models\{}'.format('img_2'))

#%% Predict new data

image_url = "https://awsimages.detik.net.id/community/media/visual/2020/02/05/3b60dc19-9034-4822-9614-7527af1015a8.jpeg?w=700&q=90"
image_path = tf.keras.utils.get_file('img-{}'.format(id_generator()), origin=image_url)

#image_path = r'E:\Kuliah\Proyek Data Science\PROYEK\frame647.jpg'

img = keras.preprocessing.image.load_img(
    image_path, target_size=(img_height, img_width)
)

#display_image(img)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "\nClass: [{}]\nConfidence: {:.2f}%"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

