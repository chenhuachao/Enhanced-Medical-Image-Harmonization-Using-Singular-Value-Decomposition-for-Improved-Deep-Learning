import os
import glob
import shutil
import json
import keras
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Defining the working directories
work_dir = '/home/sun/stream_infer/task/Molybdenum/RSNA_train_data/' 
cancer_path = os.path.join(work_dir, 'cancer')
nocancer_path = os.path.join(work_dir, 'nocancer')



IMG_SIZE = 1024  # images of size 1024x1024 1024会发生一些问题，512的GPU和CPU都可以用
size = (IMG_SIZE, IMG_SIZE)
n_CLASS = 2  # As you have two classes: cancer and nocancer



# Setting up ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # using 20% of the data for validation
)

# Preparing training and validation sets
train_set = datagen.flow_from_directory(
    directory=work_dir,
    target_size=size,
    class_mode='categorical',
    batch_size=2,# batch_size是8，如果是1024的话就是4.
    subset='training'  # set as training data
)

val_set = datagen.flow_from_directory(
    directory=work_dir,
    target_size=size,
    class_mode='categorical',
    batch_size=2,# batch_size是8，如果是1024的话就是4.
    subset='validation'  # set as validation data
)


def create_model():
    model = Sequential()
    model.add(tf.keras.applications.EfficientNetB3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation='softmax'))  # Adjusting the final layer to have two outputs
    
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer=Adam(learning_rate=2e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



EPOCHS = 50  # You can adjust the number of epochs depending on your requirement

callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True),
    ModelCheckpoint("/home/sun/stream_infer/task/Molybdenum/model/epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5", save_best_only=True, save_freq='epoch', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, mode='min')
]


history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=EPOCHS,
    callbacks=callbacks
)



plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('/home/sun/stream_infer/task/Molybdenum/training_plot.png')
plt.show()
