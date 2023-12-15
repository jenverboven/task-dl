# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
from contextlib import redirect_stdout
from streamlit.logger import get_logger
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# Load your images
image_path1 = "datasets/vehicles/training_set/cars/img_1.jpg"
image1 = Image.open(image_path1)

image_path2 = "datasets/vehicles/training_set/planes/img_128.jpg"
image2 = Image.open(image_path2)

image_path3 = "datasets/vehicles/training_set/bikes/img_251.jpg"
image3 = Image.open(image_path3)

image_path4 = "datasets/vehicles/training_set/buses/img_385.jpg"
image4 = Image.open(image_path4)

image_path5 = "datasets/vehicles/training_set/trains/img_805.jpg"
image5 = Image.open(image_path5)

st.title("Task DL")
st.write("Example images from training set:")

# Display images side by side
st.image([image1, image2, image3, image4, image5], caption=['Car', 'Plane', 'Bike', 'Bus', 'Train'], use_column_width=True)

amount_epochs = st.slider("Select the amount of epochs", min_value=1, max_value=50, value=15, step=1)

steps_epoch = st.slider("Select the amount of steps per epoch", min_value=1, max_value=50, value=20, step=1)

resolution_images = st.slider("Select the resolution of the images", min_value=32, max_value=128, value=128, step=32)

batch_size = st.slider("Select the batch size", min_value=10, max_value=40, value=40, step=10)

if st.button("Train model!"):
  with st.spinner("Training the model..."):
    train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                      rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_val_datagen.flow_from_directory('datasets/vehicles/training_set/',
                                                    subset='training',
                                                    target_size = (resolution_images, resolution_images),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical')

    validation_set = train_val_datagen.flow_from_directory('datasets/vehicles/training_set/',
                                                    subset='validation',
                                                    target_size = (resolution_images, resolution_images),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('datasets/vehicles/test_set/',
                                                target_size = (resolution_images, resolution_images),
                                                batch_size = batch_size,
                                                class_mode = 'categorical')
    
    NUM_CLASSES = 5

    # Create a sequential model with a list of layers
    model = tf.keras.Sequential([
      layers.Conv2D(32, (3, 3), input_shape = (resolution_images, resolution_images, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.2),
      layers.Conv2D(64, (3, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.2),
      layers.Conv2D(128, (3, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(64, activation="relu"),
      layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer = "adam",
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'],
                  run_eagerly=True)

    history = model.fit(training_set,
                    validation_data = validation_set,
                    steps_per_epoch = 1,
                    epochs = amount_epochs
                    )
    
    # Create a figure and a grid of subplots with a single call
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()
    # Show the figure
    st.pyplot(fig)

    test_loss, test_acc = model.evaluate(test_set)
    st.write('Test accuracy:', test_acc)

    predictions = model.predict(test_set)
    true_labels = test_set.classes
    predicted_labels = predictions.argmax(axis=1)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    st.write("Confusion matrix:")
    st.write(conf_matrix)