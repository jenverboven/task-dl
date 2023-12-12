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

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Task DL",
    )

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', name='conv_layer_1'))

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', name='conv_layer_2'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(activation="relu", units=128))

    model.add(Dense(activation="sigmoid", units=1))

    # compiling the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    print(model.summary())
    
    st.write("Model Summary:")
    # with st.echo():
    #     with st.spinner("Calculating model summary..."):
    #         with redirect_stdout(st):
    #             model.summary()


if __name__ == "__main__":
    run()
