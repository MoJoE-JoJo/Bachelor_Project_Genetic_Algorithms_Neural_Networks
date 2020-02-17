from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from src.Enums.OptimizerEnum import Optimizer
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss

convStrides = 1  # stride 1 allows us to leave all spatial down-sampling to the POOL layers
poolStrides = 2

conv_dropout = 0.2

convKernelSize = 3
convKernelSize1 = 1
poolKernelSize = 2

filterSizeS = 64
filterSizeL = 128

bn_decay = 0.95

img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(filterSizeS, (convKernelSize, convKernelSize), padding="same", input_shape=(28, 28, 1), data_format='channels_last'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Conv2D(filterSizeL, (convKernelSize, convKernelSize), padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(rate=conv_dropout),

    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss = Loss.sparse_categorical_crossentropy
optimizer = Optimizer.Adam

if loss == (Loss.categorical_crossentropy or Loss.mean_squared_error):
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

model.compile(optimizer=optimizer.name,
              loss=loss.name,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=1)

model.evaluate(x_test,  y_test, verbose=1)
