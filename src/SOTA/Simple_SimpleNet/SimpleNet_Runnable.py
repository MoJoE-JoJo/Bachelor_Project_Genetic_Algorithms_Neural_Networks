from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import time
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from src.Enums.OptimizerEnum import Optimizer
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss

class SimpleNet:
    convStrides = 1  # stride 1 allows us to leave all spatial down-sampling to the POOL layers
    poolStrides = 2

    conv_dropout = 0.2

    convKernelSize = 3
    convKernelSize1 = 1
    poolKernelSize = 2

    filterSizeS = 64
    filterSizeL = 128

    bn_decay = 0.95
    learning_rate = 0.001  #values between 0.001 and 0.00001
    weight_decay = 0.00000005 # it is very small, this is 50 nano, it converges faster the smaller it is, and thus gets stuck later

    initial_bias_constant = 0.1

    img_rows = 28
    img_cols = 28

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    train_slice = (int)(x_train.shape[0] * 0.1)
    test_slice = (int)(x_test.shape[0] * 0.1)
    (x_train, y_train), (x_test, y_test) = (x_train[:train_slice],
                                            y_train[:train_slice]), \
                                           (x_test[:test_slice],
                                            y_test[:test_slice])

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

    history = []

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters=filterSizeS, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize1, convKernelSize1), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize1, convKernelSize1), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize), strides=(convStrides, convStrides), padding="same", kernel_initializer='glorot_uniform', bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides), data_format='channels_last', padding="same"),
        tf.keras.layers.Dropout(rate=conv_dropout),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')

    ])

    def start(self, notify):
        self.history = []
        self.model.compile(optimizer=Optimizer.Adam.name,
                           loss=Loss.sparse_categorical_crossentropy.name,
                           metrics=['accuracy'])
        epochs_run = 0
        epochs_pr_it = 1
        counter = 1
        iterations = 5
        timer = 0

        tc1 = time.time()
        while counter <= iterations:
            self.model.fit(self.x_train, self.y_train, epochs=epochs_pr_it, verbose=0)
            epochs_run += epochs_pr_it
            hist = self.model.evaluate(self.x_test, self.y_test, verbose=0)

            timer += time.time()-tc1
            tc1 = time.time()

            self.history.append({"epoch": counter, "loss": hist[0], "accuracy": hist[1], "accumulated_time": timer})
            counter += 1
            notify()


