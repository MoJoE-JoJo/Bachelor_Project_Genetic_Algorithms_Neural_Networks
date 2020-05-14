from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import time
from src.Enums.OptimizerEnum import Optimizer
from src.Enums.LossEnum import Loss


class SimpleNet:

    def __init__(self, input_shape, output_shape, dataset, dataset_percentage, scaling):
        convStrides = 1  # stride 1 allows us to leave all spatial down-sampling to the POOL layers
        poolStrides = 2

        conv_dropout = 0.2

        convKernelSize = 3
        convKernelSize1 = 1
        poolKernelSize = 2

        filterSizeS = 64
        filterSizeL = 128

        bn_decay = 0.95
        weight_decay = 0.00000005  # it is very small, this is 50 nano, it converges faster the smaller it is, and thus gets stuck later

        initial_bias_constant = 0.1

        (x_train, y_train), (x_test, y_test) = dataset
        x_train, x_test = x_train / scaling, x_test / scaling
        train_slice = (int)(x_train.shape[0] * dataset_percentage)
        test_slice = (int)(x_test.shape[0] * dataset_percentage)
        (x_train, y_train), (x_test, y_test) = (x_train[:train_slice],
                                                y_train[:train_slice]), \
                                               (x_test[:test_slice],
                                                y_test[:test_slice])

        self.x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
        self.y_train = y_train
        self.x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])
        self.y_test = y_test

        self.history = []

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=filterSizeS, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides),
                                      data_format='channels_last', padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides),
                                      data_format='channels_last', padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides),
                                      data_format='channels_last', padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides),
                                      data_format='channels_last', padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides),
                                      data_format='channels_last', padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides),
                                      data_format='channels_last', padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize1, convKernelSize1),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize1, convKernelSize1),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Conv2D(filters=filterSizeL, kernel_size=(convKernelSize, convKernelSize),
                                   strides=(convStrides, convStrides), padding="same",
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer=tf.keras.initializers.Constant(value=initial_bias_constant),
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape,
                                   data_format='channels_last'),
            tf.keras.layers.MaxPool2D(pool_size=(poolKernelSize, poolKernelSize), strides=(poolStrides, poolStrides),
                                      data_format='channels_last', padding="same"),
            tf.keras.layers.Dropout(rate=conv_dropout),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_shape, activation='softmax')

        ])

    def start(self, notify):
        self.history = []
        self.model.compile(optimizer=Optimizer.Adam.name,
                           loss=Loss.sparse_categorical_crossentropy.name,
                           metrics=['accuracy'])
        epochs_run = 0
        epochs_pr_it = 1
        counter = 1
        iterations = 80
        timer = 0

        tc1 = time.time()
        while counter <= iterations:
            self.model.fit(self.x_train, self.y_train, epochs=epochs_pr_it, verbose=0)
            epochs_run += epochs_pr_it
            hist = self.model.evaluate(self.x_test, self.y_test, verbose=0)

            timer += time.time()-tc1
            tc1 = time.time()

            self.history.append({"epoch": counter, "loss": hist[0], "accuracy": hist[1], "accumulated_time": timer, "params": self.model.count_params()})
            print(self.history[-1])
            counter += 1
            notify()
