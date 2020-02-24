from threading import Thread

from src.GA.SimpleGA import SimpleGA
from src.GA.LonelyGA import LonelyGA
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer

import tensorflow as tf
from tensorflow.keras import datasets
import gc
import sys
import time


# CONSTANTS
# Data shape and running
INPUT_SHAPE = (28, 28)
OUTPUT_SHAPE = 10
SCALING = 255.0
DATASET = datasets.mnist.load_data()
EPOCHS = 5
MAX_RUNTIME = 60

# Hyper parameters
ACTIVATION_FUNCTION = Activation.relu
INITIAL_MAX_NODES = 50
LOSS_FUNCTION = Loss.sparse_categorical_crossentropy
OPTIMIZER = Optimizer.Adam

# GA parameters
POPULATION_SIZE = 10
MATINGPOOL = 10  # Must be between 1 and POPULATION_SIZE
MUTATION_RATE = 0.3  # Must be between 0 and 1


def initialize_tf():
    (x_train, y_train), (x_test, y_test) = DATASET
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Optimizer.Adam.name,
                  loss=Loss.sparse_categorical_crossentropy.name,
                  metrics=['accuracy'])

    model.fit(x_train[:10], y_train[:10], epochs=1, verbose=0)


ga = LonelyGA()


def lonely_ga():
    global ga
    LonelyGA.start(self=ga,
                   input_shape=INPUT_SHAPE,
                   output_shape=OUTPUT_SHAPE,
                   initial_max_nodes=INITIAL_MAX_NODES,
                   activation=ACTIVATION_FUNCTION,
                   optimizer=OPTIMIZER,
                   loss=LOSS_FUNCTION,
                   population_size=POPULATION_SIZE,
                   mutation_rate=MUTATION_RATE,
                   scaling=SCALING,
                   dataset=DATASET,
                   epochs=EPOCHS,
                   matingpool=MATINGPOOL)


gc.enable()
initialize_tf()


t = Thread(target=lonely_ga)
t.start()

tc_switch = True
tc1 = time.perf_counter()
tc2 = 0
time_elapsed = 0

while(True):
    if time_elapsed >= MAX_RUNTIME:
        print("exiting")
        sys.exit()
    if tc_switch:
        tc2 = time.perf_counter()
        time_elapsed += tc2-tc1
        tc_switch = False
    if not tc_switch:
        tc1 = time.perf_counter()
        time_elapsed += tc1-tc2
        tc_switch = True
