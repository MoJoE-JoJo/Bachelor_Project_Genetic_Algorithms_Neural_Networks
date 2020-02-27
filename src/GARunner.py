from threading import Thread
from multiprocessing import Process

from src.GA.SimpleGA import SimpleGA
from src.GA.LonelyGA import LonelyGA
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer

import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import gc
import sys
import time


# CONSTANTS
# Data shape and running
INPUT_SHAPE = (28, 28)
OUTPUT_SHAPE = 10
SCALING = 255.0
DATASET = datasets.mnist.load_data()
DATASET_PERCENTAGE = 0.1
EPOCHS = 5
MAX_RUNTIME = 60
REPETITIONS = 3
TEST_NAME = "joe"

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
    (x_train, y_train), (x_test, y_test) = DATASET
    train_slice = (int) (x_train.shape[0]*DATASET_PERCENTAGE)
    test_slice = (int) (x_test.shape[0]*DATASET_PERCENTAGE)
    data = (x_train[:train_slice],
            y_train[:train_slice]), \
           (x_test[:test_slice],
            y_test[:test_slice])
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
                   dataset=data,
                   epochs=EPOCHS,
                   matingpool=MATINGPOOL)


gc.enable()
initialize_tf()

t = Thread(target=lonely_ga)
t.daemon = True
t.start()
print("GA started")

def makePlot():
    ys = []
    #for i in range(0, REPETITIONS):
        # ys.append(comp()) # add the data from a single run of an experiment

    #Makes sure that they have the same length, in case some of the repetitions get to make more generations than the rest
    min_length = float("inf")
    for i in ys:
        if len(i)<min_length:
            min_length = len(i)
    ys = [i[:min_length] for i in ys]

    # Create y-axis average values
    y = [0 for n in ys[0]]
    for i in ys:
        n=0
        for j in i:
            y[n] += j
            n+=1

    y = [i/REPETITIONS for i in y]

    # Create yerror values
    yerr = []
    for i in range(0,len(y)):
        y_temp = []
        for j in ys:
            y_temp.append(j[i])
        yerr.append((max(y_temp)-min(y_temp))/2)

    x = [val for val in range(0, len(y))]

    plt.style.use('classic')
    fig = plt.figure()
    subfig = fig.add_subplot(111)
    subfig.set_ylabel('accuracy')
    subfig.set_xlabel("epochs")
    plt.xticks(x)

    subfig.set_title(TEST_NAME)
    subfig.errorbar(x, y, yerr=yerr)

    plt.savefig(fname=("../test/" + TEST_NAME + "/plot.svg"))




tc1 = time.time()
tc2 = 0
time_elapsed = 0

while True:
    if time_elapsed >= MAX_RUNTIME:
        print("exiting")
        makePlot()
        sys.exit()

    tc2 = time.time()
    time_elapsed += tc2-tc1
    tc1 = time.time()

    time.sleep(5)
