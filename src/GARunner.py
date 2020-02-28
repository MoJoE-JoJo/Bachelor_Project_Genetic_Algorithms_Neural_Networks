from threading import Thread
from multiprocessing import Process

import os.path
import sys
import gc
import csv
from ast import literal_eval as make_tuple

import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Used to find modules when running from venv
from src.FileWriter import FileWriter
from src.GA.SimpleGA import SimpleGA
from src.GA.LonelyGA import LonelyGA
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer
from src.Enums.DatasetEnum import Dataset

writer = None

def write_to_file(data):
    print("Writing")
    writer.write_to_file(data)


# READ EXPERIMENTS FROM FILE
input_file = open(sys.argv[1], "r")
input_reader = csv.DictReader(input_file)
experiments = [row for row in input_reader]
input_file.close()

# RUN EXPERIMENTS
for exp in experiments:
    folder_name = exp["folder_name"] #TODO Rename to FOLDER_NAME
    iterations = int(exp["iterations"]) #TODO Rename REPETITIONS

    # Create path to output files
    path = 'experiments/' + folder_name + '/'
    # Create folder for output files
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # CONSTANTS
    # Data shape and running
    INPUT_SHAPE = make_tuple(exp["input_shape"])
    OUTPUT_SHAPE = int(exp["output_shape"])
    SCALING = float(exp["scaling"])
    DATASET_PERCENTAGE = 0.1 #TODO read from file
    if Dataset(int(exp["data_set"])) == Dataset.mnist:
        DATA_SET = datasets.mnist.load_data()
    else:
        DATA_SET = datasets.mnist.load_data()
    EPOCHS = int(exp["epochs"])
    MAX_RUNTIME = 60 #TODO read from file
    REPETITIONS = 3 #TODO read from file
    TEST_NAME = "joe" #TODO read from file

    # Hyper parameters
    ACTIVATION_FUNCTION = Activation(int(exp["activation_function"]))
    INITIAL_MAX_NODES = int(exp["initial_max_nodes"])
    LOSS_FUNCTION = Loss(int(exp["loss_function"]))
    OPTIMIZER = Optimizer(int(exp["optimizer"]))

    # GA parameters
    POPULATION_SIZE = int(exp["population_size"])
    MATING_POOL = int(exp["mating_pool"])  # Must be between 0 and POPULATION_SIZE
    MUTATION_RATE = float(exp["mutation_rate"])  # Must be between 0 and 1

    # RUN ITERATIONS
    for i in range(iterations):
        writer = FileWriter(path+str(i)+'-', f'Experiment {i+1} ')
        writer.write_to_file([])

        writer.write_to_file(['Data shape and running'])
        writer.write_to_file(['input_shape ', INPUT_SHAPE])
        writer.write_to_file(['output_shape ', OUTPUT_SHAPE])
        writer.write_to_file(['scaling ', SCALING])
        writer.write_to_file(['epochs ', EPOCHS])
        writer.write_to_file([])

        writer.write_to_file(['Hyper parameters'])
        writer.write_to_file(['activation_function ', ACTIVATION_FUNCTION])
        writer.write_to_file(['initial_max_nodes ', INITIAL_MAX_NODES])
        writer.write_to_file(['loss_function ', LOSS_FUNCTION])
        writer.write_to_file(['optimizer ', OPTIMIZER])
        writer.write_to_file([])

        writer.write_to_file(['GA parameters'])
        writer.write_to_file(['population_size ', POPULATION_SIZE])
        writer.write_to_file(['mating_pool ', MATING_POOL])
        writer.write_to_file(['mutation_rate', MUTATION_RATE])
        writer.write_to_file([])

        writer.write_to_file(['OUTPUT'])

        writer.write_to_file(['generation_no', 'neurons_no', 'accuracy', 'loss'])

        writer.close()

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

def lonely_ga():
    global ga
    (x_train, y_train), (x_test, y_test) = DATASET
    train_slice = (int)(x_train.shape[0] * DATASET_PERCENTAGE)
    test_slice = (int)(x_test.shape[0] * DATASET_PERCENTAGE)
    data = (x_train[:train_slice],
            y_train[:train_slice]), \
           (x_test[:test_slice],
            y_test[:test_slice])

    LonelyGA(notify=write_to_file,
              input_shape=INPUT_SHAPE,
              output_shape=OUTPUT_SHAPE,
              initial_max_nodes=INITIAL_MAX_NODES,
              activation=ACTIVATION_FUNCTION,
              optimizer=OPTIMIZER,
              loss=LOSS_FUNCTION,
              population_size=POPULATION_SIZE,
              mutation_rate=MUTATION_RATE,
              scaling=SCALING,
              dataset=DATA_SET,
              epochs=EPOCHS,
              matingpool=MATING_POOL)

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
