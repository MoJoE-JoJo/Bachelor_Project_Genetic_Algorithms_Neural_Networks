from threading import Thread
from multiprocessing import Process

import os.path
import sys
import gc
import csv
import re
from ast import literal_eval as make_tuple

import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import matplotlib
import time



sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Used to find modules when running from venv
from src.FileWriter import FileWriter
from src.GA.SimpleGA import SimpleGA
from src.GA.LonelyGA import LonelyGA
from src.GA.LonelyErrorGA import LonelyErrorGA
from src.GA.LonelyLossGA import LonelyLossGA
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer
from src.Enums.DatasetEnum import Dataset
from src.SOTA.Simple_SimpleNet.SimpleNet_Runnable import SimpleNet

writer = None
ga = None

def write_to_file(data):
    print("Writing")
    writer.write_to_file(data)


def notify():
    global ga
    if ALGORITHM == "SimpleNet":
        writer.write_to_file([ga.history[-1]["epoch"],
                              ga.history[-1]["accumulated_time"],
                              ga.history[-1]["accuracy"],
                              ga.history[-1]["loss"]])
    elif ALGORITHM in ["Lonely_GA_Layers", "Lonely_GA_Layers_All",
                       "Lonely_GA_Layers_LS_3", "Lonely_GA_Layers_LS_4", "Lonely_GA_Layers_LS_5",
                       "Lonely_GA_Layers_All_LS_3", "Lonely_GA_Layers_All_LS_4", "Lonely_GA_Layers_All_LS_5",
                       "Lonely_GA_Layers_Copy_LS_3", "Lonely_GA_Layers_Copy_LS_4", "Lonely_GA_Layers_Copy_LS_5"]:
        writer.write_to_file([ga.history[-1]["generation"],
                              ga.history[-1]["params"],
                              ga.history[-1]["layers"],
                              ga.history[-1]["accuracy"],
                              ga.history[-1]["loss"]])
    else:
        writer.write_to_file([ga.history[-1]["generation"],
                              ga.history[-1]["params"],
                              ga.history[-1]["nodes"],
                              ga.history[-1]["accuracy"],
                              ga.history[-1]["loss"]])


def lonely_ga():
    global experiments, writer, ga,\
           FOLDER_NAME, REPETITIONS, \
           INPUT_SHAPE, OUTPUT_SHAPE, SCALING, DATASET_PERCENTAGE, DATASET, EPOCHS, MAX_RUNTIME, \
           ACTIVATION_FUNCTION, INITIAL_MAX_NODES, LOSS_FUNCTION, OPTIMIZER, \
           POPULATION_SIZE, MATING_POOL, MUTATION_RATE

    (x_train, y_train), (x_test, y_test) = DATASET
    train_slice = (int)(x_train.shape[0] * DATASET_PERCENTAGE)
    test_slice = (int)(x_test.shape[0] * DATASET_PERCENTAGE)
    data = (x_train[:train_slice],
            y_train[:train_slice]), \
           (x_test[:test_slice],
            y_test[:test_slice])

    ga.start(input_shape=INPUT_SHAPE,
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
             matingpool=MATING_POOL,
             notify=notify)


def simple_net():
    global experiments, writer, ga,\
           FOLDER_NAME, REPETITIONS, \
           INPUT_SHAPE, OUTPUT_SHAPE, SCALING, DATASET_PERCENTAGE, DATASET, EPOCHS, MAX_RUNTIME, \
           ACTIVATION_FUNCTION, INITIAL_MAX_NODES, LOSS_FUNCTION, OPTIMIZER, \
           POPULATION_SIZE, MATING_POOL, MUTATION_RATE

    ga.start(notify=notify)


def initialize_tf():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Optimizer.Adam.name,
                  loss=Loss.sparse_categorical_crossentropy.name,
                  metrics=['accuracy'])

    model.fit(x_train[:10], y_train[:10], epochs=1, verbose=0)


def make_plot(data):
    global experiments, writer,\
           FOLDER_NAME, REPETITIONS, \
           INPUT_SHAPE, OUTPUT_SHAPE, SCALING, DATASET_PERCENTAGE, DATASET, EPOCHS, MAX_RUNTIME, \
           ACTIVATION_FUNCTION, INITIAL_MAX_NODES, LOSS_FUNCTION, OPTIMIZER, \
           POPULATION_SIZE, MATING_POOL, MUTATION_RATE

    ys = []
    ys = data #Need to only select the relevant stuff

    #Makes sure that they have the same length, in case some of the repetitions get to make more generations than the rest
    min_length = float("inf")
    for i in ys:
        if len(i) < min_length:
            min_length = len(i)
    ys = [i[:min_length] for i in ys]

    # Create y-axis average values
    y_acc = [0 for n in ys[0]]
    y_los = [0 for n in ys[0]]
    y_params = [0 for n in ys[0]]
    for i in ys:
        n=0
        for j in i:
            y_acc[n] += j["accuracy"]
            y_los[n] += j["loss"]
            y_params[n] += j["params"]
            n+=1

    y_acc = [i/REPETITIONS for i in y_acc]
    y_los = [i/REPETITIONS for i in y_los]
    y_params = [i / REPETITIONS for i in y_params]

    # Create yerror values
    yerr_acc = [[] for i in range(0,2)]
    yerr_los = [[] for i in range(0,2)]
    yerr_params = [[] for i in range(0, 2)]
    for i in range(0,len(y_acc)):
        y_temp_acc = []
        y_temp_los = []
        y_temp_params = []
        for j in ys:
            y_temp_acc.append(j[i]['accuracy'])
            y_temp_los.append(j[i]['loss'])
            y_temp_params.append(j[i]['params'])

        yerr_acc[0].append(y_acc[i]-min(y_temp_acc))
        yerr_acc[1].append(max(y_temp_acc)-y_acc[i])

        yerr_los[0].append(y_los[i]-min(y_temp_los))
        yerr_los[1].append(max(y_temp_los)-y_los[i])

        yerr_params[0].append(y_params[i] - min(y_temp_params))
        yerr_params[1].append(max(y_temp_params) - y_params[i])

    plt.style.use('classic')

    fig, (ax_acc, ax_los, ax_params) = plt.subplots(3, figsize=(8, 10))
    fig.suptitle(FOLDER_NAME)

    if ALGORITHM == "SimpleNet":
        x = [val for val in range(1, len(y_acc)+1)]
        ax_acc.set(xlabel='', ylabel='accuracy')
        ax_los.set(xlabel='epoch', ylabel='loss')
        ax_params.set(xlabel='generation', ylabel='parameters')
    else:
        x = [val for val in range(0, len(y_acc))]
        ax_acc.set(xlabel='', ylabel='accuracy')
        ax_los.set(xlabel='', ylabel='loss')
        ax_params.set(xlabel='generation', ylabel='parameters')

    acc_bl_val = 0.984  # Found as the averages of epoch 25-80 of SimpleNet baseline
    los_bl_val = 0.0896  # Found as the averages of epoch 25-80 of SimpleNet baseline
    params_bl_val = 1442954  # Found by using param_count() on our implementation of SimpleNet

    x_bl = list(x)
    x_bl.insert(0, (min(x)-1))
    x_bl.append(max(x) + 1)
    y_bl_acc = [acc_bl_val for val in x_bl]
    y_bl_los = [los_bl_val for val in x_bl]
    y_bl_params = [params_bl_val for val in x_bl]
    yerr_bl = [0 for val in y_bl_acc]

    ax_acc.set_xticks(x)
    ax_acc.set_xlim(min(x)-0.1, max(x)+0.1)
    ax_acc.locator_params(axis='x', nbins=10)

    ax_los.set_xticks(x)
    ax_los.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax_los.locator_params(axis='x', nbins=10)

    ax_params.set_xticks(x)
    ax_params.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax_params.locator_params(axis='x', nbins=10)

    ax_acc.errorbar(x, y_acc, yerr=yerr_acc)
    ax_los.errorbar(x, y_los, yerr=yerr_los)
    ax_params.errorbar(x, y_params, yerr=yerr_params)

    ax_params.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    if ALGORITHM != 'SimpleNet':
        ax_acc.errorbar(x_bl, y_bl_acc, yerr=yerr_bl)
        ax_los.errorbar(x_bl, y_bl_los, yerr=yerr_bl)
        ax_params.errorbar(x_bl, y_bl_params, yerr=yerr_bl)

    plt.savefig(fname=(path + "plot.svg"))


def choose_GA():
    if ALGORITHM == "SimpleNet":
        return SimpleNet()
    # Lonely_GA variations
    elif ALGORITHM in ["Lonely_GA", "Lonely_GA_Validation", "Lonely_GA_PS_01", "Lonely_GA_PS_033"]:
        return LonelyGA(ALGORITHM)
    # Lonely_Loss_GA variations
    elif ALGORITHM in ["Lonely_Loss_GA", "Lonely_Loss_GA_PS_01", "Lonely_Loss_GA_PS_033", "Lonely_GA_Layers",
                       "Lonely_Loss_GA_LS_PS_2_033", "Lonely_Loss_GA_LS_PS_3_033", "Lonely_Loss_GA_LS_PS_4_033",
                       "Lonely_Loss_GA_LS_PS_5_033", "Lonely_Loss_GA_Exp_2_033", "Lonely_Loss_GA_Exp_3_033",
                       "Lonely_Loss_GA_Exp_4_033", "Lonely_Loss_GA_Exp_5_033", "Lonely_GA_Layers_All",
                       "Lonely_GA_Layers_LS_3", "Lonely_GA_Layers_LS_4", "Lonely_GA_Layers_LS_5",
                       "Lonely_GA_Layers_All_LS_3", "Lonely_GA_Layers_All_LS_4", "Lonely_GA_Layers_All_LS_5",
                       "Lonely_GA_Layers_Copy_LS_3", "Lonely_GA_Layers_Copy_LS_4", "Lonely_GA_Layers_Copy_LS_5"]:
        return LonelyLossGA(ALGORITHM)
    # Lonely_Error_GA variations
    elif ALGORITHM in ["Lonely_Error_GA", "Lonely_Error_GA_PS_01", "Lonely_Error_GA_PS_033"]:
        return LonelyErrorGA(ALGORITHM)


gc.enable()
initialize_tf()


# READ EXPERIMENTS FROM FILE
regex = re.compile('#[\\w\\W]*')

input_file = open(sys.argv[1], "r")
input_reader = csv.DictReader(input_file)
filtered = [row for row in input_reader if not(regex.fullmatch(row["folder_name"]))]
experiments = filtered
input_file.close()
path = ""


# RUN EXPERIMENTS
for exp in experiments:
    FOLDER_NAME = exp["folder_name"]
    REPETITIONS = int(exp["repetitions"])

    # Create path to output files
    path = '../experiments/' + FOLDER_NAME + '/'
    # Create folder for output files
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # CONSTANTS
    # Data shape and running
    INPUT_SHAPE = make_tuple(exp["input_shape"])
    OUTPUT_SHAPE = int(exp["output_shape"])
    SCALING = float(exp["scaling"])
    DATASET_PERCENTAGE = float(exp["dataset_percentage"])
    if Dataset(int(exp["data_set"])) == Dataset.mnist:
        DATASET = datasets.mnist.load_data()
    else:
        DATASET = datasets.mnist.load_data()
    EPOCHS = int(exp["epochs"])
    MAX_RUNTIME = int(exp["max_runtime"])

    # Hyper parameters
    ACTIVATION_FUNCTION = Activation(int(exp["activation_function"]))
    INITIAL_MAX_NODES = int(exp["initial_max_nodes"])
    LOSS_FUNCTION = Loss(int(exp["loss_function"]))
    OPTIMIZER = Optimizer(int(exp["optimizer"]))

    # GA parameters
    POPULATION_SIZE = int(exp["population_size"])
    MATING_POOL = int(exp["mating_pool"])  # Must be between 0 and POPULATION_SIZE
    MUTATION_RATE = float(exp["mutation_rate"])  # Must be between 0 and 1
    ALGORITHM = str(exp["algorithm"])

    experiment_data = []
    # RUN REPETITIONS
    for i in range(REPETITIONS):
        writer = FileWriter(path+str(i)+'-', f'Experiment {i+1} ')
        writer.write_to_file([])

        writer.write_to_file(['Data shape and running'])
        writer.write_to_file(['input_shape ', INPUT_SHAPE])
        writer.write_to_file(['output_shape ', OUTPUT_SHAPE])
        writer.write_to_file(['scaling ', SCALING])
        writer.write_to_file(['epochs ', EPOCHS])
        writer.write_to_file(['max_runtime', MAX_RUNTIME])
        writer.write_to_file(['dataset_percentage', DATASET_PERCENTAGE])
        writer.write_to_file(['dataset', Dataset(int(exp["data_set"])).name])
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

        if ALGORITHM == "SimpleNet": # TODO: SimpleNet bruger ikke nogle af de parametre der parses, bortset fra repetitions
            writer.write_to_file(['epoch', 'accumulated_time', 'accuracy', 'loss'])
        elif ALGORITHM == "Lonely_GA_Layers":
            writer.write_to_file(['generation_no', 'params_no', 'layers_no', 'accuracy', 'loss'])
        else:
            writer.write_to_file(['generation_no', 'params_no', 'neurons_no', 'accuracy', 'loss'])

        ga = choose_GA()

        if ALGORITHM in ["Lonely_GA", "Lonely_GA_Validation", "Lonely_GA_PS_01", "Lonely_GA_PS_033"]:
            t = Thread(target=lonely_ga)
            t.daemon = True
            t.start()
            t.join(MAX_RUNTIME)
            ga.alive = False
            experiment_data.append(ga.history)

        if ALGORITHM in ["Lonely_Loss_GA", "Lonely_Loss_GA_PS_01", "Lonely_Loss_GA_PS_033", "Lonely_GA_Layers",
                         "Lonely_Loss_GA_LS_PS_2_033", "Lonely_Loss_GA_LS_PS_3_033", "Lonely_Loss_GA_LS_PS_4_033",
                         "Lonely_Loss_GA_LS_PS_5_033", "Lonely_Loss_GA_Exp_2_033", "Lonely_Loss_GA_Exp_3_033",
                         "Lonely_Loss_GA_Exp_4_033", "Lonely_Loss_GA_Exp_5_033", "Lonely_GA_Layers_All",
                         "Lonely_GA_Layers_LS_3", "Lonely_GA_Layers_LS_4", "Lonely_GA_Layers_LS_5",
                         "Lonely_GA_Layers_All_LS_3", "Lonely_GA_Layers_All_LS_4", "Lonely_GA_Layers_All_LS_5",
                         "Lonely_GA_Layers_Copy_LS_3", "Lonely_GA_Layers_Copy_LS_4", "Lonely_GA_Layers_Copy_LS_5"]:
            t = Thread(target=lonely_ga)
            t.daemon = True
            t.start()
            t.join(MAX_RUNTIME)
            ga.alive = False
            experiment_data.append(ga.history)

        if ALGORITHM in ["Lonely_Error_GA", "Lonely_Error_GA_PS_01", "Lonely_Error_GA_PS_033"]:
            t = Thread(target=lonely_ga)
            t.daemon = True
            t.start()
            t.join(MAX_RUNTIME)
            ga.alive = False
            experiment_data.append(ga.history)

        if ALGORITHM == "SimpleNet":
            t = Thread(target=simple_net)
            t.daemon = True
            t.start()
            t.join()
            experiment_data.append(ga.history)

        writer.close()
    make_plot(experiment_data)

