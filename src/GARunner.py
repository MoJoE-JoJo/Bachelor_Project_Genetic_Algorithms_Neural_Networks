import os.path
import sys
import gc
import csv
from tensorflow.keras import datasets
from ast import literal_eval as make_tuple

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
    folder_name = exp["folder_name"]
    iterations = int(exp["iterations"])

    # Create path to output files
    path = 'experiments/' + folder_name + '/'
    # Create folder for output files
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # CONSTANTS
    # Data shape and running
    INPUT_SHAPE = make_tuple(exp["input_shape"])
    OUTPUT_SHAPE = int(exp["output_shape"])
    SCALING = float(exp["scaling"])
    if Dataset(int(exp["data_set"])) == Dataset.mnist:
        DATA_SET = datasets.mnist.load_data()
    else:
        DATA_SET = datasets.mnist.load_data()
    EPOCHS = int(exp["epochs"])

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

        print("GA Started")
        gc.enable()
        ga = LonelyGA(notify=write_to_file,
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

        writer.close()
