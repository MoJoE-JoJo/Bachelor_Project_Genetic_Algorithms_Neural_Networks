import os.path
import sys

import tensorflow as tf
from tensorflow.keras import datasets
import gc
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Used to find modules when running from venv
from src.FileWriter import FileWriter
from src.GA.SimpleGA import SimpleGA
from src.GA.LonelyGA import LonelyGA
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer


# READ FROM FILE
input_file = open(sys.argv[1], "r")
input_reader = csv.DictReader(input_file)

line_count = 0
for row in input_reader:
    print(row)
    # Use row["key_name"] to get specific value
    line_count += 1
print(f'Processed {line_count} lines.')

input_file.close()


# WRITE TO FILE
path = 'Test/'
writer = FileWriter(path, ['Col1', 'Col2'])


def write_to_file(data):
    print("Writing")
    writer.write_to_file(data)


# CONSTANTS
# Data shape and running
INPUT_SHAPE = (28, 28)
OUTPUT_SHAPE = 10
SCALING = 255.0
DATASET = datasets.mnist.load_data()
EPOCHS = 5

# Hyper parameters
ACTIVATION_FUNCTION = Activation.relu
INITIAL_MAX_NODES = 50
LOSS_FUNCTION = Loss.sparse_categorical_crossentropy
OPTIMIZER = Optimizer.Adam

# GA parameters
POPULATION_SIZE = 10
MATINGPOOL = 10  # Must be between 0 and POPULATION_SIZE
MUTATION_RATE = 0.3  # Must be between 0 and 1

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
              dataset=DATASET,
              epochs=EPOCHS,
              matingpool=MATINGPOOL)

writer.close()
