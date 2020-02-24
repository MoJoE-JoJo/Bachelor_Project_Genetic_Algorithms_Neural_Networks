from src.FileWriter import FileWriter
from src.GA.SimpleGA import SimpleGA
from src.GA.LonelyGA import LonelyGA
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer

import tensorflow as tf
from tensorflow.keras import datasets
import gc

# path = '../Test/'
writer = FileWriter('', ['Col1', 'Col2'])
x = 0
while x < 5:
    writer.write_to_file([x, 'test'])
    x = x + 1

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
ga = LonelyGA(input_shape=INPUT_SHAPE,
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

