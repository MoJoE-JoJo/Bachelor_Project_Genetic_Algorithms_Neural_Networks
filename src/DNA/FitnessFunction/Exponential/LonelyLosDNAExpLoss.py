import gc
import math

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

import random
from src.Enums.LossEnum import Loss
from src.Genes.DenseGene import DenseGene


# Contains a gene representing a dense layer in the neural network
# with an initial number of neurons between 1 and a max value
class LonelyLosDNAExpLoss:
    fitness = 0.0
    history = None
    evaluated = 0.0
    num_params = 0
    parameter_scaling = 0.1

    def __init__(self, initial_max_nodes, activation, optimizer, loss, mutation_rate, loss_exp):
        gc.enable()
        self.initial_max_nodes = initial_max_nodes
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.mutation_rate = mutation_rate
        self.loss_exp = loss_exp
        self.gene = DenseGene(random.randrange(1, self.initial_max_nodes+1))

    # Mutates the gene based on a given mutation rate
    def mutate(self):
        mutation = random.uniform(0.0, 1.0)
        if mutation > self.mutation_rate:
            return
        else:
            self.gene.mutate()

    # Calculates the fitness by training a neural network with the hyper parameters specified by the DNA
    def fitness_func(self, input_shape=(28, 28), output_shape=10, data=datasets.mnist.load_data(), scaling=255.0, epochs=5):
        (x_train, y_train), (x_test, y_test) = data
        x_train, x_test = x_train / scaling, x_test / scaling

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(self.gene.node_count, activation=self.activation.name),
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])

        if self.loss == (Loss.categorical_crossentropy or Loss.mean_squared_error):
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)

        model.compile(optimizer=self.optimizer.name,
                      loss=self.loss.name,
                      metrics=['accuracy'])

        hist = model.fit(x_train, y_train, epochs=epochs, verbose=0)

        self.history = hist.history
        loss = (1 / hist.history['loss'][-1])
        self.num_params = model.count_params()

        self.fitness = math.pow(loss, self.loss_exp) / (math.pow(self.num_params, self.parameter_scaling))

        result = model.evaluate(x_test, y_test, verbose=0)
        self.evaluated = dict(zip(model.metrics_names, result))
