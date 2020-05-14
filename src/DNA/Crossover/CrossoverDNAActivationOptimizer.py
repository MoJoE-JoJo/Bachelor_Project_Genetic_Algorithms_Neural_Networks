import gc
import math

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

import random

from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer
from src.Genes.DenseGeneActivation import DenseGeneActivation
from src.Genes.OverallGene import OverallGene


# DNA containing a list of genes, each gene representing a dense layer in the
# neural network with an initial number of neurons between 1 and a max value.
# The first gene is an overall gene specifying the optimizer to use when training the network.
# The second gene represents a dense layer with changeable number of neurons and activation function. More dense genes
# can be added.
class CrossoverDNAActivationOptimizer:
    history = None
    fitness = 0.0
    evaluated = 0.0
    num_params = 0
    exponent = 3
    parameter_scaling = 0.1

    def __init__(self, initial_max_nodes, loss, mutation_rate, genes=None):
        gc.enable()
        self.initial_max_nodes = initial_max_nodes
        self.loss = loss
        self.mutation_rate = mutation_rate
        if genes is None:
            self.genes = [OverallGene(Optimizer(random.choice([5, 7]))),
                          DenseGeneActivation(
                              random.randrange(1, self.initial_max_nodes+1),
                              Activation(random.choice([1, 6]))
                          )]
        else:
            self.genes = genes

    # Check if the DNA should mutate
    def mutate(self):
        mutation = random.uniform(0.0, 1.0)
        if mutation > self.mutation_rate:
            return
        else:
            self.do_mutate()

    # Cecide mutate type
    def do_mutate(self):
        mutation_type = random.choice([0, 1])
        if mutation_type == 0:
            self.mutate_gene()
        elif mutation_type == 1:
            self.mutate_gene_no()

    # Mutate a random gene (layer)
    def mutate_gene(self):
        for g in self.genes:
            g.mutate()

    # Randomly add or remove a gene (layer)
    def mutate_gene_no(self):
        # if the DNA contains only one gene (the overall gene) the only possible mutation is to add a gene (layer)
        if len(self.genes) == 1:
            self.genes = self.genes + [DenseGeneActivation(random.randrange(1, self.initial_max_nodes + 1),
                                                           Activation(random.choice([1, 6])))]
        # else either add or remove a gene (layer)
        else:
            mutation_type = random.choice([0, 1])
            # add layer
            if mutation_type == 0:
                self.genes = self.genes + [DenseGeneActivation(random.randrange(1, self.initial_max_nodes + 1),
                                                               Activation(random.choice([1, 6])))]
            # remove layer
            elif mutation_type == 1:
                self.genes.pop(random.randrange(1, len(self.genes)))  # Cannot pop first element (OverallGene)

    # Calculates the fitness by training a neural network with the hyperparameters specified by the DNA
    def fitness_func(self, input_shape=(28, 28), output_shape=10, data=datasets.mnist.load_data(), scaling=255.0, epochs=5):
        (x_train, y_train), (x_test, y_test) = data
        x_train, x_test = x_train / scaling, x_test / scaling

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        if len(self.genes) > 0:
            for gene in self.genes[1:]:
                model.add(tf.keras.layers.Dense(gene.node_count, activation=gene.activation_function.name))
        model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))

        if self.loss == (Loss.categorical_crossentropy or Loss.mean_squared_error):
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)

        model.compile(optimizer=self.genes[0].optimizer.name,
                      loss=self.loss.name,
                      metrics=['accuracy'])

        hist = model.fit(x_train, y_train, epochs=epochs, verbose=0)

        self.history = hist.history

        loss = (1 / hist.history['loss'][-1])

        self.num_params = model.count_params()
        self.fitness = math.pow(loss, self.exponent) / (math.pow(self.num_params, self.parameter_scaling))

        result = model.evaluate(x_test, y_test, verbose=0)
        self.evaluated = dict(zip(model.metrics_names, result))
