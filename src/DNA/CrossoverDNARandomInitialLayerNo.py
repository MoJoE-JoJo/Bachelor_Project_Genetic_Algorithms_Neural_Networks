import gc
import math

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

import random
from src.Enums.LossEnum import Loss
from src.Genes.LonelyGene import LonelyGene


# Contains a list of genes, each gene representing a dense layer in the
# neural network with an initial number of neurons between 1 and a max value
class CrossoverDNARandomInitialLayerNo:
    history = None
    fitness = 0.0
    evaluated = 0.0
    num_params = 0
    exponent = 3
    parameter_scaling = 0.33

    def __init__(self, initial_max_nodes, activation, optimizer, loss, mutation_rate, genes=None):
        gc.enable()
        self.initial_max_nodes = initial_max_nodes
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.mutation_rate = mutation_rate
        if genes is None:
            number_of_layers = random.randrange(1, 10+1)
            self.genes = [LonelyGene(random.randrange(1, self.initial_max_nodes+1)) for i in range(number_of_layers)]
        else:
            self.genes = genes

    # Mutates the gene based on a given mutation rate
    def mutate(self):
        mutation = random.uniform(0.0, 1.0)
        if mutation > self.mutation_rate:
            return
        else:
            self.do_mutate()

    # decide mutate type
    def do_mutate(self):
        # if the DNA contains no genes the only possible mutation is to add a gene (layer)
        if len(self.genes) == 0:
            self.genes = self.genes + [LonelyGene(random.randrange(1, self.initial_max_nodes+1))]
        # else either mutate a gene or add/remove a gene
        else:
            mutation_type = random.choice([1, 2])
            if mutation_type == 1:
                self.mutate_gene()
            elif mutation_type == 2:
                self.mutate_gene_no()

    # mutate a random gene (layer)
    def mutate_gene(self):
        for g in self.genes:
            g.mutate()

    # randomly add or remove a gene (layer)
    def mutate_gene_no(self):
        mutation_type = random.choice([1, 2])
        # add layer
        if mutation_type == 1:
            self.genes = self.genes + [LonelyGene(random.randrange(1, self.initial_max_nodes+1))]
        # remove layer
        elif mutation_type == 2:
            self.genes.pop(random.randrange(len(self.genes)))

    # Calculates the fitness by training a neural network with the hyper parameters specified by the DNA
    def fitness_func(self, input_shape=(28, 28), output_shape=10, data=datasets.mnist.load_data(), scaling=255.0, epochs=5):
        (x_train, y_train), (x_test, y_test) = data
        x_train, x_test = x_train / scaling, x_test / scaling

        #input_layer = [tf.keras.layers.Flatten(input_shape=input_shape)]
        #hidden_layers = []
        #if len(self.genes) > 0:
        #    hidden_layers = [tf.keras.layers.Dense(gene.node_count, activation=self.activation.name) for gene in self.genes]
        #output_layer = [tf.keras.layers.Dense(output_shape, activation='softmax')]

        #model = tf.keras.models.Sequential(input_layer + hidden_layers + output_layer)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        if len(self.genes) > 0:
            for gene in self.genes:
                model.add(tf.keras.layers.Dense(gene.node_count, activation=self.activation.name))
        model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))


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
        self.fitness = math.pow(loss, self.exponent) / (math.pow(self.num_params, self.parameter_scaling))

        result = model.evaluate(x_test, y_test, verbose=0)
        self.evaluated = dict(zip(model.metrics_names, result))
