import gc

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

import random
from src.Enums.LossEnum import Loss
from src.Genes.DenseGene import DenseGene


# DNA containing a gene representing a dense layer in the neural network
# with an initial number of neurons between 1 and a max value
# Fitness based on accuracy, using validation split when training the neural network
class LonelyAccDNA:
    fitness = 0.0
    history = None
    evaluated = 0.0
    num_params = 0

    def __init__(self, initial_max_nodes, activation, optimizer, loss, mutation_rate):
        gc.enable()
        self.initial_max_nodes = initial_max_nodes
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.mutation_rate = mutation_rate
        self.gene = DenseGene(random.randrange(1, self.initial_max_nodes+1))

    # Mutates the gene based on a given mutation rate
    def mutate(self):
        mutation = random.uniform(0.0, 1.0)
        if mutation > self.mutation_rate:
            return
        else:
            self.gene.mutate()

    # Calculates the fitness by training a neural network with the hyperparameters specified by the DNA
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
        self.fitness = hist.history['accuracy'][-1]
        self.history = hist.history
        self.num_params = model.count_params()

        result = model.evaluate(x_test, y_test, verbose=0)
        self.evaluated = dict(zip(model.metrics_names, result))
