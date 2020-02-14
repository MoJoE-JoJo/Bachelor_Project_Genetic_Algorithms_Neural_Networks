import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

import random
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer
from src.Genes.LonelyGene import LonelyGene
from src.Genes.SimpleGenes.DenseGene import DenseGene
from src.Genes.SimpleGenes.OverallGene import OverallGene


# Contains two genes, one overall gene and one dense gene.
class LonelyDNA:
    fitness = 0.0

    def __init__(self, initial_max_nodes, activation, optimizer, loss, mutation_rate):
        self.initial_max_nodes = initial_max_nodes
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.mutation_rate = mutation_rate
        self.gene = LonelyGene(random.randrange(0, self.initial_max_nodes+1), self.activation)

    # uses the normalized mutations rates as probabilities for the number of mutations
    def mutate(self):
        mutation = random.uniform(0.0, 1.0)
        if mutation <= self.mutation_rate:
            return
        else:
            self.gene.mutate()

    def fitness_func(self, input_shape=(28, 28), output_shape=10, data=datasets.mnist.load_data(), scaling=255.0, epochs=5):
        (x_train, y_train), (x_test, y_test) = data
        x_train, x_test = x_train / scaling, x_test / scaling

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(self.gene.node_count, activation=self.gene.activation.name),
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
