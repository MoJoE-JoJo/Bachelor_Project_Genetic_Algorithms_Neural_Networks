import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

import random
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer
from src.Genes.SimpleGenes.DenseGene import DenseGene
from src.Genes.SimpleGenes.OverallGene import OverallGene


# Contains two genes, one overall gene and one dense gene.
class SimpleDNA:
    fitness = 0.0
    mutation_rates = [0.7, 0.2, 0.1]    #Represents individual probabilities of mutating either 0, 1, or 2 genes

    def __init__(self):
        no_dense_genes = 1
        initial_max_no_nodes = 50

        overall_gene = [OverallGene(Optimizer(random.randrange(1, len(Optimizer)+1)),
                                    Loss(random.randrange(1, len(Loss)+1)))]
        dense_genes = [DenseGene(random.randrange(0, initial_max_no_nodes+1),
                                 Activation(random.randrange(1, len(Activation)+1)))
                       for i in range(no_dense_genes)]

        self.genes = overall_gene + dense_genes

    # uses the normalized mutations rates as probabilities for the number of mutations
    def mutate(self):
        n_mutation_rates = self.__normalize()
        mutation = random.uniform(0.0, 1.0)
        if mutation <= n_mutation_rates[0]:
            return
        elif mutation <= n_mutation_rates[0] + n_mutation_rates[1]:
            self.genes[random.randrange(0, len(self.genes))].mutate()
        elif mutation <= sum(n_mutation_rates):
            self.genes[0].mutate()
            self.genes[1].mutate()

    # returns normalized versions of the different mutation rates
    def __normalize(self):
        m_rate_sum = sum(self.mutation_rates)
        n_mutation_rates = [i/m_rate_sum for i in self.mutation_rates]
        return n_mutation_rates

    def fitness_func(self, input_shape=(28, 28), output_shape=10):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(self.genes[1].node_count, activation=self.genes[1].activation.name),
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ]) if self.genes[1].node_count > 0 \
            else tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(output_shape, activation='softmax')
            ])
        loss = self.genes[0].loss_function
        optimizer = self.genes[0].optimizer

        if loss == (Loss.categorical_crossentropy or Loss.mean_squared_error):
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)

        model.compile(optimizer=optimizer.name,
                      loss=loss.name,
                      metrics=['accuracy'])

        hist = model.fit(x_train, y_train, epochs=3, verbose=0)
        self.fitness = hist.history['accuracy'][-1]
        #print(self.fitness)
        #print("Optimizer: {0}, Loss: {1}, Nodes: {2}, Activation: {3}, Accuracy: {4:.4f}".format(optimizer.name, loss.name, self.genes[1].node_count, self.genes[1].activation.name, self.fitness))


    #fitness function, modtager input-shape, output shape, og datasættet, generer det neurale netværk, og køre det. Evaluerer det med cross evaluation
    #husk logik til at håndterer hvis der er 0 nodes i et gen, da der så ikke skal laves et lag