from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
import gc
import random
import copy
import time
import tensorflow as tf

from src.DNA.LonelyLoss.LonelyDNALayers import LonelyDNALayers
from src.DNA.LonelyLoss.LonelyLossDNA import LonelyLossDNA
from src.DNA.LonelyLoss.LonelyLossDNAPS import LonelyLossDNAPS


class LonelyLossGA:
    alive = True
    generation_counter = 0
    notify = None
    input_shape = None
    output_size = None
    dataset = None
    scaling = None
    population_size = None
    epochs = None
    matingpool = None
    population = []
    history = []

    def __init__(self, t):
        self.GA_type = t

    def start(self, input_shape, output_shape, initial_max_nodes, activation, optimizer, loss, population_size, mutation_rate, dataset, scaling, epochs, matingpool, notify):
        gc.enable()
        self.generation_counter = 0
        self.history = []
        self.notify = notify
        self.input_shape = input_shape
        self.output_size = output_shape
        self.dataset = dataset
        self.scaling = scaling
        self.population_size = population_size
        self.epochs = epochs
        self.matingpool = matingpool
        self.population = []

        if self.GA_type == "Lonely_Loss_GA":
            self.population = [LonelyLossDNA(initial_max_nodes, activation, optimizer, loss, mutation_rate)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Loss_GA_PS_01":
            self.population = [LonelyLossDNAPS(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.1)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Loss_GA_PS_033":
            self.population = [LonelyLossDNAPS(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.33)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_GA_Layers":
            self.population = [LonelyDNALayers(initial_max_nodes, activation, optimizer, loss, mutation_rate)
                               for i in range(self.population_size)]

        self.evolution()

    def evolution(self):
        tc1 = time.time()
        while self.alive:
            for i in self.population:
                if self.alive:
                    i.fitness_func(input_shape=self.input_shape, output_shape=self.output_size, data=self.dataset, scaling=self.scaling, epochs=self.epochs)
                    tf.keras.backend.clear_session()

            if self.alive:
                self.population.sort(key=lambda x: x.fitness, reverse=True)

                matingpool = copy.deepcopy(self.population[:self.matingpool])
                print(time.time() - tc1)
                tc1 = time.time()

                # Used to update the history of the genetic algorithm
                if self.GA_type == "Lonely_GA_Layers":
                    self.history.append({"generation": self.generation_counter,
                                         "loss": self.population[0].evaluated["loss"],
                                         "accuracy": self.population[0].evaluated["accuracy"],
                                         "layers": len(self.population[0].genes),
                                         "params": self.population[0].num_params})
                    print(
                        "Generation {0} ----- Optimizer: {1}, Loss: {2}, Layers: {3}, Activation: {4}, Loss: {5: .4f}, Params: {6}"
                        .format(self.generation_counter,
                                self.population[0].optimizer.name,
                                self.population[0].loss.name,
                                len(self.population[0].genes),
                                self.population[0].activation.name,
                                1 / self.population[0].fitness,
                                self.population[0].num_params))
                else:
                    self.history.append({"generation": self.generation_counter,
                                         "loss": self.population[0].evaluated["loss"],
                                         "accuracy": self.population[0].evaluated["accuracy"],
                                         "nodes": self.population[0].gene.node_count,
                                         "params": self.population[0].num_params})

                    print(
                        "Generation {0} ----- Optimizer: {1}, Loss: {2}, Nodes: {3}, Activation: {4}, Loss: {5: .4f}, Params: {6}"
                        .format(self.generation_counter,
                                self.population[0].optimizer.name,
                                self.population[0].loss.name,
                                self.population[0].gene.node_count,
                                self.population[0].activation.name,
                                1 / self.population[0].fitness,
                                self.population[0].num_params))

                self.notify()  # Used to notify GARunner that an update to the history has happened

                self.generation_counter += 1
                fitness = [x.fitness for x in matingpool]
                for i in range(0, self.population_size):
                    parents = random.choices(matingpool, weights=fitness, k=1)   # .choices normalizes the weights
                    self.population[i] = copy.deepcopy(parents[0])
                    self.population[i].mutate()
