from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import random
import copy

from src.DNA.LonelyDNA import LonelyDNA


class LonelyGA:
    generation_counter = 0

    def __init__(self, input_shape, output_shape, initial_max_nodes, activation, optimizer, loss, population_size, mutation_rate, dataset, scaling, epochs):
        self.input_shape = input_shape
        self.output_size = output_shape
        self.dataset = dataset
        self.scaling = scaling
        self.population_size = population_size
        self.epochs = epochs
        self.population = [LonelyDNA(initial_max_nodes, activation, optimizer, loss, mutation_rate) for i in range(self.population_size)]
        self.evolution()

    def evolution(self):
        while True:
            for i in self.population:
                i.fitness_func(input_shape=self.input_shape, output_shape=self.output_size, data=self.dataset, scaling=self.scaling, epochs=self.epochs)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            print("Generation {0} ----- Optimizer: {1}, Loss: {2}, Nodes: {3}, Activation: {4}, Accuracy: {5: .4f}"
                  .format(self.generation_counter,
                          self.population[0].optimizer.name,
                          self.population[0].loss.name,
                          self.population[0].gene.node_count,
                          self.population[0].gene.activation.name,
                          self.population[0].fitness))

            self.generation_counter += 1
            for i in self.population:
                i.mutate()