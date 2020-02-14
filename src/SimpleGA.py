from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import sys
import collections
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import time
import random
import copy

from src.SimpleDNA import SimpleDNA


class SimpleGA:
    population_size = 10
    generation_counter = 0

    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size
        self.population = [SimpleDNA() for i in range(self.population_size)]
        self.evolution()

    def evolution(self):
        while True:
            for i in self.population:
                i.fitness_func()
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            matingpool = self.mating_normalize(copy.deepcopy(self.population[:5]))
            print("Generation {0} ----- Optimizer: {1}, Loss: {2}, Nodes: {3}, Activation: {4}, Accuracy: {5: .4f}"
                  .format(self.generation_counter,
                          self.population[0].genes[0].optimizer.name,
                          self.population[0].genes[0].loss_function.name,
                          self.population[0].genes[1].node_count,
                          self.population[0].genes[1].activation.name,
                          self.population[0].fitness))

            self.generation_counter += 1
            fitness = [x.fitness for x in matingpool]
            for i in range(0, self.population_size):
                parents = random.choices(matingpool, weights=fitness, k=2)
                self.population[i] = self.crossover(parents[0], parents[1])
                self.population[i].mutate()

    def crossover(self, parent1, parent2, cross=1):
        child = SimpleDNA()
        child.genes[:cross] = copy.deepcopy(parent1.genes[:cross])
        child.genes[cross:] = copy.deepcopy(parent2.genes[cross:])
        return child

    def mating_normalize(self, pool):
        sum = 0.0
        for i in pool:
            sum += i.fitness
        for i in pool:
            try:
                i.fitness = i.fitness/sum
            except RuntimeError:
                i.fitness = i.fitness
                print("Divide by zero")
        return pool

    #    while True:
            # check fitness for all individer
            # udvælge de bedste individer
            # Skal bagefter køre på testsættet med den bedste (kaldet champion), og det er det bedste resultat fra den generation.
            #Reproduction, parrere de bedste individer

     #construct NN


