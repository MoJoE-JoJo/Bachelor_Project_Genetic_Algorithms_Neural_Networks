from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import random
import copy
import time
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Thread

from src.DNA.LonelyDNA import LonelyDNA


class LonelyGA:
    generation_counter = 0
    input_shape = None
    output_size = None
    dataset = None
    scaling = None
    population_size = None
    epochs = None
    matingpool = None
    population = None

    def start(self, input_shape, output_shape, initial_max_nodes, activation, optimizer, loss, population_size, mutation_rate, dataset, scaling, epochs, matingpool):
        self.input_shape = input_shape
        self.output_size = output_shape
        self.dataset = dataset
        self.scaling = scaling
        self.population_size = population_size
        self.epochs = epochs
        self.matingpool = matingpool
        self.population = [LonelyDNA(initial_max_nodes, activation, optimizer, loss, mutation_rate) for i in range(self.population_size)]
        self.evolution()

    def evolution(self):
        tc1 = time.time()
        while True:
            for i in self.population:
                i.fitness_func(input_shape=self.input_shape, output_shape=self.output_size, data=self.dataset, scaling=self.scaling, epochs=self.epochs)

            #with ThreadPoolExecutor(max_workers=5) as executor:
            #    for i in self.population:
            #        executor.submit(fn=i.fitness_func, input_shape=self.input_shape, output_shape=self.output_size, data=self.dataset, scaling=self.scaling, epochs=self.epochs)
            #    executor.shutdown(wait=True)


            #threads = []
            #for i in self.population:
            #    threads.append(Thread(target=i.fitness_func, kwargs={'input_shape': self.input_shape, 'output_shape': self.output_size, 'data': self.dataset, 'scaling': self.scaling, 'epochs': self.epochs}))
            #for i in threads:
            #    i.daemon = True
            #    i.start()
            #for i in threads:
            #    i.join()

            self.population.sort(key=lambda x: x.fitness, reverse=True)
            matingpool = copy.deepcopy(self.population[:self.matingpool])
            print(time.time()-tc1)
            tc1 = time.time()
            print("Generation {0} ----- Optimizer: {1}, Loss: {2}, Nodes: {3}, Activation: {4}, Accuracy: {5: .4f}"
                  .format(self.generation_counter,
                          self.population[0].optimizer.name,
                          self.population[0].loss.name,
                          self.population[0].gene.node_count,
                          self.population[0].activation.name,
                          self.population[0].fitness))

            self.generation_counter += 1
            fitness = [x.fitness for x in matingpool]
            for i in range(0, self.population_size):
                parents = random.choices(matingpool, weights=fitness, k=1)
                self.population[i] = copy.deepcopy(parents[0])
                self.population[i].mutate()