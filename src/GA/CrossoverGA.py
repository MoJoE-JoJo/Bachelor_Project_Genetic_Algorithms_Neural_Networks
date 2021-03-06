from __future__ import absolute_import, division, print_function, unicode_literals

import math
import random
import copy
import time
import gc
import tensorflow as tf

from src.DNA.Crossover.CrossoverDNAActivationOptimizer import CrossoverDNAActivationOptimizer
from src.DNA.Crossover.CrossoverDNARandomInitialLayerNo import CrossoverDNARandomInitialLayerNo
from src.DNA.Crossover.CrossoverDNAMinLayers import CrossoverDNAMinLayers
from src.DNA.Crossover.CrossoverDNAValidation import CrossoverDNAValidation
from src.DNA.Layers.LonelyLosDNALayersMutAll import LonelyLosDNALayersMutAll


class CrossoverGA:
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
    initial_max_nodes = None
    activation = None
    optimizer = None
    loss = None
    mutation_rate = None

    def __init__(self, t):
        self.GA_type = t

    def start(self, input_shape, output_shape, initial_max_nodes, activation, optimizer, loss,
              population_size, mutation_rate, dataset, scaling, epochs, matingpool, notify):
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

        self.initial_max_nodes = initial_max_nodes
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.mutation_rate = mutation_rate

        self.create_population()
        self.evolution()

    def evolution(self):
        tc1 = time.time()
        while self.alive:
            # Calculate fitness for population
            for i in self.population:
                if self.alive:
                    try:
                        i.fitness_func(input_shape=self.input_shape, output_shape=self.output_size,
                                       data=self.dataset, scaling=self.scaling, epochs=self.epochs)
                    except:
                        try:
                            i.fitness_func(input_shape=self.input_shape, output_shape=self.output_size,
                                           data=self.dataset, scaling=self.scaling, epochs=self.epochs)
                        except:
                            print("Error occured, its a bug in TensorFlow----------------------------------------" +
                                  "------------------------------------------------------------------------------" +
                                  "-------------------------------------------------------------")
                            i.fitness = 0
                    tf.keras.backend.clear_session()

            # Create mating pool and do reproduction
            if self.alive:
                self.population.sort(key=lambda x: x.fitness, reverse=True)

                matingpool = copy.deepcopy(self.population[:self.matingpool])
                print(time.time() - tc1)
                tc1 = time.time()

                self.write_data()
                self.notify()  # Used to notify GARunner that an update to the history has happened
                self.generation_counter += 1
                self.reproduction(matingpool)

    # Do reproduction based on the type of DNA of the population
    def reproduction(self, matingpool):
        new_population = []

        fitness = [x.fitness for x in matingpool]
        for i in range(math.ceil(self.population_size/2)):
            parents = random.choices(matingpool, weights=fitness, k=2)  # .choices normalizes the weights
            p1_genes = parents[0].genes
            p2_genes = parents[1].genes
            if self.GA_type in ["Crossover", "Crossover_Random_Initial_Layer_No", "Crossover_Validation"]:
                # if not possible to crossover parents do asexual reproduction
                if (len(p1_genes) == 0 or len(p2_genes) == 0) or (len(p1_genes) == 1 and len(p2_genes) == 1):
                    self.asexual_reproduction(new_population, parents)
                # else do crossover
                else:
                    self.crossover_reproduction(new_population, p1_genes, p2_genes)
            elif self.GA_type == "Crossover_Min_Layers":
                self.crossover_reproduction(new_population, p1_genes, p2_genes)
            elif self.GA_type == "Crossover_Activation_Optimizer":
                # if not possible to crossover parents do asexual reproduction
                if len(p1_genes) == 1 and len(p2_genes) == 1:
                    self.asexual_reproduction(new_population, parents)
                # else do crossover
                else:
                    self.crossover_reproduction(new_population, p1_genes, p2_genes)

        self.population = new_population

    def asexual_reproduction(self, new_population, parents):
        new_population.append(copy.deepcopy(parents[0]))
        new_population[-1].mutate()

        new_population.append(copy.deepcopy(parents[1]))
        new_population[-1].mutate()

    def crossover_reproduction(self, new_population, p1_genes, p2_genes):
        split = math.ceil(min(len(gene) for gene in [p1_genes, p2_genes]) / 2)

        c1_genes = copy.deepcopy(p1_genes[:split]) + copy.deepcopy(p2_genes[split:])
        c2_genes = copy.deepcopy(p2_genes[:split]) + copy.deepcopy(p1_genes[split:])

        # Create "children" with the correct type of DNA
        c1 = None
        c2 = None
        if self.GA_type == "Crossover":
            c1 = LonelyLosDNALayersMutAll(self.initial_max_nodes, self.activation, self.optimizer,
                                          self.loss, self.mutation_rate, 4, c1_genes)
            c2 = LonelyLosDNALayersMutAll(self.initial_max_nodes, self.activation, self.optimizer,
                                          self.loss, self.mutation_rate, 4, c2_genes)
        elif self.GA_type == "Crossover_Min_Layers":
            c1 = CrossoverDNAMinLayers(self.initial_max_nodes, self.activation, self.optimizer,
                                       self.loss, self.mutation_rate, c1_genes)
            c2 = CrossoverDNAMinLayers(self.initial_max_nodes, self.activation, self.optimizer,
                                       self.loss, self.mutation_rate, c2_genes)
        elif self.GA_type == "Crossover_Activation_Optimizer":
            c1 = CrossoverDNAActivationOptimizer(self.initial_max_nodes, self.loss, self.mutation_rate, c1_genes)
            c2 = CrossoverDNAActivationOptimizer(self.initial_max_nodes, self.loss, self.mutation_rate, c2_genes)
        elif self.GA_type == "Crossover_Random_Initial_Layer_No":
            c1 = CrossoverDNARandomInitialLayerNo(self.initial_max_nodes, self.activation, self.optimizer,
                                                  self.loss, self.mutation_rate, c1_genes)
            c2 = CrossoverDNARandomInitialLayerNo(self.initial_max_nodes, self.activation, self.optimizer,
                                                  self.loss, self.mutation_rate, c2_genes)
        elif self.GA_type == "Crossover_Validation":
            c1 = CrossoverDNAValidation(self.initial_max_nodes, self.activation, self.optimizer,
                                        self.loss, self.mutation_rate, c1_genes)
            c2 = CrossoverDNAValidation(self.initial_max_nodes, self.activation, self.optimizer,
                                        self.loss, self.mutation_rate, c2_genes)

        new_population.append(c1)
        new_population[-1].mutate()

        new_population.append(c2)
        new_population[-1].mutate()

    def write_data(self):
        self.history.append({"generation": self.generation_counter,
                             "loss": self.population[0].evaluated["loss"],
                             "accuracy": self.population[0].evaluated["accuracy"],
                             "layers": len(self.population[0].genes),
                             "params": self.population[0].num_params})

        if self.GA_type == "Crossover_Activation_Optimizer":
            print(
                "Generation {0} --- Optimizer: {1}, Loss: {2}, Layers: {3}, Fitness: {4: .4f}, Params: {5}"
                .format(self.generation_counter,
                        self.population[0].genes[0].optimizer.name,
                        self.population[0].loss.name,
                        len(self.population[0].genes),
                        self.population[0].fitness,
                        self.population[0].num_params))
        else:
            print(
                "Generation {0} --- Optimizer: {1}, Loss: {2}, Layers: {3}, Activation: {4}, Fitness: {5: .4f}, Params: {6}"
                .format(self.generation_counter,
                        self.population[0].optimizer.name,
                        self.population[0].loss.name,
                        len(self.population[0].genes),
                        self.population[0].activation.name,
                        self.population[0].fitness,
                        self.population[0].num_params))

    def create_population(self):
        if self.GA_type == "Crossover":
            self.population = [LonelyLosDNALayersMutAll(self.initial_max_nodes, self.activation, self.optimizer,
                                                        self.loss, self.mutation_rate, 4)
                               for i in range(self.population_size)]
        elif self.GA_type == "Crossover_Min_Layers":
            self.population = [CrossoverDNAMinLayers(self.initial_max_nodes, self.activation, self.optimizer,
                                                     self.loss, self.mutation_rate)
                               for i in range(self.population_size)]

        elif self.GA_type == "Crossover_Activation_Optimizer":
            self.population = [CrossoverDNAActivationOptimizer(self.initial_max_nodes, self.loss, self.mutation_rate)
                               for i in range(self.population_size)]

        elif self.GA_type == "Crossover_Random_Initial_Layer_No":
            self.population = [CrossoverDNARandomInitialLayerNo(self.initial_max_nodes, self.activation, self.optimizer,
                                                                self.loss, self.mutation_rate)
                               for i in range(self.population_size)]

        elif self.GA_type == "Crossover_Validation":
            self.population = [CrossoverDNAValidation(self.initial_max_nodes, self.activation, self.optimizer,
                                                      self.loss, self.mutation_rate)
                               for i in range(self.population_size)]
