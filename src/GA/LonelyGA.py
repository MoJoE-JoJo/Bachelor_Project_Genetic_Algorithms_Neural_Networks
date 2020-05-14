from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import random
import copy
import time
import gc

import tensorflow as tf

from src.DNA.Baseline.LonelyAccDNA import LonelyAccDNA
from src.DNA.FitnessFunction.ParScale.LonelyAccDNAParScale import LonelyAccDNAParScale
from src.DNA.Baseline.LonelyAccDNAValidation import LonelyAccDNAValidation
from src.DNA.FitnessFunction.Initial.LonelyErrDNA import LonelyErrDNA
from src.DNA.FitnessFunction.ParScale.LonelyErrDNAParScale import LonelyErrDNAParScale
from src.DNA.FitnessFunction.Exponential.LonelyLosDNAExpLoss import LonelyLosDNAExpLoss
from src.DNA.FitnessFunction.Exponential.LonelyLosDNAOverallExp import LonelyLosDNAOverallExp
from src.DNA.FitnessFunction.ParScale.LonelyLosDNAParScale import LonelyLosDNAParScale
from src.DNA.Layers.LonelyLosDNALayers import LonelyLosDNALayers
from src.DNA.Layers.LonelyLosDNALayersMutAll import LonelyLosDNALayersMutAll
from src.DNA.Layers.LonelyLosDNALayersMutAllCopy import LonelyLosDNALayersMutAllCopy
from src.DNA.FitnessFunction.Initial.LonelyLosDNA import LonelyLosDNA


class LonelyGA:
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

        self.create_population(initial_max_nodes, activation, optimizer, loss, mutation_rate)

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
                            print("Error occured, its a bug in TensorFlow-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
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
                fitness = [x.fitness for x in matingpool]
                for i in range(0, self.population_size):
                    parents = random.choices(matingpool, weights=fitness, k=1)  # .choices normalizes the weights
                    self.population[i] = copy.deepcopy(parents[0])
                    self.population[i].mutate()

    def write_data(self):
        if "Layers" in self.GA_type:
            self.history.append({"generation": self.generation_counter,
                                 "loss": self.population[0].evaluated["loss"],
                                 "accuracy": self.population[0].evaluated["accuracy"],
                                 "layers": len(self.population[0].genes),
                                 "params": self.population[0].num_params})

            print(
                "Generation {0} ----- Optimizer: {1}, Loss: {2}, Layers: {3}, Activation: {4}, Fitness: {5: .4f}, Params: {6}"
                .format(self.generation_counter,
                        self.population[0].optimizer.name,
                        self.population[0].loss.name,
                        len(self.population[0].genes),
                        self.population[0].activation.name,
                        self.population[0].fitness,
                        self.population[0].num_params))
        else:
            self.history.append({"generation": self.generation_counter,
                                 "loss": self.population[0].evaluated["loss"],
                                 "accuracy": self.population[0].evaluated["accuracy"],
                                 "nodes": self.population[0].gene.node_count,
                                 "params": self.population[0].num_params})

            print(
                "Generation {0} ----- Optimizer: {1}, Loss: {2}, Nodes: {3}, Activation: {4}, Fitness: {5: .4f}, Params: {6}"
                .format(self.generation_counter,
                        self.population[0].optimizer.name,
                        self.population[0].loss.name,
                        self.population[0].gene.node_count,
                        self.population[0].activation.name,
                        self.population[0].fitness,
                        self.population[0].num_params))

    def create_population(self, initial_max_nodes, activation, optimizer, loss, mutation_rate):

        # ----- Baseline

        if self.GA_type == "Lonely_Acc":
            self.population = [LonelyAccDNA(initial_max_nodes, activation, optimizer, loss, mutation_rate)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Acc_Validation":
            self.population = [LonelyAccDNAValidation(initial_max_nodes, activation, optimizer, loss, mutation_rate)
                               for i in range(self.population_size)]

        # ----- Fitness function - initial

        elif self.GA_type == "Lonely_Err":
            self.population = [LonelyErrDNA(initial_max_nodes, activation, optimizer, loss, mutation_rate)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los":
            self.population = [LonelyLosDNA(initial_max_nodes, activation, optimizer, loss, mutation_rate)
                               for i in range(self.population_size)]

        # ----- Fitness function - par scale

        elif self.GA_type == "Lonely_Acc_Par_Scale_0_1":
            self.population = [LonelyAccDNAParScale(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.1)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Acc_Par_Scale_0_33":
            self.population = [LonelyAccDNAParScale(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.33)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Err_Par_Scale_0_1":
            self.population = [LonelyErrDNAParScale(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.1)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Err_Par_Scale_0_33":
            self.population = [LonelyErrDNAParScale(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.33)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Par_Scale_0_1":
            self.population = [LonelyLosDNAParScale(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.1)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Par_Scale_0_33":
            self.population = [LonelyLosDNAParScale(initial_max_nodes, activation, optimizer, loss, mutation_rate, 0.33)
                               for i in range(self.population_size)]

        # ----- Fitness function - exponential loss

        elif self.GA_type == "Lonely_Los_Exp_Loss_2":
            self.population = [LonelyLosDNAExpLoss(initial_max_nodes, activation, optimizer, loss, mutation_rate, 2)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Exp_Loss_3":
            self.population = [LonelyLosDNAExpLoss(initial_max_nodes, activation, optimizer, loss, mutation_rate, 3)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Exp_Loss_4":
            self.population = [LonelyLosDNAExpLoss(initial_max_nodes, activation, optimizer, loss, mutation_rate, 4)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Exp_Loss_5":
            self.population = [LonelyLosDNAExpLoss(initial_max_nodes, activation, optimizer, loss, mutation_rate, 5)
                               for i in range(self.population_size)]

        # ----- Fitness function - overall exponential

        elif self.GA_type == "Lonely_Los_Overall_Exp_2":
            self.population = [LonelyLosDNAOverallExp(initial_max_nodes, activation, optimizer, loss, mutation_rate, 2)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Overall_Exp_3":
            self.population = [LonelyLosDNAOverallExp(initial_max_nodes, activation, optimizer, loss, mutation_rate, 3)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Overall_Exp_4":
            self.population = [LonelyLosDNAOverallExp(initial_max_nodes, activation, optimizer, loss, mutation_rate, 4)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Overall_Exp_5":
            self.population = [LonelyLosDNAOverallExp(initial_max_nodes, activation, optimizer, loss, mutation_rate, 5)
                               for i in range(self.population_size)]

        # ----- Layers

        elif self.GA_type == "Lonely_Los_Layers_3":
            self.population = [LonelyLosDNALayers(initial_max_nodes, activation, optimizer, loss, mutation_rate, 3)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_4":
            self.population = [LonelyLosDNALayers(initial_max_nodes, activation, optimizer, loss, mutation_rate, 4)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_5":
            self.population = [LonelyLosDNALayers(initial_max_nodes, activation, optimizer, loss, mutation_rate, 5)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_Mut_All_3":
            self.population = [LonelyLosDNALayersMutAll(initial_max_nodes, activation, optimizer, loss, mutation_rate, 3)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_Mut_All_4":
            self.population = [LonelyLosDNALayersMutAll(initial_max_nodes, activation, optimizer, loss, mutation_rate, 4)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_Mut_All_5":
            self.population = [LonelyLosDNALayersMutAll(initial_max_nodes, activation, optimizer, loss, mutation_rate, 5)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_Mut_All_Copy_3":
            self.population = [LonelyLosDNALayersMutAllCopy(initial_max_nodes, activation, optimizer, loss, mutation_rate, 3)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_Mut_All_Copy_4":
            self.population = [LonelyLosDNALayersMutAllCopy(initial_max_nodes, activation, optimizer, loss, mutation_rate, 4)
                               for i in range(self.population_size)]

        elif self.GA_type == "Lonely_Los_Layers_Mut_All_Copy_5":
            self.population = [LonelyLosDNALayersMutAllCopy(initial_max_nodes, activation, optimizer, loss, mutation_rate, 5)
                               for i in range(self.population_size)]
