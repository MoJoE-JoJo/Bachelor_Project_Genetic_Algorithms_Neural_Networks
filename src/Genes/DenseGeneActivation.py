import random

# A class to hold the hyperparameters for a dense layer that can be modified,
# these being the number of nodes in the layer and the activation function
from src.Enums.ActivationEnum import Activation


class DenseGeneActivation:
    __low_node_percentage = 0.77
    __high_node_percentage = 1.3

    def __init__(self, nodes, activation_function):
        self.node_count = nodes
        self.activation_function = activation_function

    def mutate(self):
        mutation_type = random.choice([0, 1])
        if mutation_type == 0:
            return
        elif mutation_type == 1:
            self.mutate_node_count()

    # mutates node count
    def mutate_activation_function(self):
        # Set activation function randomly to either Elu (1) or Relu (6)
        self.activation_function = Activation(random.choice([1, 6]))

    # mutates node count
    def mutate_node_count(self):
        self.node_count = int(
            random.uniform(self.__low_node_percentage * self.node_count,
                           self.__high_node_percentage * self.node_count)
        )
        if self.node_count <= 1:
            self.node_count = 1
