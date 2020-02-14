import random

# A class to hold the hyperparameters for a dense layer that can be modified,
# these being the number of nodes in the layer and the activation function


class LonelyGene:
    __low_node_percentage = 0.75
    __high_node_percentage = 1.5

    def __init__(self, nodes, acti):
        self.node_count = nodes
        self.activation = acti

    # mutates node count
    def mutate(self):
        self.node_count = int(
            random.uniform(self.__low_node_percentage * self.node_count,
                           self.__high_node_percentage * self.node_count)
        )
