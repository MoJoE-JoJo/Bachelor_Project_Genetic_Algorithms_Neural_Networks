import random

# A class to hold the hyperparameters for a dense layer that can be modified,
# these being the number of nodes in the layer


class DenseGene:
    __low_node_percentage = 0.77
    __high_node_percentage = 1.3

    def __init__(self, nodes):
        self.node_count = nodes

    # Mutate node count
    def mutate(self):
        self.node_count = int(
            random.uniform(self.__low_node_percentage * self.node_count,
                           self.__high_node_percentage * self.node_count)
        )
        if self.node_count <= 1:
            self.node_count = 1
