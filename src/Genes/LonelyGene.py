import random

# A class to hold the hyperparameter for a dense layer that can be modified,
# this being the number of nodes in the layer


class LonelyGene:
    __low_node_percentage = 0.5
    __high_node_percentage = 1.5

    def __init__(self, nodes):
        self.node_count = nodes

    # mutates node count
    def mutate(self):
        self.node_count = int(
            random.uniform(self.__low_node_percentage * self.node_count,
                           self.__high_node_percentage * self.node_count)
        )
