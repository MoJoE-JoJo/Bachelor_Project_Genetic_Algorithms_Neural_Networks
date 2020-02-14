import random

from src.Enums.ActivationEnum import Activation


# A class to hold the hyperparameters for a dense layer that can be modified,
# these being the number of nodes in the layer and the activation function
class DenseGene:
    __m_rate_nodes = 3.0
    __m_rate_activation = 1.0
    __m_rate_both = 1.0
    __low_node_percentage = 0.75
    __high_node_percentage = 1.5

    def __init__(self, nodes, activation):
        self.node_count = nodes
        self.activation = activation

    # uses the normalized mutations rates as probabilities for each kind of mutation
    def mutate(self):
        m_rate_nodes, m_rate_activation, m_rate_both = self.__normalize()
        mutation = random.uniform(0.0, 1.0)
        if mutation <= m_rate_nodes:
            self.__mutate_nodes()
        elif mutation <= m_rate_nodes + m_rate_activation:
            self.__mutate_activation()
        elif mutation <= m_rate_nodes + m_rate_activation + m_rate_both:
            self.__mutate_nodes()
            self.__mutate_activation()

    # returns normalized versions of the different mutation rates
    def __normalize(self):
        m_rate_sum = self.__m_rate_nodes + self.__m_rate_activation + self.__m_rate_both
        m_rate_nodes = self.__m_rate_nodes / m_rate_sum
        m_rate_activation = self.__m_rate_activation / m_rate_sum
        m_rate_both = self.__m_rate_both / m_rate_sum
        return m_rate_nodes, m_rate_activation, m_rate_both

    # sets node_count to random number between 0 and max_no_nodes
    def __mutate_nodes(self): self.node_count = int(random.uniform(self.__low_node_percentage*self.node_count, self.__high_node_percentage*self.node_count))

    # sets activation function to a random
    def __mutate_activation(self): self.activation = Activation(random.randrange(1, len(Activation)+1))