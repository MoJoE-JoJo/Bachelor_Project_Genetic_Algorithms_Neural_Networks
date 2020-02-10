from random import random

from src.Enums.ActivationEnum import Activation


class DenseGene:
    __m_rate_nodes = 1.0
    __m_rate_activation = 1.0
    __m_rate_both = 1.0

    def __init__(self, nodes, activation, max_no_nodes):
        self.node_count = nodes
        self.activation = activation
        self.max_nodes = max_no_nodes

    def mutate(self):
        m_rate_nodes, m_rate_activation, m_rate_both = self.__normalize()
        mutation = random.uniform(0.0, 1.0)
        if mutation <= m_rate_nodes:
            self.__mutate_nodes()
        elif mutation <= m_rate_activation:
            self.__mutate_activation()
        elif mutation <= m_rate_both:
            self.__mutate_nodes()
            self.__mutate_activation()

    # returns normalized versions of the different mutation rates
    def __normalize(self):
        m_rate_sum = self.__m_rate_nodes + self.__m_rate_activation + self.__m_rate_both
        m_rate_nodes = self.__m_rate_nodes / m_rate_sum
        m_rate_activation = self.__m_rate_activation / m_rate_sum
        m_rate_both = self.__m_rate_both / m_rate_sum
        return m_rate_nodes, m_rate_activation, m_rate_both

    # sets node
    def __mutate_nodes(self): self.node_count = random.randrange(0, self.max_no_nodes+1)

    def __mutate_activation(self): self.activation = Activation(random.randrange(1, len(Activation)+1))
