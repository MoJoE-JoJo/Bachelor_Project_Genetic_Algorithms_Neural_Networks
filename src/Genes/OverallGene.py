from random import random

from src.Enums.LossEnum import Loss
from src.Enums.OptimizerEnum import Optimizer

# A class to hold the overall hyperparameters of the neural network that can be modified,
# these being the optimizer and the loss function
class OverallGene:
    __m_rate_optimizer = 1.0
    __m_rate_loss = 1.0
    __m_rate_both = 1.0

    def __init__(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss_function = loss_function

    # uses the normalized mutations rates as probabilities for each kind of mutation
    def mutate(self):
        m_rate_optimizer, m_rate_loss, m_rate_both = self.__normalize()
        mutation = random.uniform(0.0, 1.0)
        if mutation <= m_rate_optimizer:
            self.__mutate_optimizer()
        elif mutation <= m_rate_loss:
            self.__mutate_loss()
        elif mutation <= m_rate_both:
            self.__mutate_optimizer()
            self.__mutate_loss()

    # returns normalized versions of the different mutation rates
    def __normalize(self):
        m_rate_sum = self.__m_rate_optimizer + self.__m_rate_loss + self.__m_rate_both
        m_rate_optimizer = self.__m_rate_optimizer / m_rate_sum
        m_rate_loss = self.__m_rate_loss / m_rate_sum
        m_rate_both = self.__m_rate_both / m_rate_sum
        return m_rate_optimizer, m_rate_loss, m_rate_both

    def __mutate_optimizer(self): self.optimizer = Optimizer(random.randrange(1, len(Optimizer) + 1))

    def __mutate_loss(self): self.optimizer = Loss(random.randrange(1, len(Loss)+1))
