import random

# A class to hold the overall hyperparameters for a neural network,
# that can vary. These being only the optimizer
from src.Enums.OptimizerEnum import Optimizer


class OverallGene:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    # mutates node count
    def mutate(self):
        # Set optimizer randomly to either Adam (5) or Nadam (7)
        self.optimizer = Optimizer(random.choice([5, 7]))
