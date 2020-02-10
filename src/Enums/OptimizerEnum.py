import enum


# Enum type for the different optimizers available in the Keras API
class Optimizer(enum.Enum):
    SGD = 1
    RMSprop = 2
    Adagrad = 3
    Adadelta = 4
    Adam = 5
    Adamax = 6
    Nadam = 7
    Ftrl = 8
