import enum


# Enum type for the different data sets
class Dataset(enum.Enum):
    mnist = 1
    fashion_mnist = 2
    cifar10 = 3