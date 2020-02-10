import enum


# Enum type for the different activation functions available in the Keras API
class Activation(enum.Enum):
    elu = 1
    softmax = 2
    selu = 3
    softplus = 4
    softsign = 5
    relu = 6
    tanh = 7
    sigmoid = 8
    hard_sigmoid = 9
    exponential = 10
    linear = 11
