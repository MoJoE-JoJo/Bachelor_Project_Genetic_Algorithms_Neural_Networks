import enum


# Enum type for the different loss functions available in the Keras API
class Loss(enum.Enum):
    squared_hinge = 1
    hinge = 2
    categorical_hinge = 3
    huber_loss = 4
    categorical_crossentropy = 5
    sparse_categorical_crossentropy = 6
    kullback_leibler_divergence = 7
    poisson = 8
