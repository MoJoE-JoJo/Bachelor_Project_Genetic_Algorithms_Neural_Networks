import enum


# Enum type for the different loss functions available in the Keras API
class Loss(enum.Enum):
    mean_squared_error = 1
    mean_absolute_error = 2
    mean_absolute_percentage_error = 3
    mean_squared_logarithmic_error = 4
    squared_hinge = 5
    hinge = 6
    categorical_hinge = 7
    logcosh = 8
    huber_loss = 9
    categorical_crossentropy = 10
    sparse_categorical_crossentropy = 11
    kullback_leibler_divergence = 12
    poisson = 13
    # is_categorical_crossentropy = 16 #(Basically returns a boolean, so it is not needed)
    # binary_crossentropy = 14 #(Not needed as it deals with binary classification)
    # cosine_proximity = 15 #(This does not exist, so we are not gonna use it)
