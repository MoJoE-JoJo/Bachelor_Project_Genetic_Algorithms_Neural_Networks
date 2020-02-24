import enum


# Enum type for the different loss functions available in the Keras API
class Loss(enum.Enum):
    # mean_squared_error = 9
    # mean_absolute_error = 10
    # mean_absolute_percentage_error = 11
    # mean_squared_logarithmic_error = 12
    # logcosh = 13
    # mean_squared_error = 14 # it doesn't wanna play nicely
    squared_hinge = 1
    hinge = 2
    categorical_hinge = 3
    huber_loss = 4
    categorical_crossentropy = 5
    sparse_categorical_crossentropy = 6
    kullback_leibler_divergence = 7
    poisson = 8
    # is_categorical_crossentropy = 16 #(Basically returns a boolean, so it is not needed)
    # binary_crossentropy = 14 #(Not needed as it deals with binary classification)
    # cosine_proximity = 15 #(This does not exist, so we are not gonna use it)
