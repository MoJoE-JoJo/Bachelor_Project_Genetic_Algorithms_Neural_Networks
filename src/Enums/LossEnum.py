import enum


# Enum type for the different loss functions available in the Keras API
class Loss(enum.Enum):
    # mean_squared_error = 1
    # mean_absolute_error = 1
    # mean_absolute_percentage_error = 1
    # mean_squared_logarithmic_error = 2
    squared_hinge = 1
    hinge = 2
    categorical_hinge = 3
    # logcosh = 6
    huber_loss = 4
    categorical_crossentropy = 5
    sparse_categorical_crossentropy = 6
    kullback_leibler_divergence = 7
    poisson = 8
    # mean_squared_error = 3 # it doesn't wanna play nicely
    # is_categorical_crossentropy = 16 #(Basically returns a boolean, so it is not needed)
    # binary_crossentropy = 14 #(Not needed as it deals with binary classification)
    # cosine_proximity = 15 #(This does not exist, so we are not gonna use it)
