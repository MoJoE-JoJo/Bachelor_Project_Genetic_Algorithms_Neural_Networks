from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import sys
import collections
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import time

from src.SimpleDNA import SimpleDNA


class SimpleGA:
    population_size = 10

    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size
        self.population = [SimpleDNA() for i in range(self.population_size)]
        self.evolution()


    # def evolution(self):
    #    while True:
            # check fitness for all individer
            # udvælge de bedste individer
            # Skal bagefter køre på testsættet med den bedste (kaldet champion), og det er det bedste resultat fra den generation.
            #Reproduction, parrere de bedste individer

     #construct NN


