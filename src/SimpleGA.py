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
    _population_size = 10
    _population = []

    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size
        self._population = [SimpleDNA() for i in range(self._population_size)]
