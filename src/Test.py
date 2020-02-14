from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from src.Enums.OptimizerEnum import Optimizer
from src.Enums.ActivationEnum import Activation
from src.Enums.LossEnum import Loss


#a = [1, 2, 3]
#b = a
#a="ello"

#one = time.perf_counter()
#first = 1000 ** 1000
#two = time.perf_counter()
#three = time.perf_counter()
#print(first)
#four = time.perf_counter()
#print('first', two - one)
#print('print', four - three)


#one= time.perf_counter()
#second = first ** 1000
#two = time.perf_counter()
#three = time.perf_counter()
#three = time.perf_counter()
#print(second)
#four = time.perf_counter()
#print('second', two - one)
#print('print', four - three)

#one= time.perf_counter()
#third = second ** 1000
#two= time.perf_counter()
#four = time.perf_counter()
#three = time.perf_counter()
#print(third)
#four = time.perf_counter()
#print('third', two - one)
#print('print', four - three)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

x = [val for val in range(1, 10)]
plt.plot(x, x)
# plt.show()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation=Activation(3).name),
  tf.keras.layers.Dense(10, activation='softmax')
])

loss = Loss.huber_loss
optimizer = Optimizer.Adam

if loss == (Loss.categorical_crossentropy or Loss.mean_squared_error):
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

model.compile(optimizer=optimizer.name,
              loss=loss.name,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, verbose=1)

model.evaluate(x_test,  y_test, verbose=1)
