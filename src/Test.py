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


mnist = tf.keras.datasets.mnist

img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
train_slice = (int) (x_train.shape[0]*0.1)
test_slice = (int) (x_test.shape[0]*0.1)
(x_train, y_train), (x_test, y_test) = (x_train[:train_slice],
                                        y_train[:train_slice]), \
                                       (x_test[:test_slice],
                                        y_test[:test_slice])

#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

loss = Loss.sparse_categorical_crossentropy
optimizer = Optimizer.Adam

if loss == (Loss.categorical_crossentropy or Loss.mean_squared_error):
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

def comp():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=Activation.relu.name),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer.name,
                  loss=loss.name,
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=5, verbose=1)
    return hist.history['accuracy']

repetitions=3
ys = []
for i in range(0, repetitions):
    ys.append(comp())
#ys = [[i for i in range(1,10)] for n in range(0,3)]

# Create y-axis average values
y = [0 for n in ys[0]]
for i in ys:
    n=0
    for j in i:
        y[n] += j
        n+=1

y = [i/repetitions for i in y]

# Create yerror values
yerr = []
for i in range(0,len(y)):
    y_temp = []
    for j in ys:
        y_temp.append(j[i])
    yerr.append((max(y_temp)-min(y_temp))/2)

x = [val for val in range(1, len(y)+1)]

test_name="joe"
fig = plt.figure()
subfig = fig.add_subplot(111)
subfig.set_ylabel('accuracy')
subfig.set_xlabel("epochs")
plt.xticks(x)

subfig.set_title(test_name)
subfig.errorbar(x, y, yerr=yerr, label='3 reps')

plt.savefig(fname=("../test/"+test_name+"/plot.svg"))