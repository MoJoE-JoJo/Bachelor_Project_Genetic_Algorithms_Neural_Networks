from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
from tensorflow.keras import datasets


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
train_slice = (int) (x_train.shape[0]*0.1)
test_slice = (int) (x_test.shape[0]*0.1)
(x_train, y_train), (x_test, y_test) = (x_train[:train_slice],
                                        y_train[:train_slice]), \
                                       (x_test[:test_slice],
                                        y_test[:test_slice])

train_label_count = [0 for n in range(0,10)]
test_label_count = [0 for n in range(0,10)]

for n in y_train:
    train_label_count[n]+=1

for n in y_test:
    test_label_count[n]+=1

print("train labels:", train_label_count)
print("test labels:", test_label_count)
