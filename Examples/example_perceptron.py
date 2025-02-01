from AIScratch.NeuralNetwork import Perceptron, Treshold, MSE
from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

"""
Example of usage of a perceptron of the library.
In this case, we are trying to find separate a set of points into
two groups. Points are labeled and the perceptron should figure out
the way to separate them.
"""
# parameters
num_of_points = 100
epoch_number = 100
a = -0.4 # slope angle
b = 0.7 # slope offset

# points labeling
set_of_points = {}
for i in range(num_of_points):
    coo = (random(), random())
    ceil = random()
    set_of_points[coo] = 1 if coo[1] > a * coo[0] + b else -1
x1 = []
y1 = []
x2 = []
y2 = []
for key in set_of_points.keys():
    if set_of_points[key] == 1:
        x1.append(key[0])
        y1.append(key[1])
        continue
    x2.append(key[0])
    y2.append(key[1])

# perceptron definition
n_in = 2
n_out = 1
eta = 0.001
af = Treshold()
weights = [af.weight_initialize(n_in, n_out) for _ in range(n_in)]
ef = MSE()
perceptron = Perceptron(eta, weights)

# training 
for i in range(epoch_number):
    for key in set_of_points.keys():
        # compute forward 
        inputs = np.array(key)
        score = perceptron.forward(inputs)
        y_est = af.forward(score)
        # compute backward
        error = ef.backward(set_of_points[key], y_est)
        gradient = af.backward(score)
        # learn
        perceptron.learn(error, gradient)

# success rate
ret = 0
for key in set_of_points.keys():
    inputs = np.array(key)
    score = perceptron.forward(inputs)
    if af.forward(score) == set_of_points[key]:
        ret += 1

# final guess
_a = -perceptron.weights[0] / perceptron.weights[1]
_b = -perceptron.bias / perceptron.weights[1]

print("Success rate : ", ret/num_of_points * 100, "%")
print(f"Target : {a}x + {b} | Guess : {_a:.2f}x + {_b:.2f}")
plt.scatter(x1, y1, c="b", label="Group 1")
plt.scatter(x2, y2, c="r", label="Group 2")
plt.grid()
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot((0,1), (_b, _a + _b), "k", label="Perceptron guess")
plt.legend()
plt.show()