from AIScratch.NeuralNetwork import Perceptron, Treshold
from random import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

"""
Example of usage of a perceptron of the library.
In this case, we are trying to find separate a set of points into
two groups. Points are labeled and the perceptron should figure out
the way to separate them.
NO_CONSTRAINT tries to find the run that is the closest to the distance.
QUICK tries to find the run that arrives at the distance the fastest.
EASY tries to find the run that asks for the less efforts.
BOTH tries to find the run that is the fastest while asking for the less efforts.
Each element is a list of time in h and each speed is in km/h.
"""
# parameters
num_of_points = 1000
epoch_number = 200
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
af = Treshold()
perceptron = Perceptron(2, 1, 0.001, af)

# training 
for i in range(epoch_number):
    for key in set_of_points.keys():
        perceptron.learn(set_of_points[key], key)

# success rate
ret = 0
for key in set_of_points.keys():
    if perceptron.estimate(key) == set_of_points[key]:
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