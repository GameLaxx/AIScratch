from AIScratch.NeuralNetwork import MLP
from AIScratch.NeuralNetwork import DenseLayer, DropoutLayer
from AIScratch.NeuralNetwork import MSE, CrossEntropy
from AIScratch.NeuralNetwork import Softmax, ReLU
from AIScratch.NeuralNetwork import ADAMOptimizer
from random import random
import numpy as np
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def number2output(num):
    ret = [0] * 10
    ret[num] = 1
    return np.array(ret)

# training
def training(network : MLP, training_xs, training_ys, epoch_number : int):
    for epoch in range(epoch_number):
        print("Training number : ", epoch)
        for i in range(len(training_xs)):
            if i % 500 == 0:
                print("Set : ", i)
            inputs = training_xs[i].flatten() / 255
            network.backward(inputs, number2output(training_ys[i]))

# compute network performance
def performance(network : MLP, test_xs, test_ys, confidence_treshold = 0.95):
    ret = 0
    for i in range(len(test_xs)):
        inputs = test_xs[i].flatten() / 255
        forward = network.forward(inputs)
        ret += 1 if np.argmax(forward) == test_ys[i] and forward[test_ys[i]] >= confidence_treshold else 0
    return ret / len(test_xs)

n_in = 784
epoch_number = 5
soft = Softmax()
relu = ReLU()
ef = CrossEntropy()
# ADAM Optimizer
optimizer_factory = lambda n_p1, n_p : ADAMOptimizer(n_p1, n_p, 0.001, 1e-8, 0.9, 0.999)
# SGD Optimizer
# optimizer_factory = lambda n_p1, n_p : SGDOptimizer(n_p1, n_p, 0.001)
load = False
if load:
    layers = []
    mlp = MLP(n_in, layers, ef, optimizer_factory)
    mlp.load("Examples/mlp_image_network.txt")
else:
    layers = [DenseLayer(128, relu), DropoutLayer(64, relu, 0.7), DenseLayer(10, soft)]
    mlp = MLP(n_in, layers, ef, optimizer_factory, 16)
    training(mlp, X_train, y_train, epoch_number)

mlp.extract("Examples/mlp_image_network.txt")

# success rate
print("Trained on a set of length : ", len(X_train))
print("Tested on a set of length : ", len(X_test))
print("Success rate on test : ", performance(mlp, X_test, y_test) * 100, "%")
print("Success rate on training : ", performance(mlp, X_train, y_train) * 100, "%")