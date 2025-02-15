from AIScratch.NeuralNetwork import MLP
from AIScratch.NeuralNetwork import DenseLayer, DropoutLayer, Conv2DLayer, PoolingLayer, FlattenLayer, PoolingType
from AIScratch.NeuralNetwork import MSE, CrossEntropy
from AIScratch.NeuralNetwork import Softmax, ReLU, Sigmoïde
from AIScratch.NeuralNetwork import ADAMOptimizer
import multiprocessing
import functools
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
            inputs = [training_xs[i] / 255]
            network.backward(inputs, number2output(training_ys[i]))

# compute network performance
def performance(network : MLP, test_xs, test_ys, confidence_treshold = 0.95):
    ret = 0
    for i in range(len(test_xs)):
        inputs = [test_xs[i] / 255]
        forward = network.forward(inputs)
        ret += 1 if np.argmax(forward) == test_ys[i] and forward[test_ys[i]] >= confidence_treshold else 0
    return ret / len(test_xs)

n_in = (1,28,28)
epoch_number = 1
soft = Softmax()
ef = CrossEntropy()
# ADAM Optimizer
optimizer_factory = lambda n_p1, n_p : ADAMOptimizer(n_p1, n_p, 0.001, 1e-8, 0.9, 0.999)
load = True
if load:
    layers = []
    mlp = MLP(n_in, layers, ef, optimizer_factory)
    mlp.load("Examples/mlp_image_network.txt")
else:
    layers = [
        Conv2DLayer(5, 0, 1, 2, ReLU()), PoolingLayer(2, 0, 2, 2, PoolingType.MAX), 
        Conv2DLayer(3, 0, 1, 4, Sigmoïde()), PoolingLayer(2, 0, 2, 4, PoolingType.MAX),
        FlattenLayer(), DenseLayer(10, Softmax())]
    mlp = MLP(n_in, layers, CrossEntropy(), optimizer_factory, batch_size=64)
    training(mlp, X_train, y_train, epoch_number)

mlp.extract("Examples/mlp_image_network.txt") 

# success rate
print("Trained on a set of length : ", len(X_train))
print("Tested on a set of length : ", len(X_test))
print("Success rate on test : ", performance(mlp, X_test, y_test, 0.9) * 100, "%")
print("Success rate on training : ", performance(mlp, X_train, y_train, 0.9) * 100, "%")