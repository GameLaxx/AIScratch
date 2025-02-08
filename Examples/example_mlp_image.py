from AIScratch.NeuralNetwork import MLP, DenseLayer, Softmax, Linear, ReLU, CrossEntropy, ADAMOptimizer, SGDOptimizer
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
        for i in range(len(training_xs) // 3):
            if i % 500 == 0:
                print("Set : ", i)
            inputs = training_xs[i].flatten() / 255
            network.backward(inputs, number2output(training_ys[i]))

# compute network performance
def performance(network : MLP, test_xs, test_ys, confidence_treshold = 0.95):
    ret = 0
    for i in range(len(test_xs)):
        inputs = test_xs[i].flatten() / 255
        ret += 1 if np.argmax(network.forward(inputs)) == test_ys[i] else 0
    return ret / len(test_xs)

n_in = 784
epoch_number = 2
soft = Softmax()
lin = Linear()
relu = ReLU()
ef = CrossEntropy()
# ADAM Optimizer
optimizer_factory = lambda n_p1, n_p : ADAMOptimizer(n_p1, n_p, 0.001, 1e-8, 0.9, 0.999)
# SGD Optimizer
# optimizer_factory = lambda n_p1, n_p : SGDOptimizer(n_p1, n_p, 0.001)
load = False
if load:
    layers = []
    name_to_layer = {"Dense": DenseLayer}
    name_to_function = {"softmax": soft, "relu" : relu}
    mlp = MLP(n_in, layers, ef, optimizer_factory)
    mlp.load("Examples/mlp_image_network.txt", name_to_layer, name_to_function)
else:
    layers = [DenseLayer(128, relu), DenseLayer(64, relu), DenseLayer(10, soft)]
    mlp = MLP(n_in, layers, ef, optimizer_factory, 1)
    training(mlp, X_train, y_train, 1)
# mlp.extract("Examples/mlp_image_network.txt")
# ------------ 0
print(mlp.forward(X_train[0].flatten() / 255), y_train[0], np.argmax(mlp.forward(X_train[0].flatten() / 255)))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
print(mlp.forward(X_train[0].flatten() / 255), y_train[0], np.argmax(mlp.forward(X_train[0].flatten() / 255)))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
print(mlp.forward(X_train[0].flatten() / 255), y_train[0], np.argmax(mlp.forward(X_train[0].flatten() / 255)))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
mlp.backward(X_train[0].flatten() / 255, number2output(y_train[0]))
print(mlp.forward(X_train[0].flatten() / 255), y_train[0], np.argmax(mlp.forward(X_train[0].flatten() / 255)))
# ------------ 1
print(mlp.forward(X_train[1].flatten() / 255), y_train[1], np.argmax(mlp.forward(X_train[1].flatten() / 255)))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
print(mlp.forward(X_train[1].flatten() / 255), y_train[1], np.argmax(mlp.forward(X_train[1].flatten() / 255)))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
print(mlp.forward(X_train[1].flatten() / 255), y_train[1], np.argmax(mlp.forward(X_train[1].flatten() / 255)))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
print(mlp.forward(X_train[1].flatten() / 255), y_train[1], np.argmax(mlp.forward(X_train[1].flatten() / 255)))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
print(mlp.forward(X_train[1].flatten() / 255), y_train[1], np.argmax(mlp.forward(X_train[1].flatten() / 255)))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
print(mlp.forward(X_train[1].flatten() / 255), y_train[1], np.argmax(mlp.forward(X_train[1].flatten() / 255)))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
mlp.backward(X_train[1].flatten() / 255, number2output(y_train[1]))
print(mlp.forward(X_train[1].flatten() / 255), y_train[1], np.argmax(mlp.forward(X_train[1].flatten() / 255)))
print(mlp.forward(X_train[0].flatten() / 255), y_train[0], np.argmax(mlp.forward(X_train[0].flatten() / 255)))
# success rate
print("Trained on a set of length : ", len(X_train))
print("Tested on a set of length : ", len(X_test))
print("Success rate on training : ", performance(mlp, X_train, y_train) * 100, "%")
print("Success rate on test : ", performance(mlp, X_test, y_test) * 100, "%")