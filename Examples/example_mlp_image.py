from AIScratch.NeuralNetwork import MLP
from AIScratch.NeuralNetwork import DenseLayer, DropoutLayer, Conv2DLayer, PoolingLayer, FlattenLayer, PoolingType
from AIScratch.NeuralNetwork import MSE, CrossEntropy
from AIScratch.NeuralNetwork import Softmax, ReLU, Sigmoïde
from AIScratch.NeuralNetwork import ADAMOptimizer
import multiprocessing
import numpy as np
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

NB_CORES = 8

def number2output(num):
    ret = [0] * 10
    ret[num] = 1
    return np.array(ret)

# training
def training(network : MLP, training_xs, training_ys, epoch_number : int):
    for epoch in range(epoch_number):
        print("Training number : ", epoch)
        comp = 0
        for i in range(0, len(training_ys), NB_CORES):
            if i >= comp:
                print("Set : ", comp)
                comp += 500
            processes = []
            for j in range(NB_CORES):
                if i + j >= len(training_ys):
                    break
                p = multiprocessing.Process(target=network.backward, args=([training_xs[i + j]/255], number2output(training_ys[i + j])))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

# compute network performance
def performance(network : MLP, test_xs, test_ys, confidence_treshold = 0.95):
    ret = 0
    for i in range(len(test_xs)):
        inputs = [test_xs[i] / 255]
        forward = network.forward(inputs)
        ret += 1 if np.argmax(forward) == test_ys[i] and forward[test_ys[i]] >= confidence_treshold else 0
    return ret / len(test_xs)

n_in = (1,28,28)
epoch_number = 10
ef = CrossEntropy()
# ADAM Optimizer
optimizer_factory = lambda n_p1, n_p : ADAMOptimizer(n_p1, n_p, 0.0005, 1e-8, 0.9, 0.999)
load = False
if load:
    layers = []
    mlp = MLP(n_in, layers, ef, optimizer_factory)
    mlp.load("Examples/mlp_image_network.txt")
    training(mlp, X_train, y_train, epoch_number)
else:
    layers = [
        Conv2DLayer(3, 0, 1, 8, ReLU()), PoolingLayer(2, 1, 2, 8, PoolingType.MAX), 
        Conv2DLayer(3, 0, 1, 16, Sigmoïde()), PoolingLayer(2, 0, 2, 16, PoolingType.MAX),
        FlattenLayer(),
        DropoutLayer(100, ReLU(), 0.5) ,DenseLayer(10, Softmax())]
    mlp = MLP(n_in, layers, CrossEntropy(), optimizer_factory, batch_size=64)
    training(mlp, X_train, y_train, epoch_number)

mlp.extract("Examples/mlp_image_network.txt") 

# success rate
print("Trained on a set of length : ", len(X_train))
print("Tested on a set of length : ", len(X_test))
print("Success rate on test : ", performance(mlp, X_test, y_test, 0.9) * 100, "%")
print("Success rate on training : ", performance(mlp, X_train, y_train, 0.9) * 100, "%")