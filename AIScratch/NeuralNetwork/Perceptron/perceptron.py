import numpy as np
from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction

class Perceptron():
    def __init__(self, n_in, n_out, eta, activation_function : ActivationFunction):
        self.n_in = n_in
        self.n_out = n_out
        self.eta = eta
        self.activation_function = activation_function
        weights = [activation_function.weight_initialize(n_in, n_out) for i in range(n_in)]
        self.weights = np.array(weights)
        self.bias = 0

    def score(self, xs):
        _input = np.array(xs)
        return np.dot(_input, self.weights) + self.bias
    
    def estimate(self, xs):
        return self.activation_function.forward(self.score(xs))
    
    def learn(self, y, xs):
        score = self.score(xs)
        gradient = self.activation_function.backward(y, score)
        _input = np.array(xs)
        self.weights += self.eta * gradient * _input
        self.bias += self.eta * gradient

