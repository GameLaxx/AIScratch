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
    
    def forward(self, inputs):
        self.last_inputs = inputs.copy() # store for learning  
        return self.activation_function.forward(np.dot(inputs, self.weights) + self.bias)
    
    def learn(self, gradient):
        self.weights += self.eta * gradient * self.last_inputs
        self.bias += self.eta * gradient

