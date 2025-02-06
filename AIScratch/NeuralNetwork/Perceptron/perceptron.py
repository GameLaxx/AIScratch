import numpy as np

class Perceptron():
    def __init__(self, eta, weights, bias = 0):
        self.eta = eta
        self.weights = np.array(weights)
        self.bias = bias
    
    def forward(self, inputs):# store for learning  
        return np.dot(inputs, self.weights) + self.bias
    
    def learn(self, error, gradient, inputs):
        self.weights -= self.eta * error * gradient * inputs
        self.bias -= self.eta * error * gradient

