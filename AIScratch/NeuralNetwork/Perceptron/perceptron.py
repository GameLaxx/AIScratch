import numpy as np

class Perceptron():
    def __init__(self, eta, weights, bias = 0):
        self.eta = eta
        self.weights = np.array(weights)
        self.bias = bias
    
    def forward(self, inputs):
        self.last_inputs = inputs.copy() # store for learning  
        return np.dot(inputs, self.weights) + self.bias
    
    def learn(self, error, gradient):
        self.weights -= self.eta * error * gradient * self.last_inputs
        self.bias -= self.eta * error * gradient

