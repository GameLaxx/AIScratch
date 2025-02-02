from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import numpy as np

class ReLU(ActivationFunction):
    def __init__(self, treshold = 0, slope = 1):
        super().__init__("relu")
        self.treshold = treshold
        self.slope = slope

    def forward(self, value):
        if value < self.treshold:
            return 0
        return self.slope * value
    
    def backward(self, value):
        if value < self.treshold:
            return 0
        return self.slope

    def weight_initialize(self, n_in = 1, n_out = 1):
        return np.random.normal(0, 2 / n_in)