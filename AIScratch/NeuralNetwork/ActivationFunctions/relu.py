from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import numpy as np

class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        if value < 0:
            return 0
        return value
    
    def backward(self, value):
        if value < 0:
            return 0
        return 1

    def weight_initialize(self, n_in):
        return np.random.normal(0, 2 / n_in)