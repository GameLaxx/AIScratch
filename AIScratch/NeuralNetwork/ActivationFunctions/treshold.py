from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import numpy as np

class Treshold(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        if value < 0:
            return -1
        return 1
    
    def backward(self, y, value):
        return y - self.forward(value)

    def weight_initialize(self, n_in = 1, n_out = 1):
        return np.random.uniform(-1, 1)