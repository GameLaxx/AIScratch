from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import numpy as np

class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, score):
        if score < 0:
            return 0
        return score
    
    def backward(self, y, score):
        if score < 0:
            return 0
        return self.forward(score) - y

    def weight_initialize(self, n_in):
        return np.random.normal(0, 2 / n_in)