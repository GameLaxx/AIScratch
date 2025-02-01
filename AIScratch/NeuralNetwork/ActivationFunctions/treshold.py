from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import random
class Treshold(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        if value < 0:
            return -1
        return 1
    
    def backward(self, value):
        return 1

    def weight_initialize(self, n_in = 1, n_out = 1):
        return random.random() * 2 - 1