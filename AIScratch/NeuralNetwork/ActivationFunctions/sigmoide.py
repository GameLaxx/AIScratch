from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import random
import numpy as np

class Sigmoïde(ActivationFunction):
    def __init__(self, up, down):
        super().__init__()
        self.up = up
        self.down = down

    def __core(self, value):
        return 1 / (1 + np.exp(-value))
    
    def forward(self, value):
        return (self.up - self.down) * self.__core(value) + self.down
    
    def backward(self, value):
        return (self.up - self.down) * self.__core(value) * (1 - self.__core(value))

    def weight_initialize(self, n_in = 1, n_out = 1):
        bound = 6 ** 0.5 / (n_in + n_out) ** 0.5
        return random.random() * 2 * bound - bound