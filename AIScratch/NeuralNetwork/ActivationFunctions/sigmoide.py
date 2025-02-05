from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import numpy as np

class Sigmoïde(ActivationFunction):
    def __init__(self, up = 1, down = 0):
        super().__init__("sigmoïde")
        self.up = up
        self.down = down

    def __core(self, value):
        return np.where(value > 500, self.up, np.where(value < -500, self.down, 1 / (1 + np.exp(-value))))
    
    def forward(self, value):
        return (self.up - self.down) * self.__core(value) + self.down
    
    def backward(self, value):
        return (self.up - self.down) * self.__core(value) * (1 - self.__core(value))

    def weight_initialize(self, n_in = 1, n_out = 1):
        return self._weight_xavier(n_in, n_out)