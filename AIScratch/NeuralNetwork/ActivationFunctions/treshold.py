from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
import random
class Treshold(ActivationFunction):
    def __init__(self, up = 1, down = 0):
        super().__init__("treshold")
        self.up = up
        self.down = down

    def forward(self, value):
        if value < 0:
            return self.down
        return self.up
    
    def backward(self, value):
        return 1 # not real derivative but works fine

    def weight_initialize(self, n_in = 1, n_out = 1):
        return random.random() * 2 - 1