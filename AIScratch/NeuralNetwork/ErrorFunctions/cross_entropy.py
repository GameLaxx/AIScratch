from AIScratch.NeuralNetwork.ErrorFunctions import ErrorFunction
import numpy as np

class CrossEntropy(ErrorFunction):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_est):
        return - y * np.log(y_est)
    
    def backward(self, y, y_est):
        return - y / y_est