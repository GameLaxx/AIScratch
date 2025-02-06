from AIScratch.NeuralNetwork.Optimizers import Optimizer
import numpy as np

class SGDOptimizer(Optimizer):
    def __init__(self, n_p1, n_p, eta):
        super().__init__()
        self.eta = eta
        self.learning_rates = eta * np.ones(n_p)

    def optimize(self, errors, gradients, inputs):
        weighted_errors = errors * gradients
        return self.learning_rates, np.outer(weighted_errors, inputs), self.eta * weighted_errors