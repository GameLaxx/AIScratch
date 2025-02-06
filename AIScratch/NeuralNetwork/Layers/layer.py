from abc import ABC, abstractmethod
from AIScratch.NeuralNetwork.Perceptron import Perceptron
from AIScratch.NeuralNetwork.Optimizers import Optimizer
from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction

class Layer(ABC):
    def __init__(self, n_out, activation_function, name):
        self.n_out : int = n_out
        self.activation_function : ActivationFunction = activation_function
        self.optimizer : Optimizer = None
        self.last_sums : list[float] = None
        self.last_activations : list[float] = None
        self.neurons : list[Perceptron] = []
        self.name = name
    
    @abstractmethod
    def _initialize(self, n_in, optimizer : Optimizer, list_of_weights = None):
        pass
    
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def learn(self, errors, gradients):
        pass

