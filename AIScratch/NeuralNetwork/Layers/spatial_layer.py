from abc import ABC, abstractmethod
from AIScratch.NeuralNetwork.Optimizers import Optimizer
from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction

class SpatialLayer(ABC):
    def __init__(self, size_in, k, padding, stride, channel_out, activation_function, name):
        self.size_in : int = size_in
        self.k : int = k
        self.padding : int = padding
        self.stride : int = stride
        self.channel_out : int = channel_out
        self.activation_function : ActivationFunction = activation_function
        self.optimizer : Optimizer = None
        self.last_sums : list[float] = None
        self.last_activations : list[float] = None
        self.filters : list[list[float]] = []
        self.biases : list[float] = []
        self.name = name
    
    @abstractmethod
    def _initialize(self, optimizer : Optimizer, list_of_filters = None):
        pass
    
    @abstractmethod
    def forward(self, inputs, is_training = False):
        pass

    @abstractmethod
    def store(self, grad_L_z):
        pass

    @abstractmethod
    def backward(self):
        pass

