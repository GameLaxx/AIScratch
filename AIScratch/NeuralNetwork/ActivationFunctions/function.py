from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, value : float) -> float:
        pass

    @abstractmethod
    def backward(self, value) -> float:
        pass

    @abstractmethod
    def weight_initialize(self, n_in = 1, n_out = 1) -> float:
        pass