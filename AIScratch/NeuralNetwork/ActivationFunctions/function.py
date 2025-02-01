from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, score : float) -> float:
        pass

    @abstractmethod
    def backward(self, y, score) -> float:
        pass

    @abstractmethod
    def weight_initialize(self, n_in = 1, n_out = 1) -> float:
        pass