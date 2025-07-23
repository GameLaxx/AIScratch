import pandas as pd
from typing import Any
from abc import ABC, abstractmethod

class PurityFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def purity(self, data_frame : pd.DataFrame, sep):
        pass
    
    @abstractmethod
    def purity_dict(self, decision_dict : dict[Any, float]):
        pass