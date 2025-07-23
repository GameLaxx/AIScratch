from AIScratch.DecisionTree.PurityFunction import PurityFunction
import pandas as pd

class GiniPurity(PurityFunction):
    def __init__(self):
        pass

    def purity(self, data_frame : pd.DataFrame, sep : str):
        class_counts = data_frame[sep].value_counts()
        total = len(data_frame)
        gini = 1.0
        for count in class_counts:
            prob = count / total
            gini -= prob ** 2
        return gini
    
    def purity_dict(self, decision_dict):
        return 1 - sum(p**2 for p in decision_dict.values())