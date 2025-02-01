from AIScratch.NeuralNetwork.ErrorFunctions import ErrorFunction

class HingeLoss(ErrorFunction):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_est):
        tmp = 1 - y * y_est
        if tmp < 0:
            return 0
        return tmp
    
    def backward(self, y, y_est):
        tmp = 1 - y * y_est
        if tmp <= 0:
            return 0
        return -y