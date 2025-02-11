import numpy as np
from enum import Enum
from AIScratch.NeuralNetwork.Layers import SpatialLayer

class PoolingType(Enum):
    MAX = 1
    AVERAGE = 2

class PoolingLayer(SpatialLayer):
    def __init__(self, size_in, k, padding, stride, channel_out, pooling_type : PoolingType):
        super().__init__(size_in, k, padding, stride, channel_out, None, "pooling")
        if pooling_type == PoolingType.MAX:
            self.pooling = np.max
        else:
            self.pooling = np.average

    def _initialize(self, optimizer, list_of_filters=None):
        pass

    def forward(self, inputs, is_training=False):
        ret = []
        if len(inputs) != self.channel_out:
            raise ValueError("The number of channels in the input must be equal to the number of channels in the layer.")
        for i in range(len(inputs)):
            ret.append([])
            for j in range(0, len(inputs[i]), self.stride):
                ret[i].append([])
                for l in range(0, len(inputs[i][0]), self.stride):
                    ret[i][-1].append(self.pooling(inputs[i][j:j+self.k, l:l+self.k]))
        return np.array(ret)
            
    
    def store(self, grad_L_z):
        pass
    
    def backward(self):
        pass