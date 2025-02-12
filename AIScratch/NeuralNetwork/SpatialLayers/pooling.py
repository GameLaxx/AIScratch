import numpy as np
from enum import Enum
from AIScratch.NeuralNetwork.SpatialLayers import SpatialLayer

class PoolingType(Enum):
    MAX = 1
    AVERAGE = 2

class PoolingLayer(SpatialLayer):
    def __init__(self, k, padding, stride, channel_out, pooling_type : PoolingType):
        super().__init__(k, padding, stride, channel_out, None, "pooling")
        self.pooling_type = pooling_type
        if pooling_type == PoolingType.MAX:
            self.pooling = np.max
        else:
            self.pooling = np.average

    def _initialize(self, size_in, optimizer, list_of_filters=None):
        self.size_in = size_in
        self.size_out = \
            (self.channel_out, int((self.size_in[1] - self.k + 2*self.padding) / self.stride) + 1,
              int((self.size_in[2] - self.k + 2*self.padding) / self.stride) + 1)
        if self.pooling_type == PoolingType.MAX:
            self.argmax = np.zeros(self.size_out+(2,), dtype=int)

    def forward(self, inputs, is_training=False):
        ret = []
        if len(inputs) != self.channel_out:
            raise ValueError("The number of channels in the input must be equal to the number of channels in the layer.")
        for i in range(self.size_out[0]):
            ret.append([])
            for j in range(self.size_out[1]):
                ret[i].append([])
                for l in range(self.size_out[2]):
                    if self.pooling_type == PoolingType.MAX:
                        vector_index = np.argmax(inputs[i][j*self.stride:j*self.stride+self.k, l*self.stride:l*self.stride+self.k])
                        self.argmax[i][j][l] = np.unravel_index(vector_index, (self.k, self.k))
                    ret[i][j].append(self.pooling(inputs[i][j*self.stride:j*self.stride+self.k, l*self.stride:l*self.stride+self.k]))
        return np.array(ret)

    def propagation(self, grad_L_z):
        ret = np.zeros(self.size_in)
        for i in range(len(grad_L_z)):
            for j in range(len(grad_L_z[i])):
                for l in range(len(grad_L_z[i][j])):
                    if self.pooling_type == PoolingType.MAX:
                        ret[i,j*self.stride + self.argmax[i,j,l][0], l*self.stride + self.argmax[i,j,l][1]] = grad_L_z[i][j][l]
                        continue
                    ret[i,j*self.stride:j*self.stride+self.k, l*self.stride:l*self.stride+self.k] = grad_L_z[i][j][l] / (self.k * self.k)
        return ret
    
    def store(self, grad_L_z):
        pass
    
    def backward(self):
        pass