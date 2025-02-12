import numpy as np
from enum import Enum
from AIScratch.NeuralNetwork.SpatialLayers import SpatialLayer
from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction

class Conv2DLayer(SpatialLayer):
    def __init__(self, k, padding, stride, channel_out, activation_function : ActivationFunction):
        super().__init__(k, padding, stride, channel_out, activation_function, "conv2d")

    def _initialize(self, size_in, optimizer, list_of_filters=None):
        self.optimizer = optimizer
        # size init
        self.size_in = size_in # n channels in of size h x w
        self.size_out = \
            (self.channel_out, int((self.size_in[1] - self.k + 2*self.padding) / self.stride) + 1,
              int((self.size_in[2] - self.k + 2*self.padding) / self.stride) + 1)
        self.filter_size = (self.channel_out, self.size_in[0], self.k, self.k)
        # batch init
        self.batch_size = 0
        self.grad_L_w = np.zeros(self.filter_size)  # Weights variations
        self.grad_L_b = np.zeros(self.channel_out) # Biases variations
        # weights init
        if list_of_filters is None:
            self.filters = np.array([self.activation_function.weight_initialize(self.size_in[0] * self.k * self.k, self.size_out[0] * self.k * self.k) for _ in np.ndindex(self.filter_size)]).reshape(self.filter_size) # c channel out for n channel in and filter of size k x k
            self.bias = np.zeros(self.channel_out)
            return
        self.filters = list_of_filters[0]
        self.bias = list_of_filters[1]

    def propagation(self):
        return 

    def forward(self, inputs, is_training=False):
        if self.padding > 0:
            self.last_inputs = np.pad(inputs, ((0, 0), (0, self.padding), (0, self.padding)), mode='constant', constant_values=0) # n channels of size h x w
        else:
            self.last_inputs = inputs # n channels of size h x w
        self.last_sums = np.zeros(self.size_out)
        for c_out in range(self.size_out[0]):  # filter iterations
            for c_in in range(self.size_in[0]):  # channel in iterations
                for i in range(self.size_out[1]):  # image height iterations
                    for j in range(self.size_out[2]):  # image width iterations
                        self.last_sums[c_out, i, j] += np.sum(
                            self.last_inputs[c_in][i*self.stride:i*self.stride+self.k, j*self.stride:j*self.stride+self.k] * self.filters[c_out, c_in, :, :]
                        )
            self.last_sums[c_out] += self.bias[c_out]
        self.last_activations = self.activation_function.forward(self.last_sums)
        return self.last_activations # n channels of size h' x w'
            
    def store(self, grad_L_z):   
        # grad_L_z is n channels of size h' x w' 
        self.grad_L_b += np.sum(grad_L_z, axis=(1, 2)) # grad_L_b is c_out in lengths
        for c_out in range(self.size_out[0]):  # filter iterations
            for c_in in range(self.size_in[0]):  # channel in iterations
                for i in range(self.size_out[1]):  # image height iterations
                    for j in range(self.size_out[2]):  # image width iterations
                        region = self.last_inputs[c_in, i*self.stride:i*self.stride+self.k, j*self.stride:j*self.stride+self.k]
                        self.grad_L_w[c_out, c_in] += region * grad_L_z[c_out, i, j]
        self.batch_size += 1
    
    def backward(self):
        learning_rates, weighted_errors, biais_update = self.optimizer.optimize(
            self.grad_L_w / self.batch_size, self.grad_L_b / self.batch_size
        )
        self.filters -= learning_rates * weighted_errors
        self.bias -= biais_update
        # update for next batch
        self.grad_L_w.fill(0)
        self.grad_L_b.fill(0)
        self.batch_size = 0