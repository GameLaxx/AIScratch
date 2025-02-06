import numpy as np
from AIScratch.NeuralNetwork.Layers import Layer
from AIScratch.NeuralNetwork.Perceptron import Perceptron

class DenseLayer(Layer):
    def __init__(self, n_out, activation_function):
        super().__init__(n_out, activation_function, "Dense")

    def _initialize(self, n_in, optimizer, list_of_weights = None):
        self.optimizer = optimizer
        for i in range(self.n_out):
            if list_of_weights == None:
                weights = [self.activation_function.weight_initialize(n_in, self.n_out) for _ in range(n_in)]
                bias = 0
            else:
                weights = list_of_weights[i][:-1]
                bias = list_of_weights[i][-1]
            self.neurons.append(Perceptron(weights, bias))

    def forward(self, inputs):
        self.last_inputs = inputs.copy() 
        self.last_sums = np.array([neuron.forward(self.last_inputs) for neuron in self.neurons])
        self.last_activations = self.activation_function.forward(self.last_sums)
        return self.last_activations
    
    def learn(self, errors, gradients):
        learning_rates, weighted_errors, biais_update = self.optimizer.optimize(errors, gradients, self.last_inputs)
        for i in range(self.n_out):
            self.neurons[i].learn(learning_rates[i], weighted_errors[i], biais_update[i])