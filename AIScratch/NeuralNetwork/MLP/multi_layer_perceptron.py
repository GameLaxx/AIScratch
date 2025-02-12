import numpy as np
from typing import Callable, Any
from AIScratch.NeuralNetwork.Perceptron import Perceptron
from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
from AIScratch.NeuralNetwork.Optimizers import Optimizer
from AIScratch.NeuralNetwork.ErrorFunctions import ErrorFunction
from AIScratch.NeuralNetwork.Layers import Layer
from AIScratch.NeuralNetwork.ActivationFunctions import activation_set
from AIScratch.NeuralNetwork.Layers import layer_set

class MLP():
    def __init__(self, input_number : int, layers : list[Layer], error_function : ErrorFunction, optimizer_factory : Callable[[int, int], Optimizer], batch_size = 1):
        self.input_number = input_number
        self.optimizer_factory = optimizer_factory
        self.layers : list[Layer] = layers
        self.error_function = error_function
        self.batch_size = batch_size
        self.batch_counter = 0
        self.__initialize()

    def __initialize(self):
        prev_size = self.input_number
        for layer in self.layers:
            layer._initialize(prev_size, self.optimizer_factory(prev_size, layer.n_out))
            prev_size = layer.n_out

    def forward(self, inputs, is_training = False):
        outputs = np.asarray(inputs)
        for layer in self.layers:
            outputs = layer.forward(outputs, is_training)
        return outputs
    
    def __errors(self, name_last_layer, expected_outputs, outputs):
        if name_last_layer == "softmax":
            return outputs - expected_outputs
        return self.error_function.backward(expected_outputs, outputs)
    
    def __gradient(self, layer : Layer, errors):
        if layer.activation_function.name == "softmax":
            gradients = 1
        else:
            gradients = np.array([layer.activation_function.backward(z) for z in layer.last_sums]) # f'p(z)
        grad_L_z = errors * gradients # dL/dz = dL/dy * f'p(z)
        layer.store(grad_L_z) # store gradient
        return grad_L_z

    def backward(self, inputs, expected_outputs):
        expected_outputs = np.asarray(expected_outputs)
        inputs = np.asarray(inputs)
        outputs = self.forward(inputs, is_training=True) # all neurons stores the inputs and all layers store activations
        errors = self.__errors(self.layers[-1].activation_function.name, expected_outputs, outputs) # compute errors for last layer
        self.batch_counter += 1
        for p in reversed(range(len(self.layers))): # each layer should compute gradient for itself and error for next
            # current layer computation
            layer = self.layers[p] # layer p
            if layer.is_spatial:
                grad_L_z = self.__gradient(layer, errors)
                if self.batch_counter == self.batch_size:
                    layer.backward()
                # next layer computation
                previous_weights = np.array([neuron.weights for neuron in layer.neurons]) # create matrix of previus layer weights (for next layer its our current one)
                errors = np.dot(grad_L_z, previous_weights) # compute next layer errors
                continue
            if layer.name == "flatten":
                errors = errors.reshape(layer.size_in)
                continue
            if layer.name == "pooling":
                continue
        if self.batch_counter == self.batch_size:
            self.batch_counter = 0

    def extract(self, file_path : str):
        with open(file_path, "w") as f:
            for layer in self.layers:
                f.write(f"*-*{layer.name}//{layer.n_out}//{layer.activation_function.name}\n")
                for neuron in layer.neurons:
                    f.write("|".join(map(str, neuron.weights)))
                    f.write("|" + str(neuron.bias) + "\n")

    def load(self, file_path : str):
        self.layers = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        layer_data = None
        prev_size = self.input_number
        list_of_weights : list[list[float]] = []
        for i in range(len(lines) + 1):
            if i == len(lines) or lines[i].startswith("*-*"):
                if layer_data != None:
                    self.layers.append(
                        layer_set[layer_data["name"]](
                            layer_data["n_out"], 
                            activation_set[layer_data["activation_function"]]
                        )
                    )
                    self.layers[-1]._initialize(prev_size, self.optimizer_factory(prev_size, layer_data["n_out"]), list_of_weights)
                    prev_size = layer_data["n_out"]
                if i == len(lines):
                    break
                layer_data = {}
                list_of_weights = []
                layer_txt = lines[i][3:].strip()
                layer_txt = layer_txt.split("//")
                layer_data["name"] = layer_txt[0]
                layer_data["n_out"] = int(layer_txt[1])
                layer_data["activation_function"] = layer_txt[2]
                continue
            if lines[i][0] == "N":
                continue
            list_of_weights.append([])
            weights_txt = lines[i].strip().split("|")
            for weight in weights_txt:
                list_of_weights[-1].append(float(weight))
